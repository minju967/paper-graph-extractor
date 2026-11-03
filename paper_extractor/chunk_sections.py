"""제목 계층(JSON)의 섹션을 입력 한도에 맞춰 청크로 묶는다. LLM은 호출하지 않는다.

규칙
  - 제목 하나와 그 바로 아래 본문을 한 섹션(원자 단위)으로 본다. 한도보다 긴 섹션만 나눈다.
  - 섹션을 문서 순서대로 채우다가 다음 섹션이 한도를 넘기면 그 섹션 앞에서 청크를 끝낸다. 섹션 중간에서 자르지 않는다.
  - 자를 수밖에 없을 때는 가능하면 더 높은 레벨 경계(예: 연구 방법 끝, 결과 시작)에서 자른다.
  - 한도보다 긴 섹션은 문단 경계에서 나눈다. 표는 중간에서 자르지 않는다(표 하나가 한도보다 크면 행 단위로 나누고 헤더를 반복).
  - 각 섹션에 제목 경로(예: 연구 방법 > 측정 항목과 방법 > 혈압)를 붙인다.
  - 참고문헌은 청크에서 제외한다(`--keep-references`로 포함).
  - 청크 본문이 언급하는 그림(Figure N·그림 N)은 PDF에서 찾아 PNG로 저장하고 해당 청크에 연결한다.
  - Markdown 표가 깨져 있으면(셀 안에 줄바꿈이 흩어짐) 그 표 영역도 PDF에서 잘라 PNG로 저장해 해당 청크에 연결한다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .chunking import estimate_tokens
from .config import Config
from .figures import layout_pages, locate, mentioned_numbers, save_png

MIN_FILL = 0.5  # 상위 경계로 자를 때 앞 청크가 최소한 채워야 하는 예산 비율
CONTINUATION_LEVEL = 99  # 나뉜 섹션의 이어지는 조각: 이 앞에서는 가급적 자르지 않는다
_REFERENCE_RE = re.compile(r"^(references?|참고\s*문헌)$", re.IGNORECASE)
_END_SECTION_RE = re.compile(
    r"^(acknowledg(?:e)?ments?|disclosures?|conflicts? of interest|"
    r"the conflict of interest statement|author contributions?|연구 참여자 세부역할)$",
    re.IGNORECASE,
)


@dataclass
class Atom:
    """청크에 들어가는 단위: 섹션 하나 또는 나뉜 섹션의 한 조각."""

    section_id: str
    level: int  # 이 앞에서 자를 때의 경계 레벨(작을수록 상위)
    title: str
    path: str
    line: int
    text: str  # 렌더링된 전체 텍스트(제목·경로·본문)
    tokens: int
    part: tuple[int, int] | None = None  # (현재, 전체) — 나뉜 섹션일 때만


@dataclass
class Chunk:
    index: int
    atoms: list[Atom]

    @property
    def tokens(self) -> int:
        return sum(a.tokens for a in self.atoms)

    def text(self) -> str:
        return "\n\n".join(a.text for a in self.atoms)


# --- 섹션 목록 -------------------------------------------------------------

def flatten_outline(outline: list[dict]) -> list[dict]:
    """계층 트리를 문서 순서의 평면 목록으로 만들고 제목 경로를 붙인다. 논문 제목(레벨 1)은 경로에서 뺀다."""
    out: list[dict] = []

    def walk(nodes: list[dict], trail: list[str]) -> None:
        for n in nodes:
            own = [] if n["level"] <= 1 else [n["title"]]
            path = trail + own
            out.append({**{k: n[k] for k in ("id", "level", "title", "line", "body")}, "path": " > ".join(path) or n["title"]})
            walk(n["children"], trail + ([] if n["level"] <= 1 else [n["title"]]))

    walk(outline, [])
    return sorted(out, key=lambda s: s["line"])


def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()


def is_reference_section(title: str) -> bool:
    return bool(_REFERENCE_RE.match(_norm_title(title)))


def is_end_section(title: str) -> bool:
    return bool(_END_SECTION_RE.match(_norm_title(title)))


# --- 렌더링 · 나누기 --------------------------------------------------------

def _render(level: int, title: str, path: str, body: str, part: tuple[int, int] | None = None) -> str:
    marks = "#" * level if level > 0 else "-"
    suffix = f" (이어짐 {part[0]}/{part[1]})" if part else ""
    head = f"{marks} {title}{suffix}\n[경로: {path}]"
    return f"{head}\n{body}" if body.strip() else head


def _is_table(par: str) -> bool:
    lines = [l for l in par.splitlines() if l.strip()]
    return bool(lines) and all(l.lstrip().startswith("|") for l in lines)


def _split_table(par: str, limit: int) -> list[str]:
    """한도보다 큰 표를 행 단위로 나누고 헤더(첫 두 줄)를 반복한다."""
    lines = [l for l in par.splitlines() if l.strip()]
    header, rows = lines[:2], lines[2:]
    out: list[list[str]] = []
    cur: list[str] = []
    for row in rows:
        if cur and estimate_tokens("\n".join(header + cur + [row])) > limit:
            out.append(cur)
            cur = []
        cur.append(row)
    if cur:
        out.append(cur)
    return ["\n".join(header + rs) for rs in out] or [par]


def _split_text(par: str, limit: int) -> list[str]:
    """한도보다 큰 문단을 줄 단위로 나눈다(한 줄이 더 길면 글자 수로 자른다)."""
    pieces: list[str] = []
    cur: list[str] = []
    for line in par.splitlines():
        if estimate_tokens(line) > limit:
            if cur:
                pieces.append("\n".join(cur))
                cur = []
            step = max(200, limit * 2)  # 토큰 한도에 맞춘 대략의 글자 수
            pieces += [line[i : i + step] for i in range(0, len(line), step)]
            continue
        if cur and estimate_tokens("\n".join(cur + [line])) > limit:
            pieces.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def split_oversize_body(body: str, limit: int) -> list[str]:
    """본문을 문단 경계에서 한도 이하 조각으로 나눈다. 표는 가능한 한 통째로 둔다."""
    units: list[str] = []
    for par in re.split(r"\n\s*\n", body):
        if not par.strip():
            continue
        if estimate_tokens(par) <= limit:
            units.append(par)
        else:
            units += _split_table(par, limit) if _is_table(par) else _split_text(par, limit)
    pieces: list[str] = []
    cur: list[str] = []
    for u in units:
        if cur and estimate_tokens("\n\n".join(cur + [u])) > limit:
            pieces.append("\n\n".join(cur))
            cur = []
        cur.append(u)
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces


_FALSE_HEADING_RE = re.compile(r"^(\s*)#{1,6}[ \t]+")


def strip_false_headings(body: str) -> str:
    """본문 줄 앞의 `#`를 뗀다. 제목으로 채택된 줄은 본문에 없고 섹션 머리로 따로 렌더링되므로,
    본문에 남은 `#`는 변환기가 잘못 붙인 문장 조각·저자명 등이다."""
    return "\n".join(_FALSE_HEADING_RE.sub(r"\1", line) for line in body.splitlines())


def make_atoms(sections: list[dict], budget: int) -> list[Atom]:
    atoms: list[Atom] = []
    for s in sections:
        s = {**s, "body": strip_false_headings(s["body"])}
        text = _render(s["level"], s["title"], s["path"], s["body"])
        tokens = estimate_tokens(text)
        if tokens <= budget:
            atoms.append(Atom(s["id"], s["level"], s["title"], s["path"], s["line"], text, tokens))
            continue
        # 머리(제목·경로)가 차지하는 몫을 빼고 본문을 나눈다.
        head_tokens = estimate_tokens(_render(s["level"], s["title"], s["path"], "", (99, 99)))
        pieces = split_oversize_body(s["body"], max(200, budget - head_tokens - 5))
        for i, piece in enumerate(pieces, start=1):
            part = (i, len(pieces))
            t = _render(s["level"], s["title"], s["path"], piece, part)
            atoms.append(
                Atom(s["id"], s["level"] if i == 1 else CONTINUATION_LEVEL, s["title"], s["path"], s["line"], t, estimate_tokens(t), part)
            )
    return atoms


# --- 묶기 -------------------------------------------------------------------

def pack(atoms: list[Atom], budget: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[Atom] = []

    def emit(upto: int) -> None:
        nonlocal cur
        chunks.append(Chunk(len(chunks) + 1, cur[:upto]))
        cur = cur[upto:]

    def best_cut(incoming: Atom) -> int:
        """cur 안에서 자를 위치. 앞 청크가 MIN_FILL 이상 차는 후보 중 뒤에 오는 섹션의 레벨이 가장 낮은(상위) 곳, 같으면 더 뒤쪽."""
        best, best_key = len(cur), (incoming.level, -len(cur))
        acc = 0
        for i in range(1, len(cur)):
            acc += cur[i - 1].tokens
            if acc < budget * MIN_FILL:
                continue
            key = (cur[i].level, -i)
            if key < best_key:
                best, best_key = i, key
        return best

    for atom in atoms:
        while cur and sum(a.tokens for a in cur) + atom.tokens > budget:
            emit(best_cut(atom))
        cur.append(atom)
    if cur:
        emit(len(cur))
    return chunks


# --- 실행 -------------------------------------------------------------------

def output_stem(path: Path) -> str:
    m = re.match(r"\d+", unicodedata.normalize("NFC", path.name))
    return m.group() if m else path.stem[:20]


def describe(chunks: list[Chunk], budget: int) -> list[str]:
    lines: list[str] = []
    for c in chunks:
        lines.append(f"  청크 {c.index}: {c.tokens} 토큰 / 섹션 {len(c.atoms)}개")
        for a in c.atoms:
            part = f" (이어짐 {a.part[0]}/{a.part[1]})" if a.part else ""
            lines.append(f"      L{a.line:<4} {'#' * max(a.level, 1) if a.level < CONTINUATION_LEVEL else '…'} {a.path}{part}  [{a.tokens}]")
    return lines


def find_pdf(pdf_dir: Path, stem: str) -> Path | None:
    for f in sorted(pdf_dir.glob("*.pdf")):
        if re.match(rf"0*{int(stem)}(?!\d)", unicodedata.normalize("NFC", f.name)) if stem.isdigit() else False:
            return f
    return None


_TABLE_LABEL_RE = re.compile(r"(?:Table|표)\s*(\d+)", re.IGNORECASE)
BROKEN_MIN_LINES = 2  # 표 블록 안에서 `|`로 시작하지 않는 줄(셀 안 줄바꿈)이 이 수 이상이면 깨진 표로 본다


def broken_tables(text: str) -> list[int]:
    """청크 텍스트의 Markdown 표 중 셀 안 줄바꿈으로 구조가 깨진 표의 번호."""
    paras = re.split(r"\n\s*\n", text)
    out: list[int] = []
    for i, par in enumerate(paras):
        lines = par.splitlines()
        if not lines or not lines[0].lstrip().startswith("|"):
            continue
        if sum(1 for l in lines if l.strip() and not l.lstrip().startswith("|")) < BROKEN_MIN_LINES:
            continue
        m = _TABLE_LABEL_RE.search(lines[0]) or (_TABLE_LABEL_RE.search(paras[i - 1]) if i > 0 else None)
        if m and int(m.group(1)) not in out:
            out.append(int(m.group(1)))
    return out


def attach_figures(chunks: list[Chunk], pdf: Path | None, fig_dir: Path, rel_dir: str) -> dict[int, list[dict]]:
    """청크별로 연결할 이미지를 PDF에서 찾아 PNG로 저장하고 목록을 돌려준다.

    - 그림: 청크 본문이 언급한 Figure N
    - 표: 청크 본문의 Markdown 표가 깨져 있는 Table N
    """
    wanted_fig = {c.index: mentioned_numbers(c.text()) for c in chunks}
    wanted_tab = {c.index: broken_tables(c.text()) for c in chunks}
    result: dict[int, list[dict]] = {c.index: [] for c in chunks}
    figs = {n for ns in wanted_fig.values() for n in ns}
    tabs = {n for ns in wanted_tab.values() for n in ns}
    if not figs and not tabs:
        return result
    pages = layout_pages(pdf) if pdf else None
    located = {
        "figure": locate(pdf, figs, "figure", pages) if pdf and figs else {},
        "table": locate(pdf, tabs, "table", pages) if pdf and tabs else {},
    }
    for kind, label in (("figure", "Figure"), ("table", "Table")):
        for num, info in located[kind].items():
            save_png(pdf, info, fig_dir / f"{kind}{num}.png")
    for idx in result:
        for kind, label, wanted in (("figure", "Figure", wanted_fig), ("table", "Table", wanted_tab)):
            for n in wanted[idx]:
                info = located[kind].get(n)
                result[idx].append(
                    {
                        "label": f"{label} {n}",
                        "kind": kind,
                        "reason": "mentioned" if kind == "figure" else "broken_markdown_table",
                        "located": info.located if info else "not_found",
                        "page": info.page if info else None,
                        "caption": info.caption if info else None,
                        "image": f"{rel_dir}/{kind}{n}.png" if info else None,
                    }
                )
    return result


def process(json_path: Path, budget: int, out_dir: Path, skip_end: bool, keep_references: bool, pdf_dir: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    sections = flatten_outline(data["outline"])
    dropped = [s["title"] for s in sections if (not keep_references and is_reference_section(s["title"])) or (skip_end and is_end_section(s["title"]))]
    sections = [s for s in sections if s["title"] not in dropped]
    atoms = make_atoms(sections, budget)
    chunks = pack(atoms, budget)
    stem = output_stem(json_path)
    pdf = find_pdf(pdf_dir, stem)
    figures = attach_figures(chunks, pdf, out_dir / f"{stem}_figures", f"{stem}_figures")
    print(f"{stem}: 섹션 {len(sections)}개 -> 청크 {len(chunks)}개 (예산 {budget} 토큰)" + (f", 제외: {', '.join(dropped)}" if dropped else ""))
    for line in describe(chunks, budget):
        print(line)
    for idx, figs in figures.items():
        for f in figs:
            print(f"  청크 {idx} 이미지: {f['label']} -> {f['located']}" + (f" (p{f['page']})" if f["page"] else "") + (" [깨진 표]" if f["kind"] == "table" else ""))
    if any(figures.values()) and not pdf:
        print(f"  [경고] {stem}번 PDF를 {pdf_dir}에서 찾지 못해 그림을 저장하지 못했습니다.")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}_chunks.json").write_text(
        json.dumps(
            {
                "source": data.get("source"),
                "budget_tokens": budget,
                "skipped_sections": dropped,
                "chunks": [
                    {
                        "index": c.index,
                        "tokens": c.tokens,
                        "sections": [
                            {"id": a.section_id, "path": a.path, "level": a.level if a.level < CONTINUATION_LEVEL else None, "line": a.line, "tokens": a.tokens, "part": list(a.part) if a.part else None}
                            for a in c.atoms
                        ],
                        "figures": figures[c.index],
                        "text": c.text(),
                    }
                    for c in chunks
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    md = []
    for c in chunks:
        figs = "".join(f"\n![{f['label']}]({f['image']})" for f in figures[c.index] if f["image"])
        md.append(f"<!-- ===== 청크 {c.index} · {c.tokens} 토큰 ===== -->\n\n{c.text()}\n{figs}\n")
    (out_dir / f"{stem}_chunks.md").write_text("\n".join(md), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="제목 계층 JSON의 섹션을 입력 한도에 맞춰 청크로 묶습니다(LLM 호출 없음).")
    parser.add_argument("paths", nargs="*", type=Path, default=[base / "outputs" / "titles"], help="*_title.json 파일 또는 폴더")
    parser.add_argument("--out", type=Path, default=base / "outputs" / "chunks", help="청크 저장 폴더")
    parser.add_argument("--budget", type=int, default=Config().chunk_target_tokens, help="청크 하나의 토큰 예산(기본: 설정의 chunk_target_tokens)")
    parser.add_argument("--pdf-dir", type=Path, default=Path(__file__).resolve().parent / "papers", help="그림을 잘라낼 원본 PDF 폴더")
    parser.add_argument("--keep-references", action="store_true", help="참고문헌을 청크에 포함(기본은 제외)")
    parser.add_argument("--skip-end-sections", action="store_true", help="참고문헌·감사의 글 등 끝부분 섹션을 제외")
    args = parser.parse_args(argv)

    files: list[Path] = []
    for p in args.paths:
        files += sorted(p.glob("*_title.json")) if p.is_dir() else [p]
    if not files:
        print("제목 계층 JSON을 찾지 못했습니다.", file=sys.stderr)
        return 1
    for f in files:
        process(f, args.budget, args.out, args.skip_end_sections, args.keep_references, args.pdf_dir)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
