"""LLM으로 논문 제목 계층을 추출하고 섹션별 본문을 매핑한다.

흐름
  1. Markdown 준비 (PDF면 convert_pdf_md로 변환, .md면 그대로 사용)
  2. 입력 한도에 맞춰 줄 단위로 청킹 (줄 번호를 붙여 LLM에 입력)
  3. 청크마다 LLM이 제목 줄(줄 번호·레벨·제목)만 반환
  4. 전체 청크 결과를 병합해 계층 트리를 만들고, 제목 사이 줄을 섹션 본문으로 묶는다
  5. 결과 점검 (제목 문구가 해당 줄에 실제로 있는지, LLM이 채택하지 않은 '#' 후보)

LLM은 원문을 다시 쓰지 않고 줄 번호만 돌려준다. 본문은 코드가 원문에서 잘라 붙이므로 변형되지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from .chunking import estimate_tokens
from .config import Config
from .convert_pdf_md import pdf_to_markdown
from .llm import BudgetedLLM, CallBudget, LLMError, OpenAIClient

# --- 프롬프트 -------------------------------------------------------------

SYSTEM_PROMPT = """\
You analyze Markdown converted from an academic paper PDF and find its section headings.

Input format: each non-empty line is shown as `L<number>| <text>`. Lines starting with `#` are
heading CANDIDATES produced by the converter. They are only hints and are often wrong:
table or figure captions, author names, running headers/footers and sentence fragments may carry
`#`, while real headings may have no `#` at all (bold text, or numbered lines like `2.1 Methods`).
The number of `#` characters also may not match the real hierarchy (for example, the paper title
may be `##`, or a subsection may have the same `#` count as a main section). Treat `#` and its
count as a candidate only; it can be wrong. Decide from the text itself.

Task: return every line in the given chunk that is a real heading, with its hierarchy level.
- level 1 = the paper title, level 2 = main sections (Abstract, Introduction, Methods, Results,
  Discussion, Conclusion, References, ...), level 3 = subsections, level 4+ = deeper.
- Use numbering (1., 2.1, I., II., A.) and the headings already found (given as context) to keep
  levels consistent across chunks. A level may increase by at most 1 from the previous heading.
- Do NOT nest headings by meaning alone. Without explicit evidence (numbering such as 1./1.1/I./A.,
  or a clearly different style), headings that look the same must get the SAME level, even if one
  topic seems to belong under another (for example, unnumbered "Body composition" and "Blood
  pressure" next to an unnumbered "Measurements" are siblings, not children).
- Do NOT report figure/table captions, running page headers/footers, journal names, or ordinary sentences.
- `line` must be the exact number after `L` of the heading line. `title` is the heading text with
  any leading `#`, `**` and list numbering markers removed but otherwise unchanged.
- Missing introduction: if, after the abstract/keywords (or the title block), body text continues and
  reaches the first main section (e.g. Methods) WITHOUT any heading line for the introduction, report
  an inferred heading: `line` = the first line of that body text, `title` = "서론" for a Korean paper
  or "Introduction" for an English paper, `level` = 2, `inferred` = true. Do this only when it is
  clearly introductory body text (not the abstract, authors, or affiliations) and no real
  introduction heading exists. Never use `inferred` for headings that are printed in the text.
  Drop caps (a large first letter) are not headings by themselves.
- If a chunk has no headings, return an empty list.
"""


class Heading(BaseModel):
    line: int = Field(description="Number after 'L' of the heading line")
    level: int = Field(description="1=paper title, 2=main section, 3=subsection, ...")
    title: str
    inferred: bool = Field(
        default=False,
        description="True only for an introduction heading that is NOT printed in the text; 'line' is then its first body line",
    )


class ChunkHeadings(BaseModel):
    headings: list[Heading]


# --- 2. 청킹 --------------------------------------------------------------

CONTEXT_RESERVE_TOKENS = 2_000  # 앞 청크에서 얻은 제목 목록(컨텍스트)이 차지할 여유분


@dataclass
class Chunk:
    index: int
    line_numbers: list[int]  # 비어 있지 않은 줄의 원본 줄 번호(1부터)
    text: str  # `L<번호>| 내용` 형태로 렌더링된 입력
    tokens: int

    @property
    def start(self) -> int:
        return self.line_numbers[0]

    @property
    def end(self) -> int:
        return self.line_numbers[-1]


def _render(lineno: int, text: str) -> str:
    return f"L{lineno}| {text.rstrip()}"


def body_budget(cfg: Config) -> int:
    """프롬프트·시스템 입력·컨텍스트를 뺀, 청크 본문에 쓸 수 있는 토큰 수."""
    room = cfg.chunk_max_tokens - estimate_tokens(SYSTEM_PROMPT) - CONTEXT_RESERVE_TOKENS
    return max(500, min(cfg.chunk_target_tokens, room))


def _is_heading_line(raw: str) -> bool:
    return raw.lstrip().startswith("#")


Item = tuple[int, str, int, bool]  # (줄 번호, 렌더링, 토큰, 제목 줄 여부)


def _paragraphs(lines: list[str]) -> list[list[Item]]:
    paragraphs: list[list[Item]] = []
    cur: list[Item] = []
    for lineno, raw in enumerate(lines, start=1):
        if not raw.strip():
            if cur:
                paragraphs.append(cur)
                cur = []
            continue
        rendered = _render(lineno, raw)
        cur.append((lineno, rendered, estimate_tokens(rendered) + 1, _is_heading_line(raw)))
    if cur:
        paragraphs.append(cur)
    return paragraphs


def _heading_only(para: list[Item]) -> bool:
    return all(is_h for *_, is_h in para)


def _sections(paragraphs: list[list[Item]]) -> list[list[list[Item]]]:
    """제목('#') 줄로 시작하는 구간을 섹션으로 묶는다. 연속된 제목은 본문이 나올 때까지 한 섹션이다."""
    sections: list[list[list[Item]]] = []
    cur: list[list[Item]] = []
    has_body = False
    for para in paragraphs:
        if para[0][3] and has_body:  # 본문이 있는 섹션 뒤에 새 제목이 나오면 새 섹션
            sections.append(cur)
            cur, has_body = [], False
        cur.append(para)
        if not _heading_only(para):
            has_body = True
    if cur:
        sections.append(cur)
    return sections


def _section_pieces(section: list[list[Item]], budget: int) -> list[list[Item]]:
    """예산보다 큰 섹션을 문단 경계에서 나눈다. 제목만 있는 문단은 다음 문단과 함께 둔다."""
    units: list[list[Item]] = []
    carry: list[Item] = []
    for para in section:
        carry = carry + para
        if _heading_only(para):
            continue
        units.append(carry)
        carry = []
    if carry:
        units.append(carry)

    pieces: list[list[Item]] = []
    cur: list[Item] = []
    total = 0
    for unit in units:
        for part in [unit] if sum(t for _, _, t, _ in unit) <= budget else _split_oversize(unit, budget):
            t = sum(x for _, _, x, _ in part)
            if cur and total + t > budget:
                pieces.append(cur)
                cur, total = [], 0
            cur.extend(part)
            total += t
    if cur:
        pieces.append(cur)
    return pieces


def _split_oversize(unit: list[Item], budget: int) -> list[list[Item]]:
    """예산보다 큰 단위는 줄 단위로 자르되, 제목 줄이 조각의 마지막에 남지 않게 한다."""
    parts: list[list[tuple[int, str, int, bool]]] = []
    cur: list[tuple[int, str, int, bool]] = []
    total = 0
    for item in unit:
        if cur and total + item[2] > budget:
            tail = []
            while cur and cur[-1][3]:  # 끝의 제목 줄은 다음 조각으로 넘긴다
                tail.insert(0, cur.pop())
            if cur:
                parts.append(cur)
                cur = tail
            else:  # 전부 제목 줄이면 그대로 둔다
                cur = tail
                parts.append(cur)
                cur = []
            total = sum(t for _, _, t, _ in cur)
        cur.append(item)
        total += item[2]
    if cur:
        parts.append(cur)
    return parts


def make_chunks(lines: list[str], budget: int) -> list[Chunk]:
    """섹션 단위로 토큰 예산 안에서 묶는다.

    - 다음 섹션이 들어가지 않으면 그 섹션 앞에서 자르고, 다음 청크를 그 섹션의 처음부터 시작한다.
    - 섹션 하나가 예산보다 크면 문단 경계에서 나눈다. 제목 줄은 바로 뒤 문단과 함께 둔다.
    - 문단 하나가 예산보다 크면 줄 단위로 자른다(제목 줄은 조각 끝에 남기지 않는다).
    """
    chunks: list[Chunk] = []
    cur: list[Item] = []
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur, cur_tokens
        if cur:
            chunks.append(
                Chunk(
                    index=len(chunks) + 1,
                    line_numbers=[n for n, *_ in cur],
                    text="\n".join(r for _, r, *_ in cur),
                    tokens=cur_tokens,
                )
            )
        cur, cur_tokens = [], 0

    for section in _sections(_paragraphs(lines)):
        items = [it for para in section for it in para]
        tokens = sum(t for _, _, t, _ in items)
        pieces = [items] if tokens <= budget else _section_pieces(section, budget)
        for piece in pieces:
            piece_tokens = sum(t for _, _, t, _ in piece)
            if cur and cur_tokens + piece_tokens > budget:
                flush()
            cur.extend(piece)
            cur_tokens += piece_tokens
    flush()
    return chunks


# --- 3. 청크별 LLM 추출 ---------------------------------------------------

MAIN_SECTION_LEVEL = 2  # 컨텍스트의 기준이 되는 상위 레벨(주요 섹션)


def context_headings(found: list[Heading]) -> list[Heading]:
    """직전까지 찾은 제목 중 가장 최근의 주요 섹션 제목과, 그 이후에 나온 제목만 돌려준다."""
    if not found:
        return []
    for i in range(len(found) - 1, -1, -1):
        if found[i].level <= MAIN_SECTION_LEVEL:
            return found[i:]
    return found[-1:]  # 주요 섹션이 아직 없으면 직전 제목만


def _context_block(found: list[Heading]) -> str:
    ctx = context_headings(found)
    if not ctx:
        return "(none yet — this is the first chunk)"
    return "\n".join(f"L{h.line} | level {h.level} | {h.title}" for h in ctx)


def extract_chunk(llm: BudgetedLLM, cfg: Config, chunk: Chunk, found: list[Heading]) -> list[Heading]:
    user = (
        f"Most recent main section and the headings after it, found in earlier chunks (line | level | title):\n{_context_block(found)}\n\n"
        f"Chunk {chunk.index} (lines L{chunk.start}-L{chunk.end}):\n{chunk.text}"
    )
    result = llm.run(
        stage=f"title-chunk{chunk.index}",
        system=SYSTEM_PROMPT,
        user=user,
        schema=ChunkHeadings,
        max_output_tokens=cfg.max_output_tokens_extract,
        retries=cfg.max_retries_per_call,
    )
    valid = set(chunk.line_numbers)
    return [h for h in result.headings if h.line in valid]  # 이 청크 밖의 줄 번호는 버린다


# --- 4. 병합 · 계층 · 본문 매핑 -------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def build_outline(lines: list[str], headings: list[Heading]) -> tuple[list[dict], list[str]]:
    """제목들을 줄 순서로 정렬해 계층 트리를 만들고 섹션 본문을 매핑한다."""
    warnings: list[str] = []
    by_line: dict[int, Heading] = {}
    for h in headings:
        if h.line in by_line:
            warnings.append(f"L{h.line}: 같은 줄이 중복 보고되어 먼저 나온 것을 사용했다.")
            continue
        by_line[h.line] = h
    ordered = sorted(by_line.values(), key=lambda h: h.line)

    # 레벨 보정: 첫 제목은 그대로 두고, 이후 제목은 직전 제목보다 2 이상 깊어질 수 없다.
    fixed: list[tuple[Heading, int]] = []
    prev = max(0, ordered[0].level - 1) if ordered else 0
    for h in ordered:
        level = max(1, h.level)
        if level > prev + 1:
            warnings.append(f"L{h.line}: 레벨 {h.level} -> {prev + 1} 보정 ('{h.title}')")
            level = prev + 1
        fixed.append((h, level))
        prev = level

    root: list[dict] = []
    stack: list[dict] = []
    flat: list[dict] = []
    for i, (h, level) in enumerate(fixed):
        next_line = fixed[i + 1][0].line if i + 1 < len(fixed) else len(lines) + 1
        start = h.line - 1 if h.inferred else h.line  # 추정 제목은 본문 첫 줄부터, 일반 제목은 다음 줄부터
        body = "\n".join(lines[start : next_line - 1]).strip()  # ~ 다음 제목 앞 줄
        node = {
            "id": f"h{i + 1}",
            "level": level,
            "title": h.title,
            "line": h.line,
            "inferred": h.inferred,
            "body": body,
            "children": [],
        }
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        (stack[-1]["children"] if stack else root).append(node)
        stack.append(node)
        flat.append(node)

    first_line = fixed[0][0].line if fixed else len(lines) + 1
    front = "\n".join(lines[: first_line - 1]).strip()
    if front:
        root.insert(0, {"id": "front", "level": 0, "title": "(제목 이전 본문)", "line": 1, "body": front, "children": []})
    return root, warnings


def check_outline(lines: list[str], headings: list[Heading]) -> list[str]:
    """5단계 점검: 제목 문구가 해당 줄에 있는지, LLM이 채택하지 않은 '#' 후보가 있는지."""
    notes: list[str] = []
    adopted = {h.line for h in headings}
    for h in headings:
        if h.inferred:
            notes.append(f"[추정 제목] L{h.line}: '{h.title}' (원문에 없음, 본문 시작: '{lines[h.line - 1].strip()[:40]}')")
            continue
        if _norm(h.title) not in _norm(lines[h.line - 1]):
            notes.append(f"[제목 불일치] L{h.line}: '{h.title}' != '{lines[h.line - 1].strip()[:60]}'")
    for n, raw in enumerate(lines, start=1):
        if raw.lstrip().startswith("#") and n not in adopted:
            notes.append(f"[미채택 # 후보] L{n}: {raw.strip()[:70]}")
    return notes


def outline_markdown(nodes: list[dict]) -> str:
    """제목 계층을 `#`, `##`, `###` 단계로 표현한다. 줄 번호와 본문 길이는 뒤에 붙인다."""
    out: list[str] = []

    def walk(items: list[dict]) -> None:
        for n in items:
            marks = "#" * n["level"] if n["level"] > 0 else "-"
            tag = ", 추정 제목" if n.get("inferred") else ""
            out.append(f"{marks} {n['title']}  (L{n['line']}{tag}, 본문 {len(n['body']):,}자)")
            walk(n["children"])

    walk(nodes)
    return "\n".join(out)


# --- 실행 -----------------------------------------------------------------

def load_env(path: Path) -> None:
    """`KEY=VALUE` 형식의 .env를 읽어 환경변수에 넣는다. 이미 설정된 환경변수와 빈 값은 건드리지 않는다."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        if value:
            os.environ.setdefault(key.strip(), value)


# 한글 덩어리(공백 없이 이어진 한글) 평균 길이. 정상 한국어 논문은 약 2.6~2.9, 띄어쓰기가 사라진 텍스트는 크게 늘어난다.
MAX_AVG_HANGUL_RUN = 5.0
MIN_HANGUL_RUNS = 200  # 한글이 거의 없는 논문(영어 논문 등)은 이 검사를 적용하지 않는다


def unparsable_reason(markdown: str) -> str | None:
    """텍스트 레이어의 띄어쓰기가 사라진 논문을 가려낸다. 파싱 가능하면 None."""
    runs = re.findall(r"[가-힣]+", markdown)
    if len(runs) < MIN_HANGUL_RUNS:
        return None
    avg = sum(map(len, runs)) / len(runs)
    if avg > MAX_AVG_HANGUL_RUN:
        return f"한글 덩어리 평균 길이 {avg:.1f} > {MAX_AVG_HANGUL_RUN} (띄어쓰기 손실)"
    return None


def output_stem(path: Path) -> str:
    """결과 파일 이름. 논문 번호(파일명 앞 숫자)로 짧게 만든다. 긴 한글 파일명은 OS 길이 제한을 넘는다."""
    m = re.match(r"\d+", unicodedata.normalize("NFC", path.stem))
    return f"{m.group()}_title" if m else f"{path.stem[:20]}_title"


def load_markdown(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return pdf_to_markdown(path)
    return path.read_text(encoding="utf-8")


def _inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(p.glob("*.md"))
        elif p.is_file() and p.suffix.lower() in {".md", ".pdf"}:
            files.append(p)
        else:
            print(f"[오류] md/pdf 파일 또는 폴더가 아닙니다: {p}", file=sys.stderr)
    return sorted(set(files))


def process(path: Path, cfg: Config, out_dir: Path, dry_run: bool) -> bool:
    markdown = load_markdown(path)
    reason = unparsable_reason(markdown)
    if reason:
        print(f"{path.name}: 파싱 제외 - {reason}")
        return True
    lines = markdown.splitlines()
    budget = body_budget(cfg)
    chunks = make_chunks(lines, budget)
    print(f"{path.name}: {len(lines)}줄 -> 청크 {len(chunks)}개 (본문 예산 {budget} 토큰)")
    for c in chunks:
        print(f"  청크 {c.index}: L{c.start}-L{c.end}, 약 {c.tokens} 토큰")
    if dry_run:
        return True
    if len(chunks) > cfg.max_api_calls:
        print(f"[오류] 청크 {len(chunks)}개가 호출 한도 {cfg.max_api_calls}회를 넘는다. --max-calls를 올리거나 청크를 키운다.", file=sys.stderr)
        return False

    client = OpenAIClient(cfg.model, cfg.request_timeout_s)
    budget_tracker = CallBudget(cfg.max_api_calls)
    llm = BudgetedLLM(client, budget_tracker, out_dir.parent / "cache", cfg.cache_fingerprint(), cfg.model)

    found: list[Heading] = []
    for c in chunks:
        try:
            got = extract_chunk(llm, cfg, c, found)
        except LLMError as exc:
            print(f"[오류] 청크 {c.index}: {exc}", file=sys.stderr)
            return False
        print(f"  청크 {c.index}: 제목 {len(got)}개")
        found.extend(got)

    outline, warnings = build_outline(lines, found)
    notes = check_outline(lines, found)
    outline_md = outline_markdown(outline)
    print(outline_md)
    for msg in warnings + notes:
        print(f"  {msg}")

    data = {
        "source": path.name,
        "model": cfg.model,
        "chunks": [{"index": c.index, "start": c.start, "end": c.end, "tokens": c.tokens} for c in chunks],
        "usage": budget_tracker.totals(),
        "warnings": warnings,
        "checks": notes,
        "outline": outline,
    }
    save_result(path, out_dir, data, outline_md)
    return True


def save_result(path: Path, out_dir: Path, data: dict, outline_md: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{output_stem(path)}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_dir / f"{output_stem(path)}.outline.md"
    md_path.write_text(outline_md + "\n", encoding="utf-8")
    print(f"  -> {out_path}\n  -> {md_path}")


# --- 확정한 제목 구조를 새 Markdown에 다시 붙이기 (LLM 호출 없음) -----------------

_MAX_NUMBERING = 10  # 제목 줄에 붙은 번호(`1.`, `Ⅰ.` 등)로 허용하는 추가 글자 수


def _flat_headings(outline: list[dict]) -> list[dict]:
    out: list[dict] = []

    def walk(nodes: list[dict]) -> None:
        for n in nodes:
            if n["level"] > 0:  # 레벨 0은 `(제목 이전 본문)` 같은 가짜 노드
                out.append(n)
            walk(n["children"])

    walk(outline)
    return sorted(out, key=lambda n: n["line"])


def _heading_texts(lines: list[str]) -> set[str]:
    return {_norm(l.lstrip().lstrip("#")) for l in lines if l.lstrip().startswith("#")}


def reanchor(old_data: dict, old_md: str, new_lines: list[str]) -> tuple[list[Heading] | None, str]:
    """이전에 확정한 제목 목록(레벨·제목·순서)을 변경된 Markdown의 줄에 다시 붙인다.

    변환 수정이 제목 자체를 바꾸지 않았을 때만 성공한다. 이전 제목을 하나라도 못 찾거나, 이전에 없던 `#` 제목 후보가
    새로 생겼으면 None과 사유를 돌려준다(그 경우는 LLM으로 다시 추출해야 한다).
    """
    new_candidates = _heading_texts(new_lines) - _heading_texts(old_md.splitlines())
    if new_candidates:
        return None, f"새 제목 후보 {len(new_candidates)}개: {sorted(new_candidates)[:3]}"
    norms = [_norm(l.lstrip().lstrip("#")) for l in new_lines]
    is_hash = [l.lstrip().startswith("#") for l in new_lines]
    found: list[Heading] = []
    pos = 0  # 직전에 맞춘 줄(0-based)의 다음부터 찾는다
    for h in _flat_headings(old_data["outline"]):
        title_n = _norm(h["title"])
        hit = None
        if h.get("inferred"):
            first = next((l for l in h["body"].splitlines() if l.strip()), "")
            key = _norm(first)[:25]
            hit = next((i for i in range(pos, len(new_lines)) if key and norms[i].startswith(key)), None)
        else:
            hit = next(
                (
                    i
                    for i in range(pos, len(new_lines))
                    if title_n
                    and len(norms[i]) <= len(title_n) + _MAX_NUMBERING
                    and (title_n in norms[i] if is_hash[i] else norms[i].endswith(title_n))  # `#` 없는 줄은 앞에 번호만 붙은 짧은 줄만 인정
                ),
                None,
            )
        if hit is None:
            return None, f"이전 제목을 찾지 못함: '{h['title'][:30]}'"
        found.append(Heading(line=hit + 1, level=h["level"], title=h["title"], inferred=bool(h.get("inferred"))))
        pos = hit + 1
    return found, f"제목 {len(found)}개 모두 재부착"


def main(argv: list[str] | None = None) -> int:
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="LLM으로 논문 제목 계층을 추출하고 섹션별 본문을 매핑합니다.")
    parser.add_argument("paths", nargs="*", type=Path, default=[base / "outputs" / "markdown"], help="md/pdf 파일 또는 폴더")
    parser.add_argument("--out", type=Path, default=base / "outputs" / "titles", help="결과 JSON 저장 폴더")
    parser.add_argument("--model", default=None, help="모델명 (생략하면 PAPER_EXTRACTOR_MODEL)")
    parser.add_argument("--max-calls", type=int, default=None, help="논문당 API 호출 한도")
    parser.add_argument("--dry-run", action="store_true", help="LLM을 호출하지 않고 청킹 결과만 출력")
    args = parser.parse_args(argv)

    load_env(base / ".env")
    cfg = Config()
    if args.model:
        cfg.model = args.model
    if args.max_calls:
        cfg.max_api_calls = args.max_calls
    cfg.validate()

    files = _inputs(args.paths)
    if not files:
        print("md/pdf 파일을 찾지 못했습니다.", file=sys.stderr)
        return 1
    ok = True
    for f in files:
        try:
            ok = process(f, cfg, args.out, args.dry_run) and ok
        except ValueError as exc:  # 모델명·API 키 누락
            print(f"[오류] {exc}", file=sys.stderr)
            return 1
        print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
