"""논문 제목 계층을 LLM으로 추출하기 위한 도구.

단계별로 구현 중이다.
  1. PDF -> Markdown 변환 (pymupdf4llm)   <- 현재 단계
  2. LLM 입력 한도에 맞춘 청킹
  3. 청크별 제목 계층 추출
  4. 전체 청크 결과 병합 및 계층 정리
  5. 결과 확인
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
import sys
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pymupdf4llm

from .pdf_to_md import _find_box, _looks_like_sentence, reorder_columns, split_blocks


# pymupdf4llm이 그림 자리에 넣는 표지 줄. 그림 안 텍스트는 보존하고 표지만 지운다.
_PICTURE_MARKER_RE = re.compile(
    r"\*\*(?:==> picture \[.*?\] intentionally omitted <==|"
    r"-----\s*(?:Start|End) of picture text\s*-----)\*\*(?:<br>)?[ \t]*",
)


_EDITOR_QUERY_RE = re.compile(r"\s*AUTHOR:\s[^?\n]{3,200}\?\s*")  # 교정 단계의 편집자 질의(`AUTHOR: …?`)가 본문에 남은 경우


def repair_symbols(text: str) -> tuple[str, dict]:
    """텍스트 레이어에서 부호(−, ±)가 제어문자로 저장돼 변환기가 `�`로 바꾼 글자를 문맥으로 복원한다.

    숫자 앞에서 단어·숫자에 붙지 않은 `�`는 마이너스(`−`), 단어·숫자 사이의 `�`는 `±`로 본다.
    """
    stats = {"minus": 0, "plus_minus": 0}

    def minus(m: re.Match) -> str:
        stats["minus"] += 1
        return "\u2212"

    def pm(m: re.Match) -> str:
        stats["plus_minus"] += 1
        return "\u00b1"

    text = re.sub(r"(?<![\w\u2212])\ufffd(?=\s?[\d.])", minus, text)  # 문장 시작·공백·괄호 뒤, 숫자 앞
    text = re.sub(r"(?<=[\w)])\ufffd(?=[\w(])", pm, text)  # 단어·숫자 사이
    stats["unrepaired"] = text.count("\ufffd")
    return text, stats


def clean_markdown(markdown: str) -> str:
    """LLM 입력에 불필요한 변환 표지를 제거한다. 본문·헤딩은 바꾸지 않는다."""
    markdown = _PICTURE_MARKER_RE.sub("", markdown)
    markdown = markdown.replace("<br>", "\n")
    markdown = _EDITOR_QUERY_RE.sub(" ", markdown)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"


_PLACEHOLDER_RE = re.compile(r"^\*\*==> picture \[.*?\] intentionally omitted <==\*\*$")
_MAX_RECOVERED_HEADING = 30


def _recovered_heading(page: "pymupdf.Page", box: dict | None) -> str | None:
    """레이아웃 모델이 그림·수식으로 오인해 지운 짧은 제목(예: `Ⅰ. 서 론`)의 글자를 PDF 텍스트 레이어에서 되살린다.

    수식(formula)으로 분류된 박스에 짧은 텍스트가 있고 수식 기호가 없을 때만 제목 후보로 돌려준다. 최종 판단은 LLM이 한다.
    """
    if not box or not box.get("bbox") or box.get("class") != "formula":  # picture는 그림 속 글자라 제외
        return None
    text = re.sub(r"\s+", " ", page.get_text("text", clip=pymupdf.Rect(*box["bbox"]))).strip()
    if not text or len(text) > _MAX_RECOVERED_HEADING or re.search(r"[=+<>/]", text):
        return None
    return text if re.search(r"[가-힣A-Za-z]{2,}", text.replace(" ", "")) else None


_DROP_CLASSES = {"page-header", "page-footer"}  # 쪽 머리글·바닥글(쪽 번호, 저널명, DOI 등)
_LICENSE_RE = re.compile(r"Creative Commons|open[- ]access article", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?。…][\"'”’)\]]*\s*$")
_COLUMN_GAP = 100  # 이 이상 가로 위치가 다르면 단이 바뀐 것으로 본다(pt)


@dataclass
class _Block:
    kind: str
    body: str
    level: int | None
    cls: str | None
    page: int
    x0: float | None
    y0: float | None = None
    y1: float | None = None
    note: bool = False  # 표 바로 밑의 주석(`M: mean; SD: …`)
    small: bool = False  # 본문보다 눈에 띄게 작은 글자(각주·표 주석 등)

    @property
    def is_float(self) -> bool:
        """본문 흐름에 끼어드는 표·캡션·표 주석·각주."""
        return self.kind == "table" or self.cls in _FLOAT_CLASSES or self.note or self.small


_FLOAT_CLASSES = {"caption", "table"}
_SMALL_FONT_GAP = 1.2  # 본문 글자 크기보다 이만큼(pt) 이상 작으면 각주·주석으로 본다
_FOOTER_SMALL_GAP = 0.3  # 쪽 아래쪽에서는 이만큼만 작아도 각주로 본다
_FOOTER_ZONE = 0.75  # 쪽 높이의 이 비율보다 아래를 각주 영역으로 본다
_NOTE_GAP = 14  # 표 바로 밑으로 보는 최대 세로 간격(pt)
_NOTE_RE = re.compile(r"^\s*[_*]*(?:M|SD|Note|Notes|주|비고|Abbreviations?)[_*]*\s*[:.]|^\s*[*†‡§]", re.IGNORECASE)


def _page_block_sizes(page: "pymupdf.Page") -> list[tuple]:
    """텍스트 레이어 블록별 (bbox, 평균 글자 크기)."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        total = weighted = 0
        for line in b["lines"]:
            for sp in line["spans"]:
                n = len(sp["text"].strip())
                total += n
                weighted += n * sp["size"]
        if total:
            out.append((b["bbox"], weighted / total))
    return out


def _size_at(bbox: list[float], sizes: list[tuple]) -> float | None:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for bb, sz in sizes:
        if bb[0] - 1 <= cx <= bb[2] + 1 and bb[1] - 1 <= cy <= bb[3] + 1:
            return sz
    return None


def _is_continuation(prev: _Block, cur: _Block, require_cross: bool = True) -> bool:
    """문장이 끊겨 이어지는 두 본문 문단인지 판단한다. 쪽·단 경계가 기본 조건이고, 사이에 표가 끼면 경계가 아니어도 인정한다."""
    if prev.kind != "paragraph" or cur.kind != "paragraph" or prev.cls != "text" or cur.cls != "text":
        return False
    if _SENTENCE_END_RE.search(prev.body) or cur.body.lstrip().startswith("**"):
        return False
    if not require_cross:
        return True
    return cur.page != prev.page or (prev.x0 is not None and cur.x0 is not None and abs(cur.x0 - prev.x0) > _COLUMN_GAP)


_CAPTION_WINDOW = 15  # 표 앞에서 떨어진 캡션을 찾는 최대 블록 수
_TABLE_CAPTION_RE = re.compile(r"^[\s*]*(?:Table|표)\s*\d+", re.IGNORECASE)


def _reattach_captions(out: list[_Block]) -> None:
    """읽기 순서 재정렬로 표 캡션이 표에서 떨어진 경우(캡션은 왼쪽 단, 표는 단을 가로지름), 캡션을 표 바로 앞으로 옮긴다."""
    idx = 0
    while idx < len(out):
        b = out[idx]
        if b.kind == "table" and not (idx > 0 and out[idx - 1].cls == "caption"):
            for j in range(idx - 1, max(-1, idx - 1 - _CAPTION_WINDOW), -1):  # 이어 붙인 문단의 쪽 번호가 바뀔 수 있어 쪽 대신 블록 거리로 제한한다
                c = out[j]
                owned = j + 1 < len(out) and out[j + 1].kind == "table"  # 이미 자기 표가 붙어 있는 캡션은 건드리지 않는다
                if c.cls == "caption" and _TABLE_CAPTION_RE.match(c.body) and not owned:
                    out.insert(idx - 1, out.pop(j))
                    break
        idx += 1


def _last_text_index(out: list[_Block]) -> int | None:
    """끝에서부터 표·캡션·표 주석을 건너뛴 첫 블록의 위치. 건너뛴 것이 없으면 None."""
    k = len(out) - 1
    while k >= 0 and out[k].is_float:
        k -= 1
    return k if 0 <= k < len(out) - 1 else None


_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s)\]>,;]+")


class _Meta:
    """본문에서 뺀 머리글·바닥글·라이선스 문구를 모아 논문 메타데이터(학술지명·연도·DOI 등)의 근거로 남긴다."""

    def __init__(self) -> None:
        self.lines: dict[str, dict] = {}
        self.licenses: list[str] = []

    def add(self, cls: str | None, body: str, page: int) -> None:
        text = re.sub(r"\s+", " ", body.replace("**", "").lstrip("> ")).strip()
        if not text or text.isdigit():  # 쪽 번호만 있는 줄은 버린다
            return
        if cls == "footnote":
            if text not in self.licenses:
                self.licenses.append(text)
            return
        key = re.sub(r"\s*\d+$", "", text)  # 끝의 쪽 번호만 다른 줄은 한 줄로 합친다
        entry = self.lines.setdefault(key, {"text": text, "class": cls, "pages": []})
        if page not in entry["pages"]:
            entry["pages"].append(page)

    def to_dict(self, pdf_path: Path, all_text: str, pdf_metadata: dict) -> dict:
        dois = sorted({m.group().rstrip(".") for m in _DOI_RE.finditer(all_text)})
        return {
            "source": pdf_path.name,
            "doi_candidates": dois,
            "running_lines": list(self.lines.values()),
            "license_notes": self.licenses,
            "pdf_metadata": {k: v for k, v in pdf_metadata.items() if v and k in ("title", "author", "subject", "keywords")},
        }


def pdf_to_markdown(pdf_path: Path) -> str:
    return convert(pdf_path)[0]


def convert(pdf_path: Path) -> tuple[str, dict]:
    """PDF를 pymupdf4llm으로 변환하고 LLM 입력에 맞게 정리한다. (Markdown, 메타데이터)를 돌려준다.

    - 2단 편집 페이지는 bbox로 읽기 순서를 바로잡는다.
    - 쪽 머리글·바닥글과 라이선스 각주는 본문에서 제외하고 메타데이터로 따로 모은다(본문 문장 한가운데 끼어들기 때문).
    - 제목으로 오인된 문장 조각은 본문으로 되돌리고, 쪽·단 경계에서 끊긴 문장은 이어 붙인다.
    - 문단 중간에 표·캡션이 끼면 문단을 먼저 완성하고 표(와 표 주석)를 그 문단 뒤로 옮긴다.
    - 그림·수식으로 오인돼 빠진 짧은 제목은 PDF 텍스트 레이어에서 되살려 제목 후보(`##`)로 넣는다.
    """
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    out: list[_Block] = []
    meta = _Meta()
    page_items: list[tuple[float, list[tuple]]] = []  # 쪽별 (쪽 높이, [(kind, body, level, box, 글자 크기)])
    body_sizes: collections.Counter = collections.Counter()
    with pymupdf.open(str(pdf_path)) as doc:
        pdf_metadata = dict(doc.metadata or {})
        for i, pg in enumerate(pages):
            text = pg.get("text", "") or ""
            boxes = pg.get("page_boxes") or []
            sizes = _page_block_sizes(doc[i])
            raw = []
            for kind, off, body, level in split_blocks(text):
                box = _find_box(boxes, off)
                if kind == "paragraph" and _PLACEHOLDER_RE.match(body):
                    recovered = _recovered_heading(doc[i], box)
                    if recovered:
                        kind, body, level = "heading", recovered, 2
                raw.append((kind, body, level, box))
            raw, _ = reorder_columns(raw)
            items = []
            for kind, body, level, box in raw:
                size = _size_at(box["bbox"], sizes) if box and box.get("bbox") else None
                if box and box.get("class") == "text" and size:
                    body_sizes[round(size * 2) / 2] += len(body)  # 본문 글자 크기 = 글자 수 가중 최빈값
                items.append((kind, body, level, box, size))
            page_items.append((doc[i].rect.height, items))
    body_size = body_sizes.most_common(1)[0][0] if body_sizes else None

    for i, (page_h, items) in enumerate(page_items):
        for kind, body, level, box, size in items:
            cls = box.get("class") if box else None
            if _PLACEHOLDER_RE.match(body):
                continue  # 복원되지 못한 그림 자리표시자
            if cls == "picture" and "Start of picture text" in body:
                continue  # 그림 안 글자(흐름도 등): 이미지로 따로 전달되고, 글자는 뒤섞여 본문 문장 사이에 끼어든다
            if cls in _DROP_CLASSES:
                meta.add(cls, body, i + 1)  # 머리글·바닥글
                continue
            if _LICENSE_RE.search(body) and len(body) < 800:
                meta.add("footnote", body, i + 1)  # 라이선스 문구(레이아웃 분류와 무관)
                continue
            if kind == "heading" and _looks_like_sentence(body):
                kind, level, cls = "paragraph", None, "text"  # 문장 조각은 본문 문단으로 되돌린다
            bbox = box["bbox"] if box and box.get("bbox") else None
            blk = _Block(kind, body, level, cls, i + 1, bbox[0] if bbox else None, bbox[1] if bbox else None, bbox[3] if bbox else None)
            if cls in ("text", "section-header") and size and body_size and bbox:
                low = bbox[1] > page_h * _FOOTER_ZONE
                if size <= body_size - _SMALL_FONT_GAP or (low and size < body_size - _FOOTER_SMALL_GAP):
                    blk.small = True
                    if blk.kind == "heading":  # 쪽 아래 각주 줄(`*Corresponding author` 등)이 제목으로 분류된 경우
                        blk.kind, blk.level, blk.cls = "paragraph", None, "text"
            if cls == "text" and out and out[-1].kind == "table":
                near = out[-1].y1 is not None and blk.y0 is not None and 0 <= blk.y0 - out[-1].y1 <= _NOTE_GAP
                if near or _NOTE_RE.match(body):
                    blk.note = True  # 표 바로 밑 주석
            target = None
            if not blk.is_float:
                k = _last_text_index(out)
                if k is not None and _is_continuation(out[k], blk, require_cross=False):
                    target = out[k]  # 표·캡션·각주가 문단 중간에 끼었다: 문단을 먼저 완성하고 그것들은 뒤에 둔다
                elif out and _is_continuation(out[-1], blk):
                    target = out[-1]
            if target is not None:
                target.body = target.body.rstrip() + " " + blk.body.lstrip()
                target.page, target.x0, target.y1 = blk.page, blk.x0, blk.y1
            else:
                out.append(blk)
    _reattach_captions(out)
    parts = [("#" * (b.level or 2) + " " + b.body) if b.kind == "heading" else b.body for b in out]
    all_text = "\n".join(pg.get("text", "") or "" for pg in pages)
    text, repairs = repair_symbols("\n\n".join(parts))
    info = meta.to_dict(pdf_path, all_text, pdf_metadata)
    info["symbol_repairs"] = repairs
    return clean_markdown(text), info


def paper_number(path: Path) -> str:
    m = re.match(r"\d+", unicodedata.normalize("NFC", path.name))
    return m.group() if m else path.stem[:20]


def _pdf_files(paths: list[Path]) -> tuple[list[Path], bool]:
    files: list[Path] = []
    had_errors = False
    for path in paths:
        if path.is_dir():
            files.extend(path.glob("*.pdf"))
        elif path.is_file() and path.suffix.lower() == ".pdf":
            files.append(path)
        else:
            had_errors = True
            print(f"[오류] PDF 파일 또는 폴더가 아닙니다: {path}", file=sys.stderr)
    return sorted(set(files)), had_errors


def main(argv: list[str] | None = None) -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="PDF를 Markdown으로 변환해 저장합니다.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[base / "papers"],
        help="PDF 파일 또는 PDF 폴더 (생략하면 paper_extractor/papers 폴더)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=base.parent / "outputs" / "markdown",
        help="Markdown 저장 폴더 (기본: outputs/markdown)",
    )
    args = parser.parse_args(argv)

    pdf_files, failed = _pdf_files(args.paths)
    if not pdf_files:
        print("PDF 파일을 찾지 못했습니다.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for pdf_path in pdf_files:
        try:
            markdown, meta = convert(pdf_path)
        except Exception as exc:  # pymupdf가 여러 예외를 내므로 파일 단위로 격리한다.
            failed = True
            print(f"[오류] {pdf_path.name}: {exc}", file=sys.stderr)
            continue
        out_path = args.out / f"{pdf_path.stem}.md"
        out_path.write_text(markdown, encoding="utf-8")
        meta_path = args.out / f"{paper_number(pdf_path)}_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{pdf_path.name} -> {out_path} ({len(markdown):,}자), 메타데이터 -> {meta_path.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
