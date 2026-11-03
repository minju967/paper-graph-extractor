"""준비 A: PDF → Markdown 변환, 블록 ID·원문 위치 관리, 판독 이상 신호, 참고문헌 구간 표시.

- 원본 PDF는 읽기만 한다.
- 변환기는 PyMuPDF4LLM이다. 대체 라이브러리로 조용히 바꾸지 않으며 설치 버전을 기록한다.
- 블록 ID는 코드가 부여한다(p{페이지}-b{순번}). LLM은 이 ID 중에서 근거를 선택할 뿐이다.
"""

from __future__ import annotations

import hashlib
import re
from importlib import metadata
from pathlib import Path

from .models import Block, Document, PageInfo

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")
_TABLE_RE = re.compile(r"^\s*\|")
_REF_HEADING_RE = re.compile(
    r"^(?:[\dIVX]+[.)]?\s*)?(references?|reference list|bibliography|literature cited|참고\s*문헌|인용\s*문헌)\s*:?$",
    re.IGNORECASE,
)
_AFTER_REF_RE = re.compile(r"appendix|supplement|부록|보충", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def converter_name() -> str:
    try:
        return f"pymupdf4llm=={metadata.version('pymupdf4llm')}"
    except metadata.PackageNotFoundError:
        return "pymupdf4llm(unknown)"


def clean_heading(text: str) -> str:
    return re.sub(r"[*_`]+", "", text).strip()


def garbled_count(text: str) -> int:
    bad = sum(1 for ch in text if ch == "\ufffd" or "\ue000" <= ch <= "\uf8ff")
    return bad + text.count("(cid:") * 6


def garbled_ratio(text: str) -> float:
    return garbled_count(text) / max(len(text), 1) if text else 0.0


_SENTENCE_END_RE = re.compile(r"(다|요|음|함|\))\.$")


def _looks_like_sentence(text: str) -> bool:
    t = clean_heading(text)
    return t.count(")") > t.count("(") or bool(_SENTENCE_END_RE.search(t))


def reorder_columns(raw: list[tuple]) -> tuple[list[tuple], bool]:
    """2단 편집 페이지의 읽기 순서를 bbox로 바로잡는다.

    변환기가 왼쪽·오른쪽 단을 섞어 내보내는 경우가 있다. 모든 블록에 bbox가 있고 양쪽 단에 블록이 2개 이상일 때만 적용하며,
    그렇지 않으면 원래 순서를 그대로 둔다. 가로로 단을 가로지르는 블록(제목·넓은 표·초록)은 흐름을 나누는 구분점으로 본다.
    """
    boxes = [b[3].get("bbox") if b[3] else None for b in raw]
    if len(raw) < 4 or any(bb is None for bb in boxes):
        return raw, False
    x0 = min(bb[0] for bb in boxes)
    x1 = max(bb[2] for bb in boxes)
    mid = (x0 + x1) / 2
    tol = 20
    wide, left, right = [], [], []
    for idx, bb in enumerate(boxes):
        if bb[0] < mid - tol and bb[2] > mid + tol:
            wide.append(idx)
        elif bb[2] <= mid + tol and (bb[0] + bb[2]) / 2 < mid:
            left.append(idx)
        else:
            right.append(idx)
    if len(left) < 2 or len(right) < 2:
        return raw, False
    wide.sort(key=lambda k: boxes[k][1])
    wide_y = [boxes[k][1] for k in wide]

    def seg(k: int) -> int:
        return sum(1 for y in wide_y if y <= boxes[k][1])

    order: list[int] = []
    for si in range(len(wide) + 1):
        if si > 0:
            order.append(wide[si - 1])
        order += sorted((k for k in left if seg(k) == si), key=lambda k: boxes[k][1])
        order += sorted((k for k in right if seg(k) == si), key=lambda k: boxes[k][1])
    # 쪽 머리글은 맨 앞, 바닥글·그림 자리표시자는 맨 뒤로 보내 본문 흐름에 끼지 않게 한다. (각주는 제자리에 둔다)
    cls = lambda k: (raw[k][3] or {}).get("class")  # noqa: E731
    head = [k for k in order if cls(k) == "page-header"]
    tail = [k for k in order if cls(k) in ("page-footer", "picture")]
    order = head + [k for k in order if k not in head and k not in tail] + tail
    moved = order != list(range(len(raw)))
    return [raw[k] for k in order], moved


def split_blocks(text: str) -> list[tuple[str, int, str, int | None]]:
    """페이지 Markdown을 (kind, 시작 오프셋, 텍스트, 제목수준) 블록으로 나눈다.

    표 줄은 한 블록으로 묶고, 제목은 줄마다 독립 블록으로 둔다.
    """
    out: list[tuple[str, int, str, int | None]] = []
    cur_kind: str | None = None
    cur_lines: list[str] = []
    cur_start = 0
    offset = 0

    def flush() -> None:
        nonlocal cur_kind, cur_lines
        if cur_kind and cur_lines:
            body = "\n".join(cur_lines).strip()
            if body:
                out.append((cur_kind, cur_start, body, None))
        cur_kind, cur_lines = None, []

    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        h = _HEADING_RE.match(stripped)
        if h:
            flush()
            out.append(("heading", offset, h.group(2).strip(), len(h.group(1))))
        elif not stripped.strip():
            if cur_kind != "table":  # 표 안의 빈 줄은 표를 끊지 않는다
                flush()
        else:
            kind = "table" if _TABLE_RE.match(stripped) else "paragraph"
            if kind != cur_kind:
                flush()
                cur_kind, cur_start = kind, offset
            cur_lines.append(stripped)
        offset += len(line)
    flush()
    return out


def _find_box(boxes: list[dict], offset: int) -> dict | None:
    for b in boxes or []:
        pos = b.get("pos")
        if pos and pos[0] <= offset < pos[1]:
            return b
    return None


def mark_references(blocks: list[Block], warnings: list[str]) -> None:
    """참고문헌 제목 이후 구간을 제외 표시한다. 제목을 찾지 못하면 아무것도 지우지 않는다."""
    in_refs = False
    found = False
    for b in blocks:
        if b.kind == "heading":
            title = clean_heading(b.text)
            if _REF_HEADING_RE.match(title):
                in_refs, found = True, True
            elif in_refs and _AFTER_REF_RE.search(title):
                in_refs = False
        if in_refs:
            b.excluded, b.exclude_reason = True, "references"
    if not found:
        warnings.append("참고문헌 제목을 찾지 못했다. 참고문헌 구간을 제외하지 않았다.")


def convert_pdf(pdf_path: Path, min_chars: int = 400, max_garbled: float = 0.02) -> Document:
    import pymupdf4llm  # 지연 import

    pdf_path = Path(pdf_path)
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    blocks: list[Block] = []
    page_infos: list[PageInfo] = []
    warnings: list[str] = []

    for i, pg in enumerate(pages, start=1):
        text = pg.get("text", "") or ""
        boxes = pg.get("page_boxes") or []
        n_chars = len(re.sub(r"\s+", "", text))
        g = garbled_ratio(text)
        flags: list[str] = []
        if n_chars < min_chars:
            flags.append("low_text")
        # 표의 부호(−)처럼 소수의 문자만 깨져도 결과 방향이 바뀌므로 비율과 절대 개수를 함께 본다.
        if g > max_garbled or garbled_count(text) >= 5:
            flags.append("garbled")
        page_infos.append(PageInfo(page=i, char_count=n_chars, garbled_ratio=round(g, 4), garbled_chars=garbled_count(text), readability_flags=flags))

        raw = []
        for kind, off, body, level in split_blocks(text):
            box = _find_box(boxes, off)
            if kind == "heading" and _looks_like_sentence(body):
                kind, level = "paragraph", None  # 레이아웃 모델이 문장 조각을 제목으로 오인한 경우
                warnings.append(f"p{i}: 문장 조각으로 보이는 제목을 본문으로 되돌림: {body[:30]!r}")
            raw.append((kind, body, level, box))
        raw, moved = reorder_columns(raw)
        if moved:
            warnings.append(f"p{i}: 다단 레이아웃으로 판단해 bbox 기준으로 읽기 순서를 재정렬함(원본과 시각 대조 필요)")
        for n, (kind, body, level, box) in enumerate(raw, start=1):
            blocks.append(
                Block(
                    id=f"p{i}-b{n}",
                    page=i,
                    kind=kind,  # type: ignore[arg-type]
                    text=body,
                    heading_level=level,
                    bbox=[float(x) for x in box["bbox"]] if box and box.get("bbox") else None,
                    box_class=box.get("class") if box else None,
                )
            )

    mark_references(blocks, warnings)

    for info in page_infos:
        pb = [b for b in blocks if b.page == info.page and b.kind != "heading"]
        if pb and all(b.excluded for b in pb):
            info.is_reference_only = True

    for info in page_infos:
        if info.readability_flags:
            warnings.append(f"p{info.page}: 판독 이상 신호 {info.readability_flags} (추출 텍스트 존재가 판독 성공을 보장하지 않음)")

    if not any(b.kind == "heading" for b in blocks):
        warnings.append("제목(heading)을 하나도 인식하지 못했다. 청킹은 문단·페이지 경계로 대체된다.")

    return Document(
        pdf_name=pdf_path.name,
        pdf_sha256=sha256_file(pdf_path),
        converter=converter_name(),
        page_count=len(pages),
        pages=page_infos,
        blocks=blocks,
        warnings=warnings,
    )


def render_page_png(pdf_path: Path, page: int, dpi: int = 150) -> bytes:
    """이미지 보완용 페이지 렌더링 (page는 1-based)."""
    import pymupdf

    with pymupdf.open(str(pdf_path)) as doc:
        pix = doc[page - 1].get_pixmap(dpi=dpi)
        return pix.tobytes("png")


def to_markdown_text(doc: Document, include_excluded: bool = True) -> str:
    """사람이 대조할 수 있는 Markdown. 블록 ID를 주석으로 남긴다."""
    lines: list[str] = []
    for b in doc.blocks:
        if b.excluded and not include_excluded:
            continue
        tag = f"<!-- {b.id} p{b.page}{' excluded:' + b.exclude_reason if b.excluded else ''}{' image' if b.source == 'image' else ''} -->"
        body = ("#" * (b.heading_level or 2) + " " + b.text) if b.kind == "heading" else b.text
        lines.append(f"{tag}\n{body}\n")
    return "\n".join(lines)
