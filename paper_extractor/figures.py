"""본문이 언급하는 그림(figure)과 깨진 표(table)를 PDF에서 찾아 이미지로 잘라낸다. LLM은 호출하지 않는다.

- 청크 텍스트에서 `Figure 1`, `Fig. 1`, `<Figure 1>`, `그림 1` 같은 언급을 찾는다(캡션 포함).
- PDF 페이지에서 캡션(블록이 `Figure N.`·`Fig. N.`·`그림 N.` 등으로 시작)을 찾고, 레이아웃 모델이 그림(picture)으로 분류한
  박스 중 캡션 바로 위·아래에 맞닿은 것을 그림 영역으로 본다. 영역과 캡션을 합쳐 PNG로 저장한다.
- 표는 같은 방식으로 `Table N.`·`표 N.` 캡션과 레이아웃의 table 박스(밑의 주석 포함)를 찾는다.
- 그림 영역을 못 찾으면 캡션이 있는 페이지 전체를 대신 저장하고 `located`에 표시한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pymupdf4llm

_MENTION_RE = re.compile(r"(?<![A-Za-z])(?:Figure|Fig\.?|그림)\s*(\d+)", re.IGNORECASE)
# 블록 맨 앞의 캡션. `<Figure 1>과 같다`처럼 문장 속 언급은 번호 뒤가 `.`·`:`·`>`+공백 등이 아니므로 제외된다.
_CAPTION_RE = re.compile(r"^\s*[<(\[]*\s*(?:Figure|Fig\.?|그림)\s*(\d+)\s*(?:[.:]|[>)\]]\s|\s*$)", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"^\s*[<(\[]*\s*(?:Table|표)\s*(\d+)\s*(?:[.:]|[>)\]]\s|\s*$)", re.IGNORECASE)
MAX_GAP = 60  # 캡션과 그림 박스 사이 최대 세로 간격(pt)
PAD = 6
X_MISMATCH_PENALTY = 10  # 캡션과 가로로 겹치지 않는 그림 박스에 주는 불이익(pt)
NOTE_GAP = 6  # 캡션 아래 범례·주석으로 볼 최대 세로 간격(pt)
MAX_NOTE_BLOCKS = 2
MIN_PICTURE = (80, 60)  # 로고 등 작은 박스를 거르는 최소 가로·세로(pt)
MIN_TABLE = (80, 30)
TABLE_NOTE_GAP = 10  # 표 밑의 주석 줄은 표와의 간격이 조금 더 넓다


@dataclass
class FigureInfo:
    number: int
    page: int  # 1부터
    bbox: list[float]
    caption: str
    located: str  # picture(그림 박스 확인) | page(그림 박스를 못 찾아 페이지 전체)


def mentioned_numbers(text: str) -> list[int]:
    return sorted({int(m.group(1)) for m in _MENTION_RE.finditer(text)})


def _overlap_x(a: list[float], b: list[float]) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > 10


def _extend_with_notes(box: list[float], blocks: list, caption_blk: tuple, note_gap: float) -> list[float]:
    for _ in range(MAX_NOTE_BLOCKS):
        nxt = next(
            (
                b
                for b in blocks
                if b is not caption_blk
                and 0 <= b[1] - box[3] <= note_gap
                and _overlap_x([b[0], b[1], b[2], b[3]], box)
                and not _CAPTION_RE.match(b[4])
                and not _TABLE_CAPTION_RE.match(b[4])
            ),
            None,
        )
        if nxt is None:
            break
        box = [min(box[0], nxt[0]), box[1], max(box[2], nxt[2]), nxt[3]]
    return box


# 종류별 설정: (캡션 정규식, 레이아웃 박스 클래스, 최소 가로·세로, 캡션 아래 주석으로 볼 최대 간격)
_KINDS = {
    "figure": (_CAPTION_RE, "picture", MIN_PICTURE, NOTE_GAP),
    "table": (_TABLE_CAPTION_RE, "table", MIN_TABLE, TABLE_NOTE_GAP),
}


def layout_pages(pdf_path: Path) -> list[dict]:
    """페이지별 레이아웃 분석(박스 클래스·좌표). 그림·표를 함께 찾을 때 한 번만 계산해 넘긴다."""
    return pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)


def locate(pdf_path: Path, wanted: set[int], kind: str = "figure", pages: list[dict] | None = None) -> dict[int, FigureInfo]:
    """원하는 번호의 그림(kind="figure") 또는 표(kind="table")를 찾는다. 찾지 못한 번호는 결과에 없다."""
    caption_re, box_class, (min_w, min_h), note_gap = _KINDS[kind]
    found: dict[int, FigureInfo] = {}
    pages = pages if pages is not None else layout_pages(pdf_path)
    with pymupdf.open(str(pdf_path)) as doc:
        for i, pg in enumerate(pages):
            boxes = [
                b["bbox"]
                for b in pg.get("page_boxes") or []
                if b.get("class") == box_class and b["bbox"][2] - b["bbox"][0] >= min_w and b["bbox"][3] - b["bbox"][1] >= min_h
            ]
            blocks = doc[i].get_text("blocks")
            for blk in blocks:
                m = caption_re.match(blk[4])
                if not m or int(m.group(1)) not in wanted or int(m.group(1)) in found:
                    continue
                num, cap = int(m.group(1)), list(blk[:4])
                best, best_key = None, (0.0, 0.0)
                for p in boxes:
                    gap = cap[1] - p[3] if p[3] <= cap[1] + 5 else p[1] - cap[3]  # 박스 위 캡션 / 박스 아래 캡션
                    if gap < -5 or gap > MAX_GAP:
                        continue
                    score = gap + (0 if _overlap_x(p, cap) else X_MISMATCH_PENALTY)  # 캡션이 박스보다 좁고 한쪽에 붙은 경우가 있어 가로 겹침은 가산점으로만 쓴다
                    if best is None or (score, gap) < best_key:  # 점수가 같으면 캡션에 더 가까운 박스
                        best, best_key = p, (score, gap)
                caption = re.sub(r"\s+", " ", blk[4]).strip()
                if best:
                    box = [min(best[0], cap[0]), min(best[1], cap[1]), max(best[2], cap[2]), max(best[3], cap[3])]
                    box = _extend_with_notes(box, blocks, blk, note_gap)  # 바로 밑에 붙은 범례·주석 줄(최대 2블록)도 포함한다
                    found[num] = FigureInfo(num, i + 1, [float(x) for x in box], caption, "picture")
                else:
                    r = doc[i].rect
                    found[num] = FigureInfo(num, i + 1, [float(r.x0), float(r.y0), float(r.x1), float(r.y1)], caption, "page")
    return found


def locate_figures(pdf_path: Path, wanted: set[int], pages: list[dict] | None = None) -> dict[int, FigureInfo]:
    return locate(pdf_path, wanted, "figure", pages)


def locate_tables(pdf_path: Path, wanted: set[int], pages: list[dict] | None = None) -> dict[int, FigureInfo]:
    return locate(pdf_path, wanted, "table", pages)


def save_png(pdf_path: Path, info: FigureInfo, out_path: Path, dpi: int = 150) -> None:
    with pymupdf.open(str(pdf_path)) as doc:
        page = doc[info.page - 1]
        rect = pymupdf.Rect(*info.bbox)
        if info.located == "picture":
            rect = pymupdf.Rect(rect.x0 - PAD, rect.y0 - PAD, rect.x1 + PAD, rect.y1 + PAD) & page.rect
        out_path.parent.mkdir(parents=True, exist_ok=True)
        page.get_pixmap(clip=rect, dpi=dpi).save(str(out_path))
