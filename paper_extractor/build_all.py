"""PDF -> Markdown -> 제목 계층 -> 청크를 한 번에 만들고, 달라진 논문만 다시 처리한다.

논문마다:
  1. 변환(`convert_pdf_md`)해 `outputs/markdown/`의 Markdown·메타데이터를 갱신한다.
  2. Markdown이 이전과 같고 제목·청크 결과도 있으면 건너뛴다.
  3. 달라졌으면 이전에 확정한 제목 구조를 새 줄에 다시 붙인다(LLM 호출 없음). 제목이 바뀌었으면(이전 제목을 못 찾거나 새 제목
     후보가 생김) 그때만 LLM으로 제목을 다시 추출한다.
  4. 제목 결과로 청크를 다시 만든다.
마지막에 논문별 처리 결과를 요약해 보여준다.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from . import chunk_sections
from .config import Config
from .convert_pdf_md import _pdf_files, convert, paper_number
from .extract_title import (
    build_outline,
    check_outline,
    load_env,
    outline_markdown,
    process as extract_titles,
    reanchor,
    save_result,
    unparsable_reason,
)


@dataclass
class Row:
    num: str
    status: str
    detail: str = ""


def build_one(pdf: Path, base: Path, cfg: Config, force_llm: bool) -> Row:
    num = paper_number(pdf)
    md_dir, titles_dir, chunks_dir = base / "markdown", base / "titles", base / "chunks"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_dir / f"{pdf.stem}.md"
    old_md = md_path.read_text(encoding="utf-8") if md_path.exists() else None

    markdown, meta = convert(pdf)
    md_path.write_text(markdown, encoding="utf-8")
    (md_dir / f"{num}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    reason = unparsable_reason(markdown)
    if reason:
        return Row(num, "제외", reason)

    titles_json = titles_dir / f"{num}_title.json"
    chunks_json = chunks_dir / f"{num}_chunks.json"
    changed = old_md != markdown
    if not changed and titles_json.exists() and chunks_json.exists() and not force_llm:
        return Row(num, "변경 없음", "Markdown이 이전과 같아 건너뜀")

    how, detail = "", ""
    if titles_json.exists() and old_md is not None and not force_llm:
        old_data = json.loads(titles_json.read_text(encoding="utf-8"))
        headings, why = reanchor(old_data, old_md, markdown.splitlines())
        if headings is not None:
            lines = markdown.splitlines()
            outline, warnings = build_outline(lines, headings)
            data = {
                "source": md_path.name,
                "model": old_data.get("model"),
                "anchored_from": "previous",
                "chunks": [],
                "usage": {"api_calls_attempted": 0},
                "warnings": warnings,
                "checks": check_outline(lines, headings),
                "outline": outline,
            }
            save_result(md_path, titles_dir, data, outline_markdown(outline))
            how, detail = "재부착", why
        else:
            detail = f"재부착 실패({why}) → LLM으로 다시 추출"
    if not how:
        ok = extract_titles(md_path, cfg, titles_dir, False)
        if not ok:
            return Row(num, "오류", "LLM 제목 추출 실패")
        how = "LLM 추출"
        data = json.loads(titles_json.read_text(encoding="utf-8"))
        detail = (detail + "; " if detail else "") + f"호출 {data['usage'].get('api_calls_attempted', 0)}회"

    chunk_sections.process(titles_json, cfg.chunk_target_tokens, chunks_dir, False, False, pdf.parent)
    return Row(num, f"갱신({how})", detail)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="변환·제목 추출·청크를 한 번에 만들고 달라진 논문만 다시 처리합니다.")
    parser.add_argument("paths", nargs="*", type=Path, default=[Path(__file__).resolve().parent / "papers"], help="PDF 파일 또는 폴더")
    parser.add_argument("--out", type=Path, default=root / "outputs", help="산출물 폴더 (기본: outputs)")
    parser.add_argument("--force-llm", action="store_true", help="이전 제목 구조를 쓰지 않고 항상 LLM으로 제목을 다시 추출")
    args = parser.parse_args(argv)

    load_env(root / ".env")
    cfg = Config()
    cfg.validate()
    pdfs, failed = _pdf_files(args.paths)
    if not pdfs:
        print("PDF 파일을 찾지 못했습니다.", file=sys.stderr)
        return 1

    rows: list[Row] = []
    for pdf in pdfs:
        print(f"\n== {pdf.name}")
        try:
            rows.append(build_one(pdf, args.out, cfg, args.force_llm))
        except Exception as exc:  # noqa: BLE001 - 논문 하나의 실패가 나머지를 막지 않게 한다
            failed = True
            rows.append(Row(paper_number(pdf), "오류", f"{type(exc).__name__}: {exc}"))
            print(f"[오류] {pdf.name}: {exc}", file=sys.stderr)

    print("\n===== 요약 =====")
    for r in rows:
        print(f"{r.num}  {r.status:<14} {r.detail}")
    return 1 if failed or any(r.status == "오류" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
