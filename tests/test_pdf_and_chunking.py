from paper_extractor.chunking import chunk_document, plan_chunks
from paper_extractor.models import Block, Document, PageInfo
from paper_extractor.pdf_to_md import convert_pdf, split_blocks


def test_convert_assigns_block_ids_pages_and_marks_references(sample_pdf):
    doc = convert_pdf(sample_pdf)
    assert doc.page_count == 2
    assert len({b.id for b in doc.blocks}) == len(doc.blocks)
    assert all(b.id.startswith(f"p{b.page}-b") for b in doc.blocks)
    refs = [b for b in doc.blocks if b.excluded]
    assert refs and all(b.exclude_reason == "references" for b in refs)
    # 참고문헌 이전의 본문은 유지된다
    assert any("Glucose fell" in b.text and not b.excluded for b in doc.blocks)
    assert doc.converter.startswith("pymupdf4llm==")


def test_low_text_pages_are_flagged(sample_pdf):
    doc = convert_pdf(sample_pdf, min_chars=100000)
    assert all("low_text" in p.readability_flags for p in doc.pages)


def test_no_reference_heading_deletes_nothing():
    text = "## Intro\n\nbody\n"
    assert [k for k, *_ in split_blocks(text)] == ["heading", "paragraph"]


def test_split_blocks_keeps_table_together():
    md = "Table 1. Results\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nNote. x\n"
    kinds = [k for k, *_ in split_blocks(md)]
    assert kinds == ["paragraph", "table", "paragraph"]


def _doc(blocks, pages=1):
    return Document(pdf_name="x", pdf_sha256="0" * 64, converter="t", page_count=pages,
                    pages=[PageInfo(page=i + 1, char_count=999, garbled_ratio=0) for i in range(pages)], blocks=blocks)


def _b(i, kind, text, page=1, level=None):
    return Block(id=f"p{page}-b{i}", page=page, kind=kind, text=text, heading_level=level)


def test_chunks_cover_every_included_block_in_order():
    blocks = [_b(1, "heading", "Intro", level=2), _b(2, "paragraph", "word " * 50),
              _b(3, "heading", "Methods", level=2), _b(4, "paragraph", "word " * 50)]
    chunks = chunk_document(_doc(blocks), 4000, 6000)
    ids = [bid for c in chunks for bid in c.block_ids]
    assert ids == ["p1-b1", "p1-b2", "p1-b3", "p1-b4"]


def test_references_excluded_from_chunks():
    blocks = [_b(1, "paragraph", "body text"), _b(2, "paragraph", "ref list")]
    blocks[1].excluded = True
    chunks = chunk_document(_doc(blocks), 4000, 6000)
    assert [bid for c in chunks for bid in c.block_ids] == ["p1-b1"]


def test_long_section_split_keeps_heading_path_and_body_not_lost():
    blocks = [_b(1, "heading", "Results", level=2)] + [_b(i, "paragraph", "sentence. " * 150) for i in range(2, 12)]
    chunks = chunk_document(_doc(blocks), 800, 1200)
    assert len(chunks) > 1
    assert all(c.heading_path == ["Results"] for c in chunks)
    assert {bid for c in chunks for bid in c.block_ids} == {b.id for b in blocks}


def test_long_table_split_repeats_header_and_marks_continuation():
    rows = "\n".join(f"| row{i} | {'x' * 40} |" for i in range(200))
    table = "| col A | col B |\n|---|---|\n" + rows
    blocks = [_b(1, "paragraph", "Table 1. Big table"), _b(2, "table", table)]
    chunks = chunk_document(_doc(blocks), 500, 700)
    assert len(chunks) > 1
    for c in chunks:
        tparts = [p for p in c.parts if p.block_id == "p1-b2"]
        assert tparts[0].text.startswith("| col A | col B |")
    assert any(p.is_table_continuation for c in chunks[1:] for p in c.parts)
    assert all(c.parts[0].text.startswith("Table 1") for c in chunks)


def test_no_headings_splits_by_page_with_warning():
    blocks = [_b(1, "paragraph", "page one text", page=1), _b(2, "paragraph", "page two text", page=2)]
    chunks = chunk_document(_doc(blocks, pages=2), 5, 6000)
    assert any("제목 인식 실패" in w for c in chunks for w in c.warnings)


def test_plan_chunks_grows_chunks_to_fit_slots_but_reports_when_impossible():
    blocks = [_b(1, "heading", "A", level=1)] + [_b(i, "paragraph", "word " * 400) for i in range(2, 40)]
    doc = _doc(blocks)
    chunks, t, m, notes = plan_chunks(doc, 1000, 1500, 24000, 6)
    assert len(chunks) <= 6 and t > 1000 and notes
    chunks, t, m, notes = plan_chunks(doc, 1000, 1500, 1500, 2)
    assert len(chunks) > 2 and any("슬롯" in n for n in notes)
