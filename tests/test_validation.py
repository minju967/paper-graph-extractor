from paper_extractor.models import Block, Document, GraphRelation, PageInfo
from paper_extractor.validation import validate_and_finalize

from conftest import make_merge


def _doc():
    blocks = [Block(id="p1-b1", page=1, kind="paragraph", text="Ten men with prediabetes"),
              Block(id="p2-b2", page=2, kind="paragraph", text="Glucose fell (p<0.05)")]
    return Document(pdf_name="x", pdf_sha256="0" * 64, converter="t", page_count=2,
                    pages=[PageInfo(page=1, char_count=9, garbled_ratio=0), PageInfo(page=2, char_count=9, garbled_ratio=0)],
                    blocks=blocks)


def _run(merged, partial=()):
    return validate_and_finalize(merged, _doc(), paper_id="x", model="m", schema_version="1", prompt_version="1",
                                 partial_reasons=list(partial))


def test_valid_graph_keeps_relations_and_is_unreviewed():
    g = _run(make_merge())
    assert len(g.relations) == 4
    assert g.status.review_status == "unreviewed" and g.status.complete


def test_fabricated_evidence_id_removed_and_reported_value_downgraded():
    m = make_merge()
    m.populations[0].conditions.evidence_block_ids = ["p9-b9"]
    g = _run(m)
    assert g.populations[0].conditions.status == "unverifiable"
    assert any(r.kind == "invalid_evidence" for r in g.review_items)


def test_relation_not_allowed_for_study_type_is_held_not_created():
    m = make_merge(study_type="meta_analysis")  # 중재용 관계가 들어 있음
    g = _run(m)
    assert g.relations == []
    assert sum(r.kind == "held_relation" for r in g.review_items) == 4


def test_wrong_direction_and_missing_evidence_are_held():
    rels = [GraphRelation(type="PARTICIPATES_IN", from_id="prg_1", to_id="pop_1", evidence_block_ids=["p1-b1"]),
            GraphRelation(type="HAS_OUTCOME", from_id="prg_1", to_id="out_1", evidence_block_ids=[])]
    g = _run(make_merge(relations=rels))
    assert g.relations == []


def test_participates_in_blocked_for_aggregate_population():
    g = _run(make_merge(extra_pop_level="aggregate"))
    assert all(r.type != "PARTICIPATES_IN" for r in g.relations)
    assert any("종합" in r.description for r in g.review_items)


def test_not_reported_becomes_unverifiable_when_partial():
    g = _run(make_merge(), partial=["처리하지 못한 청크: ['c03']"])
    assert g.paper.authors.status == "unverifiable"
    assert not g.status.complete


def test_number_not_in_cited_block_raises_support_warning():
    m = make_merge()
    m.outcomes[0].result_value.value = "decreased by 12.5 mg/dL"
    g = _run(m)
    assert any(r.kind == "support_warning" and "12.5" in r.description for r in g.review_items)
