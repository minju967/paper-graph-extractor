from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from paper_extractor.config import Config
from paper_extractor.llm import LLMError, Usage
from paper_extractor.models import (
    Attr, ChunkExtraction, ImageExtraction, MergeOutput, OutcomeNode, PageTranscript, PaperNode,
    PopulationNode, ProgramNode, Activity, GraphRelation, TypeDecision,
)


def _page(doc, lines):
    p = doc.new_page()
    y = 72
    for text, size in lines:
        p.insert_text((72, y), text, fontsize=size)
        y += size + 14
    return p


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    long_text = "Participants performed the exercise and glucose was measured repeatedly. " * 12
    _page(doc, [("Effects of Exercise on Glucose", 20), ("Abstract", 14), (long_text[:90], 10),
                (long_text[90:180], 10), ("1. Methods", 14), ("Ten men with prediabetes took part.", 10)])
    _page(doc, [("2. Results", 14), ("Glucose fell after HIIE (p<0.05).", 10), ("References", 14),
                ("1. Smith J. Some paper. 2010.", 10)])
    path = tmp_path / "sample.pdf"
    doc.save(str(path))
    return path


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    # 샘플 PDF는 짧으므로 '글자 수 부족' 신호가 기본으로 켜지지 않게 한다.
    return Config(model="fake-model", output_dir=tmp_path / "out", min_chars_per_page=10)


def make_attr(value, ids=("p1-b1",), status="reported"):
    return Attr(value=value, status=status, evidence_block_ids=list(ids))


def empty_attr():
    return Attr(value=None, status="not_reported", note="확인한 범위에서 찾지 못함")


def make_merge(study_type="intervention", relations=None, extra_pop_level="individual"):
    paper = PaperNode(
        id="paper", title=make_attr("Effects of Exercise on Glucose"), authors=empty_attr(), year=empty_attr(),
        journal=empty_attr(), doi_url=empty_attr(), study_type=make_attr(study_type), inclusion_criteria=empty_attr(),
        exclusion_criteria=empty_attr(), recruit_age=empty_attr(),
    )
    pop = PopulationNode(
        id="pop_1", name=make_attr("Prediabetic men"), level=make_attr(extra_pop_level), role=empty_attr(),
        conditions=make_attr("prediabetes"), risk_group=empty_attr(), participant_age=empty_attr(), sex=make_attr("men"),
    )
    prg = ProgramNode(
        id="prg_1", name=make_attr("HIIE"), level=make_attr("individual"), role=make_attr("중재"), duration=empty_attr(),
        activities=[Activity(activity_type=make_attr("운동"), name=empty_attr(), frequency=empty_attr(),
                             duration_per_session=empty_attr(), intensity_or_intake=empty_attr())],
    )
    out = OutcomeNode(
        id="out_1", indicator=make_attr("glucose"), result_value=make_attr("fell (p<0.05)", ids=("p2-b2",)), unit=empty_attr(),
        comparison_type=empty_attr(), timepoint=empty_attr(), statistics=make_attr("p<0.05", ids=("p2-b2",)),
    )
    rels = relations if relations is not None else [
        GraphRelation(type="HAS_POPULATION", from_id="paper", to_id="pop_1", evidence_block_ids=["p1-b1"]),
        GraphRelation(type="PARTICIPATES_IN", from_id="pop_1", to_id="prg_1", evidence_block_ids=["p1-b1"]),
        GraphRelation(type="HAS_OUTCOME", from_id="prg_1", to_id="out_1", evidence_block_ids=["p2-b2"]),
        GraphRelation(type="MEASURED_IN", from_id="out_1", to_id="pop_1", evidence_block_ids=["p2-b2"]),
    ]
    return MergeOutput(final_study_type=study_type, type_note="", paper=paper, populations=[pop], programs=[prg],
                       outcomes=[out], relations=rels, review_items=[])


class FakeClient:
    """스키마 유형별로 미리 정한 응답을 돌려주는 가짜 LLM. 호출 기록을 남긴다."""

    def __init__(self, merge=None, fail_stages=(), study_type="intervention"):
        self.calls: list[str] = []
        self.merge = merge or make_merge()
        self.fail_stages = set(fail_stages)
        self.study_type = study_type

    def parse(self, *, stage, system, user, schema, max_output_tokens, images=None):
        self.calls.append(stage)
        if stage in self.fail_stages:
            raise LLMError("fake failure", retryable=False)
        usage = Usage(input_tokens=100, output_tokens=10)
        if schema is TypeDecision:
            return TypeDecision(study_type=self.study_type, evidence_block_ids=["p1-b1"], design_description="d",
                                cautions=[], first_page_unreadable=False), usage
        if schema is ChunkExtraction:
            return ChunkExtraction(entities=[], facts=[], relations=[], unresolved=[], readability_block_ids=[],
                                   not_applicable_note=""), usage
        if schema is PageTranscript:
            return PageTranscript(transcript="transcribed", notes=[]), usage
        if schema is ImageExtraction:
            ext = ChunkExtraction(entities=[], facts=[], relations=[], unresolved=[], readability_block_ids=[],
                                  not_applicable_note="")
            return ImageExtraction(transcript="image text", extraction=ext), usage
        if schema is MergeOutput:
            return self.merge, usage
        raise AssertionError(schema)
