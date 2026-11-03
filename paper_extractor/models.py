"""데이터 모델.

- 문서 모델(Block/Page/Chunk): 코드가 관리하는 원문·위치 정보. LLM이 만들지 않는다.
- 단계 출력 스키마(TypeDecision/ChunkExtraction/...): LLM 구조화 출력용.
- 최종 그래프(PaperGraph): 논문별 JSON.

주의: 물리적 JSON 키·자료형은 계획서 9절에서 미정이므로 여기 정의는 '구현 기본안'이다.
값은 원문 표현을 보존하는 문자열로 두고, 정규화는 하지 않는다.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 문서 모델 (코드 관리)
# ---------------------------------------------------------------------------

BlockKind = Literal["heading", "paragraph", "table", "image_transcript"]


class Block(BaseModel):
    id: str  # 예: p3-b12 (페이지 3의 12번째 블록), 이미지 판독문은 p3-img
    page: int  # 1-based
    kind: BlockKind
    text: str
    heading_level: Optional[int] = None
    bbox: Optional[list[float]] = None  # 변환 결과에서 확인된 경우에만 보존
    box_class: Optional[str] = None  # PyMuPDF4LLM이 부여한 분류(있을 때)
    excluded: bool = False  # 참고문헌 구간
    exclude_reason: Optional[str] = None
    source: Literal["text", "image"] = "text"


class PageInfo(BaseModel):
    page: int
    char_count: int
    garbled_ratio: float
    garbled_chars: int = 0
    is_reference_only: bool = False
    readability_flags: list[str] = Field(default_factory=list)  # 예: low_text, garbled


class Document(BaseModel):
    pdf_name: str
    pdf_sha256: str
    converter: str
    page_count: int
    pages: list[PageInfo]
    blocks: list[Block]
    warnings: list[str] = Field(default_factory=list)

    def block_map(self) -> dict[str, Block]:
        return {b.id: b for b in self.blocks}

    def page_blocks(self, page: int) -> list[Block]:
        return [b for b in self.blocks if b.page == page]


class ChunkPart(BaseModel):
    block_id: str
    text: str  # 청크에 실제로 들어가는 텍스트(표 분할 시 일부)
    is_table_continuation: bool = False  # 긴 표를 나눈 뒤 이어지는 부분 → 중복 근거 식별


class Chunk(BaseModel):
    id: str  # c01 ...
    heading_path: list[str]
    parts: list[ChunkPart]
    token_estimate: int
    warnings: list[str] = Field(default_factory=list)
    status: Literal["pending", "done", "skipped_budget", "failed"] = "pending"

    @property
    def block_ids(self) -> list[str]:
        seen: list[str] = []
        for p in self.parts:
            if p.block_id not in seen:
                seen.append(p.block_id)
        return seen


# ---------------------------------------------------------------------------
# 1단계: 연구 유형 판정
# ---------------------------------------------------------------------------

StudyType = Literal["intervention", "meta_analysis", "other", "undetermined"]


class TypeDecision(BaseModel):
    study_type: StudyType
    evidence_block_ids: list[str]
    design_description: str  # 독립 집단 vs 동일 집단 반복 조건, 중재 범주·종합 분석 성격 등
    cautions: list[str]  # 미확인·주의 사항
    first_page_unreadable: bool  # 판독 문제로 판정이 불가능한 경우


class PageTranscript(BaseModel):
    transcript: str
    notes: list[str]


# ---------------------------------------------------------------------------
# 2단계: 청크별 정보 수집
# ---------------------------------------------------------------------------

EntityType = Literal["Paper", "Population", "Program", "Outcome"]
SourceRole = Literal["current_study", "included_study", "cited_study", "author_recommendation", "unclear"]
Level = Literal["individual", "aggregate", "subanalysis", "unspecified"]
RelationType = Literal[
    "HAS_POPULATION",
    "PARTICIPATES_IN",
    "HAS_OUTCOME",
    "MEASURED_IN",
    "COMPARED_WITH_PROGRAM",
    "REPORTS",
    "ANALYZES_POPULATION",
    "EVALUATES_PROGRAM",
]


class EntityClue(BaseModel):
    """청크 안에서 쓰는 임시 개체 단서. 최종 노드 ID가 아니다."""

    temp_id: str  # 청크 안에서만 유효. 예: pop_A
    entity_type: EntityType
    label: str  # 원문 표현 (집단명/조건명/지표명 등). 창작 금지
    level: Level
    evidence_block_ids: list[str]


class AttributeFact(BaseModel):
    entity_temp_id: str  # EntityClue.temp_id 또는 "paper"
    entity_type: EntityType
    attribute: str  # 속성 정의서의 속성명
    value: str  # 원문 표현 보존
    applies_to: str  # 어느 대상·조건·분석·지표·시점에 적용되는지
    source_role: SourceRole
    level: Level
    evidence_block_ids: list[str]


class RelationFact(BaseModel):
    """관계 연결에 필요한 참여·비교·분석 사실. 최종 간선이 아니다."""

    relation: RelationType
    from_temp_id: str
    to_temp_id: str
    statement: str  # 근거가 말하는 사실을 한 문장으로
    source_role: SourceRole
    evidence_block_ids: list[str]


class Unresolved(BaseModel):
    kind: Literal[
        "unresolved_reference",  # 다른 구간·보충자료 확인 필요
        "contradiction",
        "readability",
        "type_conflict",  # 1단계 연구 유형과 모순
        "supplement_missing",  # 본문이 참조하는 보충자료가 입력에 없음
        "other",
    ]
    description: str
    block_ids: list[str]


class ChunkExtraction(BaseModel):
    entities: list[EntityClue]
    facts: list[AttributeFact]
    relations: list[RelationFact]
    unresolved: list[Unresolved]
    # 판독 문제가 있다고 LLM이 명시한 블록 (이미지 보완 후보)
    readability_block_ids: list[str]
    not_applicable_note: str  # 이 청크에 추출할 정보가 없을 때 이유 (전체 논문 미보고 판정 아님)


class ImageExtraction(BaseModel):
    transcript: str
    extraction: ChunkExtraction


# ---------------------------------------------------------------------------
# 3단계: 최종 그래프
# ---------------------------------------------------------------------------

InfoStatus = Literal["reported", "not_reported", "not_applicable", "unverifiable"]
NodeType = Literal["Paper", "Population", "Program", "Outcome"]


class Attr(BaseModel):
    """속성값 + 정보 상태 + 근거. 원문 표현을 그대로 보존한다."""

    value: Optional[str] = None
    status: InfoStatus = "not_reported"
    evidence_block_ids: list[str] = Field(default_factory=list)
    note: Optional[str] = None  # 값이 없는 이유와 확인 범위, 충돌 설명 등


class PaperNode(BaseModel):
    id: str
    title: Attr
    authors: Attr
    year: Attr
    journal: Attr
    doi_url: Attr
    study_type: Attr
    inclusion_criteria: Attr
    exclusion_criteria: Attr
    recruit_age: Attr  # 보고된 최소·최대값을 원문 표현으로


class PopulationNode(BaseModel):
    id: str
    name: Attr
    level: Attr  # individual | aggregate
    role: Attr
    conditions: Attr  # 질병·건강 상태. 복수면 구분 설명을 value에
    risk_group: Attr  # 원문 위험군
    participant_age: Attr  # 보고 형태·수치·원문 표현
    sex: Attr


class Activity(BaseModel):
    activity_type: Attr  # 운동 | 식단
    name: Attr
    frequency: Attr
    duration_per_session: Attr
    intensity_or_intake: Attr


class ProgramNode(BaseModel):
    id: str
    name: Attr
    level: Attr  # individual | aggregate
    role: Attr  # 중재 | 대조
    duration: Attr
    activities: list[Activity]


class OutcomeNode(BaseModel):
    id: str
    indicator: Attr
    result_value: Attr  # 종류·값·부호
    unit: Attr
    comparison_type: Attr
    timepoint: Attr
    statistics: Attr


class GraphRelation(BaseModel):
    type: RelationType
    from_id: str
    to_id: str
    evidence_block_ids: list[str]
    note: Optional[str] = None


class ReviewItem(BaseModel):
    kind: str  # conflict | held_relation | invalid_evidence | unprocessed | type_conflict | support_warning ...
    description: str
    related_ids: list[str] = Field(default_factory=list)
    evidence_block_ids: list[str] = Field(default_factory=list)


class MergeOutput(BaseModel):
    """통합 호출의 LLM 출력."""

    final_study_type: StudyType
    type_note: str
    paper: PaperNode
    populations: list[PopulationNode]
    programs: list[ProgramNode]
    outcomes: list[OutcomeNode]
    relations: list[GraphRelation]
    review_items: list[ReviewItem]


class RunStatus(BaseModel):
    complete: bool
    partial_reasons: list[str] = Field(default_factory=list)
    review_status: Literal["unreviewed", "reviewed"] = "unreviewed"
    reviewer: Optional[str] = None


class PaperGraph(BaseModel):
    paper_id: str
    schema_version: str
    prompt_version: str
    model: str
    study_type: StudyType
    type_note: str
    status: RunStatus
    paper: PaperNode
    populations: list[PopulationNode]
    programs: list[ProgramNode]
    outcomes: list[OutcomeNode]
    relations: list[GraphRelation]
    review_items: list[ReviewItem]
