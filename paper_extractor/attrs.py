"""노드 속성 추출 결과의 공통 속성 형식: 값(value), 근거 문장 번호(evidence), 메모(note)."""

from __future__ import annotations

from pydantic import BaseModel, computed_field

STUDY_TYPES = ["중재 연구", "관찰 연구", "문헌 종합 연구"]


STATUS_REPORTED = "보고됨"
STATUS_NOT_REPORTED = "미보고"


class TextAttr(BaseModel):
    value: str | None
    evidence: list[str]  # 근거 문장 번호(예: h4.s1)
    note: str | None

    @computed_field  # LLM이 답하지 않는다: 값을 찾았으면 보고됨, 없으면 미보고(v1.40)
    @property
    def status(self) -> str:
        return STATUS_REPORTED if self.value is not None else STATUS_NOT_REPORTED


class NumAttr(BaseModel):
    value: int | None
    evidence: list[str]  # 근거 문장 번호(예: h4.s1)
    note: str | None

    @computed_field
    @property
    def status(self) -> str:
        return STATUS_REPORTED if self.value is not None else STATUS_NOT_REPORTED
