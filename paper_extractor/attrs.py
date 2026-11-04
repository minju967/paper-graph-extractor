"""노드 속성 추출 결과의 공통 속성 형식: 값(value), 근거 문장 번호(evidence), 메모(note)."""

from __future__ import annotations

from pydantic import BaseModel


class TextAttr(BaseModel):
    value: str | None
    evidence: list[str]  # 근거 문장 번호(예: h4.s1)
    note: str | None


class NumAttr(BaseModel):
    value: int | None
    evidence: list[str]  # 근거 문장 번호(예: h4.s1)
    note: str | None
