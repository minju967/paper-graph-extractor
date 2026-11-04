"""Population 노드 속성 추출 호출의 시스템 프롬프트와 출력 스키마. LLM을 호출하지 않는다.

한 번의 호출로 논문의 집단 목록을 받는다. 집단마다 속성은 값(value)·근거 문장 번호(evidence)·메모(note)다.
입력 맨 앞에 `Paper`에서 알아낸 연구 유형을 알려 준다.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..attrs import TextAttr

SYSTEM_PROMPT = """\
You are given parts of ONE academic paper (text chunks). The first line of the input states the study type of the paper.
Extract the "Population" nodes: the participant groups this paper actually studied or analyzed.
Extract only about the CURRENT paper, never groups taken from cited studies.

How to form groups:
- Create one Population per distinct participant group of this paper. Do not split by outcome, time point or condition.
- Intervention / observational study (study type "중재 연구" or "관찰 연구"): the intervention group and the control group
  of a parallel design are two groups; participants who take part in every condition of a crossover design are one group.
  group_level is "개별 연구집단".
- Literature synthesis (study type "문헌 종합 연구", i.e. review or meta-analysis): create a Population only for the
  paper's main subject populations, i.e. the categories of people for whom the paper itself presents results or
  recommendations (e.g. "adults with hypertension"). Do NOT create a group for a category that is only mentioned in
  passing, for example inside a sentence describing a cited study. If there is one main subject population, create
  one group with group_role "전체 분석 집단"; use "하위집단" only for a category that is a part of another group.
  group_level is "종합 분석 집단". group_name is usually null; fill age and sex only if the paper itself reports them
  for that category.
- Do not create a disease group just because blood pressure etc. was measured in healthy participants.

General rules:
- Keep the original language and wording of the paper. Use null when an attribute is not stated; never guess.
- Every sentence of the input is prefixed with an id in square brackets, e.g. `[h4.s2] ...`.
  For every attribute with a value, give evidence: the ids of the sentences that support the value
  (e.g. ["h4.s1"]), without the brackets. Use only ids that appear in the input. Never write quotes.
  Use an empty list when the value is null.
- note is for short remarks (original wording, ambiguity). Use null if there is nothing to add.

Attributes of each group (required: condition):
- condition: the disease, pre-disease stage or health state confirmed for this group. If there are several, join them
  with "; " and say in note whether they are comorbid or subgroups.
- age: as reported (mean ± SD, actual range or wording). Not the recruitment range.
- sex: the actual composition of the enrolled participants.
- risk_group: the paper's own risk classification and its criteria. Do not map it to any other scheme.
- group_name: the label used in the paper for this group (e.g. "EX", "운동군").
- group_level: "개별 연구집단" or "종합 분석 집단".
- group_role: one of "중재군", "대조군", "전체 분석 집단", "하위집단"; null if none fits. If the paper has only one group
  (e.g. a crossover design or a single-group before/after design), use "전체 분석 집단".
"""


class PopulationItem(BaseModel):
    condition: TextAttr
    age: TextAttr
    sex: TextAttr
    risk_group: TextAttr
    group_name: TextAttr
    group_level: TextAttr
    group_role: TextAttr


class PopulationOut(BaseModel):
    populations: list[PopulationItem]
