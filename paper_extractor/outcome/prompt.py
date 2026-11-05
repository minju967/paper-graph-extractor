"""Outcome 노드 속성 추출 호출의 시스템 프롬프트와 출력 스키마. LLM을 호출하지 않는다.

한 번의 호출로 논문의 결과 목록을 받는다. 결과마다 속성은 값(value)·근거 문장 번호(evidence)·메모(note)이고,
Program·Population과의 연결은 입력에 주어진 목록의 번호(program_ids 등)로 답한다.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..attrs import TextAttr

INDICATOR_CATEGORIES = ["혈압", "혈당", "간수치", "비만", "기타"]
COMPARISON_TYPES = ["집단 내 전후", "조건 간", "집단 간·상호작용", "통합 분석"]
DIRECTIONS = ["감소", "동일", "증가"]

SYSTEM_PROMPT = """\
You are given parts of ONE academic paper (text chunks), the Programs and Populations already extracted from it
(each with an id such as P1 / G1), and, in the first line, the study type of the paper.
Extract the "Outcome" nodes: the results THIS paper reports.
Never extract results of cited studies (citations such as [12], "선행연구", "et al. (year)").

How to form outcomes:
- One Outcome = one indicator x program x population x comparison target x comparison type. Time points are NOT a
  splitting unit: write them in measurement_time. Different indicators are different outcomes (e.g. SBP and DBP).
- The same result reported in the abstract, the text and a table is ONE outcome. If the text and a table disagree, keep
  the text's value and say so in note; do not correct either.
- Report non-significant results too (direction "동일", see below).
- Do not judge whether a result is an improvement; do not compute any new effect.
- Do NOT create an Outcome that only compares participant subgroups with each other (e.g. "the effect was greatest in
  hypertensive participants", "no difference by sex or age", a test for a difference between subgroups). Extract each
  subgroup's OWN effect as its own Outcome instead. If the paper states how the subgroups differ, you may mention it in
  the note of those Outcomes.

Target program (only when a line "Target program: Pk (...)" follows the study type line):
- Extract ONLY the outcomes that evaluate that Program (its id is in program_ids). Ignore outcomes of the other Programs
  (they are extracted in separate calls). The other Programs may still appear in comparison_program_ids.
- A result that compares two Programs directly is extracted ONLY in the call of its evaluated Program (the rule is
  under "direction" below). If the target is not the evaluated Program of that result, skip it: another call extracts it.
- If no outcome of that Program is reported, return an empty list.

Links to the already extracted nodes (use ONLY ids that appear in the lists; never invent ids):
- program_ids: the Program(s) whose effect the outcome evaluates.
- population_ids: the Population(s) in which the outcome was analyzed.
- comparison_program_ids: the Program(s) it is compared with (e.g. the control condition or the other exercise
  condition). Empty for a before/after comparison within one condition.
- link_evidence: the ids of the sentences (or table rows) that show which program and population the result belongs to.
- link_note: short remark about the links, or null.

General rules:
- Keep the original language and wording of the paper. Use null when an attribute is not stated; never guess.
- Every sentence of the input is prefixed with an id in square brackets, e.g. `[h14.s2] ...`.
  For every attribute with a value, give evidence: the ids of the supporting sentences (e.g. ["h14.s2"]), without the
  brackets. Use only ids that appear in the input. Never write quotes. Use an empty list when the value is null.
- note is for short remarks (original wording, ambiguity, conflicts). Use null if there is nothing to add.

Attributes of each outcome (required: indicator, direction):
- indicator: the health indicator that was evaluated, in the paper's wording (e.g. "수축기 혈압", "SBP").
- indicator_category: exactly one of "혈압", "혈당", "간수치", "비만", "기타" (Korean strings, written exactly as listed);
  anything that is not blood pressure, blood glucose (including insulin resistance), liver enzymes or obesity/body
  composition is "기타".
- direction: exactly one of "감소", "동일", "증가" (Korean strings, written exactly as listed), or null.
  It says how the indicator of the evaluated Program (program_ids) turned out compared with what it is compared with
  (comparison_program_ids such as the control or another Program, or the baseline state):
  - "감소": the indicator was lower / decreased (e.g. "significantly reduced", "larger reduction than ...").
  - "증가": the indicator was higher / increased.
  - "동일": the paper says there was NO significant difference or change. A small numeric change that the paper calls
    not significant is still "동일"; describe the trend in note (e.g. "tended to be greater, not significant").
    Significance decides, not the size or sign of the change: if the paper says the difference/change was not
    significant (also in a clause such as "reductions were larger ..., although they were not significant" or
    "significant only for SBP" -> DBP is not significant), the direction MUST be "동일", never "감소" or "증가".
    Use "감소" / "증가" only when the paper reports the change or difference as significant (or states a significant
    reduction/increase). Before answering, check that direction agrees with what you wrote in note.
  - null: the paper does not say which way it went in words. This is REQUIRED when the sentence gives only numbers
    (an estimate with a confidence interval, e.g. "-0.75 [-2.2 to 0.69]") or only "the effect was greatest in group X"
    without a before/after or Program comparison direction. NEVER guess the direction from the sign of a number: the
    program decides it from the confidence interval. Explain in note.
  Do NOT put numbers or the paper's wording into direction; numbers belong in representative_values and
  significance_statistics, the wording can be written in note.
  Comparing two Programs directly: the evaluated Program (program_ids) is the Program that the sentence names as the
  SUBJECT, i.e. the one reported to have done better / reduced more, and direction is relative to the compared Program
  (e.g. "isometric training reduced SBP more than endurance training" -> program_ids = isometric, comparison_program_ids
  = endurance, direction = "감소"). When the comparison is symmetric ("no differences among A, B and C"), direction is
  "동일" and the evaluated Program is the one with the SMALLER number in each pair (A-B: A, A-C: A, B-C: B).
- significance_statistics: the p-value / confidence interval of THIS result, with the test or analysis it belongs to,
  as written. Use a p-value only when the sentence ties it to this result. A p-value that belongs to a different test
  (e.g. a test for the difference BETWEEN subgroups) must not be copied onto each subgroup's own result: use null and
  say so in note.
- representative_values: only values the paper states (pre/post, change, pooled effect); keep signs.
- comparison_type: exactly one of "집단 내 전후", "집단 간·상호작용", "조건 간", "통합 분석" (Korean strings, written
  exactly as listed). Decide by asking "WHAT is being compared in this result?" and go through the list in this order,
  taking the first one that fits:
  1. "집단 내 전후" - Only ONE condition and ONE group: the same group before vs after (or baseline vs follow-up).
     No other group or program is involved. Example: exercise group's blood pressure before vs after 8 weeks.
  2. "통합 분석" - Only for a literature synthesis (study type "문헌 종합 연구"). The result is ONE pooled effect
     obtained by combining many studies, for one Program compared with its control or with the baseline state. This also
     covers the pooled effect inside a subgroup. Example: pooled SBP change of endurance training vs control = -3.5 mmHg.
  3. "조건 간" - Two different Programs are compared DIRECTLY with each other, or the same participants went through
     several conditions (crossover). Example: isometric vs dynamic resistance training.
  4. "집단 간·상호작용" - The groups that received different conditions are compared in a parallel-group study
     (exercise group vs control group), or a group x time interaction between such groups is reported.
     Example: "exercise group vs control group, group x time interaction p=.03".
  When two choices seem possible, follow the order above. Never invent another value.
- measurement_time: the time point(s) as written, e.g. "운동 후 40분".
- Write note and link_note in Korean (short sentences). Values (value) keep the paper's original wording.
"""


class OutcomeItem(BaseModel):
    indicator: TextAttr
    indicator_category: TextAttr
    direction: TextAttr
    significance_statistics: TextAttr
    representative_values: TextAttr
    comparison_type: TextAttr
    measurement_time: TextAttr
    program_ids: list[str]
    population_ids: list[str]
    comparison_program_ids: list[str]
    link_evidence: list[str]
    link_note: str | None


class OutcomeOut(BaseModel):
    outcomes: list[OutcomeItem]
