"""Program 노드 속성 추출 호출의 시스템 프롬프트와 출력 스키마. LLM을 호출하지 않는다.

한 번의 호출로 논문의 프로그램(조건) 목록을 받는다. 프로그램마다 속성은 값(value)·근거 문장 번호(evidence)·메모(note)이고,
활동 구성(activities)은 활동별 속성 목록이다. 입력 맨 앞에 `Paper`에서 알아낸 연구 유형을 알려 준다.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..attrs import TextAttr

SYSTEM_PROMPT = """\
You are given parts of ONE academic paper (text chunks). The first line of the input states the study type of the paper.
Extract the "Program" nodes: the exercise / diet programs or conditions of THIS paper.
Never treat a recommendation or a cited study's program as a program this paper carried out.

How to form programs:
- Create one Program per distinct condition. For example MICE, HIIE and a non-exercise control are three programs; an
  exercise group and a control group are two programs. Do not split a program by session, week or outcome.
- Literature synthesis (study type "문헌 종합 연구"): group Programs by the kind of exercise or diet (e.g. aerobic exercise,
  dynamic resistance training) AND by program_source. One Program per (kind x program_source). Do NOT split Programs by
  the organization that issued a guideline. When several guidelines or organizations recommend the same kind with
  different values (frequency, time, intensity), keep ONE entry per organization in the activities list of that same
  Program (same activity_name) and write the organization's name in the note of activity_name.
- A control condition is also a Program (condition_role "대조"). If it has no exercise or diet activity at all (rest,
  no treatment), set activity_type to null, leave activities empty and say so in note.
- program_source by study type:
    "중재 연구": "직접 시행" for every condition.
    "문헌 종합 연구": "종합 분류" when the paper groups many studies' interventions into categories (e.g. endurance,
      resistance), and "권고" when it presents what a guideline or the authors recommend.
    "관찰 연구": "관찰 습관" (habits that were observed, not delivered).
- A meal given only before a test is not a diet program.

General rules:
- Closed values (activity_type, program_source, condition_role) must be written exactly as listed, in Korean.
- Keep the original language and wording of the paper. Use null when an attribute is not stated; never guess.
- Every sentence of the input is prefixed with an id in square brackets, e.g. `[h4.s2] ...`.
  For every attribute with a value, give evidence: the ids of the sentences that support the value
  (e.g. ["h4.s1"]), without the brackets. Use only ids that appear in the input. Never write quotes.
  Use an empty list when the value is null.
- note is for short remarks (original wording, ambiguity). Use null if there is nothing to add.

Attributes of each program (required for a program whose condition_role is "중재": activity_type):
- activity_type: "운동", "식단", or "복합". Use "복합" ONLY when the program contains both exercise and diet activities.
  A program that combines several kinds of exercise (e.g. aerobic plus resistance) is still "운동".
- program_source: "직접 시행", "종합 분류", "권고" or "관찰 습관" (see above).
- program_name: the name, condition label or abbreviation used in the paper (e.g. "MICE", "걷기운동군").
- condition_role: "중재" or "대조".
- duration: the actual length of the intervention with its unit (e.g. "12주"). Not washout, visit interval or follow-up.
- activities: the distinct activities of the program, one entry per activity. Each entry has:
    activity_type ("운동" or "식단"),
    activity_name (only an activity the paper names; never infer a sport from the intensity),
    frequency (as written with its unit, e.g. "주 3회"),
    session_time (the activity time of one session only; do not add rest, measurement or waiting time),
    intensity (the paper's own basis and unit, e.g. "HRR 70%"; keep HRR vs maximal heart rate as written, or the diet target).
- Warm-up and cool-down are NOT activities and are not part of session_time: write them only in the note of
  session_time (e.g. "warm-up 5 min and cool-down 5 min are excluded").
"""


class Activity(BaseModel):
    activity_type: TextAttr
    activity_name: TextAttr
    frequency: TextAttr
    session_time: TextAttr
    intensity: TextAttr


class ProgramItem(BaseModel):
    activity_type: TextAttr
    program_source: TextAttr
    program_name: TextAttr
    condition_role: TextAttr
    duration: TextAttr
    activities: list[Activity]


class ProgramOut(BaseModel):
    programs: list[ProgramItem]
