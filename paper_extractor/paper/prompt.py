"""Paper 노드 속성 추출 호출의 시스템 프롬프트와 출력 스키마. LLM을 호출하지 않는다.

속성마다 값(value), 근거 문장 번호(evidence), 메모(note)를 받는다. 번호는 코드가 원문 문장과 섹션 경로로 바꾼다(v1.14).
정보 상태(보고됨·미보고·해당 없음·확인 불가)는 LLM이 아니라 코드가 정한다.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..attrs import NumAttr, TextAttr

SYSTEM_PROMPT = """\
You are given parts of ONE academic paper (one or more text chunks). Extract the attributes of the paper itself
(the "Paper" node). Extract only about the CURRENT paper, never about papers it cites.

General rules:
- Keep the original language and wording of the paper. Do not translate or paraphrase values.
- If an attribute is not stated in the given text, set its value to null. Never guess.
- The input may start with a block of running header/footer lines of the paper (ids `meta.s1`, `meta.s2`, ...).
  They often contain the journal name, the publication year and the DOI; use them as evidence like any other sentence.
- Every sentence of the input is prefixed with an id in square brackets, e.g. `[h4.s2] ...`.
  For every attribute with a value, give evidence: the ids of the sentences that support the value
  (e.g. ["h4.s1", "h4.s2"]), without the brackets. Use only ids that appear in the input. Never write quotes.
  Use an empty list when the value is null.
- note is for short remarks (original wording, ambiguity). Use null if there is nothing to add.
- Bilingual title or author lists (Korean and English) are not a conflict. Use the version that appears first,
  at the top of the first page, as the value (e.g. an English title above the Korean title -> the English title).

Attributes (required: title, authors, year, journal, study_type):
- title: the title of the paper. Do not give evidence for it (leave the evidence list empty).
- authors: all authors, joined with ", ".
- year: the publication year as a number.
- journal: the journal (or publisher) name.
- study_type: one of
    "중재 연구" (the researchers assigned and ran an exercise/diet intervention),
    "관찰 연구" (the researchers only observed or surveyed participants without assigning an intervention),
    "문헌 종합 연구" (the paper reviews or pools other studies).
  Decide by what the researchers actually did, not by the paper's own label. If it is ambiguous whether the researchers
  assigned the intervention (e.g. participants joined an existing exercise/diet program and were measured before and
  after), choose "중재 연구". Use null if the text is insufficient.
- review_subtype: only for "문헌 종합 연구"; one of "메타분석", "체계적 문헌고찰", "서술형 리뷰", "기타 문헌 종합".
  Otherwise null.
- doi: choose the paper's own DOI from the candidates listed in the input; null if there are none or none fits.
  Give the id of a sentence that contains it as evidence.
- inclusion_criteria: the participant inclusion conditions. Copy the sentences exactly as written in the paper:
  do not shorten, merge, reword, or drop citations and endings; join several sentences with a single space.
  A negatively worded participant condition (e.g. "no history of stroke") counts as inclusion.
- exclusion_criteria: only if the paper explicitly presents them as exclusion criteria.
- recruit_age_min / recruit_age_max: the age eligibility condition as plain numbers in years. Never use the
  participants' mean age or actual age range. Write the original wording in note.
    "만 65세 이상" -> min 65, max null; "50대 이상" -> min 50, max null; "만 60세 미만" -> min null, max 60.
    A decade range such as "40~50대" -> min 40, max 59; a single decade such as "60대" or "60세 전후" -> 60 to 69.
    Early / middle / late within a decade: early n0-n2, middle n3-n6, late n7-n9 (e.g. "60대 중반" -> 63 to 66).
"""


class PaperOut(BaseModel):
    title: TextAttr
    authors: TextAttr
    year: NumAttr
    journal: TextAttr
    study_type: TextAttr
    review_subtype: TextAttr
    doi: TextAttr
    inclusion_criteria: TextAttr
    exclusion_criteria: TextAttr
    recruit_age_min: NumAttr
    recruit_age_max: NumAttr
