"""라우팅 호출의 시스템 프롬프트와 출력 스키마. LLM을 호출하지 않는다.

라우팅 호출: 섹션 목록(`routing_input`)을 보고, 노드마다 속성값이 있을 만한 섹션(1순위와 후보)을 고른다.
"""

from __future__ import annotations

from pydantic import BaseModel

SYSTEM_PROMPT = """\
You are given the section list of ONE academic paper. Each line has the form
`<section id>  <title path> | <beginning of the section's first paragraph>`, grouped under `[청크 N]` chunk headers.

Task: for each of four nodes, choose in which sections the values of that node's attributes are most likely to be found.
Do NOT extract any values; only choose sections.

Nodes and their attributes (required attributes first):
- paper: title, authors, publication year, journal, study type (required); DOI/URL, eligibility criteria,
  recruitment age, review sub-type (optional).
- population: disease or health condition (required); age, sex, the paper's own risk group, group name (optional).
- program: activity type (exercise, diet or both) (required); activity name, intensity, frequency, duration,
  time per session, condition role (optional).
- outcome: indicator, direction of change (required); significance, statistics, representative values,
  comparison method (optional).

For each node return:
- primary: the sections to read first (usually 1-3 ids).
- fallback: other sections to read if the primary sections do not provide all required attributes, best candidate first.
  It may be empty.

Rules:
- For population, program and outcome, put body sections (such as participants, methods, results) in primary and put the
  title block / abstract (the first part of the paper) in fallback. For paper, the title block and abstract are primary.
- A section title may be ambiguous; use the beginning of its first paragraph to decide.
- Use only section ids that appear in the list. Never invent ids.
- If no section fits a node, return empty lists for that node.
"""


class NodeRoute(BaseModel):
    primary: list[str]
    fallback: list[str]


class RoutingOut(BaseModel):
    paper: NodeRoute
    population: NodeRoute
    program: NodeRoute
    outcome: NodeRoute
