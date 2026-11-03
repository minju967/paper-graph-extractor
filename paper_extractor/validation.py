"""기계 검증. 통과해도 '사람 검수 완료'가 아니다(근거 ID 존재 ≠ 원문이 값을 지지함).

수행 내용
- 근거 블록 ID 존재 검증(없는 ID 제거 + 검토 항목)
- 관계 유형·방향·연구 유형별 허용 여부, 개별/종합 수준 규칙
- 숫자 값이 인용 근거 블록에 실제 등장하는지의 약한 점검(경고만)
- 미처리 구간이 있으면 not_reported를 unverifiable로 낮춤
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from .models import (
    Attr,
    Document,
    GraphRelation,
    MergeOutput,
    PaperGraph,
    ReviewItem,
    RunStatus,
)

ALLOWED: dict[str, dict[str, tuple[str, str]]] = {
    "intervention": {
        "HAS_POPULATION": ("Paper", "Population"),
        "PARTICIPATES_IN": ("Population", "Program"),
        "HAS_OUTCOME": ("Program", "Outcome"),
        "MEASURED_IN": ("Outcome", "Population"),
        "COMPARED_WITH_PROGRAM": ("Outcome", "Program"),
    },
    "meta_analysis": {
        "REPORTS": ("Paper", "Outcome"),
        "ANALYZES_POPULATION": ("Outcome", "Population"),
        "EVALUATES_PROGRAM": ("Outcome", "Program"),
    },
    "other": {},
    "undetermined": {},
}

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _iter_attrs(node) -> list[tuple[str, Attr]]:
    out = []
    for name, val in node:
        if isinstance(val, Attr):
            out.append((name, val))
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, BaseModel):
                    for n2, v2 in item:
                        if isinstance(v2, Attr):
                            out.append((f"{name}[{i}].{n2}", v2))
    return out


def _node_id(node) -> str:
    return node.id


def validate_and_finalize(
    merged: MergeOutput,
    doc: Document,
    *,
    paper_id: str,
    model: str,
    schema_version: str,
    prompt_version: str,
    partial_reasons: list[str],
    extra_review: list[ReviewItem] | None = None,
) -> PaperGraph:
    blocks = doc.block_map()
    review: list[ReviewItem] = list(merged.review_items) + list(extra_review or [])
    study_type = merged.final_study_type
    nodes_by_id: dict[str, str] = {merged.paper.id: "Paper"}

    def check_ids(ids: list[str], owner: str) -> list[str]:
        good = [i for i in dict.fromkeys(ids) if i in blocks]
        bad = [i for i in ids if i not in blocks]
        if bad:
            review.append(
                ReviewItem(
                    kind="invalid_evidence",
                    description=f"존재하지 않는 근거 ID를 제거했다: {sorted(set(bad))}",
                    related_ids=[owner],
                )
            )
        return good

    def check_node(node, kind: str) -> None:
        nid = node.id
        if nid in nodes_by_id:
            review.append(ReviewItem(kind="duplicate_id", description=f"중복 노드 ID {nid}", related_ids=[nid]))
        nodes_by_id[nid] = kind
        for name, attr in _iter_attrs(node):
            attr.evidence_block_ids = check_ids(attr.evidence_block_ids, f"{nid}.{name}")
            if attr.status == "reported":
                if not attr.evidence_block_ids:
                    attr.status = "unverifiable"
                    attr.note = ((attr.note + " / ") if attr.note else "") + "유효한 근거 블록이 없어 확인 불가로 낮춤"
                    review.append(ReviewItem(kind="invalid_evidence", description=f"{nid}.{name}: 근거 없는 보고 값", related_ids=[nid]))
                elif attr.value:
                    _support_check(nid, name, attr)
            if partial_reasons and attr.status == "not_reported":
                attr.status = "unverifiable"
                attr.note = ((attr.note + " / ") if attr.note else "") + "처리되지 않은 구간이 있어 미보고로 단정할 수 없음"

    def _support_check(nid: str, name: str, attr: Attr) -> None:
        nums = set(_NUM_RE.findall(attr.value or ""))
        if not nums:
            return
        text = " ".join(blocks[i].text for i in attr.evidence_block_ids)
        text_nums = set(_NUM_RE.findall(text))
        missing = sorted(n for n in nums if n not in text_nums)
        if missing:
            review.append(
                ReviewItem(
                    kind="support_warning",
                    description=f"{nid}.{name}: 값의 숫자 {missing}이(가) 인용 근거 블록에 보이지 않는다. 사람이 원문과 대조해야 한다.",
                    related_ids=[nid],
                    evidence_block_ids=attr.evidence_block_ids,
                )
            )

    check_node(merged.paper, "Paper")
    for p in merged.populations:
        check_node(p, "Population")
    for p in merged.programs:
        check_node(p, "Program")
    for o in merged.outcomes:
        check_node(o, "Outcome")

    level_of: dict[str, str] = {}
    for p in merged.populations:
        level_of[p.id] = (p.level.value or "").strip().lower()
    for p in merged.programs:
        level_of[p.id] = (p.level.value or "").strip().lower()

    allowed = ALLOWED.get(study_type, {})
    kept: list[GraphRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for r in merged.relations:
        key = (r.type, r.from_id, r.to_id)
        desc = f"{r.from_id} -{r.type}-> {r.to_id}"
        if key in seen:
            continue
        seen.add(key)
        r.evidence_block_ids = check_ids(r.evidence_block_ids, desc)
        reason = None
        if r.type not in allowed:
            reason = f"연구 유형 {study_type}에서 허용되지 않는 관계"
        else:
            ft, tt = allowed[r.type]
            if nodes_by_id.get(r.from_id) != ft or nodes_by_id.get(r.to_id) != tt:
                reason = f"관계 방향/끝점 유형 불일치(기대 {ft}→{tt})"
            elif not r.evidence_block_ids:
                reason = "유효한 근거 블록이 없음"
            elif r.type == "PARTICIPATES_IN" and "aggregate" in (level_of.get(r.from_id, ""), level_of.get(r.to_id, "")):
                reason = "종합 분석 집단·중재 범주에는 PARTICIPATES_IN을 만들지 않음"
        if reason:
            review.append(
                ReviewItem(
                    kind="held_relation",
                    description=f"관계를 간선으로 만들지 않고 보류: {desc} — {reason}",
                    related_ids=[r.from_id, r.to_id],
                    evidence_block_ids=r.evidence_block_ids,
                )
            )
        else:
            kept.append(r)

    # 연구 유형별 필수 관계가 하나도 없으면 알려 준다(검증 보조).
    if allowed and not kept:
        review.append(ReviewItem(kind="no_relations", description="허용 관계가 하나도 확정되지 않았다. 사람 검토 필요."))

    status = RunStatus(complete=not partial_reasons, partial_reasons=list(partial_reasons))
    return PaperGraph(
        paper_id=paper_id,
        schema_version=schema_version,
        prompt_version=prompt_version,
        model=model,
        study_type=study_type,
        type_note=merged.type_note,
        status=status,
        paper=merged.paper,
        populations=merged.populations,
        programs=merged.programs,
        outcomes=merged.outcomes,
        relations=kept,
        review_items=review,
    )
