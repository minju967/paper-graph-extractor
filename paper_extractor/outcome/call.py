"""Outcome 호출: 입력(`inputs.py`)을 LLM에 1회 보내 결과 목록과 속성값, Program·Population 연결 번호를 받고 출력한다.

논문 한 편당 LLM을 1회 호출한다. 같은 입력은 캐시를 써서 다시 호출하지 않는다.
LLM은 근거 문장 번호와 노드 번호(P1·G1)만 출력하고, 코드가 번호를 원문 문장과 노드 이름으로 바꾸며
목록에 없는 번호를 표시한다. 연구 유형은 이후 `Paper` 결과에서 받을 값이며, 지금은 명령줄로 받는다.
Program·Population 결과는 캐시에서만 읽는다(없으면 호출하지 않고 멈춤).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..attrs import STUDY_TYPES
from ..config import Config
from ..evidence_refs import resolve
from ..extract_title import load_env
from ..llm import BudgetedLLM, CallBudget, OpenAIClient
from ..routing_call import cached_routing
from ..routing_input import section_list
from .direction import decide_direction, direction_mismatch
from .inputs import build_user, cached_nodes, outcome_chunk_indexes, target_programs
from .prompt import (
    CHANGE_DIRECTIONS,
    COMPARISON_TYPES,
    DIRECTIONS,
    INDICATOR_CATEGORIES,
    SIGNIFICANCES,
    SYSTEM_PROMPT,
    OutcomeItem,
    OutcomeOut,
)

# 결과가 많은 논문에서 출력(추론 토큰 포함)이 잘리지 않도록 Outcome 호출만 한도를 넓힌다.
OUTCOME_MAX_OUTPUT_TOKENS = 16_000

LINK_FIELDS =(("program_ids", "P"), ("population_ids", "G"), ("comparison_program_ids", "P"))


def call_outcome(user: str, cfg: Config, cache_dir: Path, max_calls: int = 1) -> tuple[OutcomeOut, dict]:
    llm = BudgetedLLM(OpenAIClient(cfg.model, cfg.request_timeout_s), CallBudget(cfg.attempts_for(max_calls)), cache_dir, cfg.cache_fingerprint(), cfg.model)
    out = llm.run(
        stage="outcome_attrs",
        system=SYSTEM_PROMPT,
        user=user,
        schema=OutcomeOut,
        max_output_tokens=OUTCOME_MAX_OUTPUT_TOKENS,
        retries=cfg.timeout_retries,
        timeout_growth=cfg.timeout_growth,
    )
    return out, llm.budget.totals()


def check_links(item: OutcomeItem, names: dict[str, str]) -> dict[str, list[str]]:
    """연결 번호 중 입력 목록에 없거나 종류가 다른 번호(Program은 P…, Population은 G…)를 찾는다."""
    bad: dict[str, list[str]] = {}
    for field, prefix in LINK_FIELDS:
        wrong = [i for i in getattr(item, field) if i not in names or not i.startswith(prefix)]
        if wrong:
            bad[field] = wrong
    return bad


CLOSED_VALUES = (
    ("direction", DIRECTIONS),
    ("change_direction", CHANGE_DIRECTIONS),
    ("significance", SIGNIFICANCES),
    ("indicator_category", INDICATOR_CATEGORIES),
    ("comparison_type", COMPARISON_TYPES),
)


def check_closed(item: OutcomeItem) -> dict[str, str]:
    """닫힌 목록이 있는 속성의 값이 목록을 어기는지 찾는다(값이 없는 것은 점검하지 않는다)."""
    bad: dict[str, str] = {}
    for field, allowed in CLOSED_VALUES:
        value = getattr(item, field).value
        if value is not None and value not in allowed:
            bad[field] = value
    return bad


def _indicator_key(item: OutcomeItem) -> str:
    """중복 비교용 지표 표기. 대소문자를 무시하고 변환 오타 DDBP를 DBP로 본다."""
    return (item.indicator.value or "").strip().lower().replace("ddbp", "dbp")


def dedupe_pairs(items: list[OutcomeItem]) -> tuple[list[OutcomeItem], int]:
    """대상 Program별 호출을 합칠 때, 같은 지표·집단에서 같은 프로그램 쌍(순서 무관)을 비교한 결과는 앞선 것만 남긴다.

    Program이 하나인 결과에 적용한다. 비교 Program이 여럿이면 쌍으로 풀어, 모든 쌍이 이미 나왔을 때만 제거한다
    (일부 쌍만 겹치면 남긴다). 비교 Program이 없는 결과(하위집단 비교 등)는 그대로 둔다.
    """
    seen: set[tuple] = set()
    kept: list[OutcomeItem] = []
    for it in items:
        if len(it.program_ids) == 1 and it.comparison_program_ids:
            base = (_indicator_key(it), tuple(sorted(it.population_ids)))
            keys = {base + (frozenset({it.program_ids[0], c}),) for c in it.comparison_program_ids}
            if keys <= seen:
                continue
            seen |= keys
        kept.append(it)
    return kept, len(items) - len(kept)


def _link_text(ids: list[str], names: dict[str, str], prefix: str) -> str:
    if not ids:
        return "없음"
    return ", ".join(f"{i}({names[i]})" if i in names and i.startswith(prefix) else f"{i}(목록에 없음)" for i in ids)


def show(out: OutcomeOut, ref_map: dict[str, dict], names: dict[str, str], full_evidence: bool = False) -> None:
    """결과의 모든 속성을 값이 없는 것(None)까지 출력한다. 속성 구간과 연결 구간을 구분해 보여준다."""
    print(f"Outcome 노드 수: {len(out.outcomes)}")
    for k, item in enumerate(out.outcomes, 1):
        print(f"\n=== Outcome 노드 {k} ===")
        print("  [Outcome 속성]")
        for name, attr in item:
            if name in {f for f, _ in LINK_FIELDS} | {"link_evidence", "link_note"}:
                continue
            evidence, invalid = resolve(attr, ref_map)
            refs = ", ".join([e["ref"] for e in evidence] + [f"{r}(입력에 없음)" for r in invalid]) or "없음"
            print(f"    {name}: {attr.value!r}  [{attr.status}] | 근거: {refs}" + (f"  | note: {attr.note}" if attr.note else ""))
            if full_evidence:
                for ev in evidence:
                    print(f"        {ev['ref']} ({ev['section'][:30]}) {ev['quote']}")
        print("  [연결: 앞서 추출한 노드 번호]")
        for field, prefix in LINK_FIELDS:
            print(f"    {field}: {_link_text(getattr(item, field), names, prefix)}")
        link_refs, link_invalid = resolve(type("A", (), {"evidence": item.link_evidence})(), ref_map)
        refs = ", ".join([e["ref"] for e in link_refs] + [f"{r}(입력에 없음)" for r in link_invalid]) or "없음"
        print(f"    link_evidence: {refs}")
        print(f"    link_note: {item.link_note!r}")
        if full_evidence:
            for ev in link_refs:
                print(f"        {ev['ref']} ({ev['section'][:30]}) {ev['quote']}")
        bad = check_links(item, names)
        if bad:
            print(f"  !! 연결 번호 오류: {bad}")
        bad_values = check_closed(item)
        if bad_values:
            print(f"  !! 닫힌 값 오류: {bad_values}")
        mismatch = direction_mismatch(item)
        if mismatch:
            print(f"  !! 유의성·방향 불일치: {mismatch}")


def collect_outcomes(selected, study_type, programs, populations, targets, cfg: Config, cache_dir: Path, budget: int, log=print):
    """대상 Program마다 호출(또는 캐시 읽기)해 합치고, 중복을 제거한 뒤 코드가 direction을 정한다.

    돌려주는 값: (Outcome 목록, 중복으로 제거한 수, direction을 정한 수). budget=0이면 캐시만 읽는다.
    """
    merged: list[OutcomeItem] = []
    for t in targets:
        user, _, _ = build_user(selected, study_type, programs, populations, t)
        out, usage = call_outcome(user, cfg, cache_dir, budget)
        log(f"[대상 {t or '전체'}] Outcome {len(out.outcomes)}개 | 사용량: {usage}")
        merged.extend(out.outcomes)
    merged, removed = dedupe_pairs(merged)
    filled = sum(decide_direction(it) for it in merged)
    return merged, removed, filled


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Outcome 속성 추출을 1회 호출합니다(LLM 호출).")
    parser.add_argument("chunks_json", type=Path)
    parser.add_argument("--study-type", choices=STUDY_TYPES, required=True)
    parser.add_argument("--evidence", action="store_true", help="근거 문장 원문까지 출력")
    parser.add_argument("--cache-only", action="store_true", help="캐시에 있는 결과만 읽는다(없으면 호출하지 않고 멈춤)")
    parser.add_argument("--target", help="이 Program(예: P1)을 평가하는 결과만 뽑는다(1회 호출)")
    parser.add_argument("--split", action="store_true", help="대조가 아닌 Program마다 1회씩 호출해 합쳐서 출력")
    args = parser.parse_args()

    load_env(root / ".env")
    cfg = Config()
    cache_dir = root / "outputs" / "cache"
    data = json.loads(args.chunks_json.read_text(encoding="utf-8"))
    rows = section_list(data)
    route = cached_routing(rows, cfg, cache_dir)
    programs, populations = cached_nodes(data, rows, route, args.study_type, cfg, cache_dir)
    idx = outcome_chunk_indexes(route, rows)
    selected = [c for c in data["chunks"] if c["index"] in idx]
    targets = target_programs(programs) if args.split else [args.target]
    merged, removed, filled = collect_outcomes(selected, args.study_type, programs, populations, targets, cfg, cache_dir, 0 if args.cache_only else 1)
    _, ref_map, names = build_user(selected, args.study_type, programs, populations)
    print(f"\n코드가 direction을 정한 결과: {filled}개")
    print(f"\n합친 뒤 중복 제거: {removed}개 제거 -> Outcome {len(merged)}개\n")
    show(OutcomeOut(outcomes=merged), ref_map, names, args.evidence)


if __name__ == "__main__":
    main()
