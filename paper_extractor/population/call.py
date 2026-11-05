"""Population 호출: 입력(`inputs.py`)을 LLM에 1회 보내 집단 목록과 속성값을 받고 출력한다.

논문 한 편당 LLM을 1회 호출한다. 같은 입력은 캐시를 써서 다시 호출하지 않는다.
LLM은 근거 문장 번호만 출력하고, 코드가 번호를 원문 문장과 섹션 경로로 바꾼다.
연구 유형은 이후 `Paper` 결과에서 받을 값이며, 지금은 명령줄로 받는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import Config
from ..evidence_refs import resolve
from ..extract_title import load_env
from ..llm import BudgetedLLM, CallBudget, OpenAIClient
from ..routing_call import cached_routing
from ..routing_input import section_list
from .inputs import STUDY_TYPES, build_user, population_chunk_indexes
from .prompt import CONDITION_CATEGORIES, GROUP_LEVELS, GROUP_ROLES, SYSTEM_PROMPT, PopulationOut


def call_population(user: str, cfg: Config, cache_dir: Path, max_calls: int = 1) -> tuple[PopulationOut, dict]:
    llm = BudgetedLLM(OpenAIClient(cfg.model, cfg.request_timeout_s), CallBudget(cfg.attempts_for(max_calls)), cache_dir, cfg.cache_fingerprint(), cfg.model)
    out = llm.run(
        stage="population_attrs",
        system=SYSTEM_PROMPT,
        user=user,
        schema=PopulationOut,
        max_output_tokens=cfg.max_output_tokens_extract,
        retries=cfg.timeout_retries,
        timeout_growth=cfg.timeout_growth,
    )
    return out, llm.budget.totals()


def check_closed(pop) -> dict[str, str]:
    """닫힌 목록이 있는 속성의 값이 목록을 어기는지 찾는다(값이 없는 것은 점검하지 않는다)."""
    allowed = {"condition": CONDITION_CATEGORIES, "group_level": GROUP_LEVELS, "group_role": GROUP_ROLES}
    return {f: getattr(pop, f).value for f, ok in allowed.items() if getattr(pop, f).value not in (None, *ok)}


def show(out: PopulationOut, ref_map: dict[str, dict]) -> None:
    print(f"집단 수: {len(out.populations)}")
    for k, pop in enumerate(out.populations, 1):
        print(f"\n[집단 {k}]")
        for name, attr in pop:
            print(f"{name}: {attr.value!r}  [{attr.status}]")
            evidence, invalid = resolve(attr, ref_map)
            for ev in evidence:
                print(f"    근거 {ev['ref']} ({ev['section'][:40]}) {ev['quote']}")
            for ref in invalid:
                print(f"    근거 {ref} (입력에 없는 번호)")
            if attr.note:
                print(f"    note: {attr.note}")
        bad = check_closed(pop)
        if bad:
            print(f"  !! 닫힌 값 오류: {bad}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Population 속성 추출을 1회 호출합니다(LLM 호출).")
    parser.add_argument("chunks_json", type=Path)
    parser.add_argument("--study-type", choices=STUDY_TYPES, required=True)
    args = parser.parse_args()

    load_env(root / ".env")
    cfg = Config()
    cache_dir = root / "outputs" / "cache"
    data = json.loads(args.chunks_json.read_text(encoding="utf-8"))
    rows = section_list(data)
    route = cached_routing(rows, cfg, cache_dir)
    idx = population_chunk_indexes(route, rows, args.study_type)
    selected = [c for c in data["chunks"] if c["index"] in idx]
    user, ref_map = build_user(selected, args.study_type)

    out, usage = call_population(user, cfg, cache_dir)
    show(out, ref_map)
    print("\n사용량:", usage)


if __name__ == "__main__":
    main()
