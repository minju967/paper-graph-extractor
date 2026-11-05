"""Program 호출: 입력(`inputs.py`)을 LLM에 1회 보내 프로그램 목록과 속성값을 받고 출력한다.

논문 한 편당 LLM을 1회 호출한다. 같은 입력은 캐시를 써서 다시 호출하지 않는다.
LLM은 근거 문장 번호만 출력하고, 코드가 번호를 원문 문장과 섹션 경로로 바꾼다.
연구 유형은 이후 `Paper` 결과에서 받을 값이며, 지금은 명령줄로 받는다.
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
from .inputs import build_user, program_chunk_indexes
from .prompt import SYSTEM_PROMPT, ProgramOut


def call_program(user: str, cfg: Config, cache_dir: Path, max_calls: int = 1) -> tuple[ProgramOut, dict]:
    llm = BudgetedLLM(OpenAIClient(cfg.model, cfg.request_timeout_s), CallBudget(cfg.attempts_for(max_calls)), cache_dir, cfg.cache_fingerprint(), cfg.model)
    out = llm.run(
        stage="program_attrs",
        system=SYSTEM_PROMPT,
        user=user,
        schema=ProgramOut,
        max_output_tokens=cfg.max_output_tokens_extract,
        retries=cfg.timeout_retries,
        timeout_growth=cfg.timeout_growth,
    )
    return out, llm.budget.totals()


def _show_attrs(item, ref_map: dict[str, dict], indent: str, full_evidence: bool, skip: tuple[str, ...] = ()) -> None:
    """속성을 값이 없는 것(None)까지 모두 출력한다. 근거는 문장 번호만, full_evidence이면 원문 문장까지."""
    for name, attr in item:
        if name in skip:
            continue
        evidence, invalid = resolve(attr, ref_map)
        refs = ", ".join([e["ref"] for e in evidence] + [f"{r}(입력에 없음)" for r in invalid]) or "없음"
        print(f"{indent}{name}: {attr.value!r}  [{attr.status}] | 근거: {refs}" + (f"  | note: {attr.note}" if attr.note else ""))
        if full_evidence:
            for ev in evidence:
                print(f"{indent}    {ev['ref']} ({ev['section'][:30]}) {ev['quote']}")


def show(out: ProgramOut, ref_map: dict[str, dict], full_evidence: bool = False) -> None:
    print(f"프로그램 노드 수: {len(out.programs)}")
    for k, prog in enumerate(out.programs, 1):
        print(f"\n=== Program 노드 {k} ===")
        print("  [Program 속성]")
        _show_attrs(prog, ref_map, "    ", full_evidence, skip=("activities",))
        print(f"  [활동 구성: 이 Program의 하위 목록, {len(prog.activities)}개]")
        if not prog.activities:
            print("    (활동 없음)")
        for j, act in enumerate(prog.activities, 1):
            print(f"    -- 활동 {j} 하위 속성 --")
            _show_attrs(act, ref_map, "      ", full_evidence)


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Program 속성 추출을 1회 호출합니다(LLM 호출).")
    parser.add_argument("chunks_json", type=Path)
    parser.add_argument("--study-type", choices=STUDY_TYPES, required=True)
    parser.add_argument("--evidence", action="store_true", help="근거 문장 원문까지 출력")
    parser.add_argument("--cache-only", action="store_true", help="캐시에 있는 결과만 읽는다(없으면 호출하지 않고 멈춤)")
    args = parser.parse_args()

    load_env(root / ".env")
    cfg = Config()
    cache_dir = root / "outputs" / "cache"
    data = json.loads(args.chunks_json.read_text(encoding="utf-8"))
    rows = section_list(data)
    route = cached_routing(rows, cfg, cache_dir)
    idx = program_chunk_indexes(route, rows)
    selected = [c for c in data["chunks"] if c["index"] in idx]
    user, ref_map = build_user(selected, args.study_type)

    out, usage = call_program(user, cfg, cache_dir, 0 if args.cache_only else 1)
    show(out, ref_map, args.evidence)
    print("\n사용량:", usage)


if __name__ == "__main__":
    main()
