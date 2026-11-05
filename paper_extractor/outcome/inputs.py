"""Outcome 호출의 입력 만들기: 앞서 추출한 Program·Population 목록에 번호를 붙이고, 라우팅의 `outcome` 1순위 청크와 함께 입력을 만든다.

LLM을 호출하지 않는다. Program·Population 결과는 캐시에서만 읽으며, 캐시에 없으면 호출하지 않고 멈춘다.
우선 `outcome` 1순위 청크만 보낸다(v1.27). 번호는 Program은 P1, P2…, Population은 G1, G2…로 목록 순서대로 붙인다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..attrs import STUDY_TYPES
from ..config import Config
from ..evidence_refs import number_chunks
from ..extract_title import load_env
from ..population.call import call_population
from ..population.inputs import build_user as population_user
from ..population.inputs import population_chunk_indexes
from ..population.prompt import PopulationOut
from ..program.call import call_program
from ..program.inputs import build_user as program_user
from ..program.inputs import program_chunk_indexes
from ..program.prompt import ProgramOut
from ..routing_call import cached_routing
from ..routing_input import section_list
from ..routing_prompt import RoutingOut


def outcome_chunk_indexes(route: RoutingOut, rows: list[dict]) -> list[int]:
    """`outcome` 1순위 섹션이 있는 청크 번호(문서 순서, 중복 제거)."""
    chunk_of = {r["id"]: r["chunk"] for r in rows}
    return sorted({chunk_of[i] for i in route.outcome.primary if i in chunk_of})


def _v(attr) -> str:
    return attr.value if attr.value is not None else "-"


def node_lists(programs: ProgramOut, populations: PopulationOut) -> tuple[str, dict[str, str]]:
    """Program(P1…)·Population(G1…) 목록 텍스트와 번호 → 이름 맵을 만든다."""
    names: dict[str, str] = {}
    prog_lines, pop_lines = [], []
    for k, p in enumerate(programs.programs, 1):
        pid = f"P{k}"
        names[pid] = _v(p.program_name)
        prog_lines.append(
            f"{pid} | name: {_v(p.program_name)} | condition_role: {_v(p.condition_role)} | "
            f"program_source: {_v(p.program_source)} | activity_type: {_v(p.activity_type)}"
        )
    for k, g in enumerate(populations.populations, 1):
        gid = f"G{k}"
        names[gid] = _v(g.group_name) if g.group_name.value else (g.condition.note or _v(g.condition))
        condition = _v(g.condition) + (f" ({g.condition.note})" if g.condition.note else "")
        pop_lines.append(f"{gid} | group_name: {_v(g.group_name)} | condition: {condition} | group_role: {_v(g.group_role)}")
    text = "Programs:\n" + ("\n".join(prog_lines) or "(none)") + "\n\nPopulations:\n" + ("\n".join(pop_lines) or "(none)")
    return text, names


def target_programs(programs: ProgramOut) -> list[str]:
    """나눠 부를 때의 대상 Program 번호. 대조 조건(condition_role이 '대조')은 대상에서 뺀다."""
    return [f"P{k}" for k, p in enumerate(programs.programs, 1) if p.condition_role.value != "대조"]


def build_user(selected: list[dict], study_type: str, programs: ProgramOut, populations: PopulationOut, target: str | None = None):
    """Outcome 호출에 보낼 사용자 입력, 문장 번호 맵, 노드 번호 → 이름 맵을 만든다.

    `target`(예: "P1")을 주면 그 Program을 평가하는 결과만 뽑도록 연구 유형 줄 다음에 대상 줄을 붙인다.
    """
    lists, names = node_lists(programs, populations)
    body, ref_map = number_chunks(selected)
    head = f"Study type of this paper: {study_type}"
    if target:
        head += f"\nTarget program: {target} ({names[target]})"
    return f"{head}\n\n{lists}\n\n{body}", ref_map, names


def cached_nodes(data: dict, rows: list[dict], route: RoutingOut, study_type: str, cfg: Config, cache_dir: Path):
    """캐시된 Population·Program 결과를 읽는다. 캐시에 없으면 BudgetExceeded가 난다(API 호출 없음)."""
    chunks = data["chunks"]
    pop_sel = [c for c in chunks if c["index"] in population_chunk_indexes(route, rows, study_type)]
    pop_user, _ = population_user(pop_sel, study_type)
    populations, _ = call_population(pop_user, cfg, cache_dir, 0)
    prog_sel = [c for c in chunks if c["index"] in program_chunk_indexes(route, rows)]
    prog_user, _ = program_user(prog_sel, study_type)
    programs, _ = call_program(prog_user, cfg, cache_dir, 0)
    return programs, populations


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Outcome 호출 입력을 만들어 확인합니다(LLM 호출 없음).")
    parser.add_argument("chunks_json", type=Path)
    parser.add_argument("--study-type", choices=STUDY_TYPES, required=True)
    parser.add_argument("--full", action="store_true", help="보낼 입력 전문을 출력")
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
    user, ref_map, names = build_user(selected, args.study_type, programs, populations)

    print("outcome 1순위:", route.outcome.primary, "/ 후보:", route.outcome.fallback)
    print("보낼 청크:", idx, "토큰:", sum(c["tokens"] for c in selected), "/ 입력 글자 수:", len(user), "/ 문장 번호:", len(ref_map))
    print("노드 번호:", names)
    if args.full:
        print("\n" + user)


if __name__ == "__main__":
    main()
