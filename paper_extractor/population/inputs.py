"""Population 호출의 입력 만들기: 연구 유형에 따라 보낼 청크를 고르고 문장 번호를 붙인다. LLM을 호출하지 않는다.

중재·관찰 연구: `population` 1순위 섹션의 청크 + 후보 첫 섹션(보통 초록)의 청크.
문헌 종합 연구: `population` 1순위 섹션의 청크 + `outcome` 1순위 섹션의 청크(분석 범주는 결과에 나온다).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import Config
from ..evidence_refs import number_chunks
from ..extract_title import load_env
from ..routing_call import cached_routing
from ..routing_input import section_list
from ..routing_prompt import RoutingOut

STUDY_TYPES = ["중재 연구", "관찰 연구", "문헌 종합 연구"]


def population_chunk_indexes(route: RoutingOut, rows: list[dict], study_type: str) -> list[int]:
    """연구 유형별로 Population 호출에 보낼 청크 번호(문서 순서, 중복 제거)."""
    chunk_of = {r["id"]: r["chunk"] for r in rows}
    if study_type == "문헌 종합 연구":
        ids = list(route.population.primary) + list(route.outcome.primary)
    else:
        ids = list(route.population.primary) + route.population.fallback[:1]
    return sorted({chunk_of[i] for i in ids if i in chunk_of})


def build_user(selected: list[dict], study_type: str) -> tuple[str, dict[str, dict]]:
    """Population 호출에 보낼 사용자 입력과 문장 번호 맵을 만든다. 첫 줄에 연구 유형을 적는다."""
    body, ref_map = number_chunks(selected)
    return f"Study type of this paper: {study_type}\n\n{body}", ref_map


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Population 호출 입력을 만들어 확인합니다(LLM 호출 없음).")
    parser.add_argument("chunks_json", type=Path)
    parser.add_argument("--study-type", choices=STUDY_TYPES, required=True)
    parser.add_argument("--full", action="store_true", help="보낼 입력 전문을 출력")
    args = parser.parse_args()

    load_env(root / ".env")
    cfg = Config()
    data = json.loads(args.chunks_json.read_text(encoding="utf-8"))
    rows = section_list(data)
    route = cached_routing(rows, cfg, root / "outputs" / "cache")
    idx = population_chunk_indexes(route, rows, args.study_type)
    selected = [c for c in data["chunks"] if c["index"] in idx]
    user, ref_map = build_user(selected, args.study_type)

    print("population 1순위:", route.population.primary, "/ 후보:", route.population.fallback, "/ outcome 1순위:", route.outcome.primary)
    print("보낼 청크:", idx, "토큰:", sum(c["tokens"] for c in selected), "/ 입력 글자 수:", len(user), "/ 문장 번호:", len(ref_map))
    if args.full:
        print("\n" + user)


if __name__ == "__main__":
    main()
