"""Program 호출의 입력 만들기: 라우팅의 `program` 1순위 섹션이 있는 청크에 문장 번호를 붙인다. LLM을 호출하지 않는다.

우선 1순위 청크만 보낸다. 결과가 부족하면 후보 1순위 청크를 추가하는 방식은 시험 결과를 본 뒤 정한다(v1.25).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..attrs import STUDY_TYPES
from ..config import Config
from ..evidence_refs import build_typed_user
from ..extract_title import load_env
from ..routing_call import cached_routing
from ..routing_input import section_list
from ..routing_prompt import RoutingOut


def program_chunk_indexes(route: RoutingOut, rows: list[dict]) -> list[int]:
    """`program` 1순위 섹션이 있는 청크 번호(문서 순서, 중복 제거)."""
    chunk_of = {r["id"]: r["chunk"] for r in rows}
    return sorted({chunk_of[i] for i in route.program.primary if i in chunk_of})


def build_user(selected: list[dict], study_type: str) -> tuple[str, dict[str, dict]]:
    """Program 호출에 보낼 사용자 입력과 문장 번호 맵을 만든다. 첫 줄에 연구 유형을 적는다."""
    return build_typed_user(selected, study_type)


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Program 호출 입력을 만들어 확인합니다(LLM 호출 없음).")
    parser.add_argument("chunks_json", type=Path)
    parser.add_argument("--study-type", choices=STUDY_TYPES, required=True)
    parser.add_argument("--full", action="store_true", help="보낼 입력 전문을 출력")
    args = parser.parse_args()

    load_env(root / ".env")
    cfg = Config()
    data = json.loads(args.chunks_json.read_text(encoding="utf-8"))
    rows = section_list(data)
    route = cached_routing(rows, cfg, root / "outputs" / "cache")
    idx = program_chunk_indexes(route, rows)
    selected = [c for c in data["chunks"] if c["index"] in idx]
    user, ref_map = build_user(selected, args.study_type)

    print("program 1순위:", route.program.primary, "/ 후보:", route.program.fallback)
    print("보낼 청크:", idx, "토큰:", sum(c["tokens"] for c in selected), "/ 입력 글자 수:", len(user), "/ 문장 번호:", len(ref_map))
    if args.full:
        print("\n" + user)


if __name__ == "__main__":
    main()
