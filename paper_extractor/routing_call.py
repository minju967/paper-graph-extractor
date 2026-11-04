"""라우팅 호출: 섹션 목록을 LLM에 보내 노드별 1순위·후보 섹션을 받고, 섹션 ID가 실제로 있는지 검사한다.

논문 한 편당 LLM을 1회 호출한다. 같은 입력은 캐시를 써서 다시 호출하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config
from .extract_title import load_env
from .llm import BudgetedLLM, CallBudget, OpenAIClient
from .routing_input import format_list, section_list
from .routing_prompt import SYSTEM_PROMPT, RoutingOut

NODES = ["paper", "population", "program", "outcome"]


class _NoCall:
    """캐시에 없으면 호출 대신 멈추기 위한 자리 표시 클라이언트(예산 0이라 호출되지 않는다)."""

    def parse(self, **_):
        raise RuntimeError("호출하지 않는다")


def cached_routing(rows: list[dict], cfg: Config, cache_dir: Path) -> RoutingOut:
    """라우팅 결과를 캐시에서만 읽는다. 캐시에 없으면 BudgetExceeded가 난다(API 호출 없음)."""
    llm = BudgetedLLM(_NoCall(), CallBudget(0), cache_dir, cfg.cache_fingerprint(), cfg.model)
    return llm.run(
        stage="routing",
        system=SYSTEM_PROMPT,
        user=format_list(rows),
        schema=RoutingOut,
        max_output_tokens=cfg.max_output_tokens_classify,
    )


def route(rows: list[dict], cfg: Config, cache_dir: Path) -> tuple[RoutingOut, dict]:
    """섹션 목록으로 라우팅을 1회 호출한다. (출력, 사용량 요약)을 돌려준다."""
    llm = BudgetedLLM(OpenAIClient(cfg.model, cfg.request_timeout_s), CallBudget(1), cache_dir, cfg.cache_fingerprint(), cfg.model)
    out = llm.run(
        stage="routing",
        system=SYSTEM_PROMPT,
        user=format_list(rows),
        schema=RoutingOut,
        max_output_tokens=cfg.max_output_tokens_classify,
        retries=0,
    )
    return out, llm.budget.totals()


def check_ids(out: RoutingOut, rows: list[dict]) -> dict[str, list[str]]:
    """노드마다 입력에 없는 섹션 ID를 찾는다. 없으면 빈 목록."""
    valid = {r["id"] for r in rows}
    return {n: [i for i in getattr(out, n).primary + getattr(out, n).fallback if i not in valid] for n in NODES}


def show(out: RoutingOut, rows: list[dict]) -> None:
    by_id = {r["id"]: r for r in rows}

    def label(i: str) -> str:
        r = by_id.get(i)
        return f"{i}(청크 {r['chunk']}, {r['path'][:30]})" if r else f"{i}(입력에 없음)"

    for n in NODES:
        route_ = getattr(out, n)
        print(f"{n}")
        print(f"  1순위: {', '.join(label(i) for i in route_.primary) or '(없음)'}")
        print(f"  후보 : {', '.join(label(i) for i in route_.fallback) or '(없음)'}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="섹션 목록으로 라우팅을 1회 호출합니다(LLM 호출).")
    parser.add_argument("chunks_json", type=Path)
    args = parser.parse_args()

    load_env(root / ".env")
    cfg = Config()
    rows = section_list(json.loads(args.chunks_json.read_text(encoding="utf-8")))
    out, usage = route(rows, cfg, root / "outputs" / "cache")
    show(out, rows)
    print("\n입력에 없는 섹션 ID:", check_ids(out, rows))
    print("사용량:", usage)


if __name__ == "__main__":
    main()
