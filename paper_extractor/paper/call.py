"""Paper 호출: 입력(`inputs.py`)을 LLM에 1회 보내 Paper 속성값을 받고 출력한다.

논문 한 편당 LLM을 1회 호출한다. 같은 입력은 캐시를 써서 다시 호출하지 않는다.
DOI 후보가 하나면 코드가 그 값을 채택해 LLM 값을 덮어쓴다(v1.07).
LLM은 근거 문장 번호만 출력하고, 코드가 번호를 원문 문장과 섹션 경로로 바꾼다(v1.14). 입력에 없는 번호는 표시한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import Config
from ..extract_title import load_env
from ..llm import BudgetedLLM, CallBudget, OpenAIClient
from ..routing_call import cached_routing
from ..routing_input import section_list
from .inputs import build_user, doi_candidates, load_meta, paper_chunk_indexes
from .prompt import SYSTEM_PROMPT, PaperOut


def call_paper(user: str, cfg: Config, cache_dir: Path) -> tuple[PaperOut, dict]:
    llm = BudgetedLLM(OpenAIClient(cfg.model, cfg.request_timeout_s), CallBudget(1), cache_dir, cfg.cache_fingerprint(), cfg.model)
    out = llm.run(
        stage="paper_attrs",
        system=SYSTEM_PROMPT,
        user=user,
        schema=PaperOut,
        max_output_tokens=cfg.max_output_tokens_extract,
        retries=0,
    )
    return out, llm.budget.totals()


def apply_single_doi(out: PaperOut, dois: list[str], ref_map: dict[str, dict]) -> None:
    """DOI 후보가 하나면 코드가 채택한다. 근거는 DOI가 들어 있는 첫 문장 번호, note에는 채택 방식을 적는다(값에는 넣지 않는다)."""
    if len(dois) == 1:
        out.doi.value = dois[0]
        out.doi.evidence = [r for r, v in ref_map.items() if dois[0] in v["text"]][:1]
        out.doi.note = "정규식 단일 후보, 코드 채택"


def resolve(attr, ref_map: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """근거 문장 번호를 원문 문장과 섹션 경로로 바꾼다. (근거 목록, 입력에 없는 번호 목록)을 돌려준다."""
    evidence, invalid = [], []
    for ref in dict.fromkeys(attr.evidence):  # 중복 제거, 순서 유지
        if ref in ref_map:
            evidence.append({"ref": ref, "section": ref_map[ref]["section"], "quote": ref_map[ref]["text"]})
        else:
            invalid.append(ref)
    return evidence, invalid


def show(out: PaperOut, ref_map: dict[str, dict]) -> None:
    for name, attr in out:
        print(f"{name}: {attr.value!r}")
        evidence, invalid = resolve(attr, ref_map)
        for ev in evidence:
            print(f"    근거 {ev['ref']} ({ev['section'][:40]}) {ev['quote']}")
        for ref in invalid:
            print(f"    근거 {ref} (입력에 없는 번호)")
        if attr.note:
            print(f"    note: {attr.note}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Paper 속성 추출을 1회 호출합니다(LLM 호출).")
    parser.add_argument("chunks_json", type=Path)
    args = parser.parse_args()

    load_env(root / ".env")
    cfg = Config()
    cache_dir = root / "outputs" / "cache"
    data = json.loads(args.chunks_json.read_text(encoding="utf-8"))
    rows = section_list(data)
    route = cached_routing(rows, cfg, cache_dir)
    idx = paper_chunk_indexes(route, rows)
    selected = [c for c in data["chunks"] if c["index"] in idx]
    meta = load_meta(args.chunks_json, root)
    dois = doi_candidates(data["chunks"], meta)
    user, ref_map = build_user(selected, dois, meta)

    out, usage = call_paper(user, cfg, cache_dir)
    apply_single_doi(out, dois, ref_map)
    show(out, ref_map)
    print("\n사용량:", usage)


if __name__ == "__main__":
    main()
