"""명령행 인터페이스.

  python -m paper_extractor plan    <pdf> ...   # 변환·청킹·호출 예상치만 (API 호출 없음)
  python -m paper_extractor extract <pdf> ...   # 전체 추출 (OPENAI_API_KEY, 모델명 필요)
  python -m paper_extractor mark-reviewed <graph.json> --reviewer 이름
  python -m paper_extractor load-neo4j <graph.json> [--replace]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .llm import OpenAIClient
from .pipeline import Pipeline


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("pdf", type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--model", default=None, help="모델명(또는 환경변수 PAPER_EXTRACTOR_MODEL)")
    p.add_argument("--max-calls", type=int, default=15)
    p.add_argument("--chunk-target", type=int, default=4000)
    p.add_argument("--chunk-max", type=int, default=6000)


def _cfg(a: argparse.Namespace) -> Config:
    cfg = Config(output_dir=a.out, max_api_calls=a.max_calls, chunk_target_tokens=a.chunk_target, chunk_max_tokens=a.chunk_max)
    if a.model:
        cfg.model = a.model
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="paper-extractor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_common(sub.add_parser("plan", help="API 호출 없이 변환·청킹·호출 예상치 확인"))
    _add_common(sub.add_parser("extract", help="논문 정보 추출 실행"))
    r = sub.add_parser("mark-reviewed", help="사람 검토를 마친 graph.json에 검토 완료를 표시")
    r.add_argument("graph", type=Path)
    r.add_argument("--reviewer", required=True)
    n = sub.add_parser("load-neo4j", help="검토 완료된 graph.json을 Neo4j에 적재")
    n.add_argument("graph", type=Path)
    n.add_argument("--replace", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd in ("plan", "extract"):
        cfg = _cfg(a)
        client = None if a.cmd == "plan" else OpenAIClient(cfg.model, cfg.request_timeout_s)
        res = Pipeline(cfg, client).run(a.pdf, dry_run=(a.cmd == "plan"))
        print(f"\n결과: {res.message}\n작업 폴더: {res.workdir}")
        if res.partial_reasons:
            print("부분 완료 사유:", *res.partial_reasons, sep="\n - ")
        return 0 if (a.cmd == "plan" or res.graph is not None) else 2
    if a.cmd == "mark-reviewed":
        from .neo4j_loader import mark_reviewed

        mark_reviewed(a.graph, a.reviewer)
        print("검토 완료로 표시했다.")
        return 0
    if a.cmd == "load-neo4j":
        from .neo4j_loader import load_graph, load_into_neo4j

        load_into_neo4j(load_graph(a.graph), replace=a.replace)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
