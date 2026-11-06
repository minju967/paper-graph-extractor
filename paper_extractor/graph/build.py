"""노드 연결: 논문 한 편의 Paper·Program·Population·Outcome 결과를 `nodes`와 `edges`가 든 JSON 하나로 묶는다(v1.43, v1.44).

LLM을 호출하지 않는다. 앞서 추출해 캐시에 있는 결과만 읽으며, 없으면 호출하지 않고 멈춘다.
연결은 새로 판단하지 않고 이미 추출된 번호에서 코드가 만든다. 관계 이름은 기획안(v0.5 문서, v1.04~v1.05)의 확정 이름을 쓴다.
  - 모든 유형: Paper -> HAS_PROGRAM -> Program, Paper -> HAS_POPULATION -> Population
  - 보고 계열(관찰·문헌 종합 연구): Paper -> REPORTS -> Outcome, Outcome -> EVALUATES_PROGRAM -> Program,
    Outcome -> ANALYZES_POPULATION -> Population
  - 시행 계열(중재 연구): Program -> HAS_OUTCOME -> Outcome, Outcome -> MEASURED_IN -> Population
  - 두 계열 모두: Outcome -> COMPARED_WITH_PROGRAM -> Program(비교 조건),
    Program -> APPLIED_TO -> Population(같은 Outcome의 평가 Program x Population에서 유도, v1.46)
시행 계열의 Population -> PARTICIPATES_IN -> Program은 만들지 않는다.
Outcome이 없는 Program은 Population과 연결하지 않고 warnings로 알리기만 한다. 대조군 Population은 Outcome에 연결되지 않아도 경고하지 않는다(v1.52). 번호 점검 결과는 값을 바꾸지 않고 warnings에 적는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from ..attrs import NumAttr, TextAttr
from ..config import Config
from ..evidence_refs import number_chunks, resolve
from ..extract_title import load_env
from ..outcome.call import collect_outcomes
from ..outcome.inputs import cached_nodes, node_lists, outcome_chunk_indexes, target_programs
from ..outcome.prompt import OutcomeItem
from ..paper.call import apply_single_doi, call_paper
from ..paper.inputs import build_user as paper_user
from ..paper.inputs import doi_candidates, load_meta, paper_chunk_indexes
from ..paper.prompt import PaperOut
from ..population.prompt import PopulationOut
from ..program.prompt import ProgramOut
from ..routing_call import cached_routing
from ..routing_input import section_list

SRC_CODE, SRC_EXTRACTED, SRC_DERIVED = "코드", "추출됨", "유도됨"
REPORT_STUDY_TYPES = ("관찰 연구", "문헌 종합 연구")  # 보고 계열. 나머지(중재 연구)는 시행 계열
OUTCOME_LINK_FIELDS = {"program_ids", "population_ids", "comparison_program_ids", "link_evidence", "link_note"}
CONTROL_ROLE = "대조"
CONTROL_GROUP_ROLE = "대조군"


def attr_json(attr, ref_map: dict[str, dict]) -> dict:
    """속성 하나를 값·정보 상태·근거 원문·메모로 바꾼다. 입력에 없는 근거 번호는 `invalid_evidence`에 적는다."""
    evidence, invalid = resolve(attr, ref_map)
    out = {"value": attr.value, "status": attr.status, "evidence": evidence, "note": attr.note}
    if invalid:
        out["invalid_evidence"] = invalid
    return out


def model_json(obj, ref_map: dict[str, dict], skip: set[str] = frozenset()):
    """속성 모델을 JSON으로 바꾼다. 속성(TextAttr/NumAttr)은 attr_json, 목록·중첩 모델은 재귀로 바꾼다."""
    if isinstance(obj, (TextAttr, NumAttr)):
        return attr_json(obj, ref_map)
    if isinstance(obj, BaseModel):
        return {name: model_json(v, ref_map) for name, v in obj if name not in skip}
    if isinstance(obj, list):
        return [model_json(v, ref_map) for v in obj]
    return obj


def _evidence(ids: list[str], ref_map: dict[str, dict]) -> list[dict]:
    class _A:
        evidence = ids

    found, invalid = resolve(_A, ref_map)
    return found + [{"ref": r, "invalid": True} for r in invalid]


def build_graph(paper: PaperOut, programs: ProgramOut, populations: PopulationOut, outcomes: list[OutcomeItem], ref_map: dict[str, dict], study_type: str) -> dict:
    """논문 한 편의 nodes·edges·warnings를 만든다. 노드 번호: paper, P1…, G1…, O1…

    연구 유형이 관찰·문헌 종합이면 보고 계열, 중재 연구이면 시행 계열 관계를 만든다.
    """
    report = study_type in REPORT_STUDY_TYPES
    nodes = [{"id": "paper", "type": "Paper", "attrs": model_json(paper, ref_map)}]
    prog_ids = [f"P{k}" for k in range(1, len(programs.programs) + 1)]
    pop_ids = [f"G{k}" for k in range(1, len(populations.populations) + 1)]
    out_ids = [f"O{k}" for k in range(1, len(outcomes) + 1)]
    for pid, p in zip(prog_ids, programs.programs):
        nodes.append({"id": pid, "type": "Program", "attrs": model_json(p, ref_map)})
    for gid, g in zip(pop_ids, populations.populations):
        nodes.append({"id": gid, "type": "Population", "attrs": model_json(g, ref_map)})
    for oid, o in zip(out_ids, outcomes):
        nodes.append({"id": oid, "type": "Outcome", "attrs": model_json(o, ref_map, OUTCOME_LINK_FIELDS)})

    role = {pid: p.condition_role.value for pid, p in zip(prog_ids, programs.programs)}
    edges: list[dict] = []
    warnings: list[dict] = []

    def edge(frm, to, etype, source, evidence=None, note=None, **extra):
        edges.append({"from": frm, "to": to, "type": etype, "source": source, "evidence": evidence or [], "note": note, **extra})

    for ids, etype in ((prog_ids, "HAS_PROGRAM"), (pop_ids, "HAS_POPULATION")):
        for nid in ids:
            edge("paper", nid, etype, SRC_CODE)
    if report:
        for oid in out_ids:
            edge("paper", oid, "REPORTS", SRC_CODE)

    applied: dict[tuple[str, str], list[str]] = {}
    applied_evidence: dict[tuple[str, str], list[str]] = {}
    for oid, o in zip(out_ids, outcomes):
        valid = {"program_ids": set(prog_ids), "comparison_program_ids": set(prog_ids), "population_ids": set(pop_ids)}
        links = {}
        pop_edge = "ANALYZES_POPULATION" if report else "MEASURED_IN"
        for field, etype, to_type in (
            ("program_ids", "EVALUATES_PROGRAM" if report else "HAS_OUTCOME", "Program"),
            ("comparison_program_ids", "COMPARED_WITH_PROGRAM", "Program"),
            ("population_ids", pop_edge, "Population"),
        ):
            good = []
            for i in dict.fromkeys(getattr(o, field)):
                if i in valid[field]:
                    good.append(i)
                    ev, note = _evidence(o.link_evidence, ref_map), o.link_note
                    if field == "program_ids" and not report:  # 시행 계열: Program -> HAS_OUTCOME -> Outcome
                        edge(i, oid, etype, SRC_EXTRACTED, ev, note)
                    else:
                        edge(oid, i, etype, SRC_EXTRACTED, ev, note)
                else:
                    warnings.append({"code": "invalid_link", "node": oid, "message": f"{field}의 {i}는 {to_type} 목록에 없는 번호라 연결하지 않음"})
            links[field] = good
        if not links["program_ids"]:
            warnings.append({"code": "outcome_without_program", "node": oid, "message": "평가 대상 Program이 연결되지 않음"})
        if not links["population_ids"]:
            warnings.append({"code": "outcome_without_population", "node": oid, "message": "Population이 연결되지 않음"})
        for pid in links["program_ids"]:
            if role.get(pid) == CONTROL_ROLE:
                warnings.append({"code": "control_evaluated", "node": oid, "message": f"대조 Program {pid}가 평가 대상으로 연결됨"})
            if pid in links["comparison_program_ids"]:
                warnings.append({"code": "self_comparison", "node": oid, "message": f"{pid}가 평가 대상이면서 비교 대상으로도 연결됨"})
        for pid in links["program_ids"]:
            for gid in links["population_ids"]:
                applied.setdefault((pid, gid), []).append(oid)
                applied_evidence.setdefault((pid, gid), []).extend(o.link_evidence)

    for (pid, gid), via in applied.items():
        edge(pid, gid, "APPLIED_TO", SRC_DERIVED, _evidence(list(dict.fromkeys(applied_evidence[(pid, gid)])), ref_map), None, via_outcomes=via)

    linked_programs = {p for p, _ in applied}
    for pid in prog_ids:
        if pid not in linked_programs and role.get(pid) != CONTROL_ROLE:
            warnings.append({"code": "program_without_population", "node": pid, "message": "Outcome이 없어 Population과 연결하지 않음"})
    linked_pops = {g for _, g in applied}
    pop_role = {gid: g.group_role.value for gid, g in zip(pop_ids, populations.populations)}
    for gid in pop_ids:
        if gid not in linked_pops and pop_role.get(gid) != CONTROL_GROUP_ROLE:  # 대조군은 추천 대상이 아니라 Outcome에 연결되지 않아도 정상(v1.52)
            warnings.append({"code": "population_without_outcome", "node": gid, "message": "어떤 Outcome에도 연결되지 않음"})
    return {"nodes": nodes, "edges": edges, "warnings": warnings}


def load_cached_graph(chunks_json: Path, root: Path) -> dict:
    """캐시된 추출 결과를 모아 연결 JSON을 만든다. 캐시에 없으면 BudgetExceeded가 난다(API 호출 없음)."""
    load_env(root / ".env")
    cfg = Config()
    cache_dir = root / "outputs" / "cache"
    data = json.loads(chunks_json.read_text(encoding="utf-8"))
    rows = section_list(data)
    route = cached_routing(rows, cfg, cache_dir)

    meta = load_meta(chunks_json, root)
    dois = doi_candidates(data["chunks"], meta)
    paper_sel = [c for c in data["chunks"] if c["index"] in paper_chunk_indexes(route, rows)]
    user, paper_refs = paper_user(paper_sel, dois, meta)
    paper, _ = call_paper(user, cfg, cache_dir, 0)
    apply_single_doi(paper, dois, paper_refs)
    study_type = paper.study_type.value
    if study_type is None:
        raise SystemExit("Paper 결과에 연구 유형(study_type)이 없어 Program·Population·Outcome 입력을 만들 수 없다.")

    programs, populations = cached_nodes(data, rows, route, study_type, cfg, cache_dir)
    selected = [c for c in data["chunks"] if c["index"] in outcome_chunk_indexes(route, rows)]
    outcomes, _, _ = collect_outcomes(selected, study_type, programs, populations, target_programs(programs), cfg, cache_dir, 0, log=lambda *_: None)
    _, ref_map = number_chunks(data["chunks"])
    graph = build_graph(paper, programs, populations, outcomes, ref_map, study_type)
    return {"source": chunks_json.name, "study_type": study_type, **graph}


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="캐시된 노드 추출 결과로 연결 JSON을 만듭니다(LLM 호출 없음).")
    parser.add_argument("chunks_json", type=Path)
    parser.add_argument("--out", type=Path, help="저장할 JSON 경로(기본: outputs/graph/<논문>.json)")
    args = parser.parse_args()

    graph = load_cached_graph(args.chunks_json, root)
    out = args.out or root / "outputs" / "graph" / (args.chunks_json.stem.replace("_chunks", "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for n in graph["nodes"]:
        counts[n["type"]] = counts.get(n["type"], 0) + 1
    etypes: dict[str, int] = {}
    for e in graph["edges"]:
        etypes[e["type"]] = etypes.get(e["type"], 0) + 1
    print(f"저장: {out}")
    print("노드:", counts)
    print("연결:", etypes)
    print(f"경고 {len(graph['warnings'])}건")
    for w in graph["warnings"]:
        print(f"  - [{w['code']}] {w['node']}: {w['message']}")


if __name__ == "__main__":
    main()
