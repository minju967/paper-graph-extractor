"""검수된 그래프 JSON을 Neo4j에 적재한다 (자동 추출과 분리된 별도 단계).

구현 기본 매핑 — 계획서 9절에서 중첩 값·근거·버전·재적재 정책이 '미정'이므로 아래는 잠정안이며 바꿀 수 있다.
- 노드 ID: "{paper_id}:{node_id}" 로 논문별 네임스페이스를 둔다. MERGE로 쓰므로 재적재해도 중복되지 않는다.
- 속성 하나(Attr)는 세 속성으로 펼친다: <키>, <키>_status, <키>_evidence(문자열 리스트). 값 없는 속성은 <키>_status만 둔다.
- 활동 구성(중첩 목록)은 `activities_json` 문자열 속성으로 보관한다. (Neo4j 속성에 중첩 구조를 직접 넣지 않는다.)
- 재적재: 같은 paper_id가 이미 있으면 기본은 중단. --replace를 주면 그 논문의 기존 노드를 지우고 다시 적재한다.
  버전 보관은 하지 않는다(필요하면 paper_id에 버전을 붙여 별도 논문으로 적재).
- 검토 항목(review_items)과 보류된 관계는 검증된 사실로 적재하지 않는다.
- 검토 상태가 reviewed가 아닌 그래프는 적재를 거부한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .models import Attr, PaperGraph

LABELS = {"Paper": "Paper", "Population": "Population", "Program": "Program", "Outcome": "Outcome"}


def load_graph(path: Path) -> PaperGraph:
    return PaperGraph.model_validate_json(Path(path).read_text("utf-8"))


def mark_reviewed(path: Path, reviewer: str) -> None:
    g = load_graph(path)
    g.status.review_status = "reviewed"
    g.status.reviewer = reviewer
    Path(path).write_text(g.model_dump_json(indent=2), "utf-8")


def _flat_attr(props: dict, key: str, attr: Attr) -> None:
    if attr.value is not None:
        props[key] = attr.value
    props[f"{key}_status"] = attr.status
    if attr.evidence_block_ids:
        props[f"{key}_evidence"] = [f"{props['paper_id']}#{b}" for b in attr.evidence_block_ids]


def node_props(graph: PaperGraph, node) -> dict:
    props: dict = {"paper_id": graph.paper_id, "node_id": node.id, "uid": f"{graph.paper_id}:{node.id}"}
    for name, val in node:
        if isinstance(val, Attr):
            _flat_attr(props, name, val)
        elif isinstance(val, list):
            props["activities_json"] = json.dumps([a.model_dump() for a in val], ensure_ascii=False)
    return props


def build_rows(graph: PaperGraph) -> tuple[list[tuple[str, dict]], list[dict]]:
    nodes: list[tuple[str, dict]] = [("Paper", node_props(graph, graph.paper))]
    nodes += [("Population", node_props(graph, n)) for n in graph.populations]
    nodes += [("Program", node_props(graph, n)) for n in graph.programs]
    nodes += [("Outcome", node_props(graph, n)) for n in graph.outcomes]
    rels = [
        {
            "type": r.type,
            "from": f"{graph.paper_id}:{r.from_id}",
            "to": f"{graph.paper_id}:{r.to_id}",
            "evidence": [f"{graph.paper_id}#{b}" for b in r.evidence_block_ids],
        }
        for r in graph.relations
    ]
    return nodes, rels


def load_into_neo4j(graph: PaperGraph, *, replace: bool = False, log=print) -> None:
    if graph.status.review_status != "reviewed":
        raise PermissionError("사람 검토가 끝나지 않은 그래프는 적재하지 않는다. 검토 후 `mark-reviewed`를 실행한다.")
    if graph.status.partial_reasons:
        log(f"[주의] 부분 완료 그래프입니다: {graph.status.partial_reasons}")
    uri, user, pw = os.environ.get("NEO4J_URI"), os.environ.get("NEO4J_USER"), os.environ.get("NEO4J_PASSWORD")
    if not (uri and user and pw):
        raise ValueError("환경변수 NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD가 필요하다.")
    from neo4j import GraphDatabase  # 지연 import

    nodes, rels = build_rows(graph)
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    try:
        with driver.session() as s:
            for label in LABELS.values():
                s.run(f"CREATE CONSTRAINT {label.lower()}_uid IF NOT EXISTS FOR (n:{label}) REQUIRE n.uid IS UNIQUE").consume()
            exists = s.run("MATCH (n {paper_id: $pid}) RETURN count(n) AS c", pid=graph.paper_id).single()["c"]
            if exists and not replace:
                raise FileExistsError(f"paper_id={graph.paper_id}가 이미 적재되어 있다. 교체하려면 --replace를 사용한다.")
            if exists:
                s.run("MATCH (n {paper_id: $pid}) DETACH DELETE n", pid=graph.paper_id).consume()
            for label, props in nodes:
                s.run(f"MERGE (n:{label} {{uid: $uid}}) SET n = $props", uid=props["uid"], props=props).consume()
            for r in rels:
                # 관계 유형은 검증된 허용 목록(Literal)에서만 오므로 문자열 보간이 안전하다.
                s.run(
                    f"MATCH (a {{uid: $f}}), (b {{uid: $t}}) MERGE (a)-[x:{r['type']}]->(b) SET x.evidence = $e",
                    f=r["from"], t=r["to"], e=r["evidence"],
                ).consume()
    finally:
        driver.close()
    log(f"[적재] 노드 {len(nodes)}개, 관계 {len(rels)}개")
