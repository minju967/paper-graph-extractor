"""Paper 호출의 입력 만들기: 캐시된 라우팅 결과로 보낼 청크를 고르고, DOI 후보를 찾아 입력 텍스트를 만든다.

LLM을 호출하지 않는다. 라우팅 결과가 캐시에 없으면 호출하지 않고 오류로 멈춘다.
규칙(v1.07, v1.08): 1순위 섹션의 청크와 후보 1순위 섹션의 청크를 함께 보낸다(같은 청크는 한 번).
DOI 후보가 하나면 코드가 채택하고, 둘 이상이면 후보를 입력에 붙여 LLM이 고르게 한다.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..config import Config
from ..evidence_refs import number_chunk
from ..extract_title import load_env
from ..routing_call import cached_routing
from ..routing_input import section_list
from ..routing_prompt import RoutingOut

META_SECTION = "(메타데이터: 머리글·바닥글)"
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>\]\)]+")


def paper_chunk_indexes(route: RoutingOut, rows: list[dict]) -> list[int]:
    """Paper용 청크 번호: 1순위 섹션들의 청크 + 후보 첫 섹션의 청크(중복 제거, 문서 순서)."""
    chunk_of = {r["id"]: r["chunk"] for r in rows}
    ids = list(route.paper.primary) + route.paper.fallback[:1]
    return sorted({chunk_of[i] for i in ids if i in chunk_of})


def doi_candidates(chunks: list[dict], meta: dict | None = None) -> list[str]:
    """논문 텍스트와 메타데이터(`doi_candidates`, 머리글 줄)에서 DOI 모양의 문자열을 중복 없이 모은다(등장 순서)."""
    texts = [c["text"] for c in chunks]
    texts += [r.get("text", "") for r in (meta or {}).get("running_lines", [])]
    found: list[str] = list((meta or {}).get("doi_candidates", []))
    for t in texts:
        for m in DOI_RE.findall(t):
            doi = m.rstrip(".,;:")
            if doi.lower() not in {f.lower() for f in found}:
                found.append(doi)
    return found


def load_meta(chunks_json: Path, root: Path) -> dict:
    """청크 파일(`NN_chunks.json`)과 같은 번호의 메타데이터(`outputs/markdown/NN_meta.json`)를 읽는다. 없으면 빈 dict."""
    path = root / "outputs" / "markdown" / f"{chunks_json.name.split('_')[0]}_meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def meta_block(meta: dict) -> tuple[str, dict[str, dict]]:
    """변환 때 제외한 머리글·바닥글 줄을 `[meta.s번호]`를 붙인 텍스트와 ref_map으로 만든다. 줄이 없으면 빈 문자열."""
    lines = [r["text"] for r in meta.get("running_lines", []) if r.get("text")]
    if not lines:
        return "", {}
    ref_map = {f"meta.s{n}": {"section": META_SECTION, "text": t} for n, t in enumerate(lines, 1)}
    body = "\n".join(f"[{ref}] {v['text']}" for ref, v in ref_map.items())
    return f"[{META_SECTION}]\n{body}", ref_map


def build_user(selected: list[dict], dois: list[str], meta: dict | None = None) -> tuple[str, dict[str, dict]]:
    """Paper 호출에 보낼 사용자 입력과 문장 번호 맵(ref_map)을 만든다.

    맨 앞에 메타데이터(머리글·바닥글 줄)를 `[meta.s번호]`로 붙이고, 이어서 청크 텍스트를 문장마다
    `[섹션ID.s번호]`를 붙여 보낸다. DOI 후보가 둘 이상일 때만 후보 목록을 붙인다.
    """
    parts = []
    ref_map: dict[str, dict] = {}
    block, meta_refs = meta_block(meta or {})
    if block:
        parts.append(block)
        ref_map.update(meta_refs)
    if len(dois) >= 2:
        parts.append("DOI candidates found in the paper:\n" + "\n".join(f"- {d}" for d in dois))
    for c in selected:
        numbered, refs = number_chunk(c)
        dup = ref_map.keys() & refs.keys()
        if dup:
            raise ValueError(f"문장 번호가 청크 사이에서 겹친다: {sorted(dup)[:3]}")
        ref_map.update(refs)
        parts.append(f"[청크 {c['index']}]\n{numbered}")
    return "\n\n".join(parts), ref_map


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Paper 호출 입력을 만들어 확인합니다(LLM 호출 없음).")
    parser.add_argument("chunks_json", type=Path)
    parser.add_argument("--full", action="store_true", help="보낼 입력 전문을 출력")
    args = parser.parse_args()

    load_env(root / ".env")
    cfg = Config()
    data = json.loads(args.chunks_json.read_text(encoding="utf-8"))
    rows = section_list(data)
    route = cached_routing(rows, cfg, root / "outputs" / "cache")

    idx = paper_chunk_indexes(route, rows)
    selected = [c for c in data["chunks"] if c["index"] in idx]
    meta = load_meta(args.chunks_json, root)
    dois = doi_candidates(data["chunks"], meta)
    user, ref_map = build_user(selected, dois, meta)

    print("paper 1순위:", route.paper.primary, "/ 후보:", route.paper.fallback)
    print("보낼 청크:", idx, "토큰:", sum(c["tokens"] for c in selected), "/ 입력 글자 수:", len(user), "/ 문장 번호:", len(ref_map))
    print("DOI 후보:", dois, "->", "코드 채택" if len(dois) == 1 else "미보고" if not dois else "LLM이 선택")
    if args.full:
        print("\n" + user)


if __name__ == "__main__":
    main()
