"""라우팅 입력 만들기: 청크 JSON에서 섹션 목록(청크 번호, 섹션 ID, 제목 경로, 첫 문단 앞부분)을 만든다.

LLM을 호출하지 않는다. 이 목록이 이후 라우팅 호출(노드별로 어느 섹션을 볼지 고르는 호출)의 입력이 된다.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SNIPPET_CHARS = 100  # 첫 문단에서 보여 줄 글자 수


def first_paragraph(body: str, limit: int = SNIPPET_CHARS) -> str:
    """본문의 첫 문단을 한 줄로 줄이고 limit 글자로 자른다. 본문이 없으면 빈 문자열."""
    for para in re.split(r"\n\s*\n", body):
        text = re.sub(r"\s+", " ", para).strip()
        if text:
            return text[:limit]
    return ""


def section_bodies(chunk: dict) -> list[str]:
    """청크 텍스트를 chunk["sections"] 순서대로 섹션별 본문으로 나눈다.

    각 섹션은 `제목 줄`, `[경로: …]` 줄, 본문 순서로 렌더링되어 있으므로, 경로 줄을 차례로 찾아
    그 다음 줄부터 다음 섹션의 제목 줄 앞까지를 본문으로 본다.
    """
    text = chunk["text"]
    starts: list[int] = []  # 각 섹션의 `[경로: …]` 줄이 끝나는 위치(본문 시작)
    cursor = 0
    for sec in chunk["sections"]:
        marker = f"[경로: {sec['path']}]"
        i = text.index(marker, cursor)
        starts.append(i + len(marker))
        cursor = i + len(marker)
    bodies = []
    for k, start in enumerate(starts):
        if k + 1 < len(starts):
            next_marker = text.rfind("\n\n", start, starts[k + 1])  # 다음 섹션 제목 줄 앞의 빈 줄
            end = next_marker if next_marker != -1 else starts[k + 1]
        else:
            end = len(text)
        bodies.append(text[start:end])
    return bodies


def section_list(chunks_data: dict) -> list[dict]:
    """논문의 모든 섹션을 문서 순서로 (청크 번호, 섹션 ID, 제목 경로, 첫 문단 앞부분) 목록으로 만든다."""
    rows = []
    for chunk in chunks_data["chunks"]:
        for sec, body in zip(chunk["sections"], section_bodies(chunk)):
            rows.append(
                {
                    "chunk": chunk["index"],
                    "id": sec["id"],
                    "path": sec["path"],
                    "snippet": first_paragraph(body),
                }
            )
    return rows


def format_list(rows: list[dict]) -> str:
    out = []
    current = None
    for r in rows:
        if r["chunk"] != current:
            current = r["chunk"]
            out.append(f"[청크 {current}]")
        out.append(f"  {r['id']:<6} {r['path']} | {r['snippet'] or '(본문 없음)'}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="청크 JSON에서 라우팅 입력용 섹션 목록을 만들어 출력합니다(LLM 호출 없음).")
    parser.add_argument("chunks_json", type=Path)
    args = parser.parse_args()
    data = json.loads(args.chunks_json.read_text(encoding="utf-8"))
    print(format_list(section_list(data)))


if __name__ == "__main__":
    main()
