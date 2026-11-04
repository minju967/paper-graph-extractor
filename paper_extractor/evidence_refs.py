"""근거 번호 붙이기: 청크의 섹션 본문을 문장으로 나누어 `섹션ID.s번호`를 붙인다. LLM을 호출하지 않는다.

LLM에는 번호가 붙은 텍스트를 보내고, LLM이 돌려준 번호는 `ref_map`으로 원문 문장과 섹션 경로로 바꾼다.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .routing_input import section_bodies

# 마침표 뒤에서 문장을 나누지 않는 약어
_ABBR = ("al.", "e.g.", "i.e.", "Fig.", "Figs.", "No.", "vs.", "approx.", "Dr.", "Eq.")
_END = re.compile(r"(?<=[.!?。])\s+")


def split_sentences(body: str) -> list[str]:
    """본문을 문장 목록으로 나눈다.

    빈 줄로 문단을 나누고, 표는 `|`로 시작하는 줄 하나를 한 단위로 보고(구분선 제외), 줄이 바뀐 셀은 앞 행에 이어 붙인다.
    문단 안의 일반 줄바꿈은 공백으로 이어 붙인 뒤 문장 끝 부호 뒤에서 나눈다.
    """
    units: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        lines = [ln.strip() for ln in para.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        if lines[0].startswith("|"):
            rows: list[str] = []
            for ln in lines:
                if ln.startswith("|") or not rows:
                    rows.append(ln)
                else:  # 셀 안에서 줄이 바뀐 경우: 앞 행에 이어 붙인다
                    rows[-1] += " " + ln
            units += [r for r in rows if not re.fullmatch(r"[|\-:\s]+", r)]
            continue
        text = " ".join(lines)
        buf = ""
        pieces = _END.split(text)
        for k, piece in enumerate(pieces):
            buf = f"{buf} {piece}" if buf else piece
            nxt = pieces[k + 1] if k + 1 < len(pieces) else ""
            if buf.endswith(_ABBR):  # 약어 뒤의 마침표는 문장 끝이 아니다
                continue
            if re.search(r"(?<![A-Za-z])[A-Z][a-z]+ [A-Z]\.$", buf):  # 이름 뒤 한 글자 약자(`Neil A.`)
                continue
            if re.search(r"\d\.$", buf) and re.match(r"\d", nxt):  # `2024. 12. 17.` 같은 날짜·번호
                continue
            units.append(buf)
            buf = ""
        if buf:
            units.append(buf)
    return units


def number_chunk(chunk: dict) -> tuple[str, dict[str, dict]]:
    """청크 텍스트의 섹션 본문을 번호 붙은 문장 줄로 바꾼다. (번호 붙은 텍스트, ref_map)을 돌려준다.

    ref_map: `h3.s4` -> {"section": 섹션 경로, "text": 문장 원문}
    """
    text = chunk["text"]
    out, cursor = [], 0
    ref_map: dict[str, dict] = {}
    for sec, body in zip(chunk["sections"], section_bodies(chunk)):
        start = text.index(body, cursor) if body else cursor
        out.append(text[cursor:start])
        cursor = start + len(body)
        lines = []
        for n, sent in enumerate(split_sentences(body), 1):
            ref = f"{sec['id']}.s{n}"
            ref_map[ref] = {"section": sec["path"], "text": sent}
            lines.append(f"[{ref}] {sent}")
        out.append("\n" + "\n".join(lines) + "\n" if lines else body)
    out.append(text[cursor:])
    return "".join(out), ref_map


def main() -> None:
    parser = argparse.ArgumentParser(description="청크에 문장 번호를 붙여 출력합니다(LLM 호출 없음).")
    parser.add_argument("chunks_json", type=Path)
    parser.add_argument("chunk", type=int, help="청크 번호")
    parser.add_argument("--limit", type=int, default=60, help="출력할 최대 줄 수")
    args = parser.parse_args()
    data = json.loads(args.chunks_json.read_text(encoding="utf-8"))
    chunk = next(c for c in data["chunks"] if c["index"] == args.chunk)
    numbered, ref_map = number_chunk(chunk)
    print(f"문장 수: {len(ref_map)}\n")
    print("\n".join(numbered.split("\n")[: args.limit]))


if __name__ == "__main__":
    main()
