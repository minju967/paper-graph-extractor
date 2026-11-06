"""Outcome의 direction(감소/차이 없음/증가)을 코드가 정한다. LLM을 호출하지 않는다(v1.37).

LLM은 변화 방향(change_direction: 감소/증가/null)과 유의성(significance: 유의/비유의/null)만 답하고, direction은 비운다.
  1. 유의성이 비유의 -> 차이 없음
  2. 유의성이 유의 -> 변화 방향(감소/증가). 변화 방향이 없으면 3번으로
  3. 신뢰구간이 정확히 하나 있으면 구간 전체가 0보다 작으면 감소, 크면 증가, 0을 지나면 차이 없음(v1.33)
  4. 아니면 null
안전장치로 신뢰구간(구간 기준 유의성)과 p값(0.05 기준)이 LLM이 답한 유의성·판정 결과와 어긋나면 값을 바꾸지 않고 알려 준다.
"""

from __future__ import annotations

import re

from .prompt import OutcomeItem

ALPHA = 0.05
_NUM = r"[+\-−–]?\s*\d+(?:\.\d+)?"
# [a to b], (a to b), [a, b], [a~b] 형태. 음수 부호는 -, −(U+2212), –(en dash)를 모두 받는다.
_CI = re.compile(rf"[\[(]\s*({_NUM})\s*(?:to|~|,|;|–(?=\s)|—)\s*({_NUM})\s*[\])]")
_P = re.compile(r"\b[pP]\s*(<=|>=|≤|≥|<|>|=)\s*(\d*\.?\d+)")


def _to_float(text: str) -> float:
    return float(text.replace("−", "-").replace("–", "-").replace(" ", ""))


def find_ci(item: OutcomeItem) -> tuple[float, float] | None:
    """대표값, 없으면 통계 문자열에서 신뢰구간을 찾는다. 정확히 하나일 때만 돌려준다."""
    for attr in (item.representative_values, item.significance_statistics):
        if not attr.value:
            continue
        found = _CI.findall(attr.value)
        if len(found) == 1:
            lo, hi = sorted((_to_float(found[0][0]), _to_float(found[0][1])))
            return lo, hi
        if len(found) > 1:
            return None
    return None


def direction_from_ci(lo: float, hi: float) -> str:
    if hi < 0:
        return "감소"
    if lo > 0:
        return "증가"
    return "차이 없음"


def significance_from_p(item: OutcomeItem) -> str | None:
    """통계 문자열의 p값을 0.05 기준으로 읽는다. 값이 여럿이면 모두 같은 결론일 때만 돌려준다."""
    text = item.significance_statistics.value
    if not text:
        return None
    results = set()
    for op, num in _P.findall(text):
        p = float(num)
        if op in ("<", "<=", "≤"):
            results.add("유의" if p <= ALPHA else None)
        elif op in (">", ">=", "≥"):
            results.add("비유의" if p >= ALPHA else None)
        else:
            results.add("유의" if p < ALPHA else "비유의")
    results.discard(None)
    return results.pop() if len(results) == 1 else None


def _evidence(item: OutcomeItem) -> list[str]:
    ids: list[str] = []
    for attr in (item.significance, item.change_direction, item.representative_values):
        for e in attr.evidence:
            if e not in ids:
                ids.append(e)
    return ids


def decide_direction(item: OutcomeItem) -> bool:
    """direction을 코드로 정해 채운다. 정했으면 True, 정하지 못해 null로 두면 False."""
    sig, change = item.significance.value, item.change_direction.value
    if sig == "비유의":
        value, basis = "차이 없음", "논문이 유의하지 않다고 보고하여 코드가 차이 없음으로 판정함."
    elif sig == "유의" and change in ("감소", "증가"):
        value, basis = change, f"논문이 유의한 {change}로 보고하여 코드가 판정함."
    else:
        ci = find_ci(item)
        if ci is None:
            item.direction.value, item.direction.evidence, item.direction.note = None, [], None
            return False
        value = direction_from_ci(*ci)
        basis = f"신뢰구간 [{ci[0]:g}, {ci[1]:g}] 기준으로 코드가 판정함."
    item.direction.value, item.direction.evidence, item.direction.note = value, _evidence(item), basis
    return True


def direction_mismatch(item: OutcomeItem) -> str | None:
    """LLM이 답한 유의성·변화 방향이 신뢰구간, p값과 어긋나면 설명을 돌려준다. 값은 바꾸지 않는다."""
    problems = []
    ci = find_ci(item)
    sig = item.significance.value
    if ci is not None and sig is not None:
        ci_sig = "비유의" if ci[0] <= 0 <= ci[1] else "유의"
        if ci_sig != sig:
            problems.append(f"유의성={sig} 이지만 신뢰구간 [{ci[0]:g}, {ci[1]:g}] 기준은 {ci_sig}")
    p_sig = significance_from_p(item)
    if p_sig is not None and sig is not None and p_sig != sig:
        problems.append(f"유의성={sig} 이지만 p값({item.significance_statistics.value}) 기준은 {p_sig}")
    if ci is not None and sig == "유의" and item.change_direction.value is not None:
        expected = direction_from_ci(*ci)
        if expected in ("감소", "증가") and expected != item.change_direction.value:
            problems.append(f"변화 방향={item.change_direction.value} 이지만 신뢰구간 기준은 {expected}")
    return "; ".join(problems) or None
