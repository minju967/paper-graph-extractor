"""신뢰구간으로 direction(감소/동일/증가)을 판정하는 코드. LLM을 호출하지 않는다.

논문에 방향 문구 없이 수치만 있으면 LLM은 direction을 null로 두고(프롬프트 규칙), 대표값·통계 문자열에 신뢰구간이
정확히 하나 있으면 코드가 채운다(v1.33). 구간 전체가 0보다 작으면 감소, 전체가 0보다 크면 증가, 0을 지나면 동일
(유의한 차이 없음)이다. LLM이 정한 값이 구간과 어긋나면 값을 바꾸지 않고 알려 준다.
"""

from __future__ import annotations

import re

from .prompt import OutcomeItem

_NUM = r"[+\-−–]?\s*\d+(?:\.\d+)?"
# [a to b], (a to b), [a, b], [a~b] 형태. 음수 부호는 -, −(U+2212), –(en dash)를 모두 받는다.
_CI = re.compile(rf"[\[(]\s*({_NUM})\s*(?:to|~|,|;|–(?=\s)|—)\s*({_NUM})\s*[\])]")


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
    return "동일"


def fill_direction(item: OutcomeItem) -> bool:
    """direction이 null이고 신뢰구간이 있으면 코드가 채운다. 채웠으면 True."""
    if item.direction.value is not None:
        return False
    ci = find_ci(item)
    if ci is None:
        return False
    item.direction.value = direction_from_ci(*ci)
    source = item.representative_values if item.representative_values.value and _CI.search(item.representative_values.value) else item.significance_statistics
    item.direction.evidence = list(source.evidence)
    item.direction.note = f"신뢰구간 [{ci[0]:g}, {ci[1]:g}] 기준으로 코드가 판정함."
    return True


def direction_mismatch(item: OutcomeItem) -> str | None:
    """LLM이 정한 direction이 신뢰구간과 어긋나면 설명을 돌려준다. 코드가 채운 값은 어긋날 수 없다."""
    ci = find_ci(item)
    if ci is None or item.direction.value is None:
        return None
    expected = direction_from_ci(*ci)
    if item.direction.value != expected:
        return f"direction={item.direction.value} 이지만 신뢰구간 [{ci[0]:g}, {ci[1]:g}] 기준은 {expected}"
    return None
