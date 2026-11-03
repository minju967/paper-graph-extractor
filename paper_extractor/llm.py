"""LLM 호출 계층.

- 모든 시도(실패·재시도 포함)를 직접 집계해 논문당 호출 한도를 지킨다. SDK 자동 재시도는 끈다.
- 완료된 결과는 캐시 키(PDF·설정·모델·스키마·프롬프트 버전·입력)로 재사용한다. 캐시 적중은 호출 예산을 쓰지 않는다.
- 불완전 출력·거절·API 오류는 성공으로 저장하지 않는다.
- API 키는 환경변수(OPENAI_API_KEY)에서만 읽고 어떤 로그에도 남기지 않는다.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BudgetExceeded(Exception):
    pass


class LLMError(Exception):
    """불완전 출력·거절·API 오류. retryable이면 남은 예산 안에서 재시도할 수 있다."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class UsageRecord:
    stage: str
    attempt: int
    model: str
    status: str  # ok | error | cache_hit
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    error: str | None = None
    elapsed_s: float = 0.0


class LLMClient(Protocol):
    def parse(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        schema: type[T],
        max_output_tokens: int,
        images: list[bytes] | None = None,
    ) -> tuple[T, Usage]: ...


class OpenAIClient:
    """OpenAI Responses API + Structured Outputs."""

    def __init__(self, model: str, timeout_s: float = 180.0):
        if not model:
            raise ValueError(
                "모델명이 설정되지 않았다. --model 또는 환경변수 PAPER_EXTRACTOR_MODEL로 지정한다."
            )
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("환경변수 OPENAI_API_KEY가 설정되지 않았다.")
        from openai import OpenAI  # 지연 import

        # max_retries=0: SDK의 숨은 자동 재시도로 호출 한도를 넘지 않도록 한다.
        self._client = OpenAI(max_retries=0, timeout=timeout_s)
        self.model = model

    def _redact(self, text: str) -> str:
        key = os.environ.get("OPENAI_API_KEY", "")
        return text.replace(key, "***") if key else text

    def parse(self, *, stage, system, user, schema, max_output_tokens, images=None):
        content: list[dict] = [{"type": "input_text", "text": user}]
        for img in images or []:
            b64 = base64.b64encode(img).decode("ascii")
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})
        try:
            resp = self._client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                text_format=schema,
                max_output_tokens=max_output_tokens,
            )
        except Exception as e:  # noqa: BLE001 - SDK 예외 계층이 넓다
            name = type(e).__name__
            retryable = name in {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}
            raise LLMError(f"{name}: {self._redact(str(e))[:500]}", retryable=retryable) from None

        if getattr(resp, "status", "completed") != "completed":
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)
            raise LLMError(f"불완전 응답(status={resp.status}, reason={reason}): 출력 한도 초과 가능", retryable=False)
        for item in getattr(resp, "output", []) or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", "") == "refusal":
                    raise LLMError(f"모델 거절: {str(getattr(part, 'refusal', ''))[:300]}", retryable=False)
        parsed = getattr(resp, "output_parsed", None)
        if parsed is None:
            raise LLMError("구조화 출력 파싱 결과가 비어 있다.", retryable=False)

        u = getattr(resp, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cached_tokens=getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0,
        )
        return parsed, usage


@dataclass
class CallBudget:
    max_calls: int
    used: int = 0
    records: list[UsageRecord] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.max_calls - self.used

    def consume(self) -> None:
        if self.remaining <= 0:
            raise BudgetExceeded(f"논문당 API 호출 한도({self.max_calls}회)를 초과할 수 없다.")
        self.used += 1

    def totals(self) -> dict:
        ok = [r for r in self.records if r.status != "cache_hit"]
        return {
            "api_calls_attempted": self.used,
            "max_calls": self.max_calls,
            "failed_attempts": sum(1 for r in self.records if r.status == "error"),
            "cache_hits": sum(1 for r in self.records if r.status == "cache_hit"),
            "input_tokens": sum(r.input_tokens for r in ok),
            "output_tokens": sum(r.output_tokens for r in ok),
            "cached_tokens": sum(r.cached_tokens for r in ok),
            "reasoning_tokens": sum(r.reasoning_tokens for r in ok),
        }


def make_key(parts: dict) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class BudgetedLLM:
    def __init__(self, client: LLMClient, budget: CallBudget, cache_dir: Path, fingerprint: dict, model: str):
        self.client = client
        self.budget = budget
        self.cache_dir = cache_dir
        self.fingerprint = fingerprint
        self.model = model
        cache_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        schema: type[T],
        max_output_tokens: int,
        images: list[bytes] | None = None,
        retries: int = 0,
        keep_free: int = 0,
    ) -> T:
        """한 논리 호출을 실행한다.

        retries: 실패 후 추가 시도 횟수. 재시도는 남은 예산에서 keep_free(이후 단계 예약분)를
        뺀 뒤에도 여유가 있을 때만 수행한다. 첫 시도는 이미 계획에 포함된 호출이므로 예약분을 뺀 검사를 하지 않는다.
        """
        img_hash = hashlib.sha256(b"".join(images or [])).hexdigest() if images else ""
        key = make_key(
            {
                "fp": self.fingerprint,
                "stage": stage,
                "system": system,
                "user": user,
                "schema": schema.__name__,
                "images": img_hash,
            }
        )
        path = self.cache_dir / f"{stage}-{key[:16]}.json"
        if path.exists():
            try:
                cached = schema.model_validate_json(json.loads(path.read_text("utf-8"))["parsed_json"])
                self.budget.records.append(UsageRecord(stage, 0, self.model, "cache_hit"))
                return cached
            except Exception:  # noqa: BLE001 - 손상된 캐시는 무시하고 재호출
                path.unlink(missing_ok=True)

        last_err: LLMError | None = None
        for attempt in range(retries + 1):
            if attempt > 0 and self.budget.remaining - keep_free < 1:
                break
            self.budget.consume()  # 시도 전에 집계 → 예외가 나도 호출 수가 누락되지 않는다
            t0 = time.time()
            try:
                parsed, usage = self.client.parse(
                    stage=stage,
                    system=system,
                    user=user,
                    schema=schema,
                    max_output_tokens=max_output_tokens,
                    images=images,
                )
            except LLMError as e:
                self.budget.records.append(
                    UsageRecord(stage, attempt, self.model, "error", error=str(e), elapsed_s=time.time() - t0)
                )
                last_err = e
                if not e.retryable:
                    break
                continue
            self.budget.records.append(
                UsageRecord(stage, attempt, self.model, "ok", **asdict(usage), elapsed_s=time.time() - t0)
            )
            path.write_text(json.dumps({"parsed_json": parsed.model_dump_json()}, ensure_ascii=False), "utf-8")
            return parsed
        raise last_err or LLMError("호출 예산 부족으로 재시도하지 못했다.")
