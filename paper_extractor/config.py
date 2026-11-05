"""실행 설정. 모델명·한도·토큰 목표는 모두 설정값이며 코드에 특정 모델을 고정하지 않는다."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 버전은 캐시 키에 포함된다. 스키마·프롬프트·규칙을 바꾸면 반드시 올린다.
SCHEMA_VERSION = "0.1"
PROMPT_VERSION = "0.1"


@dataclass
class Config:
    # --- 모델 (계획서 9절: 모델명·출력 한도 미정 → 환경변수/인자로 선택) ---
    model: str = field(default_factory=lambda: os.environ.get("PAPER_EXTRACTOR_MODEL", ""))
    max_output_tokens_classify: int = 2_000
    max_output_tokens_extract: int = 8_000
    max_output_tokens_merge: int = 16_000
    max_output_tokens_image: int = 8_000
    request_timeout_s: float = 180.0
    # 시간 초과·일시적 오류가 나면 최대 이만큼 더 시도하고, 시간 초과 때마다 제한 시간에 timeout_growth를 곱한다.
    timeout_retries: int = 2
    timeout_growth: float = 1.5

    # --- 호출 예산 (계획서 6절: 논문당 15회 이내, 이미지 최대 5페이지) ---
    max_api_calls: int = 15
    max_image_pages: int = 5
    # 이미지 보완을 위해 미리 남겨두는 호출 수 (이미지가 불필요하면 재시도 등에 쓰인다)
    image_reserve: int = 2
    # 호출 실패 시 같은 입력의 재시도 횟수. 남은 예산 안에서만 사용한다.
    max_retries_per_call: int = 1

    # --- 청킹 (계획서 준비 B: 4,000 목표 / 6,000 일반 상한은 초기 제안값) ---
    chunk_target_tokens: int = 4_000
    chunk_max_tokens: int = 6_000
    # 호출 예산 때문에 키울 수 있는 청크 상한 (모델 입력 한도에 맞춰 조정)
    chunk_hard_max_tokens: int = 24_000
    # 통합 호출 입력 상한. 넘으면 조용히 자르지 않고 부분 결과로 표시한다.
    merge_max_input_tokens: int = 60_000
    merge_excerpt_max_tokens: int = 6_000

    # --- 판독 이상 신호 ---
    min_chars_per_page: int = 400
    max_garbled_ratio: float = 0.02

    # --- 경로 ---
    output_dir: Path = Path("outputs")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        return d

    def attempts_for(self, max_calls: int) -> int:
        """계획한 호출 수 max_calls에 재시도 몫을 더한 호출 한도. 캐시 전용(0)이면 0이다."""
        return max_calls + self.timeout_retries if max_calls > 0 else 0

    def cache_fingerprint(self) -> dict:
        """결과에 영향을 주는 설정만 캐시 키에 넣는다."""
        return {
            "model": self.model,
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "chunk_target_tokens": self.chunk_target_tokens,
            "chunk_max_tokens": self.chunk_max_tokens,
        }

    def validate(self) -> None:
        if self.max_api_calls < 3:
            raise ValueError("max_api_calls는 최소 3(판정+추출+통합)이어야 한다.")
        if self.image_reserve > self.max_image_pages:
            self.image_reserve = self.max_image_pages
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("chunk_target_tokens는 chunk_max_tokens 이하여야 한다.")
