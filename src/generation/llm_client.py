"""LiteLLM 통합 클라이언트 — 멀티 모델 지원, 재시도, 토큰 추적."""

from __future__ import annotations

import logging
import time

import litellm
from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from config.settings import settings
from src.generation import LLMResponse

logger = logging.getLogger(__name__)

litellm.drop_params = True

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0

_VERTEX_LOCATION_OVERRIDES: dict[str, str] = {
    "vertex_ai/gemini-2.5-pro": "us-central1",
    "vertex_ai/gemini-3.5-flash": "global",
    "vertex_ai/gemini-3.1-pro-preview": "global",
    "vertex_ai/claude-sonnet-4-5": "us-east5",
}


class LLMError(RuntimeError):
    """LLM 호출 실패 — status_code로 HTTP 매핑."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def generate(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = 60.0,
) -> LLMResponse:
    """LLM 호출 — 재시도 + 토큰/레이턴시 추적.

    Args:
        messages: OpenAI 형식 메시지 리스트 [{"role": ..., "content": ...}].
        model: LiteLLM 모델 ID (예: "openai/gpt-4o-mini").
        temperature: 생성 온도.
        max_tokens: 최대 생성 토큰.
        timeout: API 호출 타임아웃(초).

    Returns:
        LLMResponse with content, model, token counts, latency.

    Raises:
        LLMError: 복구 불가 오류 또는 최대 재시도 초과 시.
    """
    model = model or settings.default_model
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            start = time.monotonic()
            extra: dict = {}
            loc = _VERTEX_LOCATION_OVERRIDES.get(model)
            if loc:
                extra["vertex_ai_location"] = loc
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **extra,
            )
            elapsed = time.monotonic() - start

            choice = response.choices[0]
            usage = response.usage

            return LLMResponse(
                content=choice.message.content or "",
                model=response.model or model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                latency=round(elapsed, 3),
            )

        except NotFoundError as e:
            raise LLMError(f"모델 오류 ({model}): {e}", status_code=getattr(e, "status_code", 404)) from e

        except AuthenticationError as e:
            raise LLMError(f"인증 오류 ({model}): {e}", status_code=getattr(e, "status_code", 401)) from e

        except (BadRequestError, ContextWindowExceededError) as e:
            raise LLMError(f"요청 오류 ({model}): {e}", status_code=400) from e

        except RateLimitError as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning("Rate limit (attempt %d/%d), %.1fs 대기: %s", attempt + 1, _MAX_RETRIES, delay, e)
                time.sleep(delay)
            else:
                logger.error("Rate limit 최종 실패 (재시도 소진): %s", e)

        except (APIConnectionError, Timeout, ServiceUnavailableError, InternalServerError) as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning("연결/서버 오류 (attempt %d/%d), %.1fs 대기: %s", attempt + 1, _MAX_RETRIES, delay, e)
                time.sleep(delay)
            else:
                logger.error("연결/서버 오류 최종 실패 (재시도 소진): %s", e)

    msg = f"LLM 호출 실패 ({_MAX_RETRIES}회 재시도 초과): {last_error!r}"
    raise LLMError(msg, status_code=502) from last_error
