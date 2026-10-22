"""Classify a failed LLM call so the client reacts to *why* it failed, not just *that* it did.

instructor wraps every provider error (429, timeout, refused connection) in ``InstructorRetryException``
``from`` the original, so the cause chain is walked before concluding "validation failed".

| kind          | typical cause                                       | client reaction                              |
|---------------|-----------------------------------------------------|----------------------------------------------|
| timeout       | model load + generation > ``request_timeout_s``     | next candidate (too slow for this prompt)    |
| rate_limit    | HTTP 429 — provider quota                           | provider cooldown (Retry-After), next model  |
| auth          | HTTP 401/402/403 — missing/revoked key, no credits  | provider disabled for the process            |
| connection    | refused / DNS / reset                               | Ollama re-probed; cloud provider cools down  |
| server        | HTTP 5xx, Anthropic 529 overloaded                  | next candidate                               |
| bad_request   | other 4xx — context too long, unknown model         | next candidate                               |
| empty_reply   | Ollama HTTP 200 with no content (Metal OOM)         | unload + retry once, then next candidate     |
| validation    | instructor exhausted its re-asks                    | next candidate                               |
| unknown       | anything else                                       | next candidate                               |
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from enum import StrEnum

import anthropic
import httpx
import openai
from instructor.core import InstructorRetryException


class ErrorKind(StrEnum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    CONNECTION = "connection"
    SERVER = "server"
    BAD_REQUEST = "bad_request"
    EMPTY_REPLY = "empty_reply"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class LLMFailure:
    kind: ErrorKind
    status: int | None = None
    retry_after: float | None = (
        None  # seconds, from Retry-After / X-RateLimit-Reset when the provider sent one
    )


_TIMEOUTS = (TimeoutError, httpx.TimeoutException, openai.APITimeoutError, anthropic.APITimeoutError)
_STATUS = (openai.APIStatusError, anthropic.APIStatusError, httpx.HTTPStatusError)
_CONNECTION = (openai.APIConnectionError, anthropic.APIConnectionError, httpx.TransportError, ConnectionError)


def classify(exc: BaseException) -> LLMFailure:
    for e in _chain(exc):
        if (failure := _classify_one(e)) is not None:
            return failure
    if isinstance(exc, InstructorRetryException):
        return LLMFailure(ErrorKind.EMPTY_REPLY if empty_reply(exc) else ErrorKind.VALIDATION)
    return LLMFailure(ErrorKind.UNKNOWN)


def empty_reply(e: InstructorRetryException) -> bool:
    """Ollama answers HTTP 200 with no content when llama-server hits a compute error (e.g. Metal OOM)."""
    c = e.last_completion
    try:
        content = c.choices[0].message.content  # type: ignore[union-attr]
    except (AttributeError, IndexError, TypeError):
        return "line 1 column 0" in str(e) and "input_value=''" in str(e)
    return not (content or "").strip()


def _chain(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    e: BaseException | None = exc
    while e is not None and e not in out and len(out) < 10:
        out.append(e)
        e = e.__cause__ or e.__context__
    return out


def _classify_one(e: BaseException) -> LLMFailure | None:
    # Order matters: the SDK timeout errors subclass their connection errors.
    if isinstance(e, _TIMEOUTS):
        return LLMFailure(ErrorKind.TIMEOUT)
    if isinstance(e, _STATUS):
        return _from_status(e.response.status_code, e.response.headers)
    if isinstance(e, _CONNECTION):
        return LLMFailure(ErrorKind.CONNECTION)
    return None


def _from_status(status: int, headers: Mapping[str, str]) -> LLMFailure:
    if status == 429:
        return LLMFailure(ErrorKind.RATE_LIMIT, status, retry_after(headers))
    if status in (401, 402, 403):  # 402: OpenRouter out of credits
        return LLMFailure(ErrorKind.AUTH, status)
    if status == 408:
        return LLMFailure(ErrorKind.TIMEOUT, status)
    if status >= 500:
        return LLMFailure(ErrorKind.SERVER, status, retry_after(headers))
    return LLMFailure(ErrorKind.BAD_REQUEST, status)


def retry_after(headers: Mapping[str, str], now: float | None = None) -> float | None:
    """Seconds to wait, from ``retry-after-ms`` (OpenAI), ``retry-after`` (seconds or HTTP date) or
    ``x-ratelimit-reset`` (OpenRouter: epoch milliseconds). None when the provider didn't say."""
    now = time.time() if now is None else now
    try:
        if v := headers.get("retry-after-ms"):
            return max(0.0, float(v) / 1000)
        if v := headers.get("retry-after"):
            try:
                return max(0.0, float(v))
            except ValueError:
                return max(0.0, parsedate_to_datetime(v).timestamp() - now)
        if v := headers.get("x-ratelimit-reset"):
            reset = float(v)
            if reset < 1e9:  # some gateways send "seconds until reset", not an epoch
                return max(0.0, reset)
            return max(0.0, (reset / 1000 if reset > 1e11 else reset) - now)
    except (ValueError, TypeError, OverflowError):
        return None
    return None
