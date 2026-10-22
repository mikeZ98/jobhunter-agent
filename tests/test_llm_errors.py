"""Failure classification: the client must tell a timeout from a 429 even when instructor wraps both."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest
from instructor.core import InstructorRetryException

from jobhunter.llm.errors import ErrorKind, classify, retry_after

REQ = httpx.Request("POST", "https://llm.test/v1/chat/completions")


def _status(sdk, cls: str, code: int, headers: dict[str, str] | None = None) -> Exception:
    resp = httpx.Response(code, headers=headers or {}, request=REQ)
    return getattr(sdk, cls)("boom", response=resp, body=None)


def _wrapped(cause: Exception) -> InstructorRetryException:
    """What instructor actually raises for a provider error: InstructorRetryException ``from`` it."""
    try:
        try:
            raise cause
        except Exception as e:
            raise InstructorRetryException(str(e), n_attempts=1, total_usage=None) from e
    except InstructorRetryException as wrapped:
        return wrapped


def _validation(content: str | None) -> InstructorRetryException:
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return InstructorRetryException("bad", last_completion=completion, n_attempts=3, total_usage=None)


@pytest.mark.parametrize("sdk", [openai, anthropic])
def test_timeout_is_not_a_connection_error(sdk):
    # APITimeoutError subclasses APIConnectionError in both SDKs; the order of checks matters.
    assert classify(sdk.APITimeoutError(request=REQ)).kind is ErrorKind.TIMEOUT
    assert classify(sdk.APIConnectionError(request=REQ)).kind is ErrorKind.CONNECTION


@pytest.mark.parametrize("sdk", [openai, anthropic])
def test_rate_limit_carries_retry_after(sdk):
    f = classify(_status(sdk, "RateLimitError", 429, {"retry-after": "7"}))
    assert (f.kind, f.status, f.retry_after) == (ErrorKind.RATE_LIMIT, 429, 7.0)


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        (401, ErrorKind.AUTH),
        (402, ErrorKind.AUTH),  # OpenRouter: out of credits
        (403, ErrorKind.AUTH),
        (408, ErrorKind.TIMEOUT),
        (400, ErrorKind.BAD_REQUEST),
        (404, ErrorKind.BAD_REQUEST),
        (500, ErrorKind.SERVER),
        (529, ErrorKind.SERVER),  # Anthropic overloaded
    ],
)
def test_status_codes(code, kind):
    assert classify(_status(openai, "APIStatusError", code)).kind is kind


def test_provider_errors_are_found_behind_instructor():
    assert classify(_wrapped(_status(openai, "RateLimitError", 429))).kind is ErrorKind.RATE_LIMIT
    assert classify(_wrapped(openai.APITimeoutError(request=REQ))).kind is ErrorKind.TIMEOUT
    assert classify(_wrapped(httpx.ConnectError("refused"))).kind is ErrorKind.CONNECTION


def test_instructor_without_provider_cause_is_validation_or_empty_reply():
    assert classify(_validation('{"score": "high"}')).kind is ErrorKind.VALIDATION
    assert classify(_validation("")).kind is ErrorKind.EMPTY_REPLY
    assert classify(_validation("   ")).kind is ErrorKind.EMPTY_REPLY


def test_plain_errors():
    assert classify(TimeoutError()).kind is ErrorKind.TIMEOUT
    assert classify(ConnectionRefusedError()).kind is ErrorKind.CONNECTION
    assert classify(ValueError("x")).kind is ErrorKind.UNKNOWN


def test_cyclic_cause_chain_terminates():
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a
    assert classify(a).kind is ErrorKind.UNKNOWN


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after": "12"}, 12.0),
        ({"retry-after": "Thu, 22 Oct 2026 10:00:30 GMT"}, 30.0),
        ({"x-ratelimit-reset": "1792663230000"}, 30.0),  # OpenRouter: epoch ms
        ({"x-ratelimit-reset": "1792663230"}, 30.0),  # epoch s
        ({"x-ratelimit-reset": "45"}, 45.0),  # relative s
        ({"retry-after": "-5"}, 0.0),
        ({"retry-after": "soon"}, None),
        ({}, None),
    ],
)
def test_retry_after(headers, expected):
    now = 1792663200.0  # 2026-10-22 10:00:00 UTC
    got = retry_after(httpx.Headers(headers), now=now)
    assert got == pytest.approx(expected) if expected is not None else got is None
