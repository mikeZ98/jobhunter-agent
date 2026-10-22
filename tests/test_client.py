"""LLMClient behaviour against a misbehaving local Ollama (no network: the HTTP edges are patched)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from instructor.core import InstructorRetryException
from pydantic import SecretStr

from jobhunter.llm.client import LLMClient
from jobhunter.llm.errors import empty_reply as _empty_reply
from jobhunter.llm.router import NoModelAvailable, Provider, Task, decision_matrix, local_model
from jobhunter.models import MatchAssessment

CODER, LLAMA = "qwen2.5-coder:7b", "llama3.1:8b"


def _completion(content: str | None) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _retry_exc(content: str | None) -> InstructorRetryException:
    return InstructorRetryException(
        "boom", last_completion=_completion(content), n_attempts=3, total_usage=None
    )


@pytest.fixture
def client(settings, monkeypatch):
    settings.llm.local_only = True
    settings.llm.creative_models = [LLAMA, CODER]
    settings.llm.fast_models = [CODER]
    c = LLMClient(settings)
    c.router.matrix = decision_matrix(settings.llm)

    def candidates(task, prompt, max_tokens, exclude=None):
        from jobhunter.llm.router import NoModelAvailable, RouteDecision

        out = [RouteDecision(m, []) for m in c.router.matrix[task] if m.key not in (exclude or set())]
        if not out:
            raise NoModelAvailable(f"no model for task={task}")
        return out

    monkeypatch.setattr(c.router, "candidates", candidates)
    c.unloads: list[str | None] = []  # type: ignore[misc]

    async def alias(name):
        return name

    async def unload(keep):
        c.unloads.append(keep)  # type: ignore[attr-defined]

    monkeypatch.setattr(c, "_ollama_model", alias)
    monkeypatch.setattr(c, "_unload_others", unload)
    return c


def test_empty_reply_detection():
    assert _empty_reply(_retry_exc(""))
    assert _empty_reply(_retry_exc("  "))
    assert not _empty_reply(_retry_exc('{"score": 3}'))


@pytest.mark.asyncio
async def test_switching_models_unloads_the_previous_one(client, monkeypatch):
    seen: list[str] = []

    async def create(_client, spec, task, kwargs):
        seen.append(kwargs["model"])
        return MatchAssessment(score=50, strengths=[], gaps=[], pitch="x")

    monkeypatch.setattr(client, "_create", create)
    await client.structured(Task.SCORE, MatchAssessment, "s", "u")
    await client.structured(Task.SCORE, MatchAssessment, "s", "u")  # same model: no unload
    await client.structured(Task.CREATIVE, MatchAssessment, "s", "u")
    assert seen == [CODER, CODER, LLAMA]
    assert client.unloads == [CODER, LLAMA]


@pytest.mark.asyncio
async def test_empty_replies_unload_and_retry_same_model_once_then_fall_through(client, monkeypatch):
    calls: list[str] = []
    replies = {LLAMA: ["", ""], CODER: []}

    async def create(_client, spec, task, kwargs):
        calls.append(spec.name)
        if replies[spec.name]:
            raise _retry_exc(replies[spec.name].pop(0))
        return MatchAssessment(score=70, strengths=[], gaps=[], pitch="x")

    monkeypatch.setattr(client, "_create", create)
    monkeypatch.setattr("jobhunter.llm.client.asyncio.sleep", _no_sleep)
    out = await client.structured(Task.CREATIVE, MatchAssessment, "s", "u")
    assert out.score == 70
    assert calls == [LLAMA, LLAMA, CODER]  # retried once after unloading everything, then next model
    assert None in client.unloads


@pytest.mark.asyncio
async def test_empty_reply_recovers_on_retry(client, monkeypatch):
    replies = [""]

    async def create(_client, spec, task, kwargs):
        if replies:
            raise _retry_exc(replies.pop())
        return MatchAssessment(score=61, strengths=[], gaps=[], pitch="x")

    monkeypatch.setattr(client, "_create", create)
    monkeypatch.setattr("jobhunter.llm.client.asyncio.sleep", _no_sleep)
    assert (await client.structured(Task.CREATIVE, MatchAssessment, "s", "u")).score == 61


async def _no_sleep(_s: float) -> None:
    return None


def test_judge_score_is_the_sum_of_rubric_parts():
    m = MatchAssessment(
        strengths=[], gaps=[], stack=30, role=20, seniority=8, logistics=10, score=88, pitch="x"
    )
    assert m.score == 68
    assert MatchAssessment(score=77, strengths=[], gaps=[], pitch="x").score == 77  # stored pre-rubric rows
    assert local_model(LLAMA, 0.1).name == LLAMA


# ------------------------------------------------------------------ classified failures -> routing


def _http_error(cls: str, code: int, headers: dict[str, str] | None = None) -> Exception:
    req = httpx.Request("POST", "https://llm.test/v1/chat/completions")
    return getattr(openai, cls)(
        "boom", response=httpx.Response(code, headers=headers or {}, request=req), body=None
    )


def _ok() -> MatchAssessment:
    return MatchAssessment(score=66, strengths=[], gaps=[], pitch="x")


@pytest.fixture
def routed(settings, monkeypatch):
    """Real router (stubbed probe): local CODER, then OpenRouter, then Anthropic."""
    settings.llm.fast_models = [CODER]
    settings.llm.local_only = False
    settings.openrouter_api_key = SecretStr("or-test")
    settings.anthropic_api_key = SecretStr("sk-test")
    settings.openai_api_key = None
    c = LLMClient(settings)
    monkeypatch.setattr(c.router.probe, "refresh", lambda force=False: None)
    c.router.probe._ollama_up = True
    c.router.probe._installed = {CODER}
    monkeypatch.setattr(type(c.router.probe), "free_memory_gb", staticmethod(lambda: 32.0))

    async def alias(name):
        return name

    async def unload(keep):
        return None

    monkeypatch.setattr(c, "_ollama_model", alias)
    monkeypatch.setattr(c, "_unload_others", unload)
    return c


def _script(client, monkeypatch, outcomes: dict[str, list[Exception]]) -> list[str]:
    """Each provider raises its queued errors in order, then succeeds. Returns the provider call log."""
    calls: list[str] = []

    async def create(_client, spec, task, kwargs):
        calls.append(spec.provider.value)
        if outcomes.get(spec.provider.value):
            raise outcomes[spec.provider.value].pop(0)
        return _ok()

    monkeypatch.setattr(client, "_create", create)
    return calls


@pytest.mark.asyncio
async def test_local_timeout_falls_through_to_openrouter_without_cooldown(routed, monkeypatch):
    calls = _script(
        routed, monkeypatch, {"ollama": [openai.APITimeoutError(request=httpx.Request("POST", "http://x"))]}
    )
    assert (await routed.structured(Task.SCORE, MatchAssessment, "s", "u")).score == 66
    assert calls == ["ollama", "openrouter"]
    assert routed.router.shortest_cooldown() is None  # a slow prompt doesn't bench the provider
    calls.clear()
    await routed.structured(Task.SCORE, MatchAssessment, "s", "u")
    assert calls == ["ollama"]


@pytest.mark.asyncio
async def test_rate_limit_cools_the_provider_down_for_the_whole_batch(routed, monkeypatch):
    routed.router.probe._ollama_up = False
    calls = _script(
        routed, monkeypatch, {"openrouter": [_http_error("RateLimitError", 429, {"retry-after": "45"})]}
    )
    await routed.structured(Task.SCORE, MatchAssessment, "s", "u")
    assert calls == ["openrouter", "anthropic"]
    assert 40 < routed.router.cooldown_left(Provider.OPENROUTER) <= 45
    calls.clear()
    await routed.structured(Task.SCORE, MatchAssessment, "s", "u")
    assert calls == ["anthropic"]  # the next call doesn't hit the 429 again


@pytest.mark.asyncio
async def test_rejected_key_disables_the_provider_for_the_run(routed, monkeypatch):
    routed.router.probe._ollama_up = False
    calls = _script(routed, monkeypatch, {"openrouter": [_http_error("AuthenticationError", 401)]})
    await routed.structured(Task.SCORE, MatchAssessment, "s", "u")
    await routed.structured(Task.SCORE, MatchAssessment, "s", "u")
    assert calls == ["openrouter", "anthropic", "anthropic"]


@pytest.mark.asyncio
async def test_only_rate_limited_left_waits_for_the_cooldown_once(routed, monkeypatch):
    routed.router.probe._ollama_up = False
    routed.settings.anthropic_api_key = None
    calls = _script(
        routed, monkeypatch, {"openrouter": [_http_error("RateLimitError", 429, {"retry-after": "3"})]}
    )
    slept: list[float] = []

    async def sleep(s: float) -> None:  # time passes: the cooldown expires
        slept.append(s)
        routed.router._cooldowns.clear()

    monkeypatch.setattr("jobhunter.llm.client.asyncio.sleep", sleep)
    assert (await routed.structured(Task.SCORE, MatchAssessment, "s", "u")).score == 66
    assert calls == ["openrouter", "openrouter"]
    assert len(slept) == 1 and 0 < slept[0] <= 3


@pytest.mark.asyncio
async def test_long_rate_limit_fails_fast_with_classified_reasons(routed, monkeypatch):
    routed.router.probe._ollama_up = False
    routed.settings.anthropic_api_key = None
    _script(routed, monkeypatch, {"openrouter": [_http_error("RateLimitError", 429, {"retry-after": "600"})]})
    monkeypatch.setattr("jobhunter.llm.client.asyncio.sleep", _no_sleep)
    with pytest.raises(NoModelAvailable, match=r"failures: openrouter:\S+: rate_limit"):
        await routed.structured(Task.SCORE, MatchAssessment, "s", "u")


@pytest.mark.asyncio
async def test_local_connection_error_forces_a_reprobe(routed, monkeypatch):
    _script(
        routed,
        monkeypatch,
        {"ollama": [openai.APIConnectionError(request=httpx.Request("POST", "http://x"))]},
    )
    invalidated: list[bool] = []
    monkeypatch.setattr(routed.router.probe, "invalidate", lambda: invalidated.append(True))
    await routed.structured(Task.SCORE, MatchAssessment, "s", "u")
    assert invalidated == [True]


@pytest.mark.asyncio
async def test_openrouter_requests_deny_data_collection(routed, monkeypatch):
    routed.router.probe._ollama_up = False
    seen: list[dict[str, Any]] = []

    async def create(_client, spec, task, kwargs):
        seen.append(kwargs)
        return _ok()

    monkeypatch.setattr(routed, "_create", create)
    await routed.structured(Task.SCORE, MatchAssessment, "s", "u")
    assert seen[0]["extra_body"] == {"provider": {"data_collection": "deny"}}
    routed.settings.llm.openrouter_deny_data_collection = False
    await routed.structured(Task.SCORE, MatchAssessment, "s", "u")
    assert "extra_body" not in seen[1]


@pytest.mark.asyncio
async def test_failed_call_is_logged_with_its_error_kind(settings):
    logged: list[dict[str, Any]] = []
    c = LLMClient(settings, db=SimpleNamespace(log_llm_call=lambda **kw: logged.append(kw)))

    async def boom(**_kw: Any) -> Any:
        raise _http_error("RateLimitError", 429)

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create_with_completion=boom)))
    spec = decision_matrix(settings.llm)[Task.SCORE][0]
    with pytest.raises(openai.RateLimitError):
        await c._create(fake, spec, Task.SCORE, {})  # type: ignore[arg-type]
    assert logged[0]["success"] is False and logged[0]["error"] == "rate_limit"
