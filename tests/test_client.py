"""LLMClient behaviour against a misbehaving local Ollama (no network: the HTTP edges are patched)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from instructor.core import InstructorRetryException

from jobhunter.llm.client import LLMClient, _empty_reply
from jobhunter.llm.router import Task, decision_matrix, local_model
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
