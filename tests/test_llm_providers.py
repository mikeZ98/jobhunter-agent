"""LLMClient against mocked provider HTTP endpoints — the real OpenAI SDK + instructor stack, no network.

Unlike test_client.py (which stubs ``_create``), these go through SDK error mapping and instructor's
wrapping, so they prove the classifier sees through both.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import openai
import pytest
from pydantic import SecretStr

from jobhunter.llm.client import LLMClient
from jobhunter.llm.router import NoModelAvailable, Provider, Task, decision_matrix
from jobhunter.models import MatchAssessment

CODER = "qwen2.5-coder:7b"
GOOD = {"score": 71, "strengths": ["RAG"], "gaps": [], "pitch": "x"}
Handler = Callable[[httpx2.Request], httpx2.Response]


def completion(content: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


class Providers:
    """Routes each request to a per-host script; every step is a Response or an exception to raise."""

    def __init__(self) -> None:
        self.scripts: dict[str, list[httpx2.Response | Exception]] = {"localhost": [], "openrouter.ai": []}
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        script = self.scripts[request.url.host]
        step = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(step, Exception):
            raise step
        return step

    def hosts(self) -> list[str]:
        return [r.url.host for r in self.requests]


@pytest.fixture
def providers(monkeypatch) -> Providers:
    p = Providers()
    real = openai.AsyncOpenAI

    class MockedAsyncOpenAI(real):  # a subclass: instructor.from_openai isinstance-checks the client
        def __init__(self, **kw: Any) -> None:
            # SDK-internal retries off so each scripted step is one call (the client's policy is under test).
            transport = httpx2.MockTransport(p.handler)
            super().__init__(
                **kw | {"max_retries": 0, "http_client": httpx2.AsyncClient(transport=transport)}
            )

    monkeypatch.setattr(openai, "AsyncOpenAI", MockedAsyncOpenAI)
    return p


@pytest.fixture
def client(settings, db, monkeypatch) -> LLMClient:
    settings.llm.fast_models = [CODER]
    settings.llm.local_only = False
    settings.llm.max_validation_retries = 2
    settings.openrouter_api_key = SecretStr("or-test")
    settings.anthropic_api_key = settings.openai_api_key = None
    c = LLMClient(settings, db)
    monkeypatch.setattr(c.router.probe, "refresh", lambda force=False: None)
    c.router.probe._ollama_up = True
    c.router.probe._installed = {CODER}
    monkeypatch.setattr(type(c.router.probe), "free_memory_gb", staticmethod(lambda: 32.0))

    async def alias(name: str) -> str:
        return name

    async def unload(keep: str | None) -> None:
        return None

    async def no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(c, "_ollama_model", alias)
    monkeypatch.setattr(c, "_unload_others", unload)
    monkeypatch.setattr("jobhunter.llm.client.asyncio.sleep", no_sleep)
    return c


def logged(db) -> list[tuple[str, int, str | None]]:
    return [tuple(r) for r in db._conn.execute("SELECT provider, success, error FROM llm_calls ORDER BY id")]


async def ask(client: LLMClient) -> MatchAssessment:
    return await client.structured(Task.SCORE, MatchAssessment, "system", "user")


@pytest.mark.asyncio
async def test_local_timeout_falls_back_to_openrouter(client, providers, db):
    providers.scripts["localhost"] = [httpx2.ReadTimeout("model still loading")]
    providers.scripts["openrouter.ai"] = [completion(json.dumps(GOOD))]
    assert (await ask(client)).score == 71
    assert providers.hosts() == ["localhost", "openrouter.ai"]
    assert logged(db) == [("ollama", 0, "timeout"), ("openrouter", 1, None)]


@pytest.mark.asyncio
async def test_openrouter_request_shape(client, providers):
    client.router.probe._ollama_up = False
    providers.scripts["openrouter.ai"] = [completion(json.dumps(GOOD))]
    await ask(client)
    req = providers.requests[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer or-test"
    assert req.headers["x-title"] == "jobhunter-agent"
    assert body["model"] == client.settings.llm.openrouter_fast_model
    assert body["provider"] == {"data_collection": "deny"}


@pytest.mark.asyncio
async def test_cloud_first_rate_limit_falls_back_to_local_and_cools_down(client, providers, db):
    client.settings.llm.prefer_local = False
    client.router.matrix = decision_matrix(client.settings.llm)
    providers.scripts["openrouter.ai"] = [httpx2.Response(429, headers={"retry-after": "30"}, json={})]
    providers.scripts["localhost"] = [completion(json.dumps(GOOD))]
    assert (await ask(client)).score == 71
    assert providers.hosts() == ["openrouter.ai", "localhost"]
    assert 25 < client.router.cooldown_left(Provider.OPENROUTER) <= 30
    await ask(client)
    assert providers.hosts() == ["openrouter.ai", "localhost", "localhost"]  # cooled down: not retried
    assert logged(db)[0] == ("openrouter", 0, "rate_limit")


@pytest.mark.asyncio
async def test_rejected_key_disables_openrouter(client, providers, db):
    client.router.probe._ollama_up = False
    providers.scripts["openrouter.ai"] = [httpx2.Response(401, json={"error": {"message": "bad key"}})]
    with pytest.raises(NoModelAvailable, match=r"failures: openrouter:\S+: auth"):
        await ask(client)
    with pytest.raises(NoModelAvailable, match="openrouter disabled: HTTP 401"):
        await ask(client)
    assert len(providers.requests) == 1  # the bad key is not sent again
    assert logged(db) == [("openrouter", 0, "auth")]


@pytest.mark.asyncio
async def test_invalid_output_is_reasked_then_falls_through(client, providers, db):
    bad = completion(json.dumps({**GOOD, "score": "high"}))
    providers.scripts["localhost"] = [bad]
    providers.scripts["openrouter.ai"] = [completion(json.dumps(GOOD))]
    assert (await ask(client)).score == 71
    assert providers.hosts() == ["localhost"] * 3 + ["openrouter.ai"]  # 1 try + 2 re-asks
    assert logged(db)[0] == ("ollama", 0, "validation")


@pytest.mark.asyncio
async def test_empty_ollama_reply_is_retried_once_then_falls_through(client, providers, db):
    providers.scripts["localhost"] = [completion("")]
    providers.scripts["openrouter.ai"] = [completion(json.dumps(GOOD))]
    assert (await ask(client)).score == 71
    assert providers.hosts().count("localhost") == 6  # two calls x (1 try + 2 re-asks)
    assert [e for _, _, e in logged(db)] == ["empty_reply", "empty_reply", None]


@pytest.mark.asyncio
async def test_local_connection_refused_reprobes(client, providers, monkeypatch):
    invalidated: list[bool] = []
    monkeypatch.setattr(client.router.probe, "invalidate", lambda: invalidated.append(True))
    providers.scripts["localhost"] = [httpx2.ConnectError("refused")]
    providers.scripts["openrouter.ai"] = [completion(json.dumps(GOOD))]
    await ask(client)
    assert invalidated == [True]


@pytest.mark.asyncio
async def test_local_only_never_reaches_the_cloud(client, providers):
    client.settings.llm.local_only = True
    client.router.matrix = decision_matrix(client.settings.llm)
    providers.scripts["localhost"] = [httpx2.ReadTimeout("slow")]
    providers.scripts["openrouter.ai"] = [completion(json.dumps(GOOD))]
    with pytest.raises(NoModelAvailable):
        await ask(client)
    assert "openrouter.ai" not in providers.hosts()
