"""Structured LLM calls: route -> instructor -> Pydantic-validated object, with model fallback.

``instructor`` re-asks the model with the validation error on schema/validator failures
(``max_retries``), so a Pydantic ``field_validator`` doubles as a *guardrail* (e.g. "CV may only
mention skills present in profile.json"). When retries are exhausted we fall through to the next
model in the decision matrix — usually a bigger local model, then the cloud.

Local calls go through a client-side gate (``ollama_parallel``, default 1) and keep one chat model
resident (``ollama_single_model``): on a 16 GB Mac, two or three 4-8B models at 12k context exhaust
Metal memory, and Ollama then answers HTTP 200 with an *empty* body. Such empty replies are treated as
a backend failure (unload, retry once), not as the model failing validation.
"""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from typing import Any, TypeVar

import httpx
import instructor
from instructor.core import InstructorRetryException
from pydantic import BaseModel

from jobhunter.config import Settings
from jobhunter.llm.router import ModelRouter, ModelSpec, NoModelAvailable, Provider, Task, ollama_alias
from jobhunter.telemetry import get_logger, llm_latency, parse_failures, span, tokens_counter

log = get_logger("llm")
T = TypeVar("T", bound=BaseModel)

# Per-call retry counter. instructor hooks are client-global; a ContextVar keeps concurrent calls apart.
_parse_errors: ContextVar[int] = ContextVar("parse_errors", default=0)


def _on_parse_error(provider: Provider) -> Any:
    def hook(*_a: Any, **_k: Any) -> None:
        _parse_errors.set(_parse_errors.get() + 1)
        parse_failures.add(1, {"provider": provider.value})

    return hook


class LLMClient:
    def __init__(self, settings: Settings, db: Any | None = None) -> None:
        self.settings = settings
        self.router = ModelRouter(settings)
        self.db = db
        self._clients: dict[Provider, instructor.AsyncInstructor] = {}
        self._aliases: set[str] = set()
        self._alias_lock = asyncio.Lock()
        self._local_gate = asyncio.Semaphore(max(1, settings.llm.ollama_parallel))
        self._loaded: str | None = None  # Ollama model this client last called

    # ---------- provider clients ----------

    def _client(self, provider: Provider) -> instructor.AsyncInstructor:
        if provider in self._clients:
            return self._clients[provider]
        cfg = self.settings.llm
        match provider:
            case Provider.OLLAMA | Provider.VLLM:
                from openai import AsyncOpenAI

                base = f"{cfg.ollama_base_url}/v1" if provider == Provider.OLLAMA else cfg.vllm_base_url
                # JSON mode is the most reliable structured mode for 7-24B local models; Ollama additionally
                # constrains decoding to valid JSON so we only ever retry on *semantic* validation errors.
                client = instructor.from_openai(
                    AsyncOpenAI(base_url=base, api_key=provider.value, timeout=cfg.request_timeout_s),
                    mode=instructor.Mode.JSON,
                )
            case Provider.ANTHROPIC:
                from anthropic import AsyncAnthropic

                key = self.settings.anthropic_api_key
                client = instructor.from_anthropic(
                    AsyncAnthropic(
                        api_key=key.get_secret_value() if key else None, timeout=cfg.request_timeout_s
                    ),
                    mode=instructor.Mode.ANTHROPIC_TOOLS,
                )
            case Provider.OPENAI:
                from openai import AsyncOpenAI

                key = self.settings.openai_api_key
                client = instructor.from_openai(
                    AsyncOpenAI(
                        api_key=key.get_secret_value() if key else None, timeout=cfg.request_timeout_s
                    ),
                    mode=instructor.Mode.TOOLS,
                )
        client.on("parse:error", _on_parse_error(provider))
        self._clients[provider] = client
        return client

    async def _ollama_model(self, name: str) -> str:
        """Create (once) and return '<tag>-jhNk', a manifest-only alias of ``name`` with num_ctx set."""
        num_ctx = self.settings.llm.ollama_num_ctx
        alias = ollama_alias(name, num_ctx)
        if alias in self._aliases:
            return alias
        async with self._alias_lock:
            if alias in self._aliases:
                return alias
            if alias not in self.router.probe._installed:
                try:
                    async with httpx.AsyncClient(
                        base_url=self.settings.llm.ollama_base_url, timeout=120
                    ) as c:
                        r = await c.post(
                            "/api/create",
                            json={
                                "model": alias,
                                "from": name,
                                "parameters": {"num_ctx": num_ctx},
                                "stream": False,
                            },
                        )
                        r.raise_for_status()
                    log.info("created Ollama alias {} (num_ctx={})", alias, num_ctx)
                except httpx.HTTPError as e:
                    log.warning(
                        "could not create {} ({!r}) — using {} with Ollama's default context", alias, e, name
                    )
                    return name
            self._aliases.add(alias)
        return alias

    async def _ollama(self, method: str, path: str, **kw: Any) -> httpx.Response:
        async with httpx.AsyncClient(base_url=self.settings.llm.ollama_base_url, timeout=60) as c:
            r = await c.request(method, path, **kw)
            r.raise_for_status()
            return r

    async def _unload_others(self, keep: str | None) -> None:
        """Unload every resident chat model except ``keep`` (the embedding model is tiny, it stays)."""
        try:
            ps = (await self._ollama("GET", "/api/ps")).json().get("models", [])
            emb = self.settings.llm.embedding_model
            kept: set[str] = set()
            for m in ps:
                name = m["name"]
                if name == keep or name.startswith(emb):
                    kept.add(name)
                    continue
                await self._ollama("POST", "/api/generate", json={"model": name, "keep_alive": 0})
                log.info("unloaded {} to make room for {}", name, keep)
            self.router.probe.note_resident(kept)
        except (httpx.HTTPError, ValueError, KeyError) as e:
            log.debug("could not unload Ollama models: {!r}", e)

    async def release(self) -> None:
        """Free the RAM held by local models (end of the LLM stages, before Apple Notes etc.)."""
        if self._loaded is None:
            return
        async with self._local_gate:
            await self._unload_others(None)
            self._loaded = None

    # ---------- public API ----------

    async def structured(
        self,
        task: Task,
        response_model: type[T],
        system: str,
        user: str,
        *,
        max_tokens: int = 1500,
        temperature: float | None = None,
        validation_context: dict[str, Any] | None = None,
    ) -> T:
        prompt = system + user
        tried: set[str] = set()
        last_err: Exception | None = None
        empty_retried: set[str] = set()
        while True:
            try:
                candidates = await asyncio.to_thread(self.router.candidates, task, prompt, max_tokens, tried)
            except NoModelAvailable as e:
                raise NoModelAvailable(f"{e}; last error: {last_err!r}") from last_err
            spec = candidates[0].model
            tried.add(spec.key)
            try:
                return await self._call(
                    spec, task, response_model, system, user, max_tokens, temperature, validation_context
                )
            except InstructorRetryException as e:
                last_err = e
                if spec.provider == Provider.OLLAMA and _empty_reply(e):
                    if spec.key not in empty_retried:
                        empty_retried.add(spec.key)
                        tried.discard(spec.key)
                        log.warning(
                            "{} returned empty replies for {} (Ollama out of memory?) — unloading other "
                            "models and retrying",
                            spec.key,
                            task,
                        )
                        async with self._local_gate:
                            await self._unload_others(None)
                            self._loaded = None
                        await asyncio.sleep(3)
                        continue
                    log.warning("{} keeps returning empty replies for {}", spec.key, task)
                    continue
                log.warning("{} exhausted validation retries for {}: {}", spec.key, task, str(e)[:300])
            except Exception as e:
                last_err = e
                log.warning("{} failed for {}: {!r}", spec.key, task, e)

    async def _call(
        self,
        spec: ModelSpec,
        task: Task,
        response_model: type[T],
        system: str,
        user: str,
        max_tokens: int,
        temperature: float | None,
        validation_context: dict[str, Any] | None,
    ) -> T:
        client = self._client(spec.provider)
        kwargs: dict[str, Any] = {
            "model": spec.name,
            "response_model": response_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_retries": self.settings.llm.max_validation_retries,
            "max_tokens": max_tokens,
            "temperature": spec.temperature if temperature is None else temperature,
        }
        if validation_context:
            kwargs["context"] = validation_context
        if spec.provider == Provider.OLLAMA:
            # Ollama's default context (4096) silently truncates long prompts, and its OpenAI endpoint
            # ignores options.num_ctx -> call an alias with num_ctx baked in (options kept as a hint).
            kwargs["model"] = await self._ollama_model(spec.name)
            kwargs["extra_body"] = {"options": {"num_ctx": self.settings.llm.ollama_num_ctx}}
            async with self._local_gate:
                if self.settings.llm.ollama_single_model and self._loaded != kwargs["model"]:
                    await self._unload_others(kwargs["model"])
                self._loaded = kwargs["model"]
                local: T = await self._create(client, spec, task, kwargs)
                return local
        remote: T = await self._create(client, spec, task, kwargs)
        return remote

    async def _create(
        self, client: instructor.AsyncInstructor, spec: ModelSpec, task: Task, kwargs: dict[str, Any]
    ) -> Any:
        _parse_errors.set(0)
        t0 = time.perf_counter()
        ok, retries = False, 0
        usage: tuple[int | None, int | None] = (None, None)
        with span("llm.call", task=task.value, provider=spec.provider.value, model=spec.name) as s:
            try:
                obj, completion = await client.chat.completions.create_with_completion(**kwargs)
                ok = True
                usage = _usage(completion)
                s.set_attribute("llm.prompt_tokens", usage[0] or 0)
                s.set_attribute("llm.completion_tokens", usage[1] or 0)
                return obj
            finally:
                retries = _parse_errors.get()
                ms = (time.perf_counter() - t0) * 1000
                attrs = {"task": task.value, "model": spec.name, "provider": spec.provider.value}
                llm_latency.record(ms, attrs)
                if usage[0]:
                    tokens_counter.add(usage[0], {**attrs, "kind": "prompt"})
                if usage[1]:
                    tokens_counter.add(usage[1], {**attrs, "kind": "completion"})
                log.debug("{} {} {:.0f}ms tokens={} ok={}", task, spec.key, ms, usage, ok)
                if self.db is not None:
                    self.db.log_llm_call(
                        task=task.value,
                        provider=spec.provider.value,
                        model=spec.name,
                        prompt_tokens=usage[0],
                        completion_tokens=usage[1],
                        latency_ms=ms,
                        success=ok,
                        retries=retries,
                    )


def _empty_reply(e: InstructorRetryException) -> bool:
    """Ollama answers HTTP 200 with no content when llama-server hits a compute error (e.g. Metal OOM)."""
    c = e.last_completion
    try:
        content = c.choices[0].message.content  # type: ignore[union-attr]
    except (AttributeError, IndexError, TypeError):
        return "line 1 column 0" in str(e) and "input_value=''" in str(e)
    return not (content or "").strip()


def _usage(completion: Any) -> tuple[int | None, int | None]:
    u = getattr(completion, "usage", None)
    if u is None:
        return None, None
    # OpenAI-compatible (Ollama/vLLM/OpenAI) vs Anthropic naming
    return (
        getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", None),
        getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", None),
    )
