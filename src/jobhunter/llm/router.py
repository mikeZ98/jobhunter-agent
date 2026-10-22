"""Model routing decision matrix.

Each *task* maps to an ordered list of candidate models. The router walks the list and picks the
first candidate that passes every gate, logging *why* each earlier one was skipped:

  gate 1  provider reachable / API key present / not cooling down after a 429 / not disabled (bad key)
  gate 2  model actually installed (Ollama ``/api/tags``; vLLM ``/v1/models``)
  gate 3  context window >= prompt tokens + output budget (+10 % safety)
  gate 4  memory: already resident (``/api/ps``) OR fits in free (V)RAM minus headroom (counting the
          resident chat model the client unloads first, ``ollama_single_model``). Soft: models that
          fail only this gate are still tried last, smallest first.

Local OSS models come first (cost ~0, nothing leaves the machine); cloud models are the opt-in safety
net — each cloud provider is only routed to once its API key is set. If the chosen model fails at call
time, ``LLMClient`` classifies the failure (``llm.errors``) and moves on to the next candidate; a 429
also puts the whole provider in cooldown here, so the rest of the batch doesn't hit it again. With
``prefer_local = false`` the same mechanism makes local Ollama the fallback for a rate-limited cloud.

Local tags come from ``[llm] fast_models / chat_models / creative_models / interview_models`` in
jobhunter.toml (defaults: qwen2.5-coder:7b + llama3.1:8b). ``local_only = true`` drops the cloud tail.

| Task                        | Local (config tier) | Cloud fallback, in order (unless local_only)               |
|-----------------------------|---------------------|------------------------------------------------------------|
| extract / score / validate  | fast_models         | OpenRouter llama-3.1-8b / claude-haiku-5-5 / gpt-4o-mini    |
| form_answer / messaging     | chat_models         | OpenRouter llama-3.1-8b / claude-haiku-5-5 / gpt-4o-mini    |
| creative (CV, cover letter) | creative_models     | OpenRouter llama-3.3-70b / claude-sonnet-5-5 / gpt-4o       |
| interview intel             | interview_models    | OpenRouter llama-3.3-70b / claude-sonnet-5-5 / gpt-4o       |
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum

import httpx
import psutil

from jobhunter.config import LLMConfig, Settings
from jobhunter.telemetry import get_logger

log = get_logger("llm.router")


class Task(StrEnum):
    EXTRACT = "extract"
    SCORE = "score"
    VALIDATE = "validate"
    FORM_ANSWER = "form_answer"
    MESSAGING = "messaging"
    CREATIVE = "creative"
    INTERVIEW = "interview"


class Provider(StrEnum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    provider: Provider
    name: str
    context_window: int
    mem_gb: float = 0.0  # resident size at default quant (Q4_K_M) incl. ~16k KV cache
    temperature: float = 0.2

    @property
    def local(self) -> bool:
        return self.provider in (Provider.OLLAMA, Provider.VLLM)

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.name}"


# ---- Local open-source catalogue (Ollama tags) ----
# (context window, resident GB at Q4_K_M incl. ~16k KV cache). Tags not listed here get both from
# Ollama's /api/tags metadata at routing time.
KNOWN_LOCAL: dict[str, tuple[int, float]] = {
    "qwen2.5-coder:7b": (32_768, 6.0),
    "llama3.1:8b": (131_072, 6.5),
    "gemma3:4b": (131_072, 4.5),
    "qwen2.5:7b": (32_768, 6.0),
    "qwen2.5:14b": (32_768, 10.5),
    "mistral-small:24b": (32_768, 16.0),
}


def ollama_alias(name: str, num_ctx: int) -> str:
    """'gemma3:4b' -> 'gemma3:4b-jh12k': same weights, num_ctx baked in (see LLMConfig.ollama_num_ctx)."""
    base = name if ":" in name else f"{name}:latest"
    return f"{base}-jh{num_ctx // 1024}k"


def local_model(tag: str, temperature: float) -> ModelSpec:
    tag = tag.removesuffix(":latest")
    ctx, mem = KNOWN_LOCAL.get(tag, (0, 0.0))
    return ModelSpec(Provider.OLLAMA, tag, ctx, mem, temperature)


def _cloud(cfg: LLMConfig) -> dict[str, ModelSpec]:
    ctx = cfg.openrouter_context_window
    return {
        "openrouter_fast": ModelSpec(Provider.OPENROUTER, cfg.openrouter_fast_model, ctx, temperature=0.1),
        "openrouter_creative": ModelSpec(
            Provider.OPENROUTER, cfg.openrouter_creative_model, ctx, temperature=0.6
        ),
        "anthropic_fast": ModelSpec(Provider.ANTHROPIC, cfg.anthropic_fast_model, 200_000, temperature=0.1),
        "anthropic_creative": ModelSpec(
            Provider.ANTHROPIC, cfg.anthropic_creative_model, 200_000, temperature=0.6
        ),
        "openai_fast": ModelSpec(Provider.OPENAI, cfg.openai_fast_model, 128_000, temperature=0.1),
        "openai_creative": ModelSpec(Provider.OPENAI, cfg.openai_creative_model, 128_000, temperature=0.6),
    }


def decision_matrix(cfg: LLMConfig) -> dict[Task, list[ModelSpec]]:
    c = _cloud(cfg)
    fast = [local_model(t, 0.1) for t in cfg.fast_models]
    chat = [local_model(t, 0.3) for t in cfg.chat_models]
    creative = [local_model(t, 0.6) for t in cfg.creative_models]
    interview = [local_model(t, 0.5) for t in cfg.interview_models]
    fast_cloud = [c["openrouter_fast"], c["anthropic_fast"], c["openai_fast"]]
    creative_cloud = [c["openrouter_creative"], c["anthropic_creative"], c["openai_creative"]]
    matrix = {
        Task.EXTRACT: fast + fast_cloud,
        Task.SCORE: fast + fast_cloud,
        Task.VALIDATE: fast + fast_cloud,
        Task.FORM_ANSWER: [*chat, *fast_cloud],
        Task.MESSAGING: [*chat, *fast_cloud],
        Task.CREATIVE: creative + creative_cloud,
        Task.INTERVIEW: interview + creative_cloud,
    }
    if cfg.vllm_base_url:  # a vLLM box (usually a real GPU) outranks the laptop for creative work
        vllm_creative = ModelSpec(Provider.VLLM, "Qwen/Qwen2.5-14B-Instruct", 32_768, 0.0, 0.6)
        for t in (Task.CREATIVE, Task.INTERVIEW):
            matrix[t].insert(0, vllm_creative)
    if cfg.local_only:
        matrix = {t: [m for m in ms if m.local] for t, ms in matrix.items()}
    elif not cfg.prefer_local:
        matrix = {t: sorted(ms, key=lambda m: m.local) for t, ms in matrix.items()}
    return matrix


def estimate_tokens(text: str) -> int:
    # ~3.5 chars/token is a decent average for mixed PL/EN technical text across Qwen/Llama tokenizers.
    return int(len(text) / 3.5) + 16


@dataclass
class ResourceProbe:
    """Caches provider state for ``ttl`` seconds so routing a batch of 40 calls costs one probe."""

    cfg: LLMConfig
    ttl: float = 60.0
    down_ttl: float = 5.0
    _at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _start_attempted: bool = False
    _installed: set[str] = field(default_factory=set)
    _resident: set[str] = field(default_factory=set)
    _resident_gb: dict[str, float] = field(default_factory=dict)  # /api/ps name -> resident size
    _meta: dict[str, tuple[int, float]] = field(default_factory=dict)  # tag -> (context, size GB)
    _ollama_up: bool = False
    _vllm_models: set[str] = field(default_factory=set)

    def refresh(self, force: bool = False) -> None:
        # Called from worker threads (asyncio.to_thread) by concurrent LLM calls: the lock makes them wait
        # for one probe instead of reading half-initialised state (that race reported "ollama not
        # reachable" for every call while Ollama answered in 20 ms). "Down" is re-checked after 5 s.
        with self._lock:
            ttl = self.ttl if self._ollama_up else self.down_ttl
            if not force and self._at and time.monotonic() - self._at < ttl:
                return
            if not self._probe_ollama() and self._maybe_start_ollama():
                self._probe_ollama()
            if self.cfg.vllm_base_url:
                try:
                    r = httpx.get(f"{self.cfg.vllm_base_url.rstrip('/')}/models", timeout=2.0)
                    self._vllm_models = {m["id"] for m in r.json().get("data", [])}
                except (httpx.HTTPError, ValueError):
                    self._vllm_models = set()
            self._at = time.monotonic()

    def invalidate(self) -> None:
        """A call just failed to connect: re-probe on the next routing decision instead of trusting the cache."""
        with self._lock:
            self._at = 0.0

    def _probe_ollama(self) -> bool:
        try:
            with httpx.Client(base_url=self.cfg.ollama_base_url, timeout=5.0) as c:
                tags = c.get("/api/tags").json().get("models", [])
                resident = c.get("/api/ps").json().get("models", [])
        except (httpx.HTTPError, ValueError) as e:
            log.debug("ollama probe failed: {!r}", e)
            self._ollama_up, self._installed, self._resident = False, set(), set()
            self._resident_gb = {}
            return False
        self._installed = {m["name"] for m in tags}
        self._meta = {
            m["name"].removesuffix(":latest"): (
                int((m.get("details") or {}).get("context_length") or 0),
                int(m.get("size") or 0) / 1024**3,
            )
            for m in tags
        }
        self._resident = {m["name"] for m in resident}
        self._resident_gb = {m["name"]: int(m.get("size") or 0) / 1024**3 for m in resident}
        self._ollama_up = True
        return True

    def _maybe_start_ollama(self) -> bool:
        """Once per process: launch Ollama.app in the background (macOS, local URL) and wait for it."""
        local = any(h in self.cfg.ollama_base_url for h in ("localhost", "127.0.0.1"))
        if self._start_attempted or not (self.cfg.ollama_autostart and local and sys.platform == "darwin"):
            return False
        self._start_attempted = True
        log.warning("Ollama not running — starting Ollama.app…")
        try:
            subprocess.run(["open", "-ga", "Ollama"], check=True, capture_output=True, timeout=10)
        except (subprocess.SubprocessError, OSError) as e:
            log.error("could not start Ollama.app: {!r}", e)
            return False
        for _ in range(30):
            time.sleep(1)
            try:
                httpx.get(f"{self.cfg.ollama_base_url}/api/version", timeout=1.0).raise_for_status()
                log.info("Ollama is up")
                return True
            except httpx.HTTPError:
                continue
        log.error("Ollama.app didn't come up within 30 s")
        return False

    @staticmethod
    def free_memory_gb() -> float:
        """Free VRAM on NVIDIA, else available unified/system RAM (Apple Silicon shares it with Metal)."""
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=True,
                ).stdout
                return max(int(x) for x in out.split()) / 1024
            except (subprocess.SubprocessError, ValueError):
                pass
        return float(psutil.virtual_memory().available) / 1024**3

    def installed(self, m: ModelSpec) -> bool:
        if m.provider == Provider.VLLM:
            return m.name in self._vllm_models
        # Ollama normalizes "name" -> "name:latest"
        return m.name in self._installed or f"{m.name}:latest" in self._installed

    def resident(self, m: ModelSpec) -> bool:
        names = {m.name, f"{m.name}:latest", ollama_alias(m.name, self.cfg.ollama_num_ctx)}
        return bool(names & self._resident)

    def note_resident(self, names: set[str]) -> None:
        """The client just loaded/unloaded models: keep routing in sync until the next probe."""
        with self._lock:
            self._resident = set(names)
            self._resident_gb = {n: g for n, g in self._resident_gb.items() if n in names}

    def reclaimable_gb(self) -> float:
        """Memory freed when the client unloads resident chat models before loading another one."""
        if not self.cfg.ollama_single_model:
            return 0.0
        emb = self.cfg.embedding_model
        return sum(g for n, g in self._resident_gb.items() if not n.startswith(emb))

    def resolve(self, m: ModelSpec) -> ModelSpec:
        """Fill context window / memory for tags missing from KNOWN_LOCAL from Ollama's own metadata."""
        if not m.local:
            return m
        ctx, size_gb = self._meta.get(m.name, (0, 0.0))
        window = m.context_window or ctx or 8_192
        if m.provider == Provider.OLLAMA:  # what the request actually gets, not the model's maximum
            window = min(window, self.cfg.ollama_num_ctx)
        return replace(
            m,
            context_window=window,
            mem_gb=m.mem_gb or (size_gb + 1.5 if size_gb else 0.0),  # weights + ~16k KV cache
        )


@dataclass
class RouteDecision:
    model: ModelSpec
    skipped: list[tuple[str, str]]  # (model key, reason)


class NoModelAvailable(RuntimeError):
    pass


class ModelRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cfg = settings.llm
        self.matrix = decision_matrix(self.cfg)
        self.probe = ResourceProbe(self.cfg)
        self._cooldowns: dict[Provider, tuple[float, str]] = {}  # provider -> (monotonic deadline, why)
        self._disabled: dict[Provider, str] = {}

    # ---------- provider health (fed by LLMClient from classified call failures) ----------

    def cooldown(self, provider: Provider, seconds: float, reason: str) -> None:
        """Skip ``provider`` for ``seconds`` (a later, longer cooldown wins over a shorter one)."""
        until = time.monotonic() + max(0.0, seconds)
        if until > self._cooldowns.get(provider, (0.0, ""))[0]:
            self._cooldowns[provider] = (until, reason)

    def disable(self, provider: Provider, reason: str) -> None:
        """Skip ``provider`` for the rest of the process (e.g. a rejected API key)."""
        self._disabled[provider] = reason

    def cooldown_left(self, provider: Provider) -> float:
        until, _ = self._cooldowns.get(provider, (0.0, ""))
        return max(0.0, until - time.monotonic())

    def shortest_cooldown(self) -> float | None:
        """Seconds until the first cooling-down provider is usable again; None if none is cooling down."""
        left = [x for p in self._cooldowns if (x := self.cooldown_left(p)) > 0]
        return min(left) if left else None

    def _provider_ready(self, m: ModelSpec) -> str | None:
        if reason := self._disabled.get(m.provider):
            return f"{m.provider} disabled: {reason}"
        if (left := self.cooldown_left(m.provider)) > 0:
            return f"{m.provider} cooling down {left:.0f}s ({self._cooldowns[m.provider][1]})"
        match m.provider:
            case Provider.OLLAMA:
                return None if self.probe._ollama_up else "ollama not reachable"
            case Provider.VLLM:
                return None if self.probe._vllm_models else "vllm not reachable"
            case Provider.ANTHROPIC:
                return None if self.settings.anthropic_api_key else "ANTHROPIC_API_KEY not set"
            case Provider.OPENAI:
                return None if self.settings.openai_api_key else "OPENAI_API_KEY not set"
            case Provider.OPENROUTER:
                return None if self.settings.openrouter_api_key else "OPENROUTER_API_KEY not set"
        return "unknown provider"

    def candidates(
        self, task: Task, prompt: str, max_output_tokens: int, exclude: set[str] | None = None
    ) -> list[RouteDecision]:
        """All viable models for this call, best first (the client walks this on failures)."""
        self.probe.refresh()
        need = int((estimate_tokens(prompt) + max_output_tokens) * 1.1)
        free = 0.0
        if any(m.local for m in self.matrix[task]):
            free = self.probe.free_memory_gb() + self.probe.reclaimable_gb()
        skipped: list[tuple[str, str]] = []
        viable: list[RouteDecision] = []
        tight: list[ModelSpec] = []  # local models that only failed the memory gate
        for m in map(self.probe.resolve, self.matrix[task]):
            if exclude and m.key in exclude:
                skipped.append((m.key, "failed earlier in this call"))
                continue
            if reason := self._provider_ready(m):
                skipped.append((m.key, reason))
                continue
            if m.local and not self.probe.installed(m):
                skipped.append((m.key, f"not installed (ollama pull {m.name})"))
                continue
            if need > m.context_window:
                skipped.append((m.key, f"context {need} > {m.context_window}"))
                continue
            if (
                m.provider == Provider.OLLAMA
                and not self.probe.resident(m)
                and m.mem_gb > free - self.cfg.memory_headroom_gb
            ):
                skipped.append((m.key, f"needs {m.mem_gb:.1f} GB, free {free:.1f} GB"))
                tight.append(m)
                continue
            viable.append(RouteDecision(m, list(skipped)))
        # Memory is a soft gate: macOS/Ollama can still load a model by compressing/swapping (slower, not
        # fatal). Smallest-first after everything that fits, so local_only never dead-ends on a busy laptop.
        # ("free" already counts the models the client unloads before a switch, see reclaimable_gb.)
        for m in sorted(tight, key=lambda m: m.mem_gb):
            viable.append(RouteDecision(m, [*skipped, (m.key, "memory tight — may swap")]))
        if not viable:
            raise NoModelAvailable(f"no model for task={task}: " + "; ".join(f"{k}: {r}" for k, r in skipped))
        return viable

    def route(self, task: Task, prompt: str, max_output_tokens: int = 1024) -> RouteDecision:
        decision = self.candidates(task, prompt, max_output_tokens)[0]
        if decision.skipped:
            log.debug("task={} -> {} (skipped: {})", task, decision.model.key, decision.skipped)
        return decision
