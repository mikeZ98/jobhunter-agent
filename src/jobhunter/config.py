"""Runtime configuration.

Precedence (highest first): init kwargs -> env vars (``JH_`` prefix, ``__`` nesting) -> ``.env`` ->
``jobhunter.toml`` -> defaults below. Secrets (API keys, LinkedIn cookies) belong in ``.env`` only.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

ROOT = Path.cwd()


class SearchConfig(BaseModel):
    keywords: list[str] = [
        "Python Developer",
        "Python Backend",
        "LLM Engineer",
        "GenAI Engineer",
        "Machine Learning Engineer",
        "AI Engineer",
    ]
    jjit_categories: list[str] = ["python", "ai", "data"]
    nfj_requirements: list[str] = ["python"]
    nfj_categories: list[str] = ["backend", "artificial-intelligence", "data"]
    seniority: list[Literal["junior", "mid"]] = ["junior", "mid"]
    locations: list[str] = ["Warszawa", "Remote"]
    linkedin_location: str = "Poland"
    max_age_days: int = 3
    max_pages_per_board: int = 3


class ScraperConfig(BaseModel):
    enabled_boards: list[str] = ["justjoinit", "nofluffjobs", "pracuj", "rocketjobs", "linkedin"]
    # Gaussian inter-request delay (seconds). Clamped to [min_delay, max_delay].
    delay_mean: float = 3.2
    delay_stddev: float = 1.1
    min_delay: float = 1.2
    max_delay: float = 9.0
    # Probability of an extra "reading" pause between pages — breaks the metronome pattern.
    long_pause_probability: float = 0.08
    long_pause_range: tuple[float, float] = (12.0, 35.0)
    max_retries: int = 4
    backoff_base: float = 4.0
    backoff_cap: float = 120.0
    # After N WAF challenges in one run a board is circuit-broken until the next run.
    challenge_circuit_threshold: int = 2
    headless: bool = True
    # Detail-page fetches per board per run (only for listings never seen before).
    max_enrich_per_board: int = 60
    browser_channel: str | None = "chrome"  # use installed Chrome (better fingerprint) if present


class LLMConfig(BaseModel):
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str | None = None  # e.g. http://gpu-box:8000/v1
    prefer_local: bool = True
    # true -> never call OpenRouter/Anthropic/OpenAI (chat or embeddings), even if API keys are set.
    # Cloud is opt-in either way: a cloud provider is only routed to once its API key is set.
    local_only: bool = False
    embedding_model: str = "nomic-embed-text"
    # Ollama tags per task tier, best first (must match `ollama list` exactly; ":latest" may be omitted).
    fast_models: list[str] = ["qwen2.5-coder:7b", "llama3.1:8b"]  # extract / score / validate
    chat_models: list[str] = ["llama3.1:8b", "qwen2.5-coder:7b"]  # form answers / LinkedIn replies
    creative_models: list[str] = ["llama3.1:8b", "qwen2.5-coder:7b"]  # Polish cover letter + CV
    interview_models: list[str] = ["llama3.1:8b", "qwen2.5-coder:7b"]
    anthropic_fast_model: str = "claude-haiku-5-5"
    anthropic_creative_model: str = "claude-sonnet-5-5"
    openai_fast_model: str = "gpt-4o-mini"
    openai_creative_model: str = "gpt-4o"
    # OpenRouter: one OpenAI-compatible key for many hosted open-weight models; first cloud tier.
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_fast_model: str = "meta-llama/llama-3.1-8b-instruct"
    openrouter_creative_model: str = "meta-llama/llama-3.3-70b-instruct"
    # Conservative: OpenRouter serves a model through several hosts, some with a shorter context.
    openrouter_context_window: int = 32_768
    # Ask OpenRouter to route only to hosts that don't store/train on prompts (CV + profile data).
    openrouter_deny_data_collection: bool = True
    # Context per Ollama request. Ollama's OpenAI endpoint ignores options.num_ctx (it ran at 4096 and
    # silently cut prompts), so the client creates a "<tag>-jh12k" alias with num_ctx baked in.
    ollama_num_ctx: int = 12_288
    # Start Ollama.app automatically (macOS) when it isn't running.
    ollama_autostart: bool = True
    # Headroom (GB) kept free when deciding whether a local model fits in (unified) memory.
    memory_headroom_gb: float = 2.5
    # Per HTTP request, i.e. model load + one generation (local calls are queued client-side first, so
    # waiting for another job's call doesn't count). Loading an 8B model on a busy 16 GB Mac takes ~2 min.
    request_timeout_s: float = 300.0
    max_validation_retries: int = 2
    # A provider that answered 429 is skipped for Retry-After seconds (this when it sent none) …
    rate_limit_cooldown_s: float = 60.0
    # … and a call with no other model left waits for it once, if that is at most this long.
    rate_limit_max_wait_s: float = 20.0
    # A cloud provider that refused the connection is skipped this long (Ollama is re-probed instead).
    connection_cooldown_s: float = 30.0
    # Parallel LLM calls. Ollama serves 1-2 requests at a time per model on a laptop; raise for vLLM/cloud.
    max_concurrency: int = 2
    # Requests sent to Ollama at once (it runs -np 1 by default; extra requests only queue there and time out).
    ollama_parallel: int = 1
    # Keep a single chat model resident: unload the others before switching. Two or three 4-8B models at
    # 12k context don't fit in 16 GB — Metal runs out of memory and Ollama answers with empty replies.
    ollama_single_model: bool = True


class ScoringConfig(BaseModel):
    min_daily: int = 10
    max_daily: int = 15
    min_score: float = 55.0
    llm_shortlist_size: int = 40  # only the top-N by cheap scores go to the LLM judge
    weight_embedding: float = 0.25
    weight_skills: float = 0.30
    weight_llm: float = 0.45
    # Rule-of-thumb UoP gross -> B2B net equivalence (PL, 2025/26). Tune to your tax situation.
    uop_to_b2b_factor: float = 1.25
    hours_per_month: int = 168


class CriticConfig(BaseModel):
    """Adversarial fact-check of every cover letter / CV draft (agents/critic.py). Fails closed."""

    enabled: bool = True
    # Second, LLM pass (fast tier). Off -> deterministic checks only, and any job-ad technology the
    # draft mentions but the profile lacks rejects the draft, since nothing can clear it.
    llm_review: bool = True
    # Rewrites with the critic's findings before a draft is quarantined instead of saved.
    max_revisions: int = Field(default=1, ge=0, le=3)


class NotesConfig(BaseModel):
    enabled: bool = True
    folder: str = "Job Search"
    account: str | None = None  # None -> default account (usually iCloud)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JH_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        toml_file="jobhunter.toml",
    )

    data_dir: Path = ROOT / "data"
    output_dir: Path = ROOT / "output"
    state_dir: Path = ROOT / ".state"  # cookies, browser profiles — gitignored
    db_path: Path = ROOT / "job_hunter.db"
    profile_path: Path = ROOT / "data" / "profile.json"
    base_cv_path: Path = ROOT / "data" / "base_cv.md"
    log_level: str = "INFO"

    search: SearchConfig = Field(default_factory=SearchConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    notes: NotesConfig = Field(default_factory=NotesConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)

    # Secrets — standard names so the SDKs also pick them up.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    linkedin_li_at: SecretStr | None = Field(default=None, validation_alias="LINKEDIN_LI_AT")
    linkedin_jsessionid: SecretStr | None = Field(default=None, validation_alias="LINKEDIN_JSESSIONID")
    otel_endpoint: str | None = Field(default=None, validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, dotenv_settings, TomlConfigSettingsSource(settings_cls))

    @property
    def cookies_dir(self) -> Path:
        return self.state_dir / "cookies"

    @property
    def browser_profile_dir(self) -> Path:
        """Persistent, *visible* Chrome profile used for logins and form autofill."""
        return self.state_dir / "browser-profile"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.output_dir, self.state_dir, self.cookies_dir, ROOT / "logs"):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
