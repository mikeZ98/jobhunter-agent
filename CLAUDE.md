# CLAUDE.md

Guidance for AI coding agents working in this repository. The README explains the product. This file
lists the rules that must survive every change.

## Invariants (do not break; each is enforced by tests)

1. **Never submit.** `agents/autofill.py` fills forms and stops. It has no click / press / submit
   calls (`test_autofill.py` parses the AST), never ticks consent boxes, never overwrites pre-filled
   fields. LinkedIn replies are drafted, never sent.
2. **Local-first, cloud opt-in.** Ollama comes first. A cloud provider (OpenRouter → Anthropic → OpenAI) is
   routed to only once its API key is set, and never with `[llm] local_only = true`. Don't add a code
   path that sends profile/CV data to a network service without going through `ModelRouter`.
   OpenRouter requests carry `provider.data_collection = "deny"`.
3. **The profile is canonical truth.** `data/profile.json` (+ `base_cv.md`) is the only source of facts
   about the candidate. Drafting validators (`agents/assets.py`) and the critic ground every claim in it.
4. **The critic fails closed** (`agents/critic.py`, `workflow/drafting.py`):
   - approval is computed in code from findings, **never** read from a model field;
   - a critic that cannot run (no model, timeout, malformed verdict) rejects the draft;
   - every suspect term must be answered by the LLM verdict (validator re-asks), else reject;
   - only approved drafts become assets (`cover_letter.*`, `cv.*`). Rejected or unverified ones go to
     `<job>/quarantine/` with `<doc>.critic.json`. Autofill reads only `cv_pdf` / `cover_letter_text`.
   - *quarantined* (findings after the last revision) jobs are excluded from `Database.failed_prepares()`.
     *Unverified* (critic unavailable) jobs stay retryable.
5. **Strict transition contracts.** Every graph node returns `SomeOutput(...).update()` from
   `workflow/schemas.py` (`strict`, `extra="forbid"`, `frozen`, domain objects as `InstanceOf[...]`).
   To add a state channel, update `PipelineState` (or `DraftState`), the node's output model, and
   `NODE_OUTPUTS` / `DRAFT_NODE_OUTPUTS`. The contract tests check that the channels exist and that
   every node has a model.
6. **Classified LLM failures.** Handle provider errors through `llm.errors.classify()`, never by
   string-matching messages. instructor wraps provider errors in `InstructorRetryException`, so the
   classifier walks `__cause__`. Rate limits cool the *provider* down (`ModelRouter.cooldown`), a
   rejected key disables it, timeouts move on without a cooldown. Local SDK clients use `max_retries=0`.
7. **Idempotent nodes.** Every node can be re-run against SQLite. A crashed run resumes. Listings are
   upserted, status moves only along `models.STATUS_TRANSITIONS`.
8. **Memory discipline on 16 GB.** One local chat model resident (`ollama_single_model`), local calls
   serialized (`ollama_parallel`), drafting is sequential per job. Don't parallelise local LLM calls.

## Privacy

Never commit or bake into an image: `.env`, `data/`, `jobhunter.toml`, `*.db`, `.state/`, `output/`,
`logs/`. They are git-ignored, and `.dockerignore` is an **allow-list**: if a new file must enter the
image, add it there explicitly. Examples use the fictional profile in `src/jobhunter/examples/`.

## Commands

Run everything through `uv run` (direnv pins the venv and every cache to `.cache/` on this volume; no
global installs). The gate, run before every commit, all must pass:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest
```

- mypy is `strict` over `src/`; ruff line length 110; Python ≥ 3.12.
- Tests are offline and deterministic: fake boards, a fake LLM (`tests/test_graph.py::FakeLLM`),
  mocked provider HTTP (`httpx2.MockTransport` behind the real OpenAI SDK, `tests/test_llm_providers.py`).
  Never call a real model or job board from a test.
- Docker: `docker compose build && docker compose run --rm jobhunter doctor` (Ollama stays on the host).

## Map

| Path | What |
|------|------|
| `graph.py` | top-level LangGraph: collect → normalize → score → prepare ×N → inbox → publish → autofill |
| `workflow/schemas.py` | node output contracts, `RunRequest` |
| `workflow/drafting.py` | per-job sub-graph: draft → critique → revise ⟲ → approve / hold |
| `agents/critic.py` | deterministic + adversarial LLM fact-check |
| `llm/router.py` · `llm/client.py` · `llm/errors.py` | candidate gates and cooldowns · instructor calls and fallback · failure classification |
| `config.py` | settings: env `JH_<SECTION>__<KEY>` > `.env` > `jobhunter.toml` (nested keys deep-merge) |

## Conventions

- Conventional Commits (`feat(scope):`, `fix:`, `test:`, `build:`, `docs:`), one logical change each,
  with the gate green at every commit.
- Comments explain *why* (constraints, incidents), not what. Match the density of the surrounding code.
- User-facing text in generated documents is Polish. Code, logs and docs are English.
