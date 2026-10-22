# syntax=docker/dockerfile:1.7
#
# Headless daily pipeline in a container; the LLMs stay on the host (Ollama), see docker-compose.yml.
#   docker compose build && docker compose run --rm jobhunter doctor
#
# Not containerised on purpose: `jobhunter login` / `apply` (headed browser, you at the keyboard) and
# Apple Notes (osascript, macOS only) — run those natively.

ARG PYTHON_VERSION=3.13

# ── builder: the locked environment, resolved by uv ────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src
# Dependencies first: this layer is reused until uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --all-extras --no-install-project

COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --all-extras --no-editable

# ── runtime: venv + Chromium, non-root ────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright

COPY --from=builder /opt/venv /opt/venv

# Chromium (scraping + HTML->PDF fallback) and its system libraries, at the version uv.lock pins.
# apt needs root, so this runs before USER.
RUN playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 jobhunter \
    && useradd --system --uid 10001 --gid jobhunter --home-dir /app --shell /usr/sbin/nologin jobhunter \
    && mkdir -p /app/data /app/output /app/var /app/logs \
    && chown -R jobhunter:jobhunter /app

WORKDIR /app
# Settings resolve paths against the working directory; mutable state lives in one volume (/app/var).
ENV HOME=/app \
    XDG_CACHE_HOME=/tmp/.cache \
    JH_DB_PATH=/app/var/job_hunter.db \
    JH_STATE_DIR=/app/var/state \
    JH_LLM__OLLAMA_AUTOSTART=false \
    JH_NOTES__ENABLED=false \
    JH_SCRAPER__HEADLESS=true

USER jobhunter

ENTRYPOINT ["jobhunter"]
CMD ["doctor"]
