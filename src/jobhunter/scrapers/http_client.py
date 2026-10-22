"""Strategy 1: direct JSON endpoints over ``curl_cffi`` with an impersonated TLS fingerprint."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from curl_cffi.requests import AsyncSession, Response
from curl_cffi.requests.exceptions import RequestException

from jobhunter.config import ScraperConfig
from jobhunter.scrapers.antiban import (
    BlockedError,
    ChallengeKind,
    CircuitBreaker,
    HostRateLimiter,
    backoff_delay,
    browser_headers,
    detect_challenge,
    notify,
    pick_profile,
)
from jobhunter.telemetry import antibot_counter, get_logger, span

log = get_logger("http")


def header_safe_url(url: str) -> str:
    """HTTP headers are latin-1: percent-encode non-ASCII (e.g. "Inżynier" in a search URL), keep the rest."""
    return quote(url, safe=":/?#[]@!$&'()*+,;=%")


class StealthHttpClient:
    """One instance per board. Keeps one TLS session alive and persists cookies between runs, so a
    daily run looks like a returning visitor (warm ``__cf_bm`` etc.) rather than a fresh handshake."""

    def __init__(
        self,
        board: str,
        cfg: ScraperConfig,
        cookies_dir: Path,
        limiter: HostRateLimiter,
        breaker: CircuitBreaker,
        on_antibot: Any = None,  # callable(board, kind, status, url) -> None (DB sink)
    ) -> None:
        self.board = board
        self.cfg = cfg
        self.limiter = limiter
        self.breaker = breaker
        self.on_antibot = on_antibot
        # Same "device" for a whole ISO week: returning-visitor cookies stay coherent with the fingerprint.
        self.profile = pick_profile(seed=f"{board}:{date.today().isocalendar()[:2]}")
        self.cookie_file = cookies_dir / f"{board}.json"
        self._session: AsyncSession[Response] | None = None

    async def __aenter__(self) -> StealthHttpClient:
        # The UA we send is the one baked into the profile, i.e. the same browser build that the TLS
        # ClientHello / HTTP2 SETTINGS impersonate — the two must never disagree.
        self._session = AsyncSession(
            impersonate=self.profile.impersonate,
            headers={"User-Agent": self.profile.user_agent},
            timeout=30,
        )
        self._load_cookies()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._save_cookies()
        if self._session:
            await self._session.close()

    # ---------- cookie persistence ----------

    def _load_cookies(self) -> None:
        if not self.cookie_file.exists() or self._session is None:
            return
        try:
            for c in json.loads(self.cookie_file.read_text()):
                self._session.cookies.set(
                    c["name"], c["value"], domain=c.get("domain", ""), path=c.get("path", "/")
                )
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("discarding corrupt cookie file {}: {}", self.cookie_file, e)

    def _save_cookies(self) -> None:
        if self._session is None:
            return
        jar = self._session.cookies.jar
        data = [
            {"name": c.name, "value": c.value, "domain": c.domain, "path": c.path, "expires": c.expires}
            for c in jar
            if not c.is_expired()
        ]
        self.cookie_file.parent.mkdir(parents=True, exist_ok=True)
        self.cookie_file.write_text(json.dumps(data, indent=1))

    # ---------- requests ----------

    async def request(
        self,
        method: Literal["GET", "POST"],
        url: str,
        *,
        params: dict[str, Any] | list[tuple[str, Any]] | None = None,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        expect_json: bool = True,
        referer: str | None = None,
    ) -> Response:
        assert self._session is not None, "use `async with StealthHttpClient(...)`"
        if self.breaker.is_open(self.board):
            raise BlockedError(self.board, "circuit_open", url)

        if referer:
            referer = header_safe_url(referer)
        hdrs = browser_headers(self.profile, referer=referer, json=expect_json)
        if self.profile.sec_ch_ua:
            hdrs |= {
                "sec-ch-ua": self.profile.sec_ch_ua,
                "sec-ch-ua-platform": self.profile.platform,
                "sec-ch-ua-mobile": "?0",
            }
        hdrs |= headers or {}

        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            await self.limiter.wait(url)
            with span("http.request", board=self.board, url=url, attempt=attempt) as s:
                try:
                    resp = await self._session.request(
                        method, url, params=params, json=json_body, headers=hdrs
                    )
                except RequestException as e:  # DNS, TLS reset, timeout
                    last_exc = e
                    wait = backoff_delay(attempt, self.cfg.backoff_base, self.cfg.backoff_cap)
                    log.warning("{} transport error ({}), retry in {:.1f}s", self.board, e, wait)
                    await asyncio.sleep(wait)
                    continue
                s.set_attribute("http.status_code", resp.status_code)

            headers = {k: v for k, v in resp.headers.items() if v is not None}
            kind = detect_challenge(resp.status_code, headers, resp.text[:20000])
            if kind is None and resp.status_code < 500:
                return resp

            if kind:
                self._record_antibot(kind, resp.status_code, url)
                if kind != ChallengeKind.RATE_LIMIT and self.breaker.record(self.board):
                    notify(
                        "JobHunter: scraping paused", f"{self.board}: {kind} — circuit opened for this run"
                    )
                    raise BlockedError(self.board, kind, url, resp.status_code)
            wait = backoff_delay(
                attempt, self.cfg.backoff_base, self.cfg.backoff_cap, resp.headers.get("retry-after")
            )
            log.warning(
                "{} HTTP {} ({}), backoff {:.1f}s [{}/{}]",
                self.board,
                resp.status_code,
                kind or "server error",
                wait,
                attempt + 1,
                self.cfg.max_retries,
            )
            await asyncio.sleep(wait)

        raise BlockedError(self.board, "retries_exhausted", url) from last_exc

    async def get_json(self, url: str, **kw: Any) -> Any:
        resp = await self.request("GET", url, **kw)
        resp.raise_for_status()
        return resp.json()

    async def post_json(self, url: str, body: Any, **kw: Any) -> Any:
        resp = await self.request("POST", url, json_body=body, **kw)
        resp.raise_for_status()
        return resp.json()

    async def get_text(self, url: str, **kw: Any) -> str:
        resp = await self.request("GET", url, expect_json=False, **kw)
        resp.raise_for_status()
        return resp.text

    def _record_antibot(self, kind: str, status: int, url: str) -> None:
        antibot_counter.add(1, {"board": self.board, "kind": kind})
        log.bind(antibot=True).warning(
            "anti-bot trigger: board={} kind={} status={}", self.board, kind, status
        )
        if self.on_antibot:
            self.on_antibot(self.board, kind, status, url)
