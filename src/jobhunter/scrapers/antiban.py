"""Anti-detection primitives shared by the HTTP and browser strategies.

Principles (in order of how much they matter in practice):
1. **Consistency beats randomness.** A Chrome JA3/JA4 TLS fingerprint with a Firefox User-Agent is the
   #1 bot tell. We rotate *coherent fingerprint profiles* (TLS target + UA + client hints + viewport)
   per session, never individual headers per request.
2. **Volume and rhythm.** A handful of paged requests per board per day, Gaussian-jittered with
   occasional long "reading" pauses, looks like a person. Concurrency against one host does not.
3. **Back off, don't fight.** On a WAF challenge we retry with exponential backoff a few times, then
   open a per-board circuit and notify the human — we never try to solve challenges automatically.
"""

from __future__ import annotations

import asyncio
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from urllib.parse import urlparse

from jobhunter.config import ScraperConfig
from jobhunter.telemetry import antibot_counter, get_logger

log = get_logger("antiban")


@dataclass(frozen=True, slots=True)
class FingerprintProfile:
    impersonate: str  # curl_cffi target; its TLS + HTTP/2 SETTINGS + header order match this UA
    user_agent: str
    sec_ch_ua: str | None
    platform: str
    viewport: tuple[int, int]
    device_scale_factor: float


# Keep UA versions aligned with the curl_cffi impersonation target (bump both together when upgrading
# curl_cffi — `python -c "from curl_cffi.requests import BrowserType; print(list(BrowserType))"`).
_MAC = "Macintosh; Intel Mac OS X 10_15_7"  # Chrome/Safari freeze the macOS version token at 10_15_7
_WIN = "Windows NT 10.0; Win64; x64"


def _chrome(
    target: str, os_token: str, platform: str, viewport: tuple[int, int], dpr: float
) -> FingerprintProfile:
    major = target.removeprefix("chrome")
    return FingerprintProfile(
        target,
        f"Mozilla/5.0 ({os_token}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36",
        f'"Google Chrome";v="{major}", "Chromium";v="{major}", "Not_A Brand";v="24"',
        platform,
        viewport,
        dpr,
    )


PROFILES: tuple[FingerprintProfile, ...] = (
    _chrome("chrome150", _MAC, '"macOS"', (1512, 982), 2.0),
    _chrome("chrome150", _WIN, '"Windows"', (1920, 1080), 1.0),
    _chrome("chrome146", _MAC, '"macOS"', (1440, 900), 2.0),
    _chrome("chrome146", _WIN, '"Windows"', (1536, 864), 1.25),
    FingerprintProfile(
        "safari260",
        f"Mozilla/5.0 ({_MAC}) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15",
        None,  # Safari sends no client hints — sending them would be a tell
        '"macOS"',
        (1512, 982),
        2.0,
    ),
    FingerprintProfile(
        "firefox147",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:147.0) Gecko/20100101 Firefox/147.0",
        None,  # Firefox sends no client hints either
        '"macOS"',
        (1680, 1050),
        2.0,
    ),
)


def pick_profile(seed: str | None = None) -> FingerprintProfile:
    """Stable per (board, day) when seeded — a 'returning visitor' keeps the same device for the day."""
    rng = random.Random(seed) if seed else random
    return rng.choice(PROFILES)


def browser_headers(
    profile: FingerprintProfile, *, referer: str | None = None, json: bool = False
) -> dict[str, str]:
    """Only headers a real browser would send that the impersonation layer doesn't already set."""
    h = {
        "Accept": "application/json, text/plain, */*"
        if json
        else "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    if referer:
        h["Referer"] = referer
    if json:
        h.update({"Sec-Fetch-Dest": "empty", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Site": "same-origin"})
    return h


class HumanDelay:
    def __init__(self, cfg: ScraperConfig) -> None:
        self.cfg = cfg

    def sample(self) -> float:
        d = random.gauss(self.cfg.delay_mean, self.cfg.delay_stddev)
        d = min(max(d, self.cfg.min_delay), self.cfg.max_delay)
        if random.random() < self.cfg.long_pause_probability:
            d += random.uniform(*self.cfg.long_pause_range)
        return d

    async def sleep(self) -> float:
        d = self.sample()
        await asyncio.sleep(d)
        return d


class HostRateLimiter:
    """Serializes requests per host and enforces the human delay between them.

    Different boards still run concurrently; requests to the *same* host never overlap.
    """

    def __init__(self, delay: HumanDelay) -> None:
        self.delay = delay
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last: dict[str, float] = {}

    async def wait(self, url: str) -> None:
        host = urlparse(url).netloc
        async with self._locks[host]:
            target_gap = self.delay.sample()
            elapsed = time.monotonic() - self._last.get(host, 0.0)
            if elapsed < target_gap:
                await asyncio.sleep(target_gap - elapsed)
            self._last[host] = time.monotonic()


# ---------- WAF / challenge detection ----------

CHALLENGE_MARKERS = (
    "just a moment",
    "cf-chl-",
    "challenge-platform",
    "cf_chl_opt",
    "attention required! | cloudflare",
    "px-captcha",
    "_incapsula_resource",
    "datadome",
    "captcha-delivery.com",
    "g-recaptcha",
    "hcaptcha.com",
)


class ChallengeKind:
    CLOUDFLARE = "cloudflare"
    CAPTCHA = "captcha"
    RATE_LIMIT = "rate_limit"
    BLOCKED = "blocked"


def detect_challenge(status: int, headers: dict[str, str], body_head: str) -> str | None:
    h = {k.lower(): v for k, v in headers.items()}
    body = body_head[:20000].lower()
    if status == 429:
        return ChallengeKind.RATE_LIMIT
    is_cf = "cf-ray" in h or h.get("server", "").lower() == "cloudflare" or "cf-mitigated" in h
    if any(m in body for m in CHALLENGE_MARKERS):
        return ChallengeKind.CLOUDFLARE if is_cf or "cf-" in body else ChallengeKind.CAPTCHA
    if status in (403, 503) and is_cf:
        return ChallengeKind.CLOUDFLARE
    if status in (401, 403, 999):  # 999 = LinkedIn's "go away"
        return ChallengeKind.BLOCKED
    return None


def backoff_delay(attempt: int, base: float, cap: float, retry_after: str | None = None) -> float:
    """Exponential backoff with full jitter (AWS architecture blog); honours Retry-After."""
    if retry_after:
        try:
            return min(float(retry_after), cap)
        except ValueError:
            pass
    return random.uniform(0, min(cap, base * 2**attempt))


@dataclass
class CircuitBreaker:
    threshold: int
    _hits: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, board: str) -> bool:
        """Returns True if this hit just opened the circuit."""
        self._hits[board] += 1
        return self._hits[board] == self.threshold

    def is_open(self, board: str) -> bool:
        return self._hits[board] >= self.threshold


class BlockedError(RuntimeError):
    def __init__(self, board: str, kind: str, url: str, status: int | None = None) -> None:
        super().__init__(f"{board}: {kind} at {url} (HTTP {status})")
        self.board, self.kind, self.url, self.status = board, kind, url, status


def notify(title: str, message: str) -> None:
    """Native macOS notification; silently no-op elsewhere."""
    antibot_counter.add(1, {"title": title})
    script = f'display notification {_as_str(message)} with title {_as_str(title)} sound name "Basso"'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5, capture_output=True)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        log.warning("notification unavailable: {} — {}", title, message)


def _as_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
