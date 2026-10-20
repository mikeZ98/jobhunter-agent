"""Strategy 2: Playwright + playwright-stealth for DOM-heavy / WAF-protected boards, and the *visible*
persistent profile used for manual logins and form autofill.

Two kinds of contexts:
* ``scraping_context(board)`` — headless by default, fingerprint-coherent, per-board ``storage_state``
  (cookies + localStorage) persisted to ``.state/cookies/<board>.storage.json``.
* ``persistent_profile()`` — a real on-disk Chrome profile (``.state/browser-profile``), headed. Log in to
  LinkedIn / Pracuj / ATSes once by hand (``jobhunter login``); the session survives across runs.
"""

from __future__ import annotations

import asyncio
import random
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright
from playwright.async_api import Error as PlaywrightError

from jobhunter.config import Settings
from jobhunter.scrapers.antiban import FingerprintProfile, detect_challenge
from jobhunter.telemetry import get_logger

log = get_logger("browser")

_VIEWPORTS = [
    ((1440, 900), 2.0),
    ((1512, 982), 2.0),
    ((1680, 1050), 2.0),
    ((1920, 1080), 1.0),
    ((1536, 864), 1.25),
]


def real_browser_profile(version: str, branded: bool, seed: str) -> FingerprintProfile:
    """Fingerprint derived from the *actual* engine: real major version, host OS, matching client hints.

    Lying about the version in a real browser is counter-productive (JS feature probes, Client Hints and
    the TLS ClientHello would disagree with the UA). The only edit is dropping the "HeadlessChrome" token.
    """
    major = version.split(".")[0]
    if sys.platform == "darwin":
        os_token, platform = "Macintosh; Intel Mac OS X 10_15_7", '"macOS"'  # frozen by Chrome since 2021
    elif sys.platform.startswith("win"):
        os_token, platform = "Windows NT 10.0; Win64; x64", '"Windows"'
    else:
        os_token, platform = "X11; Linux x86_64", '"Linux"'
    ua = f"Mozilla/5.0 ({os_token}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"
    brand = "Google Chrome" if branded else "Chromium"
    hints = (
        f'"{brand}";v="{major}", "Chromium";v="{major}", "Not_A Brand";v="24"'
        if branded
        else f'"Chromium";v="{major}", "Not_A Brand";v="24"'
    )
    viewport, dpr = random.Random(seed).choice(_VIEWPORTS)
    return FingerprintProfile("n/a", ua, hints, platform, viewport, dpr)


async def apply_stealth(target: Page | BrowserContext, profile: FingerprintProfile | None = None) -> None:
    """Patch automation tells (navigator.webdriver, chrome.runtime, plugins, WebGL vendor...).

    Every override must agree with the claimed UA: playwright-stealth defaults ``navigator.platform`` to
    "Win32", which contradicts a macOS UA. With ``profile=None`` (real installed Chrome, persistent
    profile) we keep the browser's own UA/platform and only patch the automation leaks.
    """
    try:
        from playwright_stealth import Stealth
    except ImportError:  # playwright-stealth < 2.0
        from playwright_stealth import stealth_async  # type: ignore[attr-defined]

        for p in target.pages if isinstance(target, BrowserContext) else [target]:
            await stealth_async(p)
        return
    if profile is None:
        stealth = Stealth(
            navigator_languages_override=("pl-PL", "pl"),
            navigator_platform=False,
            navigator_user_agent=False,
            navigator_user_agent_data=False,
            sec_ch_ua=False,
        )
    else:
        platform = (
            "MacIntel"
            if "Macintosh" in profile.user_agent
            else "Win32"
            if "Windows" in profile.user_agent
            else "Linux x86_64"
        )
        stealth = Stealth(
            navigator_languages_override=("pl-PL", "pl"),
            navigator_platform_override=platform,
            navigator_user_agent_override=profile.user_agent,
            sec_ch_ua_override=profile.sec_ch_ua,
        )
    await stealth.apply_stealth_async(target)


def _context_kwargs(profile: FingerprintProfile) -> dict[str, Any]:
    w, h = profile.viewport
    # Inner window = screen minus browser chrome; jitter so sessions don't share an exact size.
    rng = random.Random(profile.user_agent + str(w))
    w, h = w - rng.randint(0, 40), h - rng.randint(70, 140)
    return {
        "user_agent": profile.user_agent,
        "viewport": {"width": w, "height": h},
        "screen": {"width": profile.viewport[0], "height": profile.viewport[1]},
        "device_scale_factor": profile.device_scale_factor,
        "locale": "pl-PL",
        "timezone_id": "Europe/Warsaw",
        "color_scheme": rng.choice(["light", "light", "dark"]),
        "extra_http_headers": {"Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7"},
    }


class BrowserManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pw: Playwright | None = None

    async def __aenter__(self) -> BrowserManager:
        self._pw = await async_playwright().start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._pw:
            await self._pw.stop()

    @property
    def pw(self) -> Playwright:
        assert self._pw is not None
        return self._pw

    def _storage_path(self, board: str) -> Path:
        return self.settings.cookies_dir / f"{board}.storage.json"

    @asynccontextmanager
    async def scraping_context(
        self, board: str, *, headless: bool | None = None
    ) -> AsyncIterator[BrowserContext]:
        cfg = self.settings.scraper
        launch_kw: dict[str, Any] = {
            "headless": cfg.headless if headless is None else headless,
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        try:
            # "chromium" = full Chromium in new-headless mode: the same binary/TLS fingerprint as the headed
            # `jobhunter login` window, so a cf_clearance earned there stays valid (headless-shell differs).
            browser = await self.pw.chromium.launch(channel=cfg.browser_channel or "chromium", **launch_kw)
            branded = bool(cfg.browser_channel) and cfg.browser_channel != "chromium"
        except PlaywrightError:
            log.info("channel {!r} unavailable, falling back to bundled Chromium", cfg.browser_channel)
            browser = await self.pw.chromium.launch(**launch_kw)
            branded = False
        # Stable per board (no date): cf_clearance is bound to UA+IP, a new "device" daily would void it.
        profile = real_browser_profile(browser.version, branded, seed=board)

        storage = self._storage_path(board)
        ctx = await browser.new_context(
            storage_state=str(storage) if storage.exists() else None, **_context_kwargs(profile)
        )
        await apply_stealth(ctx, profile)
        try:
            yield ctx
        finally:
            # If the browser already died (crash / OOM), these raise TargetClosedError — swallow it here so
            # the *original* error from the board is what gets logged, and partial results survive.
            for step in (lambda: ctx.storage_state(path=str(storage)), ctx.close, browser.close):
                try:
                    await step()
                except PlaywrightError as e:
                    log.warning("{} browser cleanup: {}", board, str(e).splitlines()[0])

    @asynccontextmanager
    async def persistent_profile(self) -> AsyncIterator[BrowserContext]:
        """Headed, authenticated profile. Never headless: a human must be able to see and submit."""
        kw: dict[str, Any] = {
            "headless": False,
            "locale": "pl-PL",
            "timezone_id": "Europe/Warsaw",
            "viewport": None,  # use the real window size
            "args": ["--disable-blink-features=AutomationControlled", "--start-maximized"],
            "accept_downloads": True,
        }
        user_data = self.settings.browser_profile_dir
        user_data.mkdir(parents=True, exist_ok=True)
        try:
            ctx = await self.pw.chromium.launch_persistent_context(
                str(user_data), channel=self.settings.scraper.browser_channel or "chromium", **kw
            )
        except PlaywrightError:
            ctx = await self.pw.chromium.launch_persistent_context(str(user_data), **kw)
        await apply_stealth(ctx)
        try:
            yield ctx
        finally:
            await ctx.close()


async def human_scroll(page: Page, steps: int | None = None) -> None:
    """Scroll in uneven increments with pauses — triggers lazy loading and looks like reading."""
    for _ in range(steps or random.randint(3, 7)):
        await page.mouse.wheel(0, random.randint(350, 900))
        await asyncio.sleep(random.uniform(0.4, 1.6))


async def page_challenge(page: Page, status: int | None) -> str | None:
    try:
        html = await page.content()
    except PlaywrightError:
        return None
    return detect_challenge(status or 200, {}, html[:20000])


async def wait_out_challenge(page: Page, timeout_s: float = 25.0) -> bool:
    """Cloudflare's passive JS challenge often resolves itself in a real browser within seconds.
    We wait (never click/solve). Returns True if the page cleared."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(2.5)
        if await page_challenge(page, None) is None:
            return True
    return False
