"""Strategy 2 boards: real (stealth) Chromium.

* **Pracuj.pl** sits behind Cloudflare's managed challenge (plain HTTP gets 403 "Just a moment").
  A persistent, fingerprint-stable browser with warm ``cf_clearance`` passes passively. We read the
  Next.js ``__NEXT_DATA__`` blob first (structured, stable) and fall back to ``data-test`` DOM hooks.
* **RocketJobs** is a client-rendered SPA on the JustJoin candidate-api. We let the page do its own
  XHRs and *intercept the JSON responses* — the most resilient way to scrape any SPA: no selectors.

If a challenge doesn't clear on its own we stop, notify, and ask the human to run
``jobhunter login <board>`` once (headed) — we never automate challenge solving.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, ClassVar
from urllib.parse import quote, quote_plus

from playwright.async_api import BrowserContext, Page, Response
from playwright.async_api import Error as PlaywrightError

from jobhunter.config import SearchConfig
from jobhunter.models import Board, ContractType, JobListing, RawListing, RemoteMode, Seniority
from jobhunter.salary import parse_salary_text
from jobhunter.scrapers.antiban import BlockedError, CircuitBreaker, HumanDelay, notify
from jobhunter.scrapers.base import Scraper
from jobhunter.scrapers.browser import BrowserManager, human_scroll, page_challenge, wait_out_challenge
from jobhunter.scrapers.justjoinit import parse_candidate_offer
from jobhunter.utils import html_to_text, parse_dt


class BrowserScraper(Scraper):
    strategy: ClassVar[str] = "browser"

    def __init__(
        self,
        search: SearchConfig,
        browser: BrowserManager,
        breaker: CircuitBreaker,
        delay: HumanDelay,
        on_antibot: Any = None,
    ) -> None:
        super().__init__(search)
        self.browser = browser
        self.breaker = breaker
        self.delay = delay
        self.on_antibot = on_antibot
        self._ctx_cm: Any = None
        self.ctx: BrowserContext | None = None
        self.page: Page | None = None

    async def __aenter__(self) -> BrowserScraper:
        self._ctx_cm = self.browser.scraping_context(self.board.value)
        self.ctx = await self._ctx_cm.__aenter__()
        self.page = await self.ctx.new_page()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._ctx_cm:
            await self._ctx_cm.__aexit__(*exc)

    async def goto(self, url: str) -> Page:
        assert self.page is not None
        if self.breaker.is_open(self.board.value):
            raise BlockedError(self.board.value, "circuit_open", url)
        await self.delay.sleep()
        resp = None
        for attempt in range(3):  # net::ERR_CONNECTION_CLOSED / RESET / TIMED_OUT are usually transient
            try:
                resp = await self.page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                break
            except PlaywrightError as e:
                if "net::ERR_" not in str(e) or attempt == 2:
                    raise
                wait = 5 * 3**attempt
                self.log.warning("{} on {} — retrying in {} s", str(e).split()[1], url, wait)
                await asyncio.sleep(wait)
        kind = await page_challenge(self.page, resp.status if resp else None)
        if kind:
            self.log.warning("challenge ({}) on {}, waiting for passive clearance", kind, url)
            if self.on_antibot:
                self.on_antibot(self.board.value, kind, resp.status if resp else None, url)
            if not await wait_out_challenge(self.page):
                self.breaker.record(self.board.value)
                notify(
                    "JobHunter: action needed",
                    f"{self.board.value} challenge didn't clear. Run: jobhunter login {self.board.value}",
                )
                raise BlockedError(self.board.value, kind, url, resp.status if resp else None)
        return self.page

    async def in_page_fetch_json(self, url: str) -> Any:
        """``fetch`` from inside the page: real browser TLS + cookies + same-origin headers."""
        assert self.page is not None
        return await self.page.evaluate(
            """async (u) => { const r = await fetch(u, {credentials: 'include',
                 headers: {'Accept': 'application/json'}}); if (!r.ok) throw new Error('HTTP ' + r.status);
                 return await r.json(); }""",
            url,
        )


# ------------------------------------------------------------------ RocketJobs


# Card level labels -> config seniority keys ("Specjalista / Mid" -> mid).
ROCKET_LEVELS = [("staż", "intern"), ("junior", "junior"), ("mid", "mid"), ("specjalista", "mid"),
                 ("senior", "senior"), ("ekspert", "senior"), ("menedżer", "lead"), ("kierownik", "lead"),
                 ("dyrektor", "lead")]  # fmt: skip

# Every card link on the list page (2026-10, React Server Components — no JSON XHRs): the title anchor
# has text, the card anchor wraps company / location / salary / level lines.
ROCKET_CARDS_JS = """els => {
  const by = new Map();
  for (const a of els) {
    const slug = new URL(a.href).pathname.split('/').filter(Boolean).pop();
    if (!slug) continue;
    const c = by.get(slug) || {slug, title: '', card: ''};
    const t = a.innerText.trim();
    const card = (a.closest('li, article, [data-index]') || a).innerText.trim();
    if (t && (!c.title || t.length < c.title.length)) c.title = t.split('\\n')[0];
    if (card.length > c.card.length) c.card = card;
    by.set(slug, c);
  }
  return [...by.values()];
}"""


def rocket_card_level(card: str) -> str | None:
    low = card.lower()
    for line in low.splitlines():
        if hits := [lvl for key, lvl in ROCKET_LEVELS if key in line and len(line) < 40]:
            return hits[0] if "mid" not in hits else "mid"
    return None


class RocketJobsScraper(BrowserScraper):
    board: ClassVar[Board] = Board.ROCKETJOBS
    base_url: ClassVar[str] = "https://rocketjobs.pl"

    async def search(self) -> AsyncIterator[RawListing]:
        """List pages only (one per keyword). Detail JSON is fetched in ``enrich``, i.e. only for
        listings that survive DB dedup and the title filter."""
        assert self.page is not None
        captured: list[dict[str, Any]] = []

        async def on_response(resp: Response) -> None:  # in case the SPA goes back to JSON XHRs
            if "/api/candidate-api/offers" in resp.url and resp.request.method == "GET" and resp.ok:
                try:
                    body = await resp.json()
                except Exception:
                    return
                if isinstance(body, dict) and isinstance(body.get("data"), list):
                    captured.extend(body["data"])

        self.page.on("response", on_response)
        seen: set[str] = set()
        wanted = (*self.search_cfg.seniority, None)
        failed: list[str] = []
        try:
            for kw in self.search_cfg.keywords:
                captured.clear()
                try:
                    page = await self.goto(
                        f"{self.base_url}/oferty-pracy/wszystkie-lokalizacje?keyword={quote_plus(kw)}"
                    )
                except PlaywrightError as e:  # one dead keyword page shouldn't cost the whole board
                    failed.append(kw)
                    self.log.warning("rocketjobs '{}' skipped: {}", kw, str(e).splitlines()[0])
                    if len(failed) == len(self.search_cfg.keywords):
                        raise
                    continue
                await human_scroll(page)
                await asyncio.sleep(1.0)
                n_before = len(seen)
                for o in list(captured):
                    if o.get("slug") and o["slug"] not in seen and o.get("experienceLevel") in wanted:
                        seen.add(o["slug"])
                        yield self._raw(o["slug"], o["title"], o.get("companyName", ""), o)
                if captured:
                    continue
                for c in await page.eval_on_selector_all("a[href*='/oferta-pracy/']", ROCKET_CARDS_JS):
                    if not c["title"] or c["slug"] in seen or rocket_card_level(c["card"]) not in wanted:
                        continue
                    seen.add(c["slug"])
                    company = next((ln.strip() for ln in c["card"].splitlines() if ln.strip()), "")
                    yield self._raw(c["slug"], c["title"], company if company != c["title"] else "", {})
                self.log.info("rocketjobs '{}': {} new cards", kw, len(seen) - n_before)
        finally:
            self.page.remove_listener("response", on_response)

    def _raw(self, slug: str, title: str, company: str, payload: dict[str, Any]) -> RawListing:
        return RawListing(
            board=self.board,
            external_id=slug,
            title=title,
            company=company,
            url=f"{self.base_url}/oferta-pracy/{slug}",
            payload=payload,
        )

    async def enrich(self, raw: RawListing) -> RawListing:
        if raw.payload.get("body") is None:
            detail = await self.in_page_fetch_json(
                f"{self.base_url}/api/candidate-api/offers/{raw.external_id}"
            )
            await self.delay.sleep()
            raw.payload = {**detail, **{k: v for k, v in raw.payload.items() if v is not None}}
            raw.company = raw.company or detail.get("companyName", "")
        raw.description = html_to_text(raw.payload.get("body") or "")
        return raw

    def parse(self, raw: RawListing) -> JobListing:
        if "title" not in raw.payload:
            raise ValueError("offer detail not fetched")
        return parse_candidate_offer(self.board, raw)


# ------------------------------------------------------------------ Pracuj.pl

PRACUJ_LEVELS = {"junior": "17", "mid": "4"}  # et= : 1 trainee, 17 junior, 4 mid/regular, 18 senior
PRACUJ_SENIORITY = [
    ("junior", Seniority.JUNIOR),
    ("mid", Seniority.MID),
    ("regular", Seniority.MID),
    ("senior", Seniority.SENIOR),
    ("expert", Seniority.LEAD),
    ("praktykant", Seniority.INTERN),
    ("stażysta", Seniority.INTERN),
    ("asystent", Seniority.JUNIOR),
    ("specjalista", Seniority.MID),
]


def _find_offer_dicts(node: Any, out: list[dict[str, Any]]) -> None:
    """Walk ``__NEXT_DATA__`` and collect every dict that looks like an offer group. Schema-agnostic
    on purpose: pracuj reshuffles its dehydrated react-query state regularly."""
    if isinstance(node, dict):
        if "jobTitle" in node and "companyName" in node:
            out.append(node)
            return
        for v in node.values():
            _find_offer_dicts(v, out)
    elif isinstance(node, list):
        for v in node:
            _find_offer_dicts(v, out)


class PracujScraper(BrowserScraper):
    board: ClassVar[Board] = Board.PRACUJ
    base_url: ClassVar[str] = "https://it.pracuj.pl"

    def _search_urls(self) -> list[str]:
        et = "%2C".join(PRACUJ_LEVELS[s] for s in self.search_cfg.seniority)
        urls = [f"{self.base_url}/praca?et={et}&itth=37"]  # itth=37 -> every Python offer
        # + configured keywords (PLC, automation, ML…); Python ones are already covered by itth=37.
        # it.pracuj.pl only: www.pracuj.pl serves a Cloudflare challenge (403) to automated browsers.
        urls += [
            f"{self.base_url}/praca/{quote(kw.lower())};kw?et={et}"
            for kw in dict.fromkeys(self.search_cfg.keywords)
            if "python" not in kw.lower()
        ]
        return urls

    async def search(self) -> AsyncIterator[RawListing]:
        seen: set[str] = set()
        for base in self._search_urls():
            for pn in range(1, self.search_cfg.max_pages_per_board + 1):
                page = await self.goto(base + (f"&pn={pn}" if pn > 1 else ""))
                await human_scroll(page, steps=3)
                offers = await self._from_next_data(page) or await self._from_dom(page)
                if not offers:
                    break
                new = 0
                for o in offers:
                    if o["external_id"] in seen:
                        continue
                    seen.add(o["external_id"])
                    new += 1
                    yield RawListing(
                        board=self.board,
                        external_id=o["external_id"],
                        url=o["url"],
                        title=o["title"],
                        company=o["company"],
                        payload=o,
                    )
                if new == 0:
                    break

    async def _from_next_data(self, page: Page) -> list[dict[str, Any]]:
        raw = await page.evaluate("() => document.getElementById('__NEXT_DATA__')?.textContent || null")
        if not raw:
            return []
        groups: list[dict[str, Any]] = []
        _find_offer_dicts(json.loads(raw), groups)
        out = []
        for g in groups:
            for off in g.get("offers") or [g]:
                url = off.get("offerAbsoluteUri") or off.get("offerUrl")
                if not url:
                    continue
                out.append(
                    {
                        "external_id": str(off.get("partitionId") or g.get("groupId") or url),
                        "url": url,
                        "title": g["jobTitle"],
                        "company": g["companyName"],
                        "salary_text": g.get("salaryDisplayText") or "",
                        "contracts": g.get("typesOfContract") or [],
                        "work_modes": g.get("workModes") or [],
                        "levels": g.get("positionLevels") or [],
                        "location": off.get("displayWorkplace") or "",
                        "published": g.get("lastPublicated") or g.get("lastPublicatedUTC"),
                        "technologies": g.get("technologies") or [],
                        "snippet": g.get("jobDescription") or "",
                    }
                )
        return out

    async def _from_dom(self, page: Page) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = await page.eval_on_selector_all(
            "[data-test='default-offer'], [data-test='positioned-offer']",
            """els => els.map(e => {
                const q = s => e.querySelector(s)?.innerText?.trim() || '';
                const a = e.querySelector("a[data-test='link-offer'], a[href*='/praca/']");
                const url = a ? a.href.split('?')[0] : '';
                return { external_id: e.getAttribute('data-test-offerid') || url, url,
                         title: q("[data-test='offer-title']"), company: q("[data-test='text-company-name']"),
                         salary_text: q("[data-test='offer-salary']"), location: q("[data-test='text-region']"),
                         contracts: [], work_modes: [q("[data-test='offer-additional-info-0']")], levels: [] };
            }).filter(o => o.url && o.title)""",
        )
        return rows

    async def enrich(self, raw: RawListing) -> RawListing:
        page = await self.goto(raw.url)
        details = await page.evaluate(
            """() => {
              const txt = s => [...document.querySelectorAll(s)].map(e => e.innerText.trim()).filter(Boolean);
              return {
                expected: txt("[data-test='item-technologies-expected'], [data-test='section-technologies-expected'] li"),
                optional: txt("[data-test='item-technologies-optional'], [data-test='section-technologies-optional'] li"),
                sections: txt("[data-test^='section-'][data-test$='-about-project'], [data-test='section-responsibilities'],"
                              + " [data-test='section-requirements'], [data-test='section-offered']"),
                apply: document.querySelector("[data-test='anchor-apply'], a[data-test*='apply']")?.href || null,
                main: document.querySelector('main')?.innerText || '' };
            }"""
        )
        raw.payload |= {
            "expected": details["expected"],
            "optional": details["optional"],
            "apply_url": details["apply"],
        }
        raw.description = ("\n\n".join(details["sections"]) or details["main"])[:12000]
        return raw

    def parse(self, raw: RawListing) -> JobListing:
        p = raw.payload
        contracts = " ".join(p.get("contracts") or [])
        sal = parse_salary_text(p.get("salary_text", ""), contract_hint=contracts)
        modes = " ".join(p.get("work_modes") or []).lower()
        remote = (
            RemoteMode.REMOTE
            if "zdaln" in modes
            else RemoteMode.HYBRID
            if "hybryd" in modes
            else RemoteMode.ONSITE
            if "stacjonarn" in modes
            else RemoteMode.UNKNOWN
        )
        levels = " ".join(p.get("levels") or []).lower() or raw.title.lower()
        seniority = next((s for key, s in PRACUJ_SENIORITY if key in levels), Seniority.UNKNOWN)
        if sal and sal.contract == ContractType.OTHER and "b2b" in contracts.lower():
            sal.contract = ContractType.B2B
        return JobListing(
            board=self.board,
            external_id=raw.external_id,
            url=raw.url,
            apply_url=p.get("apply_url") or raw.url,
            title=raw.title,
            company=raw.company,
            description=raw.description or p.get("snippet", ""),
            seniority=seniority,
            remote=remote,
            locations=[p["location"]] if p.get("location") else [],
            required_skills=p.get("expected") or p.get("technologies") or [],
            nice_to_have_skills=p.get("optional") or [],
            salaries=[sal] if sal else [],
            published_at=parse_dt(p.get("published")),
        )
