"""LinkedIn — logged-out "jobs-guest" endpoints (HTML fragments, no auth, verified 2026-10).

Search: ``/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords=..&location=..&f_E=2,3&f_TPR=r86400&start=0``
Detail: ``/jobs-guest/jobs/api/jobPosting/{id}``

Deliberately *not* using the logged-in Voyager API for search: it ties scraping to your account.
Volume is kept very low (LinkedIn returns HTTP 999 / 429 quickly) and failures are non-fatal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import ClassVar
from urllib.parse import quote_plus, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from jobhunter.models import Board, JobListing, RawListing, RemoteMode, Seniority
from jobhunter.scrapers.base import ApiScraper
from jobhunter.utils import html_to_text, parse_dt

# f_E experience filter: 1 internship, 2 entry, 3 associate, 4 mid-senior, 5 director, 6 executive
EXPERIENCE = {"junior": "2", "mid": "3,4"}
LEVEL_MAP = {
    "internship": Seniority.INTERN,
    "entry level": Seniority.JUNIOR,
    "associate": Seniority.MID,
    "mid-senior level": Seniority.MID,  # LinkedIn lumps mid+senior; title inference decides
    "director": Seniority.LEAD,
    "executive": Seniority.LEAD,
}


class LinkedInGuestScraper(ApiScraper):
    board: ClassVar[Board] = Board.LINKEDIN
    base_url: ClassVar[str] = "https://www.linkedin.com"
    page_size: ClassVar[int] = 25  # fixed by LinkedIn (~10 per HTML fragment; start advances by 25)

    async def search(self) -> AsyncIterator[RawListing]:
        f_e = ",".join(sorted({c for s in self.search_cfg.seniority for c in EXPERIENCE[s].split(",")}))
        seen: set[str] = set()
        for kw in self.search_cfg.keywords:
            for page in range(min(2, self.search_cfg.max_pages_per_board)):
                html = await self.http.get_text(
                    f"{self.base_url}/jobs-guest/jobs/api/seeMoreJobPostings/search",
                    params={
                        "keywords": kw,
                        "location": self.search_cfg.linkedin_location,
                        "f_E": f_e,
                        "f_TPR": f"r{86400 * self.search_cfg.max_age_days}",
                        "start": page * self.page_size,
                    },
                    referer=f"{self.base_url}/jobs/search?keywords={quote_plus(kw)}",
                )
                cards = list(self._parse_cards(html))
                for raw in cards:
                    if raw.external_id not in seen:
                        seen.add(raw.external_id)
                        yield raw
                if len(cards) < 10:
                    break

    def _parse_cards(self, html: str) -> list[RawListing]:
        out = []
        soup = BeautifulSoup(html, "lxml")
        for card in soup.select("div.base-search-card[data-entity-urn]"):
            job_id = str(card["data-entity-urn"]).rsplit(":", 1)[-1]
            link = card.select_one("a.base-card__full-link")
            title = card.select_one(".base-search-card__title")
            company = card.select_one(".base-search-card__subtitle")
            location = card.select_one(".job-search-card__location")
            posted = card.select_one("time")
            if not (link and title and company):
                continue
            parts = urlsplit(str(link["href"]))
            out.append(
                RawListing(
                    board=self.board,
                    external_id=job_id,
                    url=urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")),  # drop tracking params
                    title=title.get_text(strip=True),
                    company=company.get_text(strip=True),
                    payload={
                        "location": location.get_text(strip=True) if location else "",
                        "posted": posted.get("datetime") if posted else None,
                    },
                )
            )
        return out

    async def enrich(self, raw: RawListing) -> RawListing:
        html = await self.http.get_text(
            f"{self.base_url}/jobs-guest/jobs/api/jobPosting/{raw.external_id}", referer=raw.url
        )
        soup = BeautifulSoup(html, "lxml")
        desc = soup.select_one(".show-more-less-html__markup") or soup.select_one(".description__text")
        criteria = {
            (h.get_text(strip=True) if (h := li.select_one("h3")) else "").lower(): (
                v.get_text(strip=True) if (v := li.select_one("span")) else ""
            )
            for li in soup.select("li.description__job-criteria-item")
        }
        raw.description = html_to_text(str(desc)) if desc else ""
        # Guests only get a "sign in to apply" link; the job view URL works in the logged-in autofill profile.
        raw.payload |= {"criteria": criteria}
        return raw

    def parse(self, raw: RawListing) -> JobListing:
        p = raw.payload
        loc = p.get("location", "")
        level = (p.get("criteria") or {}).get("seniority level", "").lower()
        text = f"{raw.title} {loc}".lower()
        remote = (
            RemoteMode.REMOTE
            if "remote" in text or "zdal" in text
            else RemoteMode.HYBRID
            if "hybrid" in text
            else RemoteMode.UNKNOWN
        )
        return JobListing(
            board=self.board,
            external_id=raw.external_id,
            url=raw.url,
            apply_url=raw.url,
            title=raw.title,
            company=raw.company,
            description=raw.description or "",
            seniority=LEVEL_MAP.get(level, Seniority.UNKNOWN),
            remote=remote,
            locations=[loc.split(",")[0]] if loc else [],
            published_at=parse_dt(p.get("posted")),
            # LinkedIn exposes no structured skills/salary to guests -> the normalizer's LLM pass fills them.
        )
