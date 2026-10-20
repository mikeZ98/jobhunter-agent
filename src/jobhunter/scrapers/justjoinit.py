"""JustJoin.it + RocketJobs — both run on the same "candidate-api" platform.

Endpoint (verified 2026-10): ``GET https://justjoin.it/api/candidate-api/offers``
  ?categories=python&experienceLevels=junior&experienceLevels=mid&from=0&itemsCount=100
  -> {"data": [...], "meta": {"totalItems", "next": {"cursor", "itemsCount"}}}
Detail: ``GET /api/candidate-api/offers/{slug}`` -> adds ``body`` (HTML description), ``applyUrl``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

from jobhunter.models import Board, ContractType, JobListing, RawListing, RemoteMode, SalaryRange, Seniority
from jobhunter.scrapers.base import ApiScraper
from jobhunter.utils import html_to_text, parse_dt

CONTRACT_MAP = {
    "b2b": ContractType.B2B,
    "permanent": ContractType.UOP,
    "mandate_contract": ContractType.UZ,
    "contract": ContractType.UOD,
    "specific-task": ContractType.UOD,
}
REMOTE_MAP = {"remote": RemoteMode.REMOTE, "hybrid": RemoteMode.HYBRID, "office": RemoteMode.ONSITE}
SENIORITY_MAP = {
    "junior": Seniority.JUNIOR,
    "mid": Seniority.MID,
    "senior": Seniority.SENIOR,
    "c_level": Seniority.LEAD,
    "manager": Seniority.LEAD,
}


class JustJoinItScraper(ApiScraper):
    board: ClassVar[Board] = Board.JUSTJOINIT
    base_url: ClassVar[str] = "https://justjoin.it"
    page_size: ClassVar[int] = 100

    def _categories(self) -> list[str]:
        return self.search_cfg.jjit_categories

    async def search(self) -> AsyncIterator[RawListing]:
        cutoff = datetime.now(UTC) - timedelta(days=self.search_cfg.max_age_days)
        seen: set[str] = set()
        for category in self._categories():
            cursor = 0
            for _ in range(self.search_cfg.max_pages_per_board):
                params: list[tuple[str, Any]] = [("categories", category)]
                params += [("experienceLevels", lvl) for lvl in self.search_cfg.seniority]
                params += [
                    ("from", cursor),
                    ("itemsCount", self.page_size),
                    ("sortBy", "publishedAt"),
                    ("orderBy", "descending"),
                ]
                data = await self.http.get_json(
                    f"{self.base_url}/api/candidate-api/offers",
                    params=params,
                    referer=f"{self.base_url}/job-offers/all-locations/{category}",
                )
                offers = data.get("data", [])
                stale = 0
                for o in offers:
                    published = parse_dt(o.get("publishedAt") or o.get("lastPublishedAt"))
                    if published and published < cutoff:
                        stale += 1
                        continue
                    if o["slug"] in seen:
                        continue
                    seen.add(o["slug"])
                    yield RawListing(
                        board=self.board,
                        external_id=o["slug"],
                        title=o["title"],
                        company=o["companyName"],
                        url=f"{self.base_url}/job-offer/{o['slug']}",
                        payload=o,
                    )
                nxt = (data.get("meta") or {}).get("next") or {}
                # Sorted by date: a page that's mostly stale means everything after it is too.
                if not offers or not nxt.get("itemsCount") or stale > len(offers) // 2:
                    break
                cursor = nxt["cursor"]

    async def enrich(self, raw: RawListing) -> RawListing:
        detail = await self.http.get_json(
            f"{self.base_url}/api/candidate-api/offers/{raw.external_id}", referer=raw.url
        )
        raw.payload = {**raw.payload, **{k: detail.get(k) for k in ("applyUrl", "companyUrl", "languages")}}
        raw.description = html_to_text(detail.get("body") or "")
        return raw

    def parse(self, raw: RawListing) -> JobListing:
        return parse_candidate_offer(self.board, raw)


def parse_candidate_offer(board: Board, raw: RawListing) -> JobListing:
    """Shared by JustJoinIT and RocketJobs (same candidate-api payload)."""
    o = raw.payload
    salaries = []
    for et in o.get("employmentTypes") or []:
        if et.get("currencySource") == "conversion":  # API adds FX-converted duplicates; keep original
            continue
        if et.get("from") is None and et.get("to") is None:
            continue
        # ``from``/``to`` are already monthly (JJIT multiplies hourly x168, daily x21); the advertised
        # rate is in ``fromPerUnit``/``toPerUnit`` with ``unit`` = Hour | Day | Month.
        salaries.append(
            SalaryRange(
                min=et.get("from"),
                max=et.get("to"),
                currency=(et.get("currency") or "PLN").upper(),
                period="month",
                contract=CONTRACT_MAP.get(et.get("type"), ContractType.OTHER),
                is_gross=et.get("gross"),
            )
        )
    cities = [loc["city"] for loc in o.get("locations") or [] if loc.get("city")] or [o.get("city") or ""]
    return JobListing(
        board=board,
        external_id=raw.external_id,
        url=raw.url,
        apply_url=o.get("applyUrl") or raw.url,
        title=raw.title,
        company=raw.company,
        description=raw.description or "",
        seniority=SENIORITY_MAP.get(o.get("experienceLevel", ""), Seniority.UNKNOWN),
        remote=REMOTE_MAP.get(o.get("workplaceType", ""), RemoteMode.UNKNOWN),
        locations=[c for c in cities if c],
        required_skills=[s["name"] for s in o.get("requiredSkills") or []],
        nice_to_have_skills=[s["name"] for s in o.get("niceToHaveSkills") or []],
        salaries=salaries,
        published_at=parse_dt(o.get("publishedAt")),
        languages=[lang.get("code", "") for lang in o.get("languages") or [] if isinstance(lang, dict)],
    )
