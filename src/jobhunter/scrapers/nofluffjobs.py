"""NoFluffJobs — JSON search API (verified 2026-10).

``POST https://nofluffjobs.com/api/search/posting?page=N&salaryCurrency=PLN&salaryPeriod=month&region=pl``
  Content-Type: application/infiniteSearch+json
  body: {"criteriaSearch": {"requirement": [...], "category": [...], "seniority": [...]}, "page": N}
  -> {"postings": [...], "totalCount", "totalPages"}
Detail: ``GET /api/posting/{id}`` -> requirements.{musts,nices,description}, details.description, apply.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import ClassVar

from jobhunter.models import Board, ContractType, JobListing, RawListing, RemoteMode, SalaryRange, Seniority
from jobhunter.scrapers.base import ApiScraper
from jobhunter.utils import html_to_text, parse_dt

CONTRACT_MAP = {
    "b2b": ContractType.B2B,
    "permanent": ContractType.UOP,
    "zlecenie": ContractType.UZ,
    "uod": ContractType.UOD,
}
SENIORITY_MAP = {
    "trainee": Seniority.INTERN,
    "junior": Seniority.JUNIOR,
    "mid": Seniority.MID,
    "senior": Seniority.SENIOR,
    "expert": Seniority.LEAD,
}
REMOTE_MAP = {"remote": RemoteMode.REMOTE, "hybrid": RemoteMode.HYBRID, "onsite": RemoteMode.ONSITE}


class NoFluffJobsScraper(ApiScraper):
    board: ClassVar[Board] = Board.NOFLUFFJOBS
    base_url: ClassVar[str] = "https://nofluffjobs.com"

    async def search(self) -> AsyncIterator[RawListing]:
        cutoff = datetime.now(UTC) - timedelta(days=self.search_cfg.max_age_days)
        criteria = {
            "requirement": self.search_cfg.nfj_requirements,
            "category": self.search_cfg.nfj_categories,
            "seniority": list(self.search_cfg.seniority),
        }
        for page in range(1, self.search_cfg.max_pages_per_board + 1):
            data = await self.http.post_json(
                f"{self.base_url}/api/search/posting",
                {"criteriaSearch": criteria, "page": page},
                params={  # "page" alone is ignored (always page 1) — the API pages by pageFrom/pageTo
                    "pageFrom": page,
                    "pageTo": page,
                    "salaryCurrency": "PLN",
                    "salaryPeriod": "month",
                    "region": "pl",
                    "language": "pl-PL",
                },
                headers={"Content-Type": "application/infiniteSearch+json", "Origin": self.base_url},
                referer=f"{self.base_url}/pl/praca-it/python",
            )
            postings = data.get("postings", [])
            for p in postings:
                # "renewed" bumps old posts to the top; we key freshness off the later of the two.
                ts = parse_dt(max(p.get("posted") or 0, p.get("renewed") or 0) or None)
                if ts and ts < cutoff:
                    continue
                yield RawListing(
                    board=self.board,
                    external_id=p["id"],
                    title=p["title"],
                    company=p["name"],
                    url=f"{self.base_url}/pl/job/{p['url']}",
                    payload=p,
                )
            if page >= (data.get("totalPages") or 1) or not postings:
                break

    async def enrich(self, raw: RawListing) -> RawListing:
        d = await self.http.get_json(f"{self.base_url}/api/posting/{raw.external_id}", referer=raw.url)
        req = d.get("requirements") or {}
        apply = d.get("apply") or {}
        raw.payload = {
            **raw.payload,
            "musts": [m["value"] for m in req.get("musts") or []],
            "nices": [n["value"] for n in req.get("nices") or []],
            "languages": [lang.get("code") for lang in req.get("languages") or [] if isinstance(lang, dict)],
            "apply_url": apply.get("referenceNumber") if apply.get("option") == "url" else None,
            "apply_form_controls": [c["name"] for c in (d.get("applyForm") or {}).get("controls", [])],
            "daily_tasks": (d.get("specs") or {}).get("dailyTasks") or [],
        }
        body = "\n".join(
            filter(
                None,
                [
                    (d.get("details") or {}).get("description"),
                    req.get("description"),
                ],
            )
        )
        tasks = "\n".join(f"• {t}" for t in raw.payload["daily_tasks"])
        raw.description = (html_to_text(body) + ("\nDaily tasks:\n" + tasks if tasks else "")).strip()
        return raw

    def parse(self, raw: RawListing) -> JobListing:
        p = raw.payload
        sal = p.get("salary") or {}
        salaries = []
        if sal.get("from") or sal.get("to"):
            contract = CONTRACT_MAP.get(sal.get("type", ""), ContractType.OTHER)
            salaries.append(
                SalaryRange(
                    min=sal.get("from"),
                    max=sal.get("to"),
                    currency=sal.get("currency", "PLN"),
                    period=(sal.get("period") or "month").lower(),
                    contract=contract,
                    is_gross=contract != ContractType.B2B,
                )
            )
        loc = p.get("location") or {}
        places = [pl.get("city") for pl in loc.get("places") or [] if pl.get("city")]
        remote = (
            RemoteMode.REMOTE
            if loc.get("fullyRemote")
            else REMOTE_MAP.get(loc.get("type", ""), RemoteMode.UNKNOWN)
        )
        levels = [SENIORITY_MAP.get(s.lower(), Seniority.UNKNOWN) for s in p.get("seniority") or []]
        seniority = min(levels, key=lambda s: list(Seniority).index(s)) if levels else Seniority.UNKNOWN
        return JobListing(
            board=self.board,
            external_id=raw.external_id,
            url=raw.url,
            apply_url=p.get("apply_url") or raw.url,
            title=raw.title,
            company=raw.company,
            description=raw.description or "",
            seniority=seniority,
            remote=remote,
            locations=sorted(set(places)),
            required_skills=p.get("musts") or ([p["technology"]] if p.get("technology") else []),
            nice_to_have_skills=p.get("nices") or [],
            salaries=salaries,
            published_at=parse_dt(p.get("posted")),
            languages=[x for x in p.get("languages") or [] if x],
        )
