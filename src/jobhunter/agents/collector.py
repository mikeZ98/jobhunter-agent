"""Agent 1 — Collector & Normalizer.

collect:   all boards concurrently (each board strictly sequential + rate-limited), dedupe against
           SQLite (board key + cross-board fingerprint) *before* fetching detail pages, so a daily
           run only spends requests on genuinely new postings.
normalize: deterministic salary/seniority/remote normalization; an LLM extraction pass only for
           listings that survived the cheap prefilter and still lack structured facts.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from jobhunter.agents.scorer import REACHABLE
from jobhunter.db import Database
from jobhunter.llm import LLMClient, Task
from jobhunter.models import (
    Board,
    CandidateProfile,
    ContractType,
    JobListing,
    RawListing,
    RemoteMode,
    Seniority,
)
from jobhunter.salary import normalize as normalize_salary
from jobhunter.salary import parse_salary_text
from jobhunter.scrapers import Scraper
from jobhunter.scrapers.antiban import BlockedError
from jobhunter.telemetry import Progress, get_logger

log = get_logger("agent.collector")

# When the same role appears on several boards keep the richest source.
BOARD_PRIORITY = [Board.JUSTJOINIT, Board.NOFLUFFJOBS, Board.ROCKETJOBS, Board.PRACUJ, Board.LINKEDIN]


RawFilter = Callable[[RawListing], str | None]


async def _collect_board(
    scraper: Scraper, db: Database, max_enrich: int, skip: RawFilter | None = None
) -> tuple[list[JobListing], dict[str, Any]]:
    board = scraper.board.value
    stats: dict[str, Any] = {
        "seen": 0, "known": 0, "duplicates": 0, "filtered": 0, "resumed": 0, "enriched": 0, "error": None
    }  # fmt: skip
    listings: list[JobListing] = []
    try:
        async with scraper:
            log.info("{}: searching…", board)
            raws: list[RawListing] = []
            async for r in scraper.search():
                raws.append(r)
                if len(raws) % 50 == 0:
                    log.info("{}: {} listed so far…", board, len(raws))
            stats["seen"] = len(raws)
            known = db.known_keys(r.source_key for r in raws)
            known_fp = db.known_fingerprints(r.fingerprint for r in raws)
            pending = db.pending_listings(r.source_key for r in raws)
            fresh: list[RawListing] = []
            fps: set[str] = set()
            for r in raws:
                if r.source_key in known or r.fingerprint in known_fp:
                    stats["known"] += 1
                elif r.fingerprint in fps:  # NFJ/JJIT list one posting per location
                    stats["duplicates"] += 1
                elif skip and skip(r):  # cheap title check before spending a detail request
                    stats["filtered"] += 1
                else:
                    fps.add(r.fingerprint)
                    if r.source_key in pending:
                        listings.append(pending[r.source_key])
                        stats["resumed"] += 1
                    else:
                        fresh.append(r)
            todo = fresh[:max_enrich]
            log.info(
                "{}: {} listed → {} to fetch ({} known, {} duplicates, {} filtered by title, "
                "{} resumed from an interrupted run, {} over the per-board cap)",
                board, len(raws), len(todo), stats["known"], stats["duplicates"], stats["filtered"],
                stats["resumed"], len(fresh) - len(todo),
            )  # fmt: skip
            progress = Progress(log, f"{board} details", len(todo), every=10)
            for raw in todo:
                try:
                    raw = await scraper.enrich(raw)
                    stats["enriched"] += 1
                except BlockedError as e:
                    # Keep what we already have; listings not yet enriched are retried tomorrow.
                    stats["error"] = f"blocked during enrich: {e.kind}"
                    break
                except Exception as e:
                    log.warning("{} enrich failed for {}: {!r}", board, raw.url, e)
                try:
                    job = scraper.parse(raw)
                    listings.append(job)
                    db.upsert_listings([job])  # saved immediately: Ctrl-C mid-run loses nothing
                except Exception as e:
                    log.warning("{} parse failed for {}: {!r}", board, raw.external_id, e)
                progress.step()
    except BlockedError as e:
        stats["error"] = f"blocked: {e.kind}"
        log.error("{} aborted: {}", board, e)
    except Exception as e:
        stats["error"] = repr(e)
        log.exception("{} crashed", board)
    log.info(
        "{}: done — kept {} ({} fetched, {} resumed){}",
        board,
        len(listings),
        stats["enriched"],
        stats["resumed"],
        f" [error: {stats['error']}]" if stats["error"] else "",
    )
    return listings, stats


async def collect(
    scrapers: list[Scraper], db: Database, max_enrich: int, skip: RawFilter | None = None
) -> tuple[list[JobListing], dict[str, Any]]:
    results = await asyncio.gather(*(_collect_board(s, db, max_enrich, skip) for s in scrapers))
    by_fp: dict[str, JobListing] = {}
    for listings, _ in results:
        for job in listings:
            prev = by_fp.get(job.fingerprint)
            if prev is None or BOARD_PRIORITY.index(job.board) < BOARD_PRIORITY.index(prev.board):
                by_fp[job.fingerprint] = job
    stats = {s.board.value: st for s, (_, st) in zip(scrapers, results, strict=True)}
    return list(by_fp.values()), stats


# ------------------------------------------------------------------ normalization

SENIORITY_PATTERNS = [
    (re.compile(r"\b(intern|internship|staż|stażysta|praktyk|trainee)\b", re.I), Seniority.INTERN),
    (re.compile(r"\b(junior|jr\.?|entry)\b", re.I), Seniority.JUNIOR),
    (re.compile(r"\b(lead|principal|staff|head|architect|manager|kierownik)\b", re.I), Seniority.LEAD),
    (re.compile(r"\b(senior|sr\.?|starszy|expert|ekspert)\b", re.I), Seniority.SENIOR),
    (re.compile(r"\b(mid|regular|middle)\b", re.I), Seniority.MID),
]
REMOTE_PATTERNS = [
    (re.compile(r"\b(fully remote|100% remote|praca zdalna|zdalnie|remote)\b", re.I), RemoteMode.REMOTE),
    (re.compile(r"\b(hybrid|hybrydow\w*)\b", re.I), RemoteMode.HYBRID),
    (re.compile(r"\b(on-?site|stacjonarn\w*|w biurze)\b", re.I), RemoteMode.ONSITE),
]


def infer_seniority(title: str, current: Seniority) -> Seniority:
    # Title wins over board metadata: "Senior Python Dev" tagged "mid" by a recruiter is still senior.
    for pat, level in SENIORITY_PATTERNS:
        if pat.search(title):
            return level
    return current


def title_prefilter(raw: RawListing, profile: CandidateProfile) -> str | None:
    """Rejection reason from title/company alone (no detail page needed), or None to keep."""
    level = infer_seniority(raw.title, Seniority.UNKNOWN)
    if level not in REACHABLE.get(profile.seniority, set(Seniority)):
        return f"seniority {level.value}"
    title, company = raw.title.lower(), raw.company.lower()
    if any(c.lower() in company for c in profile.excluded_companies):
        return "excluded company"
    if kw := next((k for k in profile.excluded_keywords if k.lower() in f"{title} "), None):
        return f"excluded keyword '{kw}'"
    return None


def normalize_listing(job: JobListing, *, uop_to_b2b: float, hours_per_month: int) -> JobListing:
    seniority = infer_seniority(job.title, job.seniority)
    remote = job.remote
    if remote == RemoteMode.UNKNOWN:
        hay = f"{job.title} {' '.join(job.locations)} {job.description[:3000]}"
        remote = next((m for p, m in REMOTE_PATTERNS if p.search(hay)), RemoteMode.UNKNOWN)
    salaries = job.salaries
    if not salaries and job.description:
        m = re.search(r"(?:salary|wynagrodzenie|stawka|widełki|rate)[^\n]{0,120}", job.description, re.I)
        if m and (s := parse_salary_text(m.group(0))):
            salaries = [s]
    salaries = [normalize_salary(s, uop_to_b2b=uop_to_b2b, hours_per_month=hours_per_month) for s in salaries]
    skills = list(dict.fromkeys(s.strip() for s in job.required_skills if s.strip()))
    nice = [
        s for s in dict.fromkeys(x.strip() for x in job.nice_to_have_skills if x.strip()) if s not in skills
    ]
    return job.model_copy(
        update={
            "seniority": seniority,
            "remote": remote,
            "salaries": salaries,
            "required_skills": skills,
            "nice_to_have_skills": nice,
        }
    )


class ExtractedFacts(BaseModel):
    """LLM extraction schema for listings without structured metadata (LinkedIn, some Pracuj)."""

    required_skills: list[str] = Field(max_length=20, description="Must-have technologies, canonical names.")
    nice_to_have_skills: list[str] = Field(default_factory=list, max_length=20)
    seniority: Seniority
    remote: RemoteMode
    salary_text: str | None = Field(None, description="Verbatim salary range from the text, or null.")
    contract: ContractType | None = None


EXTRACT_SYSTEM = (
    "You extract structured facts from Polish/English IT job postings. Use only information present "
    "in the text. Canonical skill names (e.g. 'PostgreSQL' not 'postgres'). If unknown, use 'unknown'/null."
)


async def llm_fill_missing(
    jobs: list[JobListing], llm: LLMClient, *, uop_to_b2b: float, hours_per_month: int, concurrency: int
) -> list[JobListing]:
    sem = asyncio.Semaphore(concurrency)
    need = [
        j
        for j in jobs
        if not (j.required_skills and j.salaries and j.seniority != Seniority.UNKNOWN)
        and len(j.description) >= 200
    ]
    if not need:
        return jobs
    progress = Progress(log, "LLM fact extraction", len(need), every=5)
    need_keys = {j.source_key for j in need}

    async def one(job: JobListing) -> JobListing:
        if job.source_key not in need_keys:
            return job
        try:
            return await _extract(job)
        finally:
            progress.step()

    async def _extract(job: JobListing) -> JobListing:
        async with sem:
            try:
                facts = await llm.structured(
                    Task.EXTRACT,
                    ExtractedFacts,
                    EXTRACT_SYSTEM,
                    f"TITLE: {job.title}\nCOMPANY: {job.company}\n\n{job.description[:9000]}",
                    max_tokens=600,
                )
            except Exception as e:
                log.warning("extraction failed for {}: {!r}", job.url, e)
                return job
        update: dict[str, Any] = {}
        if not job.required_skills:
            update["required_skills"] = facts.required_skills
            update["nice_to_have_skills"] = job.nice_to_have_skills or facts.nice_to_have_skills
        if job.seniority == Seniority.UNKNOWN:
            update["seniority"] = facts.seniority
        if job.remote == RemoteMode.UNKNOWN:
            update["remote"] = facts.remote
        if (
            not job.salaries
            and facts.salary_text
            and (
                s := parse_salary_text(
                    facts.salary_text, contract_hint=facts.contract.value if facts.contract else None
                )
            )
        ):
            update["salaries"] = [normalize_salary(s, uop_to_b2b=uop_to_b2b, hours_per_month=hours_per_month)]
        return job.model_copy(update=update)

    return list(await asyncio.gather(*(one(j) for j in jobs)))


def apply_profile_prefilter(
    jobs: list[JobListing], profile: CandidateProfile, prefilter: Any
) -> tuple[list[JobListing], list[tuple[JobListing, str]]]:
    kept, dropped = [], []
    for j in jobs:
        reason = prefilter(j, profile)
        (dropped.append((j, reason)) if reason else kept.append(j))
    return kept, dropped
