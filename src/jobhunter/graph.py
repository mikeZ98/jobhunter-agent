"""LangGraph orchestration.

    START
      │
   collect ──► normalize ──► score ──┬──► prepare ×N  (Send fan-out: assets ‖ interview intel per job)
                                      │        │
                                      └────────┴──► inbox ──► publish ──┬──► autofill (HITL) ──► END
                                                                        └──────────────────────► END

* Every node is idempotent w.r.t. SQLite, so a crashed run can simply be re-run.
* ``prepare`` runs N parallel branches (bounded by a semaphore so a laptop Ollama isn't swamped);
  results fan back in through the ``items`` reducer.
* ``autofill`` is opt-in (``--apply``) and interactive; it never submits (see agents/autofill.py).
"""

from __future__ import annotations

import asyncio
import json
import operator
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from jobhunter.agents import assets as assets_agent
from jobhunter.agents import collector, scorer
from jobhunter.agents.autofill import review_loop
from jobhunter.agents.interview import InterviewIntel, generate_intel
from jobhunter.agents.linkedin_messaging import InboxMessage, RecruiterTriage, draft_replies, fetch_inbox
from jobhunter.config import Settings
from jobhunter.db import Database
from jobhunter.integrations.apple_notes import AppleNotes, AppleNotesError
from jobhunter.integrations.notes_render import Digest, DigestItem, render
from jobhunter.llm import Embedder, LLMClient
from jobhunter.models import CandidateProfile, JobListing, ListingStatus, ScoredListing, slugify
from jobhunter.salary import refresh_fx_rates
from jobhunter.scrapers import ScrapeInfra, build_scrapers
from jobhunter.scrapers.browser import BrowserManager
from jobhunter.telemetry import get_logger, traced_node
from jobhunter.workflow.schemas import (
    AutofillOutput,
    BoardStats,
    CollectOutput,
    InboxOutput,
    NormalizeOutput,
    PrepareOutput,
    PublishOutput,
    ScoreOutput,
)

log = get_logger("graph")


# ------------------------------------------------------------------ state


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {**a, **b}


class PipelineState(TypedDict, total=False):
    run_date: str
    boards: list[str] | None
    skip_collect: bool  # reuse the DB only: re-prepare failed jobs + publish (``run --skip-collect``)
    with_inbox: bool
    apply: bool
    listings: list[JobListing]
    scored: list[ScoredListing]
    shortlist: list[ScoredListing]
    items: Annotated[list[DigestItem], operator.add]  # fan-in target for parallel ``prepare`` branches
    inbox: list[tuple[InboxMessage, RecruiterTriage]]
    board_stats: Annotated[dict[str, Any], _merge]
    errors: Annotated[list[str], operator.add]
    note_id: str | None


class PrepareInput(TypedDict):
    item: ScoredListing
    rank: int
    run_date: str


@dataclass
class Deps:
    settings: Settings
    db: Database
    llm: LLMClient
    embedder: Embedder
    profile: CandidateProfile
    base_cv: str
    browser: BrowserManager
    notes: AppleNotes | None
    prepare_sem: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.prepare_sem = asyncio.Semaphore(max(1, self.settings.llm.max_concurrency))


# ------------------------------------------------------------------ graph


DailyGraph = CompiledStateGraph[PipelineState, None, PipelineState, PipelineState]


def build_daily_graph(deps: Deps) -> DailyGraph:
    s = deps.settings

    @traced_node("collect")
    async def collect(state: PipelineState) -> dict[str, Any]:
        if state.get("skip_collect"):
            log.info("collect: skipped (--skip-collect)")
            return CollectOutput(listings=[]).update()
        await asyncio.to_thread(refresh_fx_rates)
        infra = ScrapeInfra.create(s, deps.browser, on_antibot=deps.db.log_antibot)
        scrapers = build_scrapers(infra, state.get("boards"))
        listings, stats = await collector.collect(
            scrapers,
            deps.db,
            s.scraper.max_enrich_per_board,
            skip=lambda raw: collector.title_prefilter(raw, deps.profile),
        )
        log.info("collect: {} listings from {} boards", len(listings), len(scrapers))
        errors = [f"{b}: {st['error']}" for b, st in stats.items() if st.get("error")]
        return CollectOutput(
            listings=listings,
            board_stats={b: BoardStats.model_validate(st) for b, st in stats.items()},
            errors=errors,
        ).update()

    @traced_node("normalize")
    async def normalize(state: PipelineState) -> dict[str, Any]:
        jobs = [
            collector.normalize_listing(
                j, uop_to_b2b=s.scoring.uop_to_b2b_factor, hours_per_month=s.scoring.hours_per_month
            )
            for j in state.get("listings", [])
        ]
        kept, dropped = collector.apply_profile_prefilter(jobs, deps.profile, scorer.prefilter)
        kept = await collector.llm_fill_missing(
            kept,
            deps.llm,
            uop_to_b2b=s.scoring.uop_to_b2b_factor,
            hours_per_month=s.scoring.hours_per_month,
            concurrency=s.llm.max_concurrency,
        )
        # LLM extraction may reveal seniority/salary that fails the prefilter -> second pass is free.
        kept, dropped2 = collector.apply_profile_prefilter(kept, deps.profile, scorer.prefilter)
        dropped += dropped2
        deps.db.upsert_listings(jobs)
        deps.db.save_scores(ScoredListing(listing=j, rejected_reason=r) for j, r in dropped)
        log.info("normalize: {} kept, {} rejected by prefilter", len(kept), len(dropped))
        return NormalizeOutput(listings=kept).update()

    @traced_node("score")
    async def score(state: PipelineState) -> dict[str, Any]:
        scored: list[ScoredListing] = []
        if listings := state.get("listings", []):
            scored = await scorer.score_listings(
                listings,
                deps.profile,
                deps.base_cv,
                deps.llm,
                deps.embedder,
                s.scoring,
                s.llm.max_concurrency,
            )
            deps.db.upsert_listings(sl.listing for sl in scored)  # persist LLM-extracted facts
            deps.db.save_scores(scored)
        # Shortlisted earlier but never prepared (LLM failure / Ctrl-C): already "known" to the collector,
        # so without this they'd never get a cover letter.
        fresh = {sl.listing.source_key for sl in scored}
        retry = [
            sl
            for sl in deps.db.failed_prepares(limit=s.scoring.max_daily)
            if sl.listing.source_key not in fresh
        ]
        pool = sorted([*scored, *retry], key=lambda sl: sl.final_score, reverse=True)
        shortlist = scorer.select_daily(pool, s.scoring)
        log.info(
            "score: {} scored, {} retried from earlier runs, {} shortlisted (embeddings: {})",
            len(scored),
            len(retry),
            len(shortlist),
            deps.embedder.backend,
        )
        return ScoreOutput(scored=scored, shortlist=shortlist).update()

    def fan_out(state: PipelineState) -> list[Send] | str:
        shortlist = state.get("shortlist") or []
        if not shortlist:
            return "inbox"
        return [
            Send("prepare", PrepareInput(item=sl, rank=i, run_date=state["run_date"]))
            for i, sl in enumerate(shortlist, 1)
        ]

    @traced_node("prepare")
    async def prepare(inp: PrepareInput) -> dict[str, Any]:
        sl, job = inp["item"], inp["item"].listing
        out_dir = (
            s.output_dir
            / inp["run_date"]
            / f"{inp['rank']:02d}-{slugify(job.company)[:30]}-{slugify(job.title)[:40]}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        async with deps.prepare_sem:
            log.info(
                "prepare #{}: cover letter + CV + interview notes — {} @ {}",
                inp["rank"],
                job.title,
                job.company,
            )
            results = await asyncio.gather(
                assets_agent.generate_assets(sl, deps.profile, deps.base_cv, deps.llm),
                generate_intel(sl, deps.profile, deps.llm),
                return_exceptions=True,
            )
        assets_res, intel_res = results
        files: dict[str, Any] = {}
        errors: list[str] = []
        if isinstance(assets_res, BaseException):
            errors.append(f"assets: {assets_res!r}"[:300])
        else:
            letter, cv = assets_res
            if isinstance(letter, BaseException):
                errors.append(f"cover letter: {letter!r}"[:300])
            else:
                letter_md = letter.to_markdown(deps.profile)
                (out_dir / "cover_letter.md").write_text(letter_md)
                files["cover_letter_md"] = out_dir / "cover_letter.md"
                files["cover_letter_text"] = "\n\n".join(
                    [letter.greeting, *letter.paragraphs, letter.closing]
                )
                if await assets_agent.markdown_to_pdf(letter_md, out_dir / "cover_letter.pdf"):
                    files["cover_letter_pdf"] = out_dir / "cover_letter.pdf"
            if isinstance(cv, BaseException):
                errors.append(f"cv: {cv!r}"[:300])
            else:
                files |= await assets_agent.write_cv_files(cv, deps.profile, out_dir)
                files["cv_keywords"] = cv.mirrored_keywords
        intel: InterviewIntel | None = None
        if isinstance(intel_res, BaseException):
            errors.append(f"intel: {intel_res!r}"[:300])
        else:
            intel = intel_res
            (out_dir / "interview.md").write_text(intel.to_markdown(f"{job.title} @ {job.company}"))
            files["interview_md"] = out_dir / "interview.md"
        (out_dir / "listing.json").write_text(sl.model_dump_json(indent=2))

        deps.db.save_assets(
            job.source_key, {k: str(v) if isinstance(v, Path) else v for k, v in files.items()}
        )
        if "cover_letter_md" in files:  # no letter => stays SCRAPED, re-queued by failed_prepares
            try:
                deps.db.set_status(job.source_key, ListingStatus.PREPARED, note=str(out_dir))
            except Exception as e:
                log.debug("status unchanged for {}: {}", job.source_key, e)
        item = DigestItem(scored=sl, intel=intel, output_dir=str(out_dir), error="; ".join(errors) or None)
        log.info("prepare #{}: done{}", inp["rank"], f" with errors: {item.error}" if item.error else "")
        return PrepareOutput(items=[item], errors=[f"{job.company}: {e}" for e in errors]).update()

    @traced_node("inbox")
    async def inbox(state: PipelineState) -> dict[str, Any]:
        if not state.get("with_inbox"):
            return InboxOutput().update()
        if not (s.linkedin_li_at and s.linkedin_jsessionid):
            log.info("inbox: skipped — set LINKEDIN_LI_AT and LINKEDIN_JSESSIONID in .env to enable")
            return InboxOutput().update()
        try:
            msgs = await asyncio.to_thread(fetch_inbox, s)
            drafts = await draft_replies(msgs, deps.profile, deps.llm, deps.db, s.llm.max_concurrency)
            return InboxOutput(inbox=drafts).update()
        except Exception as e:
            log.warning("inbox skipped: {!r}", e)
            return InboxOutput(errors=[f"linkedin inbox: {e}"]).update()

    @traced_node("publish")
    async def publish(state: PipelineState) -> dict[str, Any]:
        items = sorted(state.get("items", []), key=lambda it: it.scored.final_score, reverse=True)
        digest = Digest(
            day=date.fromisoformat(state["run_date"]),
            items=items,
            board_stats=state.get("board_stats", {}),
            inbox=state.get("inbox", []),
        )
        html = render(digest)
        day_dir = s.output_dir / state["run_date"]
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "digest.html").write_text(html)
        log.info("publish: digest written to {}", day_dir / "digest.html")
        await deps.llm.release()  # give the RAM held by Ollama back before driving Notes
        note_id = None
        if deps.notes and s.notes.enabled:
            try:
                note_id = (await asyncio.to_thread(deps.notes.upsert, digest.title, html)).note_id
            except Exception as e:  # the digest is on disk; never lose the run at the last step
                msg = str(e) if isinstance(e, AppleNotesError) else repr(e)
                log.error("Apple Notes failed: {}", msg)
                return PublishOutput(errors=[f"apple notes: {msg}"]).update()
        return PublishOutput(note_id=note_id).update()

    def after_publish(state: PipelineState) -> str:
        return "autofill" if state.get("apply") else END

    @traced_node("autofill")
    async def autofill(state: PipelineState) -> dict[str, Any]:
        rows = deps.db.by_status(ListingStatus.PREPARED, limit=s.scoring.max_daily)
        if rows:
            async with deps.browser.persistent_profile() as ctx:
                await review_loop(ctx, deps.db, deps.profile, deps.llm, rows)
        return AutofillOutput().update()

    g = StateGraph(PipelineState)
    g.add_node("collect", collect)
    g.add_node("normalize", normalize)
    g.add_node("score", score)
    g.add_node("prepare", prepare)  # type: ignore[arg-type]  # Send payload != PipelineState by design
    g.add_node("inbox", inbox)
    g.add_node("publish", publish)
    g.add_node("autofill", autofill)
    g.add_edge(START, "collect")
    g.add_edge("collect", "normalize")
    g.add_edge("normalize", "score")
    g.add_conditional_edges("score", fan_out, ["prepare", "inbox"])
    g.add_edge("prepare", "inbox")
    g.add_edge("inbox", "publish")
    g.add_conditional_edges("publish", after_publish, ["autofill", END])
    g.add_edge("autofill", END)
    return g.compile()


def load_profile(settings: Settings) -> tuple[CandidateProfile, str]:
    if not settings.profile_path.exists():
        raise FileNotFoundError(f"{settings.profile_path} missing — run `jobhunter init` and fill it in")
    profile = CandidateProfile.model_validate(json.loads(settings.profile_path.read_text()))
    base_cv = settings.base_cv_path.read_text() if settings.base_cv_path.exists() else ""
    if not base_cv:
        base_cv = re.sub(r"\s+", " ", profile.model_dump_json())
    return profile, base_cv
