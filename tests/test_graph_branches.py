"""Every conditional branch of the daily graph, deterministically: fake boards, fake LLM, fake integrations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from jobhunter import graph as graph_mod
from jobhunter.agents.linkedin_messaging import InboxMessage, Intent, RecruiterTriage
from jobhunter.graph import Deps, build_daily_graph, load_profile
from jobhunter.integrations.apple_notes import AppleNotesError
from jobhunter.models import ListingStatus
from tests.test_graph import FakeBoard, FakeEmbedder, FakeLLM

RUN_DATE = date.today().isoformat()


class CrashingBoard(FakeBoard):
    async def search(self) -> AsyncIterator[Any]:
        raise RuntimeError("board down")
        yield  # pragma: no cover


class FakeBrowser:
    def __init__(self) -> None:
        self.opened = 0

    @asynccontextmanager
    async def persistent_profile(self) -> AsyncIterator[str]:
        self.opened += 1
        yield "ctx"


class FakeNotes:
    def __init__(self, error: Exception | None = None) -> None:
        self.error, self.calls = error, 0

    def upsert(self, title: str, html: str) -> SimpleNamespace:
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(note_id="note-1")


class CountingLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.released = 0

    async def release(self) -> None:
        self.released += 1


@pytest.fixture
def harness(settings, db, monkeypatch):
    """(deps, llm, boards_seen); boards come from ``harness_boards`` (default: one healthy FakeBoard)."""
    boards_seen: list[Any] = []
    board_cls: list[type[FakeBoard]] = [FakeBoard]

    def build_scrapers(infra, boards):
        boards_seen.append(boards)
        return [cls(settings.search) for cls in board_cls]

    monkeypatch.setattr(graph_mod, "build_scrapers", build_scrapers)
    monkeypatch.setattr(graph_mod, "refresh_fx_rates", lambda: None)
    settings.scoring.min_daily, settings.scoring.min_score = 1, 0
    profile, base_cv = load_profile(settings)
    llm = CountingLLM()
    deps = Deps(settings, db, llm, FakeEmbedder(), profile, base_cv, browser=FakeBrowser(), notes=None)  # type: ignore[arg-type]
    return SimpleNamespace(
        deps=deps, llm=llm, boards_seen=boards_seen, board_cls=board_cls, db=db, s=settings
    )


async def _run(h, **inp: Any) -> dict[str, Any]:
    return await build_daily_graph(h.deps).ainvoke({"run_date": RUN_DATE, **inp})


# ------------------------------------------------------------------ collect / score / fan-out


@pytest.mark.asyncio
async def test_board_filter_reaches_the_scrapers(harness):
    await _run(harness, boards=["justjoinit"])
    assert harness.boards_seen == [["justjoinit"]]


@pytest.mark.asyncio
async def test_crashed_board_is_reported_and_the_run_completes(harness):
    harness.board_cls[:] = [CrashingBoard]
    out = await _run(harness)
    assert out["board_stats"]["justjoinit"]["error"] == "RuntimeError('board down')"
    assert out["errors"] == ["justjoinit: RuntimeError('board down')"]
    assert out["items"] == [] and "CoverLetter" not in harness.llm.calls  # fan_out went straight to inbox
    assert (harness.s.output_dir / RUN_DATE / "digest.html").exists()


@pytest.mark.asyncio
async def test_skip_collect_on_an_empty_db_publishes_an_empty_digest(harness):
    out = await _run(harness, skip_collect=True)
    assert harness.boards_seen == [] and harness.llm.calls == []
    assert out["listings"] == [] and out["shortlist"] == []
    assert (harness.s.output_dir / RUN_DATE / "digest.html").exists()


# ------------------------------------------------------------------ prepare: critic outcomes


@pytest.mark.asyncio
async def test_unverified_drafts_are_retried_on_the_next_run(harness):
    harness.llm.broken = {"CriticVerdict"}  # the critic's model is down
    out = await _run(harness)
    assert out["quarantined"] == []  # not the draft's fault...
    assert all("unverified (critic unavailable)" in (it.error or "") for it in out["items"])
    assert harness.db.by_status(ListingStatus.PREPARED) == []  # ...but nothing unverified ships
    assert len(harness.db.failed_prepares()) == 3  # ...so it is retried

    harness.llm.broken.clear()
    out2 = await _run(harness, skip_collect=True)
    assert len(out2["items"]) == 3 and not any(it.error for it in out2["items"])
    assert len(harness.db.by_status(ListingStatus.PREPARED)) == 3


@pytest.mark.asyncio
async def test_disabled_critic_ships_unchecked_drafts(harness):
    harness.s.critic.enabled = False
    harness.llm.lie = True
    out = await _run(harness)
    assert "CriticVerdict" not in harness.llm.calls
    assert out["quarantined"] == [] and len(harness.db.by_status(ListingStatus.PREPARED)) == 3


@pytest.mark.asyncio
async def test_deterministic_critic_alone_still_catches_invented_figures(harness):
    harness.s.critic.llm_review = False
    harness.llm.lie = True
    out = await _run(harness)
    assert "CriticVerdict" not in harness.llm.calls
    assert len(out["quarantined"]) == 3


# ------------------------------------------------------------------ inbox


@pytest.mark.asyncio
async def test_inbox_without_linkedin_cookies_is_skipped(harness, monkeypatch):
    monkeypatch.setattr(graph_mod, "fetch_inbox", lambda s: pytest.fail("must not fetch"))
    harness.s.linkedin_li_at = harness.s.linkedin_jsessionid = None
    out = await _run(harness, with_inbox=True, skip_collect=True)
    assert out["inbox"] == []


@pytest.mark.asyncio
async def test_inbox_drafts_replies(harness, monkeypatch):
    harness.s.linkedin_li_at, harness.s.linkedin_jsessionid = SecretStr("a"), SecretStr("b")
    msg = InboxMessage(conversation_id="c1", sender="Recruiter", text="Hi, Python role?")
    triage = RecruiterTriage(
        is_recruiter_outreach=True, intent=Intent.JOB_OFFER, fit_score=80, language="en", reply="Hi!"
    )
    monkeypatch.setattr(graph_mod, "fetch_inbox", lambda s: [msg])

    async def draft_replies(msgs, profile, llm, db, concurrency):
        return [(m, triage) for m in msgs]

    monkeypatch.setattr(graph_mod, "draft_replies", draft_replies)
    out = await _run(harness, with_inbox=True, skip_collect=True)
    assert out["inbox"] == [(msg, triage)]


@pytest.mark.asyncio
async def test_inbox_contract_rejects_untyped_replies(harness, monkeypatch):
    """A drafting bug returning raw dicts is caught at the inbox node, not when the digest renders."""
    harness.s.linkedin_li_at, harness.s.linkedin_jsessionid = SecretStr("a"), SecretStr("b")
    monkeypatch.setattr(graph_mod, "fetch_inbox", lambda s: ["raw"])

    async def draft_replies(msgs, profile, llm, db, concurrency):
        return [({"text": "raw"}, {"reply": "Hi"})]

    monkeypatch.setattr(graph_mod, "draft_replies", draft_replies)
    out = await _run(harness, with_inbox=True, skip_collect=True)
    assert out["inbox"] == [] and out["errors"][0].startswith("linkedin inbox: 2 validation errors")


@pytest.mark.asyncio
async def test_inbox_failure_is_an_error_not_a_crash(harness, monkeypatch):
    harness.s.linkedin_li_at, harness.s.linkedin_jsessionid = SecretStr("a"), SecretStr("b")

    def boom(s):
        raise RuntimeError("cookie expired")

    monkeypatch.setattr(graph_mod, "fetch_inbox", boom)
    out = await _run(harness, with_inbox=True, skip_collect=True)
    assert out["inbox"] == [] and out["errors"] == ["linkedin inbox: cookie expired"]


# ------------------------------------------------------------------ publish


@pytest.mark.asyncio
async def test_publish_updates_apple_notes_and_frees_local_models(harness):
    harness.deps.notes = FakeNotes()
    out = await _run(harness, skip_collect=True)
    assert out["note_id"] == "note-1" and harness.llm.released == 1


@pytest.mark.asyncio
async def test_publish_respects_notes_disabled(harness):
    harness.deps.notes = notes = FakeNotes()
    harness.s.notes.enabled = False
    out = await _run(harness, skip_collect=True)
    assert out["note_id"] is None and notes.calls == 0


@pytest.mark.asyncio
async def test_apple_notes_failure_keeps_the_digest(harness):
    harness.deps.notes = FakeNotes(error=AppleNotesError("Notes is locked"))
    out = await _run(harness, skip_collect=True)
    assert out["note_id"] is None and out["errors"] == ["apple notes: Notes is locked"]
    assert (harness.s.output_dir / RUN_DATE / "digest.html").exists()


# ------------------------------------------------------------------ autofill (after_publish)


@pytest.fixture
def review_calls(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    async def review_loop(ctx, db, profile, llm, rows):
        calls.append([r["title"] for r in rows])

    monkeypatch.setattr(graph_mod, "review_loop", review_loop)
    return calls


@pytest.mark.asyncio
async def test_without_apply_autofill_never_runs(harness, review_calls):
    await _run(harness)
    assert review_calls == [] and harness.deps.browser.opened == 0


@pytest.mark.asyncio
async def test_apply_reviews_prepared_listings(harness, review_calls):
    await _run(harness, apply=True)
    assert len(review_calls) == 1 and len(review_calls[0]) == 3
    assert harness.deps.browser.opened == 1


@pytest.mark.asyncio
async def test_apply_with_nothing_prepared_does_not_open_a_browser(harness, review_calls):
    harness.llm.lie = True  # every letter quarantined -> nothing PREPARED -> nothing to fill
    await _run(harness, apply=True)
    assert review_calls == [] and harness.deps.browser.opened == 0
