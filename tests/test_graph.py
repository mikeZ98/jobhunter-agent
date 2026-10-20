"""End-to-end LangGraph run with a fake board and a fake LLM (no network, no Ollama)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from typing import Any, ClassVar

import pytest

from jobhunter import graph as graph_mod
from jobhunter.agents.assets import CoverLetter, TailoredCV
from jobhunter.agents.interview import InterviewIntel
from jobhunter.graph import Deps, build_daily_graph, load_profile
from jobhunter.llm.embeddings import _hashed_bow
from jobhunter.models import Board, ListingStatus, MatchAssessment, RawListing, Seniority
from jobhunter.scrapers.base import Scraper
from tests.conftest import make_job

TITLES = [
    "Python Developer",
    "Senior Python Developer",
    "LLM Engineer",
    "Java Developer",
    "Python Backend Engineer",
]


class FakeBoard(Scraper):
    board: ClassVar[Board] = Board.JUSTJOINIT
    strategy: ClassVar[str] = "api"

    async def search(self) -> AsyncIterator[RawListing]:
        for i, t in enumerate(TITLES):
            yield RawListing(
                board=self.board, external_id=f"job-{i}", url=f"https://x/{i}", title=t, company=f"Company{i}"
            )

    def parse(self, raw: RawListing):
        return make_job(
            external_id=raw.external_id,
            url=raw.url,
            title=raw.title,
            company=raw.company,
            seniority=Seniority.MID,
            required_skills=["Python", "FastAPI", "LangGraph", "RAG"],
        )


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.broken: set[str] = set()  # response models that fail, like an out-of-memory Ollama

    async def release(self) -> None:
        pass

    async def structured(self, task, response_model, system, user, *, validation_context=None, **_: Any):
        self.calls.append(response_model.__name__)
        if response_model.__name__ in self.broken:
            raise RuntimeError("empty reply")
        ctx = validation_context or {}
        company = ctx.get("company", "Acme")
        if response_model is MatchAssessment:
            return MatchAssessment(score=82, strengths=["DocuMind RAG"], gaps=[], pitch="DocuMind ↔ RAG")
        if response_model is CoverLetter:
            p = (
                f"W {company} rozwijacie usługi na FastAPI — w DocuMind obniżyłem p95 z 9,1 s do 2,4 s "
                "dzięki hybrydowemu wyszukiwaniu i streamingowi odpowiedzi. "
            ) * 7
            return CoverLetter.model_validate(
                {
                    "subject": "s",
                    "greeting": "Dzień dobry,",
                    "closing": "Pozdrawiam",
                    "paragraphs": [p, p[:250], "Dostępny od listopada."],
                },
                context=ctx,
            )
        if response_model is TailoredCV:
            return TailoredCV.model_validate(
                {
                    "headline": "Python & GenAI Engineer",
                    "summary": "RAG + FastAPI w produkcji.",
                    "skills_primary": ["Python", "FastAPI", "LangGraph", "RAG"],
                    "projects": [
                        {"name": "DocuMind", "bullets": ["p95 9.1s → 2.4s", "eval harness w CI"]},
                        {"name": "Pricing API", "bullets": ["1.8k req/s", "zero-downtime migracja"]},
                    ],
                },
                context=ctx,
            )
        if response_model is InterviewIntel:
            q = {
                "question": "Jak ewaluować RAG?",
                "why_likely": "RAG w ofercie",
                "answer_outline": ["RAGAS", "golden set"],
            }
            return InterviewIntel.model_validate(
                {
                    "business_model": "prawdopodobnie SaaS B2B",
                    "product_guess": "platforma",
                    "elevator_pitch": "pitch",
                    "tech_deep_dives": ["asyncio", "pgvector", "LangGraph"],
                    "likely_questions": [
                        q | {"kind": k} for k in ["system_design", "framework", "llm_ml"] * 2
                    ],
                    "reverse_questions": [
                        "Jak mierzycie jakość odpowiedzi?",
                        "Kto ma on-call?",
                        "Jaki jest budżet na GPU?",
                    ],
                }
            )
        raise AssertionError(f"unexpected model {response_model}")


class FakeEmbedder:
    backend = "hashed-bow"

    async def embed(self, texts):
        return _hashed_bow(texts)


@pytest.mark.asyncio
async def test_daily_graph_end_to_end(settings, db, monkeypatch):
    monkeypatch.setattr(graph_mod, "build_scrapers", lambda infra, boards: [FakeBoard(settings.search)])
    monkeypatch.setattr(graph_mod, "refresh_fx_rates", lambda: None)
    settings.scoring.min_daily, settings.scoring.min_score = 1, 0
    profile, base_cv = load_profile(settings)
    llm = FakeLLM()
    deps = Deps(settings, db, llm, FakeEmbedder(), profile, base_cv, browser=None, notes=None)  # type: ignore[arg-type]
    g = build_daily_graph(deps)
    run_date = date.today().isoformat()
    out = await g.ainvoke({"run_date": run_date, "boards": None, "with_inbox": False, "apply": False})

    titles = sorted(it.scored.listing.title for it in out["items"])
    # Senior (seniority) and Java (excluded keyword) are dropped by the prefilter.
    assert titles == ["LLM Engineer", "Python Backend Engineer", "Python Developer"]
    assert not out["errors"], out["errors"]
    for it in out["items"]:
        d = settings.output_dir / run_date / it.output_dir.rsplit("/", 1)[-1]
        assert (d / "cover_letter.md").exists() and (d / "cv.md").exists() and (d / "cv.typ").exists()
        assert (d / "interview.md").read_text().count("## Pytania do zadania rekruterowi") == 1
        assert (d / "cv.pdf").exists()  # typst (python pkg) or Chromium fallback
    assert (settings.output_dir / run_date / "digest.html").read_text().startswith("<h1>Job Hunt")
    assert len(db.by_status(ListingStatus.PREPARED)) == 3
    # ...and both are dropped from the title alone, before any detail request, so never stored.
    assert out["board_stats"]["justjoinit"]["filtered"] == 2
    assert db._conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == 3

    # Second run: everything is known -> no new LLM work, empty digest, statuses untouched.
    llm.calls.clear()
    out2 = await g.ainvoke({"run_date": run_date, "boards": None, "with_inbox": False, "apply": False})
    assert out2.get("items", []) == [] and llm.calls == []
    assert len(db.by_status(ListingStatus.PREPARED)) == 3


@pytest.mark.asyncio
async def test_failed_prepare_is_retried_without_scraping(settings, db, monkeypatch):
    monkeypatch.setattr(graph_mod, "build_scrapers", lambda infra, boards: [FakeBoard(settings.search)])
    monkeypatch.setattr(graph_mod, "refresh_fx_rates", lambda: None)
    settings.scoring.min_daily, settings.scoring.min_score = 1, 0
    profile, base_cv = load_profile(settings)
    llm = FakeLLM()
    llm.broken = {"CoverLetter", "InterviewIntel"}
    deps = Deps(settings, db, llm, FakeEmbedder(), profile, base_cv, browser=None, notes=None)  # type: ignore[arg-type]
    g = build_daily_graph(deps)
    run_date = date.today().isoformat()
    out = await g.ainvoke({"run_date": run_date, "boards": None, "with_inbox": False, "apply": False})
    assert len(out["items"]) == 3 and all(it.error for it in out["items"])
    assert db.by_status(ListingStatus.PREPARED) == []

    # The jobs are now "known", so a normal re-run wouldn't collect them again; --skip-collect (and any
    # later run) picks the failed ones up from the DB and prepares them without re-judging.
    llm.broken.clear()
    llm.calls.clear()
    out2 = await g.ainvoke(
        {"run_date": run_date, "boards": None, "skip_collect": True, "with_inbox": False, "apply": False}
    )
    assert sorted(it.scored.listing.title for it in out2["items"]) == [
        "LLM Engineer",
        "Python Backend Engineer",
        "Python Developer",
    ]
    assert not any(it.error for it in out2["items"])
    assert "MatchAssessment" not in llm.calls
    assert len(db.by_status(ListingStatus.PREPARED)) == 3
    assert db.failed_prepares() == []
