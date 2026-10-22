"""Drafting sub-graph: draft → critic → revise loop, and what ends up approved, held or quarantined."""

from __future__ import annotations

from typing import Any, get_type_hints

import pytest

from jobhunter.agents.assets import CoverLetter, TailoredCV
from jobhunter.agents.critic import CriticVerdict
from jobhunter.config import CriticConfig
from jobhunter.models import ScoredListing
from jobhunter.workflow.drafting import DraftState, build_drafting_graph, run_drafting
from jobhunter.workflow.schemas import DRAFT_NODE_OUTPUTS, DraftOutput
from tests.conftest import make_job

SENTENCE = (
    "W Acme rozwijacie usługi na FastAPI — w DocuMind obniżyłem p95 z 9,1 s do 2,4 s "
    "dzięki hybrydowemu wyszukiwaniu i streamingowi odpowiedzi. "
)
HONEST = SENTENCE * 7
INVENTED = HONEST.replace("2,4 s", "0,8 s")  # a figure the profile never states


class ScriptedLLM:
    """Letters/CVs come from per-document scripts (str = paragraph text, Exception = failure);
    the critic approves unless ``critic_error`` is set."""

    def __init__(self, letters: list[Any], cvs: list[Any] | None = None) -> None:
        self.letters, self.cvs = letters, cvs if cvs is not None else [None]
        self.prompts: dict[str, list[str]] = {"CoverLetter": [], "TailoredCV": [], "CriticVerdict": []}
        self.critic_error: Exception | None = None

    async def structured(self, task, response_model, system, user, *, validation_context=None, **_: Any):
        name = response_model.__name__
        self.prompts[name].append(user)
        ctx = validation_context or {}
        if response_model is CriticVerdict:
            if self.critic_error:
                raise self.critic_error
            terms = [{"term": t, "claims_experience": False} for t in ctx.get("suspect_terms", [])]
            return CriticVerdict.model_validate({"terms": terms}, context=ctx)
        script = self.letters if response_model is CoverLetter else self.cvs
        step = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(step, Exception):
            raise step
        if response_model is CoverLetter:
            return CoverLetter.model_validate(
                {
                    "subject": "s",
                    "greeting": "Dzień dobry,",
                    "closing": "Pozdrawiam",
                    "paragraphs": [step, step[:250], "Dostępny od listopada."],
                },
                context=ctx,
            )
        return TailoredCV.model_validate(
            {
                "headline": "Python & GenAI Engineer",
                "summary": "RAG + FastAPI w produkcji.",
                "skills_primary": ["Python", "FastAPI", "LangGraph", "RAG"],
                "projects": [
                    {"name": "DocuMind", "bullets": ["p95 9.1s → 2.4s", step or "eval harness w CI"]},
                    {"name": "Pricing API", "bullets": ["1.8k req/s", "zero-downtime migracja"]},
                ],
            },
            context=ctx,
        )


@pytest.fixture
def item() -> ScoredListing:
    return ScoredListing(listing=make_job(required_skills=["Python", "FastAPI"]))


async def _run(profile, llm, item, **cfg: Any):
    return await run_drafting(build_drafting_graph(profile, "", llm, CriticConfig(**cfg)), item)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_approved_on_first_draft(profile, item):
    llm = ScriptedLLM([HONEST])
    out = await _run(profile, llm, item)
    assert out.letter is not None and out.cv is not None
    assert out.held == {} and out.errors == {} and out.rounds == 1
    assert len(llm.prompts["CriticVerdict"]) == 2  # both documents checked


@pytest.mark.asyncio
async def test_rejected_letter_is_rewritten_with_the_findings(profile, item):
    llm = ScriptedLLM([INVENTED, HONEST])
    out = await _run(profile, llm, item)
    assert out.letter is not None and "2,4 s" in out.letter.paragraphs[0]
    assert out.held == {} and out.rounds == 2
    assert len(llm.prompts["TailoredCV"]) == 1  # the approved CV is not rewritten
    retry = llm.prompts["CoverLetter"][1]
    assert "FACT-CHECK OF YOUR PREVIOUS DRAFT" in retry and "- 0,8: figure not found in the profile" in retry


@pytest.mark.asyncio
async def test_still_rejected_after_last_revision_is_quarantined(profile, item):
    llm = ScriptedLLM([INVENTED])
    out = await _run(profile, llm, item, max_revisions=1)
    assert out.letter is None and out.cv is not None
    held = out.held["cover_letter"]
    assert held.quarantined and not held.report.approved
    assert len(llm.prompts["CoverLetter"]) == 2 and out.rounds == 2


@pytest.mark.asyncio
async def test_zero_revisions_quarantines_immediately(profile, item):
    llm = ScriptedLLM([INVENTED])
    out = await _run(profile, llm, item, max_revisions=0)
    assert out.held["cover_letter"].quarantined and len(llm.prompts["CoverLetter"]) == 1


@pytest.mark.asyncio
async def test_critic_unavailable_holds_without_revising(profile, item):
    llm = ScriptedLLM([HONEST])
    llm.critic_error = RuntimeError("ollama down")
    out = await _run(profile, llm, item)
    assert out.letter is None and out.cv is None  # fail closed: nothing unverified becomes an asset
    assert set(out.held) == {"cover_letter", "cv"}
    assert not any(h.quarantined for h in out.held.values())  # unverified, so retried next run
    assert len(llm.prompts["CoverLetter"]) == 1  # rewriting can't fix a missing critic


@pytest.mark.asyncio
async def test_critic_disabled_approves_unchecked(profile, item):
    llm = ScriptedLLM([INVENTED])
    out = await _run(profile, llm, item, enabled=False)
    assert out.letter is not None and out.held == {} and llm.prompts["CriticVerdict"] == []


@pytest.mark.asyncio
async def test_deterministic_only_mode_skips_the_llm_critic(profile, item):
    llm = ScriptedLLM([INVENTED, HONEST])
    out = await _run(profile, llm, item, llm_review=False)
    assert out.letter is not None and llm.prompts["CriticVerdict"] == []


@pytest.mark.asyncio
async def test_failed_revision_keeps_the_rejected_draft_held(profile, item):
    llm = ScriptedLLM([INVENTED, RuntimeError("model crashed")])
    out = await _run(profile, llm, item)
    assert out.held["cover_letter"].quarantined
    assert "model crashed" in out.errors["cover_letter"]


@pytest.mark.asyncio
async def test_first_draft_failure_does_not_block_the_other_document(profile, item):
    llm = ScriptedLLM([RuntimeError("empty reply")])
    out = await _run(profile, llm, item)
    assert out.letter is None and out.held == {} and "empty reply" in out.errors["cover_letter"]
    assert out.cv is not None


@pytest.mark.asyncio
async def test_cv_number_moved_between_projects_is_quarantined(profile, item):
    llm = ScriptedLLM([HONEST], cvs=["1.8k req/s w DocuMind"])  # Pricing API's figure
    out = await _run(profile, llm, item)
    assert out.cv is None and out.held["cv"].quarantined and out.letter is not None


def test_drafting_contracts_match_the_graph(profile):
    g = build_drafting_graph(profile, "", ScriptedLLM([HONEST]), CriticConfig())  # type: ignore[arg-type]
    assert set(g.get_graph().nodes) - {"__start__", "__end__"} == set(DRAFT_NODE_OUTPUTS)
    for out in DRAFT_NODE_OUTPUTS.values():
        assert out.channels() <= set(get_type_hints(DraftState))


def test_draft_update_omits_documents_not_drafted_this_round():
    assert DraftOutput(attempt=2).update() == {"attempt": 2, "errors": {}}
    assert set(DraftOutput(attempt=1, letter=None).update()) == {"attempt", "errors", "letter"}
