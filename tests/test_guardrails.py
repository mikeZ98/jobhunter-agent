"""Pydantic validators are the anti-hallucination guardrails — instructor re-asks on these errors."""

import pytest
from pydantic import ValidationError

from jobhunter.agents.assets import (
    CoverLetter,
    TailoredCV,
    TailoredProject,
    _typst_escape,
    render_cv_markdown,
    render_cv_typst,
)
from jobhunter.agents.interview import InterviewIntel, LikelyQuestion, QuestionKind
from jobhunter.agents.scorer import canon

CTX = {
    "project_names": ["DocuMind", "Pricing API"],
    "company": "Acme",
    "allowed_skills": {canon(s) for s in ["Python", "FastAPI", "PostgreSQL", "Docker", "LangGraph", "RAG"]},
}

BODY = (
    "W Acme budujecie platformę RAG na FastAPI — w projekcie DocuMind zszedłem z p95 9,1 s do 2,4 s dzięki "
    "hybrydowemu wyszukiwaniu BM25 i pgvector. "
) * 6


def letter(paragraphs):
    return {
        "subject": "Python Developer — Jan Kowalski",
        "greeting": "Dzień dobry,",
        "paragraphs": paragraphs,
        "closing": "Pozdrawiam,",
    }


def test_cover_letter_accepts_concrete_text():
    CoverLetter.model_validate(letter([BODY, BODY[:300], "Mogę zacząć od 1 listopada, B2B."]), context=CTX)


def test_cover_letter_rejects_boilerplate():
    with pytest.raises(ValidationError, match="boilerplate"):
        CoverLetter.model_validate(
            letter([BODY + " Jestem osobą komunikatywną.", BODY[:200], "x y z"]), context=CTX
        )


def test_cover_letter_must_name_a_real_project_and_the_company():
    generic = "Mam doświadczenie w Pythonie i budowaniu API w różnych firmach produkcyjnych. " * 16
    with pytest.raises(ValidationError, match="real project"):
        CoverLetter.model_validate(letter([generic, generic[:300], "Dostępny od zaraz."]), context=CTX)


def cv(**kw):
    base = {
        "headline": "Python & GenAI Engineer",
        "summary": "Builds RAG systems.",
        "skills_primary": ["Python", "FastAPI", "PostgreSQL", "RAG"],
        "projects": [
            {"name": "DocuMind", "bullets": ["a", "b"]},
            {"name": "Pricing API", "bullets": ["c", "d"]},
        ],
    }
    return base | kw


def test_cv_rejects_invented_skills():
    with pytest.raises(ValidationError, match="not in candidate profile"):
        TailoredCV.model_validate(cv(skills_primary=["Python", "Rust", "FastAPI", "Go"]), context=CTX)


def test_cv_rejects_invented_projects():
    bad = cv(
        projects=[
            {"name": "DocuMind", "bullets": ["a", "b"]},
            {"name": "SpaceX Autopilot", "bullets": ["c", "d"]},
        ]
    )
    with pytest.raises(ValidationError, match="Unknown projects"):
        TailoredCV.model_validate(bad, context=CTX)


def test_cv_renderers(profile):
    t = TailoredCV.model_validate(cv(), context=CTX)
    md = render_cv_markdown(t, profile)
    assert md.startswith("# Jan Kowalski") and "### DocuMind" in md
    typ = render_cv_typst(t, profile)
    assert "#set page" in typ and "DocuMind" in typ and "\\@" in typ  # '@' in e-mail escaped for Typst
    assert _typst_escape("~1.2k users") == "\\~1.2k users"  # '~' is a non-breaking space in Typst


def _q(kind):
    return LikelyQuestion(kind=kind, question="q", why_likely="w", answer_outline=["a", "b"])


def test_interview_intel_requires_three_specific_reverse_questions():
    base = {
        "business_model": "b",
        "product_guess": "p",
        "tech_deep_dives": ["a", "b", "c"],
        "elevator_pitch": "e",
        "likely_questions": [
            _q(k) for k in [QuestionKind.SYSTEM_DESIGN, QuestionKind.FRAMEWORK, QuestionKind.ALGORITHMS] * 2
        ],
    }
    InterviewIntel.model_validate(
        base | {"reverse_questions": ["Jak ewaluujecie RAG?", "Kto ma on-call?", "Jaki p95?"]}
    )
    with pytest.raises(ValidationError, match="exactly 3"):
        InterviewIntel.model_validate(base | {"reverse_questions": ["a", "b"]})
    with pytest.raises(ValidationError, match="specific"):
        InterviewIntel.model_validate(base | {"reverse_questions": ["Jak wygląda typowy dzień?", "b", "c"]})


def test_tailored_project_bullet_bounds():
    with pytest.raises(ValidationError):
        TailoredProject(name="DocuMind", bullets=["only one"])
