"""Adversarial critic: catches what the drafting validators can't, and fails closed when it can't check."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jobhunter.agents.assets import CoverLetter, TailoredCV, TailoredProject
from jobhunter.agents.critic import (
    CriticVerdict,
    ViolationKind,
    deterministic_review,
    mentions,
    numbers,
    review,
)
from tests.conftest import make_job

HONEST = (
    "W DocuMind obniżyłem p95 z 9,1 s do 2,4 s dzięki hybrydowemu wyszukiwaniu BM25 i pgvector, "
    "a w Pricing API utrzymałem 1.8k req/s przy p99 < 60 ms. Mam 4 lata doświadczenia z Pythonem."
)


def letter(*paragraphs: str) -> CoverLetter:
    # model_construct: the critic must judge drafts on facts, independent of the drafting validators.
    return CoverLetter.model_construct(
        subject="Python Developer — Jan Kowalski",
        greeting="Dzień dobry,",
        paragraphs=list(paragraphs) or [HONEST, "Acme buduje platformę danych.", "Mogę zacząć od 1.11.2026."],
        closing="Pozdrawiam",
    )


def cv(*, skills=None, projects=None) -> TailoredCV:
    return TailoredCV(
        headline="Python & GenAI Engineer",
        summary="Backend i RAG w produkcji.",
        skills_primary=skills or ["Python", "FastAPI", "PostgreSQL", "RAG"],
        projects=projects
        or [
            TailoredProject(name="DocuMind", bullets=["p95 9.1s → 2.4s", "~1.2k DAU, 99.9% uptime"]),
            TailoredProject(name="Pricing API", bullets=["1.8k req/s", "p99 < 60 ms"]),
        ],
    )


JOB = make_job(
    required_skills=["Python", "FastAPI", "Kafka"],
    nice_to_have_skills=["Docker", "Go"],
    description="Zespół 40 inżynierów. Python, Kafka, FastAPI.",
)


class FakeCritic:
    def __init__(self, payload: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.payload, self.error, self.calls = payload or {}, error, 0

    async def structured(self, task, response_model, system, user, *, validation_context=None, **_: Any):
        self.calls += 1
        if self.error:
            raise self.error
        return response_model.model_validate(self.payload, context=validation_context)


def kinds(violations) -> list[ViolationKind]:
    return [v.kind for v in violations]


# ------------------------------------------------------------------ parsing helpers


def test_number_parsing():
    assert numbers("9,1 s i 2.4s") == {9.1, 2.4}
    assert numbers("1.8k req/s, 1,2 tys. użytkowników, 2M klientów, 40k dokumentów") == {1800, 1200, 2e6, 4e4}
    assert numbers("18 000 PLN") == {18000}
    assert numbers("99.9% uptime") == {99.9}
    assert numbers("p95, S3, OAuth2") == set()  # glued to letters: identifiers, not figures


@pytest.mark.parametrize(
    ("text", "term", "hit"),
    [
        ("pracowałem z Kafką", "Kafka", True),
        ("kolejki w Kafki", "Kafka", True),
        ("deploy w Dockerze", "Docker", True),
        ("Google Cloud", "Go", False),
        ("serwisy w Go i Pythonie", "Go", True),
        ("C++ i Python", "C++", True),
        ("", "Kafka", False),
    ],
)
def test_mentions_handles_polish_inflection(text, term, hit):
    assert mentions(text, term) is hit


# ------------------------------------------------------------------ deterministic pass: letters


def test_honest_letter_passes(profile):
    violations, suspects = deterministic_review(letter(), profile, "", JOB)
    assert violations == [] and suspects == []


def test_invented_number_is_rejected(profile):
    bad = letter(HONEST.replace("2,4 s", "0,8 s"), "Acme.", "Od zaraz.")
    violations, _ = deterministic_review(bad, profile, "", JOB)
    assert kinds(violations) == [ViolationKind.NUMBER] and violations[0].claim == "0,8"


def test_letter_may_quote_the_job_ad(profile):
    ok = letter(HONEST, "Wasz zespół 40 inżynierów pracuje na FastAPI.", "Od zaraz.")
    assert deterministic_review(ok, profile, "", JOB)[0] == []


def test_inflated_experience_is_rejected(profile):
    bad = letter(HONEST.replace("4 lata", "7 lat"), "Acme.", "Od zaraz.")
    violations, _ = deterministic_review(bad, profile, "", JOB)
    assert kinds(violations) == [ViolationKind.EXPERIENCE_YEARS]


def test_trivial_numbers_are_not_claims(profile):
    ok = letter(HONEST, "Pokażę 3 projekty, w 2024 wdrożyłem DocuMind.", "Dostępny od 1.11.2026.")
    assert deterministic_review(ok, profile, "", JOB)[0] == []


def test_job_ad_tech_missing_from_profile_is_a_suspect(profile):
    claim = letter(HONEST, "Budowałem przetwarzanie strumieniowe na Kafce i Kafką.", "Od zaraz.")
    violations, suspects = deterministic_review(claim, profile, "", JOB)
    assert violations == [] and suspects == ["Kafka"]
    # Profile skills (Python, FastAPI, Docker) are never suspects, mentioned or not.
    assert (
        deterministic_review(letter("Python FastAPI Docker " + HONEST, "x", "y"), profile, "", JOB)[1] == []
    )


def test_base_cv_counts_as_ground_truth(profile):
    claim = letter(HONEST, "Prowadziłem 25 code review tygodniowo.", "Od zaraz.")
    assert kinds(deterministic_review(claim, profile, "", JOB)[0]) == [ViolationKind.NUMBER]
    assert deterministic_review(claim, profile, "Code review: 25 tygodniowo.", JOB)[0] == []


# ------------------------------------------------------------------ deterministic pass: CVs


def test_honest_cv_passes(profile):
    assert deterministic_review(cv(), profile, "", JOB) == ([], [])


def test_cv_number_moved_to_another_project_is_rejected(profile):
    moved = cv(
        projects=[
            TailoredProject(name="DocuMind", bullets=["1.8k req/s na produkcji", "eval harness w CI"]),
            TailoredProject(name="Pricing API", bullets=["p99 < 60 ms", "migracja bez przestojów"]),
        ]
    )
    violations, _ = deterministic_review(moved, profile, "", JOB)
    assert kinds(violations) == [ViolationKind.PROJECT_NUMBER]
    assert "DocuMind" in violations[0].detail


def test_cv_unknown_skill_and_project_are_rejected(profile):
    bad = cv(
        skills=["Python", "FastAPI", "Kafka", "RAG"],
        projects=[
            TailoredProject(name="DocuMind", bullets=["RAG", "eval"]),
            TailoredProject(name="StreamHub", bullets=["Kafka", "Flink"]),
        ],
    )
    violations, _ = deterministic_review(bad, profile, "", JOB)
    assert sorted(kinds(violations)) == [ViolationKind.PROJECT, ViolationKind.SKILL]


def test_cv_aliases_count_as_profile_skills(profile):
    assert deterministic_review(cv(skills=["python3", "postgres", "k8s", "GenAI"]), profile, "", JOB)[0] == []


# ------------------------------------------------------------------ review(): fail closed


@pytest.mark.asyncio
async def test_without_llm_unclearable_suspects_reject(profile):
    claim = letter(HONEST, "Znam Kafkę.", "Od zaraz.")
    report = await review(claim, profile=profile, base_cv="", job=JOB, llm=None)
    assert not report.approved and kinds(report.violations) == [ViolationKind.UNSUPPORTED_TERM]
    clean = await review(letter(), profile=profile, base_cv="", job=JOB, llm=None)
    assert clean.approved and not clean.llm_checked


@pytest.mark.asyncio
async def test_critic_failure_rejects_the_draft(profile):
    llm = FakeCritic(error=RuntimeError("no model"))
    report = await review(letter(), profile=profile, base_cv="", job=JOB, llm=llm)  # type: ignore[arg-type]
    assert not report.approved
    assert kinds(report.violations) == [ViolationKind.CRITIC_UNAVAILABLE] and "no model" in report.feedback()


@pytest.mark.asyncio
async def test_deterministic_rejection_skips_the_llm(profile):
    llm = FakeCritic()
    bad = letter(HONEST.replace("2,4 s", "0,8 s"), "Acme.", "Od zaraz.")
    report = await review(bad, profile=profile, base_cv="", job=JOB, llm=llm)  # type: ignore[arg-type]
    assert not report.approved and llm.calls == 0


@pytest.mark.asyncio
async def test_llm_findings_reject(profile):
    llm = FakeCritic(
        {
            "unsupported_claims": [
                {"claim": "kierowałem zespołem 5 osób", "reason": "profile lists no leadership role"},
                {"claim": "DocuMind na GCP", "reason": "profile says AWS", "contradicts_profile": True},
            ]
        }
    )
    report = await review(letter(), profile=profile, base_cv="", job=JOB, llm=llm)  # type: ignore[arg-type]
    assert report.llm_checked and not report.approved
    assert kinds(report.violations) == [ViolationKind.LLM_UNSUPPORTED, ViolationKind.LLM_CONTRADICTED]
    assert "- DocuMind na GCP: profile says AWS" in report.feedback()


@pytest.mark.asyncio
async def test_llm_clears_or_convicts_each_suspect(profile):
    draft = letter(HONEST, "Widzę, że używacie Kafki — chętnie ją poznam.", "Od zaraz.")
    cleared = FakeCritic({"terms": [{"term": "Kafka", "claims_experience": False}]})
    report = await review(draft, profile=profile, base_cv="", job=JOB, llm=cleared)  # type: ignore[arg-type]
    assert report.approved and report.model_dump()["approved"] is True

    convicted = FakeCritic({"terms": [{"term": "kafka", "claims_experience": True}]})
    report = await review(draft, profile=profile, base_cv="", job=JOB, llm=convicted)  # type: ignore[arg-type]
    assert kinds(report.violations) == [ViolationKind.UNSUPPORTED_TERM]


@pytest.mark.asyncio
async def test_verdict_that_skips_a_suspect_is_not_accepted(profile):
    draft = letter(HONEST, "Znam Kafkę.", "Od zaraz.")
    llm = FakeCritic({"unsupported_claims": [], "terms": []})  # rubber stamp
    report = await review(draft, profile=profile, base_cv="", job=JOB, llm=llm)  # type: ignore[arg-type]
    assert kinds(report.violations) == [ViolationKind.CRITIC_UNAVAILABLE]


def test_verdict_validator_demands_every_suspect():
    with pytest.raises(ValidationError, match="kafka"):
        CriticVerdict.model_validate({"terms": []}, context={"suspect_terms": ["Kafka"]})
    CriticVerdict.model_validate(
        {"terms": [{"term": " KAFKA ", "claims_experience": False}]}, context={"suspect_terms": ["Kafka"]}
    )
