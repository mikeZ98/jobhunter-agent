"""Agent 3b — Adversarial critic: fact-check a cover letter / CV draft against the canonical profile.

The drafting validators in ``agents.assets`` keep a draft *well-formed* (real project names, profile
skills only). They cannot see an invented number, a tool claimed from the job ad, or a metric moved
from one project to another. The critic assumes the writer oversold and looks for exactly that.

Two passes, cheapest first:

1. **Deterministic** (no LLM, cannot be talked out of a verdict):
   * every number must occur in the profile / base CV (a letter may also quote the job ad);
   * CV project bullets may only use numbers from *that* project's summary and highlights;
   * "N lat / years" experience claims must match a figure from the profile;
   * CV skills and project names must exist in the profile;
   * job-ad technologies mentioned in the draft but absent from the profile become *suspects*.
2. **Adversarial LLM review** (skipped when pass 1 already rejected): lists every statement about the
   candidate that the profile does not support, and must answer for every suspect term whether the draft
   claims experience with it.

**Fail closed.** Approval is computed here from the findings, never taken from the model. A critic that
cannot run (no model, timeout, malformed verdict) rejects the draft; a suspect nobody could clear is a
violation. An unverified draft is never sent to an employer as if it were checked.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, computed_field, model_validator

from jobhunter.agents.assets import CoverLetter, TailoredCV
from jobhunter.agents.scorer import canon
from jobhunter.llm import LLMClient, Task
from jobhunter.models import CandidateProfile, JobListing
from jobhunter.telemetry import critic_verdicts, get_logger

log = get_logger("agent.critic")

DocumentKind = Literal["cover_letter", "cv"]


class ViolationKind(StrEnum):
    NUMBER = "number"  # a figure that occurs nowhere in the profile
    PROJECT_NUMBER = "project_number"  # a real figure attributed to the wrong project
    EXPERIENCE_YEARS = "experience_years"
    SKILL = "skill"
    PROJECT = "project"
    UNSUPPORTED_TERM = "unsupported_term"  # job-ad technology the profile lacks, claimed or not cleared
    LLM_UNSUPPORTED = "llm_unsupported"
    LLM_CONTRADICTED = "llm_contradicted"
    CRITIC_UNAVAILABLE = "critic_unavailable"


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ViolationKind
    claim: str
    detail: str


class CriticReport(BaseModel):
    document: DocumentKind
    violations: list[Violation] = Field(default_factory=list)
    llm_checked: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def approved(self) -> bool:
        return not self.violations

    def feedback(self) -> str:
        """The violations as instructions for a rewrite."""
        return "\n".join(f"- {v.claim}: {v.detail}" if v.claim else f"- {v.detail}" for v in self.violations)


# ------------------------------------------------------------------ LLM verdict schema


class TermCheck(BaseModel):
    term: str
    claims_experience: bool = Field(
        description="True if the draft says or implies the candidate has used, built with or knows it."
    )


class UnsupportedClaim(BaseModel):
    claim: str = Field(description="The draft's statement about the candidate, quoted.")
    reason: str = Field(description="What the profile says instead, or that it says nothing about it.")
    contradicts_profile: bool = Field(False, description="True if the profile states something different.")


class CriticVerdict(BaseModel):
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list, max_length=20)
    terms: list[TermCheck] = Field(default_factory=list, description="One entry per SUSPECT TERM.")

    @model_validator(mode="after")
    def every_suspect_answered(self, info: ValidationInfo) -> CriticVerdict:
        expected = {t.lower() for t in (info.context or {}).get("suspect_terms", [])}
        missing = expected - {t.term.strip().lower() for t in self.terms}
        if missing:
            raise ValueError(f"Answer claims_experience for every SUSPECT TERM; missing: {sorted(missing)}")
        return self


CRITIC_SYSTEM = """You are an adversarial fact-checker for job-application documents (cover letters, CVs).
The writer was told never to invent facts. Assume they tried to oversell anyway.
The CANDIDATE PROFILE and BASE CV are the only source of truth about the candidate.

List in "unsupported_claims" every statement in the DRAFT about the candidate that the profile does not
state or directly imply: skills, tools, years of experience, employers, roles, team sizes, projects,
numbers, responsibilities, education, certifications. Set contradicts_profile when the profile says
something different (a wrong number, a tool or result attributed to the wrong project, a higher level).
Statements about the employer or the job ad, intentions ("chcę", "I would"), and availability or
logistics taken from the profile are not claims to check. Do not report style issues.

For every term in SUSPECT TERMS (job-ad technologies missing from the profile) add one "terms" entry:
claims_experience = true if the draft says or implies the candidate has used or knows it, false if the
draft only mentions it as the employer's stack or as something to learn.
If everything is supported, return empty "unsupported_claims"."""


# ------------------------------------------------------------------ deterministic pass

_DATE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b")
# Not preceded by a letter/digit/separator, so "p95", "S3", "OAuth2" and the ".9" of "99.9" are not numbers.
_NUMBER = re.compile(r"(?<![\w.,/])(\d{1,3}(?:[  ]\d{3})+|\d+(?:[.,]\d+)?)(?:\s?(k|K|tys\.?|mln|M)\b)?")
_YEARS = re.compile(
    r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*\+?\s*(?:lat|lata|roku|years?|yrs)\b(?!\s*(?:temu|ago))", re.I
)
_SCALE = {"k": 1e3, "K": 1e3, "tys": 1e3, "tys.": 1e3, "mln": 1e6, "M": 1e6}


def _value(digits: str, suffix: str | None) -> float:
    v = float(re.sub(r"[  ]", "", digits).replace(",", "."))
    return v * _SCALE.get(suffix or "", 1.0)


def numbers(text: str) -> set[float]:
    return {v for _, v in _number_claims(text, keep_trivial=True)}


def _number_claims(text: str, *, keep_trivial: bool = False) -> list[tuple[str, float]]:
    """(raw, value) for every figure in ``text``. Trivial ones (counts <= 12, calendar years, dates)
    are skipped unless ``keep_trivial`` — they are rarely claims, and flagging them would reject
    "3 projekty" or "od 1 listopada"."""
    text = _DATE.sub(" ", text)
    out: list[tuple[str, float]] = []
    for m in _NUMBER.finditer(text):
        digits, suffix = m.group(1), m.group(2)
        v = _value(digits, suffix)
        plain_int = suffix is None and re.fullmatch(r"\d+", digits) is not None
        if not keep_trivial and plain_int and (v <= 12 or 1900 <= v <= 2100):
            continue
        out.append((m.group(0).strip(), v))
    return out


def _grounded(v: float, ground: set[float]) -> bool:
    return any(abs(v - g) <= 1e-6 * max(1.0, abs(g)) for g in ground)


def _term_pattern(term: str) -> re.Pattern[str]:
    """Match a technology name including Polish inflection: Kafka -> Kafki/Kafką, Docker -> Dockerze."""
    t = term.strip()
    if len(t) >= 4 and t[-1].isalpha():
        stem = t[:-1] if t[-1].lower() in "aeiouy" else t
        return re.compile(rf"(?<!\w){re.escape(stem)}\w{{0,4}}(?!\w)", re.I)
    return re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.I)


def mentions(text: str, term: str) -> bool:
    return bool(term.strip()) and _term_pattern(term).search(text) is not None


def draft_text(draft: CoverLetter | TailoredCV) -> str:
    if isinstance(draft, CoverLetter):
        return "\n".join([draft.subject, draft.greeting, *draft.paragraphs, draft.closing])
    parts = [
        draft.headline,
        draft.summary,
        ", ".join(draft.skills_primary),
        ", ".join(draft.skills_secondary),
    ]
    for p in draft.projects:
        parts += [p.name, *p.bullets]
    return "\n".join(parts)


class _Ground:
    """What the profile actually says, precomputed once per draft."""

    def __init__(self, profile: CandidateProfile, base_cv: str) -> None:
        self.corpus = profile.model_dump_json() + "\n" + base_cv
        self.numbers = numbers(self.corpus)
        self.vocab = {canon(k) for k in profile.skill_index()} | {
            canon(t) for p in profile.projects for t in p.stack
        }
        self.projects = {p.name.lower(): p for p in profile.projects}

    def supports_term(self, term: str) -> bool:
        return canon(term) in self.vocab or mentions(self.corpus, term)


def deterministic_review(
    draft: CoverLetter | TailoredCV, profile: CandidateProfile, base_cv: str, job: JobListing
) -> tuple[list[Violation], list[str]]:
    """Hard violations, and job-ad terms the draft mentions but the profile lacks (for the LLM pass)."""
    g = _Ground(profile, base_cv)
    text = draft_text(draft)
    violations: list[Violation] = []

    def check_numbers(fragment: str, ground: set[float], kind: ViolationKind, where: str) -> None:
        for raw, v in _number_claims(fragment):
            if not _grounded(v, ground):
                violations.append(Violation(kind=kind, claim=raw, detail=f"figure not found in {where}"))

    if isinstance(draft, CoverLetter):
        # A letter may quote the employer's figures ("zespół 40 osób"); those are not candidate claims.
        check_numbers(
            text, g.numbers | numbers(job.text_for_matching(20_000)), ViolationKind.NUMBER, "the profile"
        )
    else:
        check_numbers(f"{draft.headline}\n{draft.summary}", g.numbers, ViolationKind.NUMBER, "the profile")
        for tp in draft.projects:
            src = g.projects.get(tp.name.lower())
            if src is None:
                violations.append(
                    Violation(kind=ViolationKind.PROJECT, claim=tp.name, detail="project not in the profile")
                )
                continue
            own = numbers("\n".join([src.summary, *src.highlights]))
            for b in tp.bullets:
                check_numbers(b, own, ViolationKind.PROJECT_NUMBER, f"{src.name}'s summary/highlights")
        for skill in [*draft.skills_primary, *draft.skills_secondary]:
            if canon(skill) not in g.vocab:
                violations.append(
                    Violation(kind=ViolationKind.SKILL, claim=skill, detail="skill not in the profile")
                )

    for m in _YEARS.finditer(text):
        if not _grounded(_value(m.group(1), None), g.numbers):
            violations.append(
                Violation(
                    kind=ViolationKind.EXPERIENCE_YEARS,
                    claim=m.group(0),
                    detail="no matching years of experience in the profile",
                )
            )

    suspects = _dedupe(
        t
        for t in [*job.required_skills, *job.nice_to_have_skills]
        if mentions(text, t) and not g.supports_term(t)
    )
    return violations, suspects


def _dedupe(terms: Iterable[str]) -> list[str]:
    seen: dict[str, str] = {}
    for t in terms:
        seen.setdefault(canon(t), t.strip())
    return list(seen.values())


# ------------------------------------------------------------------ review


async def review(
    draft: CoverLetter | TailoredCV,
    *,
    profile: CandidateProfile,
    base_cv: str,
    job: JobListing,
    llm: LLMClient | None,
) -> CriticReport:
    """Fact-check ``draft``. ``llm=None`` runs the deterministic pass only — and then fails closed on suspects."""
    doc: DocumentKind = "cover_letter" if isinstance(draft, CoverLetter) else "cv"
    violations, suspects = deterministic_review(draft, profile, base_cv, job)
    report = await _llm_pass(doc, draft, profile, base_cv, job, llm, violations, suspects)
    critic_verdicts.add(1, {"document": doc, "approved": report.approved, "llm": report.llm_checked})
    if report.approved:
        log.info("critic: {} for {} approved", doc, job.company)
    else:
        log.warning("critic: {} for {} rejected — {}", doc, job.company, report.feedback().replace("\n", " "))
    return report


async def _llm_pass(
    doc: DocumentKind,
    draft: CoverLetter | TailoredCV,
    profile: CandidateProfile,
    base_cv: str,
    job: JobListing,
    llm: LLMClient | None,
    violations: list[Violation],
    suspects: list[str],
) -> CriticReport:
    if violations:  # already rejected; don't spend a local model call confirming it
        return CriticReport(document=doc, violations=violations)
    if llm is None:
        return CriticReport(
            document=doc,
            violations=[
                Violation(
                    kind=ViolationKind.UNSUPPORTED_TERM,
                    claim=t,
                    detail="job-ad technology missing from the profile; no LLM review to clear it",
                )
                for t in suspects
            ],
        )
    user = (
        f"CANDIDATE PROFILE:\n{profile.model_dump_json(exclude={'extra_answers'})}\n\n"
        f"BASE CV:\n{base_cv[:6000]}\n\n"
        f"JOB AD (context only — facts about the employer, not the candidate):\n{job.text_for_matching(4000)}\n\n"
        f"DRAFT ({doc.replace('_', ' ')}):\n{draft_text(draft)}\n\n"
        f"SUSPECT TERMS: {', '.join(suspects) if suspects else '(none)'}"
    )
    try:
        verdict = await llm.structured(
            Task.VALIDATE,
            CriticVerdict,
            CRITIC_SYSTEM,
            user,
            max_tokens=900,
            temperature=0.0,
            validation_context={"suspect_terms": suspects},
        )
    except Exception as e:  # fail closed
        detail = f"critic could not run, draft unverified: {e!r}"[:300]
        return CriticReport(
            document=doc,
            violations=[Violation(kind=ViolationKind.CRITIC_UNAVAILABLE, claim="", detail=detail)],
        )
    found = [
        Violation(
            kind=ViolationKind.LLM_CONTRADICTED if c.contradicts_profile else ViolationKind.LLM_UNSUPPORTED,
            claim=c.claim,
            detail=c.reason,
        )
        for c in verdict.unsupported_claims
    ]
    found += [
        Violation(
            kind=ViolationKind.UNSUPPORTED_TERM,
            claim=t.term,
            detail="draft claims experience with a technology missing from the profile",
        )
        for t in verdict.terms
        if t.claims_experience and t.term.strip().lower() in {s.lower() for s in suspects}
    ]
    return CriticReport(document=doc, violations=found, llm_checked=True)
