"""Domain models shared by scrapers, agents, persistence and renderers.

Two layers:
* ``RawListing``  — what a scraper saw, loosely typed, board-specific payload preserved for re-parsing.
* ``JobListing``  — normalized, comparable, scoreable. Everything downstream works on this.
LLM I/O schemas live next to the agents that own them (``jobhunter.agents.*``).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, computed_field, field_validator, model_validator


class Board(StrEnum):
    JUSTJOINIT = "justjoinit"
    NOFLUFFJOBS = "nofluffjobs"
    PRACUJ = "pracuj"
    ROCKETJOBS = "rocketjobs"
    LINKEDIN = "linkedin"


class ContractType(StrEnum):
    B2B = "b2b"
    UOP = "uop"  # umowa o pracę (permanent)
    UZ = "uz"  # umowa zlecenie (mandate)
    UOD = "uod"  # umowa o dzieło
    OTHER = "other"


class Seniority(StrEnum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    UNKNOWN = "unknown"


class RemoteMode(StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class ListingStatus(StrEnum):
    SCRAPED = "scraped"
    PREPARED = "prepared"
    APPLIED = "applied"
    INTERVIEW = "interview"
    OFFER = "offer"
    REJECTED = "rejected"
    SKIPPED = "skipped"


# Allowed forward transitions. Anything else is a bug or a manual override (``force=True``).
STATUS_TRANSITIONS: dict[ListingStatus, set[ListingStatus]] = {
    ListingStatus.SCRAPED: {ListingStatus.PREPARED, ListingStatus.SKIPPED},
    ListingStatus.PREPARED: {ListingStatus.APPLIED, ListingStatus.SKIPPED},
    ListingStatus.APPLIED: {ListingStatus.INTERVIEW, ListingStatus.REJECTED},
    ListingStatus.INTERVIEW: {ListingStatus.INTERVIEW, ListingStatus.OFFER, ListingStatus.REJECTED},
    ListingStatus.OFFER: set(),
    ListingStatus.REJECTED: set(),
    ListingStatus.SKIPPED: {ListingStatus.PREPARED},
}


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\((?:k|m|f|x|\s|/|-)+\)", " ", text.lower())  # "(k/m)", "(f/m/x)"
    text = re.sub(r"\b(sp\.?\s*z\s*o\.?\s*o\.?|s\.a\.|gmbh|ltd|inc|poland|polska)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def source_key_of(board: str, external_id: str) -> str:
    return hashlib.sha256(f"{board}:{external_id}".encode()).hexdigest()[:24]


def fingerprint_of(company: str, title: str) -> str:
    return hashlib.sha256(f"{slugify(company)}|{slugify(title)}".encode()).hexdigest()[:24]


class RawListing(BaseModel):
    board: Board
    external_id: str
    url: str
    title: str
    company: str
    payload: dict[str, Any] = Field(default_factory=dict, repr=False)
    description: str | None = Field(default=None, repr=False)
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def source_key(self) -> str:
        return source_key_of(self.board.value, self.external_id)

    @property
    def fingerprint(self) -> str:
        return fingerprint_of(self.company, self.title)


class SalaryRange(BaseModel):
    """A salary band as advertised, plus a normalized comparable view."""

    min: float | None = None
    max: float | None = None
    currency: str = "PLN"
    period: str = "month"  # hour | day | month | year
    contract: ContractType = ContractType.OTHER
    is_gross: bool | None = None  # UoP is gross; B2B is net (excl. VAT)
    # Normalized: monthly, PLN, *B2B-net-equivalent* so UoP and B2B bands are comparable.
    monthly_pln_min: float | None = None
    monthly_pln_max: float | None = None
    b2b_equiv_min: float | None = None
    b2b_equiv_max: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        if self.min is None and self.max is None:
            return "—"
        lo = f"{self.min:,.0f}".replace(",", " ") if self.min else "?"
        hi = f"{self.max:,.0f}".replace(",", " ") if self.max else "?"
        tax = "netto" if self.is_gross is False else "brutto" if self.is_gross else ""
        return f"{lo}–{hi} {self.currency}/{self.period} {self.contract.value.upper()} {tax}".strip()


class JobListing(BaseModel):
    board: Board
    external_id: str
    url: str
    apply_url: str | None = None
    title: str
    company: str
    description: str = ""
    seniority: Seniority = Seniority.UNKNOWN
    remote: RemoteMode = RemoteMode.UNKNOWN
    locations: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    salaries: list[SalaryRange] = Field(default_factory=list)
    published_at: datetime | None = None
    languages: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_key(self) -> str:
        """Stable id for *this board's* posting — primary key in SQLite."""
        return source_key_of(self.board.value, self.external_id)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fingerprint(self) -> str:
        """Cross-board identity: the same role reposted on JJIT and NFJ collapses to one."""
        return fingerprint_of(self.company, self.title)

    @property
    def best_salary(self) -> SalaryRange | None:
        ranked = [s for s in self.salaries if s.b2b_equiv_max or s.b2b_equiv_min]
        return max(ranked, key=lambda s: s.b2b_equiv_max or s.b2b_equiv_min or 0, default=None)

    def text_for_matching(self, max_chars: int = 6000) -> str:
        parts = [
            f"{self.title} @ {self.company}",
            f"Seniority: {self.seniority.value}; Remote: {self.remote.value}; Locations: {', '.join(self.locations)}",
            f"Required: {', '.join(self.required_skills)}",
            f"Nice to have: {', '.join(self.nice_to_have_skills)}",
            self.description,
        ]
        return "\n".join(parts)[:max_chars]


class MatchAssessment(BaseModel):
    """LLM judge output (validated). Combined with deterministic signals into ``ScoredListing``."""

    # Evidence first, numbers after (JSON is generated in field order). A 7B model asked for one holistic
    # number collapses onto 85/88 for everything; four bounded parts summed in code spread out properly.
    strengths: list[str] = Field(max_length=5, description="Concrete overlaps: projects, stack, domain.")
    gaps: list[str] = Field(max_length=5, description="Missing must-haves or seniority mismatch.")
    red_flags: list[str] = Field(default_factory=list, max_length=5)
    stack: int = Field(
        0, ge=0, le=40, description="0-40: must-have stack covered by the candidate's projects."
    )
    role: int = Field(
        0, ge=0, le=25, description="0-25: role type (backend / GenAI-LLM / ML) and domain match."
    )
    seniority: int = Field(
        0, ge=0, le=20, description="0-20: seniority fit; low if years required > candidate's."
    )
    logistics: int = Field(
        0, ge=0, le=15, description="0-15: location/remote, contract, salary vs expectations."
    )
    score: int = Field(
        0, ge=0, le=100, description="Leave 0: computed as stack + role + seniority + logistics."
    )
    pitch: str = Field("", max_length=400, description="One-sentence tailored pitch in Polish.")

    @field_validator("pitch", mode="before")
    @classmethod
    def _clip_pitch(cls, v: object) -> object:
        return v[:400].rstrip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _total(self) -> MatchAssessment:
        if parts := self.stack + self.role + self.seniority + self.logistics:
            self.score = parts  # older stored assessments only have ``score``
        return self


class ScoredListing(BaseModel):
    listing: JobListing
    embedding_score: float = 0.0
    skill_score: float = 0.0
    llm: MatchAssessment | None = None
    final_score: float = 0.0
    rejected_reason: str | None = None


# ---------- Candidate profile (profile.json) ----------


class ProfileSkill(BaseModel):
    name: str
    years: float = 0
    level: int = Field(default=3, ge=1, le=5)
    aliases: list[str] = Field(default_factory=list)


class ProfileProject(BaseModel):
    name: str
    summary: str
    stack: list[str]
    highlights: list[str] = Field(default_factory=list, description="Quantified, production-grade facts.")
    url: HttpUrl | None = None
    role_tags: list[str] = Field(default_factory=list, description="backend | genai | ml")


class CandidateProfile(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: str
    city: str
    linkedin_url: str | None = None
    github_url: str | None = None
    portfolio_url: str | None = None
    headline: str
    target_roles: list[str]
    seniority: Seniority = Seniority.MID
    skills: list[ProfileSkill]
    projects: list[ProfileProject]
    languages: list[str] = ["Polish (native)", "English (C1)"]
    remote_preference: list[RemoteMode] = [RemoteMode.REMOTE, RemoteMode.HYBRID]
    acceptable_cities: list[str] = Field(default_factory=list)
    min_b2b_monthly_pln: float | None = None
    min_uop_monthly_pln: float | None = None
    expected_b2b_monthly_pln: float | None = None
    notice_period: str = "1 month"
    availability: str = "Available from next month"
    excluded_companies: list[str] = Field(default_factory=list)
    excluded_keywords: list[str] = Field(default_factory=list)
    extra_answers: dict[str, str] = Field(
        default_factory=dict, description="Canned answers for recurring form questions (regex -> answer)."
    )

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def skill_index(self) -> dict[str, ProfileSkill]:
        idx: dict[str, ProfileSkill] = {}
        for s in self.skills:
            for key in (s.name, *s.aliases):
                idx[key.lower().strip()] = s
        return idx
