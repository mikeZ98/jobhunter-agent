"""Agent 4 — Interview Intel & Simulation: stack-specific likely questions + company cheat sheet."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from jobhunter.llm import LLMClient, Task
from jobhunter.models import CandidateProfile, ScoredListing


class QuestionKind(StrEnum):
    SYSTEM_DESIGN = "system_design"
    ALGORITHMS = "algorithms"
    FRAMEWORK = "framework"
    LLM_ML = "llm_ml"
    BEHAVIORAL = "behavioral"


class LikelyQuestion(BaseModel):
    kind: QuestionKind
    question: str
    why_likely: str = Field(description="Which line of the ad / stack element makes this probable.")
    answer_outline: list[str] = Field(
        min_length=2, max_length=5, description="Bullet outline tied to the candidate's own projects."
    )


class InterviewIntel(BaseModel):
    business_model: str = Field(
        description="Probable business model and how engineering creates value there. "
        "Hedge clearly when inferring ('prawdopodobnie')."
    )
    product_guess: str
    tech_deep_dives: list[str] = Field(
        min_length=3,
        max_length=6,
        description="Topics to revise, e.g. 'FastAPI dependency injection & lifespan'.",
    )
    likely_questions: list[LikelyQuestion] = Field(min_length=6, max_length=10)
    reverse_questions: list[str] = Field(description="Exactly 3 smart questions for the hiring manager.")
    red_flags_to_probe: list[str] = Field(default_factory=list, max_length=4)
    elevator_pitch: str = Field(
        max_length=500, description="30-second Polish self-intro tailored to this role."
    )

    @field_validator("reverse_questions")
    @classmethod
    def exactly_three(cls, v: list[str]) -> list[str]:
        if len(v) != 3:
            raise ValueError(f"Provide exactly 3 reverse-interview questions, got {len(v)}.")
        generic = (
            "jak wygląda typowy dzień",
            "what does a typical day",
            "jaka jest kultura",
            "company culture",
        )
        if any(g in q.lower() for q in v for g in generic):
            raise ValueError("Reverse questions must be specific to this team's stack/product, not generic.")
        return v

    @field_validator("likely_questions")
    @classmethod
    def balanced(cls, v: list[LikelyQuestion]) -> list[LikelyQuestion]:
        kinds = {q.kind for q in v}
        if len(kinds) < 3:
            raise ValueError(
                "Cover at least 3 question kinds (system_design, framework, algorithms, llm_ml...)."
            )
        return v

    def to_markdown(self, title: str) -> str:
        out = [
            f"# Interview cheat sheet — {title}",
            "",
            "## Firma",
            f"**Model biznesowy:** {self.business_model}",
            f"**Produkt:** {self.product_guess}",
            "",
            "## Elevator pitch",
            self.elevator_pitch,
            "",
            "## Do powtórki",
            *[f"- {t}" for t in self.tech_deep_dives],
            "",
            "## Prawdopodobne pytania",
        ]
        for q in self.likely_questions:
            out += [
                f"### [{q.kind.value}] {q.question}",
                f"*Dlaczego:* {q.why_likely}",
                *[f"- {a}" for a in q.answer_outline],
                "",
            ]
        out += [
            "## Pytania do zadania rekruterowi",
            *[f"{i}. {q}" for i, q in enumerate(self.reverse_questions, 1)],
        ]
        if self.red_flags_to_probe:
            out += ["", "## Do wybadania", *[f"- {r}" for r in self.red_flags_to_probe]]
        return "\n".join(out)


SYSTEM = """You are a staff engineer who has interviewed hundreds of Python / GenAI / ML engineers in Poland.
Given a job ad and a candidate profile, predict what THIS company will most likely ask, based on its stack,
domain and seniority. Prefer concrete, stack-specific questions (e.g. "How would you make a RAG pipeline's
retrieval evaluable?", "Explain asyncio event loop starvation in FastAPI") over generic ones.
Answer outlines must reference the candidate's real projects. Write in Polish unless the ad is in English.
Infer the business model from the company name/ad only; clearly hedge inferences."""


async def generate_intel(item: ScoredListing, profile: CandidateProfile, llm: LLMClient) -> InterviewIntel:
    job = item.listing
    projects = "\n".join(
        f"- {p.name} [{', '.join(p.stack)}]: {p.summary}; {'; '.join(p.highlights)}" for p in profile.projects
    )
    gaps = ", ".join(item.llm.gaps) if item.llm else "unknown"
    user = (
        f"JOB AD:\n{job.text_for_matching(7000)}\n\nCANDIDATE ({profile.seniority.value}): {profile.headline}\n"
        f"PROJECTS:\n{projects}\nKNOWN GAPS vs this role: {gaps}"
    )
    return await llm.structured(Task.INTERVIEW, InterviewIntel, SYSTEM, user, max_tokens=2600)
