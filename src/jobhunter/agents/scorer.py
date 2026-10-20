"""Agent 2 — Matching & Scoring.

Funnel (cheap -> expensive):
  1. hard prefilter     seniority, exclusions, location/remote, salary floor        O(n), no I/O
  2. skill overlap      alias-canonicalized must-have / nice-to-have coverage       O(n)
  3. semantic sim       embedding cosine(job, base_cv), backend-calibrated          1 batched call
  4. LLM judge          rubric score + strengths/gaps/pitch, only top-K by (2)+(3)   K calls
  final = w_emb*emb + w_skill*skill + w_llm*llm   ->  top 10–15 above threshold
"""

from __future__ import annotations

import asyncio
import re
import unicodedata

import numpy as np

from jobhunter.config import ScoringConfig
from jobhunter.llm import Embedder, LLMClient, Task
from jobhunter.llm.embeddings import cosine
from jobhunter.models import (
    CandidateProfile,
    JobListing,
    MatchAssessment,
    RemoteMode,
    ScoredListing,
    Seniority,
)
from jobhunter.telemetry import Progress, get_logger

log = get_logger("agent.scorer")

CANONICAL = {
    "postgres": "postgresql",
    "psql": "postgresql",
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "gcp": "google cloud",
    "google cloud platform": "google cloud",
    "amazon web services": "aws",
    "rest": "rest api",
    "restful": "rest api",
    "restful api": "rest api",
    "fast api": "fastapi",
    "py": "python",
    "python 3": "python",
    "python3": "python",
    "ci/cd": "ci",
    "continuous integration": "ci",
    "gitlab ci": "ci",
    "github actions": "ci",
    "large language models": "llm",
    "llms": "llm",
    "genai": "llm",
    "generative ai": "llm",
    "langchain": "langchain",
    "lang chain": "langchain",
    "hugging face": "huggingface",
    "transformers": "huggingface",
    "scikit-learn": "sklearn",
    "scikit learn": "sklearn",
    "ml": "machine learning",
    "nosql": "nosql",
    "mongo": "mongodb",
    "rabbit": "rabbitmq",
    "sql alchemy": "sqlalchemy",
    "english": "_lang",
    "angielski": "_lang",
    "polish": "_lang",
    "polski": "_lang",
}

# Seniority the candidate can credibly apply for, given their own level.
REACHABLE = {
    Seniority.JUNIOR: {Seniority.INTERN, Seniority.JUNIOR, Seniority.MID, Seniority.UNKNOWN},
    Seniority.MID: {Seniority.JUNIOR, Seniority.MID, Seniority.UNKNOWN},
    Seniority.SENIOR: {Seniority.MID, Seniority.SENIOR, Seniority.UNKNOWN},
}

# Embedding cosine ranges observed for "unrelated" vs "near-identical" job/CV pairs, per backend.
CALIBRATION = {"ollama": (0.40, 0.82), "openai": (0.15, 0.62), "hashed-bow": (0.03, 0.40)}


CITY_ALIASES = {
    "warsaw": "warszawa",
    "cracow": "krakow",
    "breslau": "wroclaw",
    "lodz": "lodz",
    "tricity": "trojmiasto",
    "gdynia": "trojmiasto",
    "gdansk": "trojmiasto",
    "sopot": "trojmiasto",
}


def city_key(city: str) -> str:
    """'Kraków' / 'Cracow, Małopolskie, Poland' / 'krakow' -> 'krakow'."""
    c = city.split(",")[0].strip().lower().replace("ł", "l")
    c = unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode()
    return CITY_ALIASES.get(c, c)


def canon(skill: str) -> str:
    s = re.sub(r"\s+", " ", skill.lower().strip())
    s = re.sub(r"\s*\(.*?\)\s*", "", s)
    return CANONICAL.get(s, s)


def prefilter(job: JobListing, profile: CandidateProfile) -> str | None:
    """Return a rejection reason, or None to keep."""
    if job.seniority not in REACHABLE.get(profile.seniority, set(Seniority)):
        return f"seniority {job.seniority.value}"
    low_title = job.title.lower()
    company = job.company.lower()
    if any(c.lower() in company for c in profile.excluded_companies):
        return "excluded company"
    if kw := next((k for k in profile.excluded_keywords if k.lower() in f"{low_title} "), None):
        return f"excluded keyword '{kw}'"
    cities = {city_key(c) for c in profile.acceptable_cities}
    job_cities = {city_key(c) for c in job.locations}
    if job.remote not in (*profile.remote_preference, RemoteMode.UNKNOWN):
        return f"{job.remote.value} not in remote preference"
    office_bound = job.remote in (RemoteMode.ONSITE, RemoteMode.HYBRID)
    if office_bound and cities and job_cities and not (cities & job_cities):
        return f"{job.remote.value} outside acceptable cities ({', '.join(job.locations)})"
    best = job.best_salary
    if best and profile.min_b2b_monthly_pln:
        top = best.b2b_equiv_max or best.b2b_equiv_min
        if top and top < profile.min_b2b_monthly_pln:
            return f"salary {top:,.0f} < floor {profile.min_b2b_monthly_pln:,.0f}"
    return None


def skill_score(job: JobListing, profile: CandidateProfile) -> tuple[float, list[str], list[str]]:
    mine = {canon(k) for k in profile.skill_index()}
    req = [canon(s) for s in job.required_skills if canon(s) != "_lang"]
    nice = [canon(s) for s in job.nice_to_have_skills if canon(s) != "_lang"]
    if not req and not nice:
        return 50.0, [], []  # unknown, neutral
    hit_req = [s for s in req if s in mine]
    hit_nice = [s for s in nice if s in mine]
    missing = [s for s in req if s not in mine]
    r = len(hit_req) / len(req) if req else 0.6
    n = len(hit_nice) / len(nice) if nice else 0.5
    return round(100 * (0.8 * r + 0.2 * n), 1), hit_req + hit_nice, missing


def _calibrate(sim: np.ndarray, backend: str) -> np.ndarray:
    lo, hi = CALIBRATION.get(backend.split(":")[0], (0.3, 0.8))
    return np.clip((sim - lo) / (hi - lo), 0, 1) * 100


JUDGE_SYSTEM = """You are a pragmatic senior tech recruiter in Poland evaluating candidate-role fit.
First list strengths, gaps and red flags, then score each rubric part separately:
- stack 0-40: must-have stack coverage (exact tech used in the candidate's real projects counts most;
  every missing must-have costs ~8 pts)
- role 0-25: role type match (backend / GenAI-LLM / ML) and domain relevance of past projects
  (data analyst, finance, PLC or pure frontend roles score <= 8)
- seniority 0-20: seniority fit (roles demanding more years than the candidate has score <= 8)
- logistics 0-15: remote/location, contract type, salary vs expectations
Use the whole range and be strict: a total of 80+ means "would pass CV screening with high
probability", 50 is a long shot. Never invent candidate facts.
The pitch must be ONE concrete Polish sentence linking a named project to the role's core need."""


def _profile_brief(profile: CandidateProfile, base_cv: str) -> str:
    projects = "\n".join(f"- {p.name} [{', '.join(p.stack)}]: {p.summary}" for p in profile.projects)
    skills = ", ".join(f"{s.name} ({s.years:g}y)" for s in profile.skills)
    return (
        f"CANDIDATE: {profile.headline} | level: {profile.seniority.value} | "
        f"min B2B: {profile.min_b2b_monthly_pln} PLN | remote pref: {[r.value for r in profile.remote_preference]}\n"
        f"SKILLS: {skills}\nPROJECTS:\n{projects}\n\nCV:\n{base_cv[:4000]}"
    )


async def score_listings(
    jobs: list[JobListing],
    profile: CandidateProfile,
    base_cv: str,
    llm: LLMClient,
    embedder: Embedder,
    cfg: ScoringConfig,
    concurrency: int,
) -> list[ScoredListing]:
    if not jobs:
        return []
    scored = []
    for j in jobs:
        s, _hits, _missing = skill_score(j, profile)
        scored.append(ScoredListing(listing=j, skill_score=s))

    vecs = await embedder.embed([base_cv[:8000]] + [j.text_for_matching() for j in jobs])
    sims = _calibrate(cosine(vecs[1:], vecs[:1]).ravel(), embedder.backend)
    for sl, e in zip(scored, sims, strict=True):
        sl.embedding_score = round(float(e), 1)

    # Only the most promising go to the (slower) LLM judge.
    pre = sorted(scored, key=lambda s: 0.55 * s.skill_score + 0.45 * s.embedding_score, reverse=True)
    shortlist, rest = pre[: cfg.llm_shortlist_size], pre[cfg.llm_shortlist_size :]
    brief = _profile_brief(profile, base_cv)
    sem = asyncio.Semaphore(concurrency)

    async def judge(sl: ScoredListing) -> None:
        j = sl.listing
        sal = j.best_salary
        user = (
            f"{brief}\n\n=== ROLE ===\n{j.text_for_matching(5000)}\n"
            f"Salary: {sal.label if sal else 'not disclosed'} "
            f"(≈{sal.b2b_equiv_max if sal else '?'} PLN B2B-equivalent max)"
        )
        async with sem:
            try:
                sl.llm = await llm.structured(Task.SCORE, MatchAssessment, JUDGE_SYSTEM, user, max_tokens=700)
            except Exception as e:
                log.warning("judge failed for {}: {!r}", j.url, e)
            progress.step(f"{j.title} @ {j.company}" + (f" → {sl.llm.score:.0f}" if sl.llm else " → failed"))

    progress = Progress(log, "LLM judge", len(shortlist), every=1)
    await asyncio.gather(*(judge(s) for s in shortlist))

    for sl in shortlist:
        if sl.llm:
            sl.final_score = round(
                cfg.weight_embedding * sl.embedding_score
                + cfg.weight_skills * sl.skill_score
                + cfg.weight_llm * sl.llm.score,
                1,
            )
        else:  # renormalize weights without the LLM term
            w = cfg.weight_embedding + cfg.weight_skills
            sl.final_score = round(
                (cfg.weight_embedding * sl.embedding_score + cfg.weight_skills * sl.skill_score) / w, 1
            )
    for sl in rest:
        sl.final_score = round(
            (0.55 * sl.skill_score + 0.45 * sl.embedding_score) * 0.9, 1
        )  # unjudged discount
        sl.rejected_reason = "below LLM shortlist cut"
    return sorted(scored, key=lambda s: s.final_score, reverse=True)


def select_daily(scored: list[ScoredListing], cfg: ScoringConfig) -> list[ScoredListing]:
    judged = [s for s in scored if s.llm is not None or s.rejected_reason is None]
    top = [s for s in judged if s.final_score >= cfg.min_score][: cfg.max_daily]
    if len(top) < cfg.min_daily:  # top up with near-misses, but never below threshold - 10
        extra = [s for s in judged if s not in top and s.final_score >= cfg.min_score - 10]
        top += extra[: cfg.min_daily - len(top)]
    return top
