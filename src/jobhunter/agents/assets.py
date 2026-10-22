"""Agent 3 — Application Assets: Polish cover letter + tailored CV (Markdown, Typst, PDF).

Anti-hallucination is enforced in the *schema*, not the prompt: validators receive the candidate
profile through instructor's validation ``context`` and reject any skill or project that is not in
``profile.json``. instructor feeds the error back to the model and it rewrites. A CV that invents
experience is worse than no CV.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from markdown_it import MarkdownIt
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from jobhunter.agents.scorer import canon
from jobhunter.llm import LLMClient, Task
from jobhunter.models import CandidateProfile, ScoredListing
from jobhunter.telemetry import get_logger

log = get_logger("agent.assets")

BANNED_PHRASES = [
    "dynamicznie rozwijając",
    "jestem osobą komunikatywną",
    "szybko się uczę",
    "z dużym zainteresowaniem",
    "chciałbym wyrazić",
    "chciałabym wyrazić",
    "idealnym kandydatem",
    "pasjonatem",
    "w odpowiedzi na ogłoszenie",
    "jestem przekonany",
    "jestem przekonana",
    "ambitny",
    "ambitna",
    "z wielką przyjemnością",
    "team player",
    "passionate",
]


class CoverLetter(BaseModel):
    subject: str = Field(description="E-mail subject, e.g. 'Python Backend Developer — Jan Kowalski'")
    greeting: str = Field(description="'Dzień dobry,' or 'Szanowni Państwo,' or a named greeting")
    paragraphs: list[str] = Field(min_length=3, max_length=4)
    closing: str

    @field_validator("paragraphs")
    @classmethod
    def no_boilerplate(cls, v: list[str]) -> list[str]:
        text = " ".join(v).lower()
        hits = [p for p in BANNED_PHRASES if p in text]
        if hits:
            raise ValueError(f"Remove corporate boilerplate phrases: {hits}. Replace with concrete facts.")
        words = len(text.split())
        if not 150 <= words <= 360:
            raise ValueError(f"Body has {words} words; target 180-320.")
        return v

    @model_validator(mode="after")
    def references_real_project(self, info: ValidationInfo) -> CoverLetter:
        ctx = info.context or {}
        projects: list[str] = ctx.get("project_names", [])
        body = " ".join(self.paragraphs).lower()
        if projects and not any(p.lower() in body for p in projects):
            raise ValueError(f"Name at least one real project from: {projects}.")
        if (company := (ctx.get("company") or "").strip()) and company.lower().split()[0] not in body:
            raise ValueError(f"Mention the company '{company}' and why its product/stack is relevant.")
        return self

    def to_markdown(self, profile: CandidateProfile) -> str:
        body = "\n\n".join(self.paragraphs)
        contact = " · ".join(
            filter(None, [profile.email, profile.phone, profile.linkedin_url, profile.github_url])
        )
        return f"**{self.subject}**\n\n{self.greeting}\n\n{body}\n\n{self.closing}\n\n{profile.full_name}  \n{contact}\n"


class TailoredProject(BaseModel):
    name: str = Field(description="Exact project name from the profile")
    bullets: list[str] = Field(
        min_length=2, max_length=4, description="Achievement bullets: action + tech + measurable result."
    )


class TailoredCV(BaseModel):
    headline: str = Field(max_length=90)
    summary: str = Field(max_length=600, description="3 sentences, first person omitted, mirrors role needs.")
    skills_primary: list[str] = Field(min_length=4, max_length=10, description="Most relevant first.")
    skills_secondary: list[str] = Field(default_factory=list, max_length=14)
    projects: list[TailoredProject] = Field(min_length=2, max_length=4, description="Most relevant first.")
    mirrored_keywords: list[str] = Field(
        default_factory=list, max_length=12, description="Job-ad keywords that the CV now genuinely covers."
    )

    @field_validator("headline", "summary", mode="before")
    @classmethod
    def _clip_text(cls, v: object, info: ValidationInfo) -> object:
        # Small models overshoot length limits by a few characters; trimming beats burning retries.
        limit = 90 if info.field_name == "headline" else 600
        if isinstance(v, str) and len(v) > limit:
            cut = v[:limit]
            if info.field_name == "summary" and (end := max(cut.rfind(". "), cut.rfind("."))) > limit // 2:
                return cut[: end + 1]
            return cut.rstrip()
        return v

    @field_validator("skills_primary", "skills_secondary", mode="before")
    @classmethod
    def skills_must_exist(cls, v: object, info: ValidationInfo) -> object:
        """Drop skills absent from the profile (never invent), then clip to the field's limit."""
        if not isinstance(v, list):
            return v
        allowed: set[str] = (info.context or {}).get("allowed_skills", set())
        if allowed:
            dropped = [s for s in v if isinstance(s, str) and canon(s) not in allowed]
            if dropped:
                log.debug("CV: dropping skills not in profile: {}", dropped)
            v = [s for s in v if not isinstance(s, str) or canon(s) in allowed]
            if dropped and info.field_name == "skills_primary" and len(v) < 4:
                raise ValueError(f"Skills not in candidate profile (do not invent): {dropped}")
        return v[: 10 if info.field_name == "skills_primary" else 14]

    @field_validator("mirrored_keywords", mode="before")
    @classmethod
    def _clip_keywords(cls, v: object) -> object:
        return v[:12] if isinstance(v, list) else v

    @field_validator("projects")
    @classmethod
    def projects_must_exist(cls, v: list[TailoredProject], info: ValidationInfo) -> list[TailoredProject]:
        allowed = {n.lower() for n in (info.context or {}).get("project_names", [])}
        if allowed:
            unknown = [p.name for p in v if p.name.lower() not in allowed]
            if unknown:
                raise ValueError(f"Unknown projects {unknown}; choose only from {sorted(allowed)}")
        return v


COVER_SYSTEM = """Piszesz list motywacyjny po polsku dla programisty aplikującego do firmy IT.
Zasady:
- Zero korpomowy i ogólników. Każde zdanie zawiera konkret: nazwę projektu, technologię, liczbę, decyzję techniczną.
- Akapit 1: jedno zdanie po co piszesz + najmocniejsze dopasowanie do roli (konkretny projekt).
- Akapit 2: 1-2 projekty produkcyjne kandydata, wybory techniczne i ich uzasadnienie, odniesione do wymagań oferty.
- Akapit 3: co kandydat zrobi w pierwszych tygodniach / dlaczego ta firma (produkt, stack, domena) — bez lizusostwa.
- Opcjonalny akapit 4: logistyka (dostępność, forma współpracy).
- Ton: rzeczowy, pewny, uprzejmy. 180-320 słów. Nie wymyślaj faktów spoza profilu.
- Pole "paragraphs": dokładnie 3 lub 4 osobne akapity (lista 3-4 stringów), nie jeden długi tekst.
- ZAKAZANE frazy i ich odmiany (odrzucimy list): {banned}."""

CV_SYSTEM = """You tailor a software engineer's CV to a specific job ad.
- Reorder and rephrase; NEVER invent skills, employers, projects, numbers or years.
- Pick the 2-4 most relevant projects; rewrite bullets to foreground the job's must-haves the project truly used.
- Mirror the ad's vocabulary where it is genuinely accurate (ATS keyword matching).
- Write the CV in the language of the job ad (Polish ad -> Polish CV, English ad -> English CV)."""


COVER_SYSTEM = COVER_SYSTEM.replace("{banned}", ", ".join(f"„{p}”" for p in BANNED_PHRASES))


def _profile_payload(profile: CandidateProfile, base_cv: str) -> str:
    return (
        profile.model_dump_json(exclude={"extra_answers", "excluded_companies", "excluded_keywords"})
        + f"\n\nBASE CV (markdown):\n{base_cv[:6000]}"
    )


async def generate_assets(
    item: ScoredListing, profile: CandidateProfile, base_cv: str, llm: LLMClient
) -> tuple[CoverLetter | BaseException, TailoredCV | BaseException]:
    job = item.listing
    ctx = {
        "project_names": [p.name for p in profile.projects],
        "allowed_skills": {canon(k) for k in profile.skill_index()},
        "company": job.company,
    }
    assessment = item.llm.model_dump_json() if item.llm else "{}"
    user = (
        f"CANDIDATE PROFILE:\n{_profile_payload(profile, base_cv)}\n\n"
        f"JOB AD:\n{job.text_for_matching(7000)}\n\nFIT ASSESSMENT:\n{assessment}"
    )
    # One document failing must not discard the other: the letter is the flakier of the two.
    # Sequential on purpose — concurrent fallbacks would thrash models in and out of a 16 GB Mac's memory.
    letter: CoverLetter | BaseException
    cv: TailoredCV | BaseException
    try:
        letter = await llm.structured(
            Task.CREATIVE, CoverLetter, COVER_SYSTEM, user, max_tokens=1400, validation_context=ctx
        )
    except Exception as e:
        letter = e
    try:
        cv = await llm.structured(
            Task.CREATIVE, TailoredCV, CV_SYSTEM, user, max_tokens=1800, validation_context=ctx
        )
    except Exception as e:
        cv = e
    return letter, cv


# ------------------------------------------------------------------ rendering


def render_cv_markdown(cv: TailoredCV, profile: CandidateProfile) -> str:
    by_name = {p.name.lower(): p for p in profile.projects}
    links = " · ".join(
        filter(
            None,
            [
                profile.email,
                profile.phone,
                profile.city,
                profile.linkedin_url,
                profile.github_url,
                profile.portfolio_url,
            ],
        )
    )
    lines = [
        f"# {profile.full_name}",
        f"**{cv.headline}**",
        "",
        links,
        "",
        "## Profil",
        cv.summary,
        "",
        "## Kluczowe technologie",
        ", ".join(cv.skills_primary),
    ]
    if cv.skills_secondary:
        lines += ["", f"*Również:* {', '.join(cv.skills_secondary)}"]
    lines += ["", "## Projekty"]
    for tp in cv.projects:
        src = by_name.get(tp.name.lower())
        stack = f" — *{', '.join(src.stack)}*" if src else ""
        url = f" ([link]({src.url}))" if src and src.url else ""
        lines += [f"### {tp.name}{stack}{url}", *[f"- {b}" for b in tp.bullets], ""]
    lines += [
        "## Języki",
        ", ".join(profile.languages),
        "",
        "<small>Wyrażam zgodę na przetwarzanie moich danych osobowych w celu prowadzenia rekrutacji.</small>",
    ]
    return "\n".join(lines)


def _typst_escape(s: str) -> str:
    # "~" is a non-breaking space in Typst markup ("~1.2k users" would lose its tilde)
    return re.sub(r"([#$*_<>@~\[\]\\`])", r"\\\1", s)


def render_cv_typst(cv: TailoredCV, profile: CandidateProfile) -> str:
    e = _typst_escape
    by_name = {p.name.lower(): p for p in profile.projects}
    parts = [
        '#set page(paper: "a4", margin: (x: 1.6cm, y: 1.4cm))',
        '#set text(font: ("Inter", "Helvetica Neue", "Arial"), size: 10pt, lang: "pl")',
        "#show heading.where(level: 2): it => block(above: 1.1em, below: .5em)[#text(size: 11pt, "
        'weight: "bold", fill: rgb("#1f4e79"))[#upper(it.body)] #line(length: 100%, stroke: .5pt)]',
        f'#align(center)[#text(size: 20pt, weight: "bold")[{e(profile.full_name)}] \\ '
        f"#text(size: 11pt)[{e(cv.headline)}] \\ "
        f"#text(size: 9pt)[{e(' · '.join(filter(None, [profile.email, profile.phone, profile.city, profile.linkedin_url, profile.github_url])))}]]",
        "== Profil",
        e(cv.summary),
        "== Kluczowe technologie",
        e(", ".join(cv.skills_primary)),
    ]
    if cv.skills_secondary:
        parts.append(f"#text(fill: luma(90))[{e(', '.join(cv.skills_secondary))}]")
    parts.append("== Projekty")
    for tp in cv.projects:
        src = by_name.get(tp.name.lower())
        stack = f' #h(1fr) #text(size: 8.5pt, style: "italic")[{e(", ".join(src.stack))}]' if src else ""
        parts.append(f"*{e(tp.name)}*{stack}")
        parts += [f"- {e(b)}" for b in tp.bullets]
    parts += [
        "== Języki",
        e(", ".join(profile.languages)),
        "#v(1fr) #text(size: 7pt, fill: luma(120))[Wyrażam zgodę na przetwarzanie moich danych "
        "osobowych w celu prowadzenia rekrutacji.]",
    ]
    return "\n\n".join(parts) + "\n"


CV_CSS = """
@page { size: A4; margin: 14mm 16mm; }
body { font: 10pt/1.45 -apple-system, "Helvetica Neue", Inter, Arial, sans-serif; color: #1b1b1b; }
h1 { font-size: 22pt; margin: 0; } h1 + p { margin: 2px 0 6px; font-size: 11.5pt; }
h2 { font-size: 10.5pt; text-transform: uppercase; letter-spacing: .06em; color: #1f4e79;
     border-bottom: 1px solid #c9d6e3; padding-bottom: 2px; margin: 14px 0 6px; }
h3 { font-size: 10.5pt; margin: 8px 0 2px; } h3 em { font-weight: 400; color: #555; }
ul { margin: 2px 0 6px 16px; padding: 0; } li { margin: 1px 0; }
small { color: #888; font-size: 7pt; } a { color: #1f4e79; text-decoration: none; }
"""


async def markdown_to_pdf(md: str, dst: Path, css: str = CV_CSS) -> bool:
    """Render Markdown -> HTML -> PDF with headless Chromium (already a dependency via Playwright)."""
    html = f"<!doctype html><meta charset='utf-8'><style>{css}</style>"
    html += MarkdownIt("commonmark", {"html": True}).render(md)
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html, wait_until="load")
            await page.pdf(path=str(dst), format="A4", print_background=True)
            await browser.close()
        return True
    except Exception as e:
        log.warning("PDF render failed for {} ({!r})", dst.name, e)
        return False


async def write_cv_files(cv: TailoredCV, profile: CandidateProfile, out_dir: Path) -> dict[str, Path]:
    """Writes cv.md + cv.typ always; cv.pdf via (1) typst python pkg / CLI, else (2) headless Chromium."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_cv_markdown(cv, profile)
    paths = {"cv_md": out_dir / "cv.md", "cv_typ": out_dir / "cv.typ", "cv_pdf": out_dir / "cv.pdf"}
    paths["cv_md"].write_text(md)
    paths["cv_typ"].write_text(render_cv_typst(cv, profile))
    if not _compile_typst(paths["cv_typ"], paths["cv_pdf"]) and not await markdown_to_pdf(
        md, paths["cv_pdf"]
    ):
        paths.pop("cv_pdf")
    return paths


def _compile_typst(src: Path, dst: Path) -> bool:
    try:
        import typst

        typst.compile(str(src), output=str(dst))
        return True
    except ImportError:
        pass
    except Exception as e:
        log.debug("typst python compile failed: {!r}", e)
    if shutil.which("typst"):
        r = subprocess.run(
            ["typst", "compile", str(src), str(dst)], capture_output=True, text=True, check=False
        )
        if r.returncode == 0:
            return True
        log.debug("typst CLI failed: {}", r.stderr[:500])
    return False
