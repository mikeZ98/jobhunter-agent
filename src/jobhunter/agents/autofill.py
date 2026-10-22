"""Agent 5 — Form Autofill (Playwright, human-in-the-loop).

Hard safety contract — enforced by construction, not by convention:
  * This module never calls ``click``, ``press``, ``dispatch_event`` or ``submit`` on the page.
    It only uses ``fill`` / ``press_sequentially`` (text), ``set_input_files`` and ``select_option``.
    Typed values are stripped of newlines first, so typing can never emit an Enter keypress.
    Nothing here can submit an application.
  * Consent / GDPR / terms checkboxes are never ticked; they're listed for the human.
  * Fields that already have a value (ATS autofill from a LinkedIn import) are never overwritten.
  * After filling, execution blocks on a CLI prompt until the human has reviewed and clicked Submit.

Field detection is selector-agnostic: an in-page scan collects every visible control with its label
text (label[for], wrapping label, aria-label(ledby), placeholder, name, autocomplete), tags it with a
``data-jh-id``, and Python maps descriptors to profile values via ordered PL/EN regex rules. Unknown
free-text questions are answered in one batched LLM call, validated against the field list.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Frame, Locator, Page
from playwright.async_api import Error as PlaywrightError
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from rich.console import Console
from rich.table import Table

from jobhunter.db import Database
from jobhunter.llm import LLMClient, Task
from jobhunter.models import CandidateProfile, ListingStatus
from jobhunter.telemetry import get_logger, span

log = get_logger("agent.autofill")
console = Console()

REVIEW_PROMPT = (
    "[ACTION REQUIRED] Form pre-filled for review. Check the browser window and click Submit manually. "
    "Press ENTER to mark as Applied and continue."
)

SCAN_JS = r"""
() => {
  const SKIP = new Set(['hidden', 'submit', 'button', 'image', 'reset', 'password', 'search']);
  const text = n => (n?.innerText || n?.textContent || '').replace(/\s+/g, ' ').trim();
  const labelOf = el => {
    const parts = [];
    if (el.id) { try { const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`); if (l) parts.push(text(l)); } catch (e) {} }
    const wrap = el.closest('label'); if (wrap) parts.push(text(wrap));
    if (el.getAttribute('aria-label')) parts.push(el.getAttribute('aria-label'));
    (el.getAttribute('aria-labelledby') || '').split(/\s+/).filter(Boolean)
      .forEach(id => { const n = document.getElementById(id); if (n) parts.push(text(n)); });
    if (!parts.length) {
      const box = el.closest('fieldset, [class*="field"], [class*="question"], li, .form-group, div');
      const cand = box?.querySelector('legend, label, [class*="label"], h3, h4, p, span');
      if (cand && cand !== el) parts.push(text(cand));
    }
    return [...new Set(parts)].join(' | ').slice(0, 400);
  };
  const out = [];
  document.querySelectorAll('input, textarea, select').forEach((el, i) => {
    const type = (el.tagName === 'INPUT' ? (el.getAttribute('type') || 'text') : el.tagName).toLowerCase();
    if (SKIP.has(type) || el.disabled || el.readOnly) return;
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    const visible = (r.width > 0 && r.height > 0 && st.visibility !== 'hidden' && st.display !== 'none');
    if (!visible && type !== 'file' && type !== 'checkbox' && type !== 'radio') return;  // custom-styled inputs are often 0x0
    const id = el.getAttribute('data-jh-id') || `jh-${i}-${Math.random().toString(36).slice(2, 7)}`;
    el.setAttribute('data-jh-id', id);
    out.push({
      id, tag: el.tagName.toLowerCase(), type, name: el.name || '', el_id: el.id || '',
      placeholder: el.placeholder || '', autocomplete: el.getAttribute('autocomplete') || '',
      label: labelOf(el), required: !!(el.required || el.getAttribute('aria-required') === 'true'),
      value: (type === 'checkbox' || type === 'radio') ? (el.checked ? 'checked' : '') : (type === 'file' ? (el.files?.length ? 'file' : '') : (el.value || '')),
      options: el.tagName === 'SELECT' ? [...el.options].map(o => o.text.trim()).filter(Boolean).slice(0, 60) : [],
      accept: el.getAttribute('accept') || '', maxlength: el.maxLength > 0 ? el.maxLength : null,
    });
  });
  return out;
}
"""

BANNER_JS = r"""
(msg) => {
  document.getElementById('jh-banner')?.remove();
  const b = document.createElement('div');
  b.id = 'jh-banner';
  b.textContent = msg;
  b.style.cssText = 'position:fixed;z-index:2147483647;left:50%;top:8px;transform:translateX(-50%);' +
    'background:#1f4e79;color:#fff;font:600 13px -apple-system,sans-serif;padding:8px 14px;border-radius:8px;' +
    'box-shadow:0 4px 16px rgba(0,0,0,.25);pointer-events:none;max-width:90vw';
  document.documentElement.appendChild(b);
  document.querySelectorAll('[data-jh-filled]').forEach(e => { e.style.outline = '2px solid #f5b301'; e.style.outlineOffset = '1px'; });
  document.querySelectorAll('[data-jh-attention]').forEach(e => { e.style.outline = '2px dashed #d93025'; });
}
"""


@dataclass
class FieldDesc:
    id: str
    tag: str
    type: str
    name: str
    el_id: str
    placeholder: str
    autocomplete: str
    label: str
    required: bool
    value: str
    options: list[str]
    accept: str
    maxlength: int | None
    frame: Frame | None = field(default=None, repr=False)

    @property
    def hay(self) -> str:
        return " ".join([self.label, self.name, self.el_id, self.placeholder, self.autocomplete]).lower()


@dataclass
class FillPlan:
    text: dict[str, str] = field(default_factory=dict)  # field id -> value
    files: dict[str, Path] = field(default_factory=dict)
    selects: dict[str, str] = field(default_factory=dict)
    consents: list[FieldDesc] = field(default_factory=list)
    unresolved: list[FieldDesc] = field(default_factory=list)
    skipped_prefilled: int = 0


Getter = Callable[[CandidateProfile, dict[str, Any]], str | None]

# Ordered: first match wins. ``autocomplete`` tokens are checked before label regexes (most reliable).
AUTOCOMPLETE: dict[str, Getter] = {
    "given-name": lambda p, _: p.first_name,
    "family-name": lambda p, _: p.last_name,
    "name": lambda p, _: p.full_name,
    "email": lambda p, _: p.email,
    "tel": lambda p, _: p.phone,
    "tel-national": lambda p, _: p.phone,
    "address-level2": lambda p, _: p.city,
    "url": lambda p, _: p.portfolio_url,
}
RULES: list[tuple[re.Pattern[str], Getter]] = [
    (re.compile(rx, re.I), g)
    for rx, g in [
        (
            r"imi[eę] i nazwisko|full.?name|your name|^name\b|\bname \||imię/nazwisko",
            lambda p, _: p.full_name,
        ),
        (r"first.?name|given.?name|\bimi[eę]\b|\bfname\b", lambda p, _: p.first_name),
        (r"last.?name|surname|family.?name|nazwisk|\blname\b", lambda p, _: p.last_name),
        (r"e-?mail", lambda p, _: p.email),
        (r"phone|\btel\b|telefon|mobile|kom[oó]rk|numer kontaktowy", lambda p, _: p.phone),
        (r"linked\s?in", lambda p, _: p.linkedin_url),
        (r"git\s?hub|gitlab", lambda p, _: p.github_url),
        (
            r"portfolio|website|strona (www|internetowa)|personal site|\bwww\b|\burl\b",
            lambda p, _: p.portfolio_url or p.github_url,
        ),
        (r"\bcity\b|miasto|location|lokalizac|miejscowo|where are you based", lambda p, _: p.city),
        (
            r"salary|wynagrodz|oczekiwan\w* finans|stawk|rate expect|compensation|expected (rate|pay)",
            lambda p, _: (
                f"{p.expected_b2b_monthly_pln:,.0f} PLN netto/mies. (B2B)".replace(",", " ")
                if p.expected_b2b_monthly_pln
                else None
            ),
        ),
        (r"notice|okres wypowiedz", lambda p, _: p.notice_period),
        (
            r"availab|dost[eę]pno|start date|kiedy (mo[zż]esz|mo[zż]e pan)|earliest start|data rozpocz",
            lambda p, _: p.availability,
        ),
        (
            r"cover.?letter|list motywac|motivation|message to (the )?recruiter|wiadomo[sś][cć] (do|dla) rekrut|dodatkowe informacje"
            r"|additional information|anything else|about you|o sobie",
            lambda _p, a: a.get("cover_letter_text"),
        ),
    ]
]
CONSENT_RX = re.compile(
    r"rodo|gdpr|zgod|consent|privacy|prywatno|regulamin|terms|przetwarzan|klauzul|marketing", re.I
)
CV_FILE_RX = re.compile(r"\bcv\b|resume|życiorys|zyciorys|curriculum|attach|załącz|zalacz|upload|plik", re.I)
COVER_FILE_RX = re.compile(r"cover|motywac", re.I)


def plan_fills(fields: list[FieldDesc], profile: CandidateProfile, assets: dict[str, Any]) -> FillPlan:
    plan = FillPlan()
    for f in fields:
        if f.type == "file":
            if f.value:
                plan.skipped_prefilled += 1
            elif COVER_FILE_RX.search(f.hay) and assets.get("cover_letter_pdf"):
                plan.files[f.id] = Path(assets["cover_letter_pdf"])
            elif (CV_FILE_RX.search(f.hay) or "pdf" in f.accept or not f.hay.strip()) and assets.get(
                "cv_pdf"
            ):
                plan.files[f.id] = Path(assets["cv_pdf"])
            continue
        if f.type in ("checkbox", "radio"):
            if CONSENT_RX.search(f.hay) and not f.value:
                plan.consents.append(f)
            continue
        if f.value.strip():
            plan.skipped_prefilled += 1
            continue
        value = _match_value(f, profile, assets)
        if f.tag == "select":
            if value and (opt := _best_option(value, f.options)):
                plan.selects[f.id] = opt
            elif f.required or f.options:
                plan.unresolved.append(f)
            continue
        if value:
            plan.text[f.id] = value[: f.maxlength] if f.maxlength else value
        elif f.required or f.tag == "textarea" or len(f.label) > 12:
            plan.unresolved.append(f)
    return plan


def _match_value(f: FieldDesc, profile: CandidateProfile, assets: dict[str, Any]) -> str | None:
    for token in f.autocomplete.lower().split():
        if token in AUTOCOMPLETE and (v := AUTOCOMPLETE[token](profile, assets)):
            return v
    for pattern, answer in profile.extra_answers.items():  # user-curated answers win over heuristics
        if re.search(pattern, f.hay, re.I):
            return answer
    if f.type == "email":
        return profile.email
    if f.type == "tel":
        return profile.phone
    for rx, getter in RULES:
        if rx.search(f.hay):
            return getter(profile, assets)
    return None


def _best_option(value: str, options: list[str]) -> str | None:
    v = value.lower()
    for o in options:
        if o.lower() == v:
            return o
    for o in options:
        if v in o.lower() or o.lower() in v:
            return o
    return None


# ------------------------------------------------------------------ LLM for unknown questions


class FieldAnswer(BaseModel):
    field_id: str
    value: str = Field(
        description="Answer text, or for selects one of the given options verbatim. "
        "Empty string if the candidate must answer personally."
    )


class FormAnswers(BaseModel):
    answers: list[FieldAnswer]

    @field_validator("answers")
    @classmethod
    def valid_ids_and_options(cls, v: list[FieldAnswer], info: ValidationInfo) -> list[FieldAnswer]:
        fields: dict[str, list[str]] = (info.context or {}).get("fields", {})
        for a in v:
            if a.field_id not in fields:
                raise ValueError(f"Unknown field_id {a.field_id}; valid: {list(fields)}")
            opts = fields[a.field_id]
            if opts and a.value and a.value not in opts:
                raise ValueError(f"Field {a.field_id}: '{a.value}' is not one of {opts}")
        return v


FORM_SYSTEM = """You fill job application form fields on behalf of a candidate, using ONLY facts from the profile.
- Short factual answers (1-3 sentences for questions, a single value for simple fields), in the form's language.
- For selects, return one option verbatim. For yes/no eligibility questions answer truthfully from the profile.
- If the answer requires information not in the profile (e.g. personal opinions, legal status unknown),
  return an empty string — the human will fill it."""


def _describe(f: FieldDesc) -> str:
    line = f"- id={f.id} type={f.type} required={f.required} label={(f.label or f.name or f.placeholder)!r}"
    return line + (f" options={f.options}" if f.options else "")


async def answer_unknowns(
    unresolved: list[FieldDesc], profile: CandidateProfile, job_summary: str, llm: LLMClient
) -> dict[str, str]:
    if not unresolved:
        return {}
    spec = "\n".join(_describe(f) for f in unresolved)
    user = f"PROFILE:\n{profile.model_dump_json()}\n\nJOB:\n{job_summary}\n\nFIELDS:\n{spec}"
    try:
        res = await llm.structured(
            Task.FORM_ANSWER,
            FormAnswers,
            FORM_SYSTEM,
            user,
            max_tokens=1200,
            validation_context={"fields": {f.id: f.options for f in unresolved}},
        )
    except Exception as e:
        log.warning("form-answer LLM failed: {!r}", e)
        return {}
    return {a.field_id: a.value for a in res.answers if a.value.strip()}


# ------------------------------------------------------------------ browser execution


async def scan(page: Page) -> list[FieldDesc]:
    fields: list[FieldDesc] = []
    for frame in page.frames:
        try:
            for d in await frame.evaluate(SCAN_JS):
                fields.append(FieldDesc(**d, frame=frame))
        except PlaywrightError:  # detached / navigating frame
            continue
    return fields


async def _locator(f: FieldDesc) -> Locator:
    assert f.frame is not None
    return f.frame.locator(f'[data-jh-id="{f.id}"]').first


async def execute(plan: FillPlan, by_id: dict[str, FieldDesc], extra_text: dict[str, str]) -> int:
    filled = 0
    for fid, value in {**plan.text, **extra_text}.items():
        f = by_id[fid]
        try:
            loc = await _locator(f)
            if f.tag == "select":
                await loc.select_option(label=value)
            elif len(value) <= 80 and f.tag != "textarea":
                # Keystroke cadence instead of an instant paste; newlines removed so no Enter is ever sent.
                await loc.fill("")
                await loc.press_sequentially(re.sub(r"[\r\n]+", " ", value), delay=random.randint(18, 55))
            else:
                await loc.fill(value)  # long text: humans paste cover letters too
            await f.frame.evaluate(  # type: ignore[union-attr]
                "id => document.querySelector(`[data-jh-id='${id}']`)?.setAttribute('data-jh-filled','1')",
                fid,
            )
            filled += 1
            await asyncio.sleep(random.uniform(0.15, 0.6))
        except PlaywrightError as e:
            log.warning("could not fill {} ({}): {}", fid, f.label[:60], e.message[:120])
    for fid, opt in plan.selects.items():
        try:
            await (await _locator(by_id[fid])).select_option(label=opt)
            filled += 1
        except PlaywrightError as e:
            log.warning("could not select {}: {}", fid, e.message[:120])
    for fid, path in plan.files.items():
        if not path.exists():
            log.warning("asset missing: {}", path)
            continue
        try:
            await (await _locator(by_id[fid])).set_input_files(str(path))
            filled += 1
        except PlaywrightError as e:
            log.warning("upload failed for {}: {}", fid, e.message[:120])
    filled_ids = {*plan.text, *extra_text, *plan.selects, *plan.files}
    needs_human = [f for f in by_id.values() if f.required and not f.value and f.id not in filled_ids]
    for f in plan.consents + needs_human:
        with contextlib.suppress(PlaywrightError):
            await f.frame.evaluate(  # type: ignore[union-attr]
                "id => document.querySelector(`[data-jh-id='${id}']`)?.setAttribute('data-jh-attention','1')",
                f.id,
            )
    return filled


def _summary(
    job_title: str, plan: FillPlan, filled: int, by_id: dict[str, FieldDesc], extra: dict[str, str]
) -> None:
    t = Table(title=f"Autofill — {job_title}", show_lines=False)
    t.add_column("Field", overflow="fold", max_width=48)
    t.add_column("Value / status", overflow="fold", max_width=60)
    for fid, v in {**plan.text, **extra}.items():
        t.add_row(by_id[fid].label[:48] or by_id[fid].name, (v[:57] + "…") if len(v) > 60 else v)
    for fid, v in plan.selects.items():
        t.add_row(by_id[fid].label[:48], f"[select] {v}")
    for fid, p in plan.files.items():
        t.add_row(by_id[fid].label[:48] or "file", f"[upload] {p.name}")
    for c in plan.consents:
        t.add_row(c.label[:48], "[red]consent — tick yourself if you agree[/red]")
    for f in plan.unresolved:
        if f.id not in extra:
            t.add_row(f.label[:48] or f.name, "[yellow]needs your answer[/yellow]")
    console.print(t)
    console.print(
        f"[dim]{filled} fields filled, {plan.skipped_prefilled} pre-filled by the site left untouched[/dim]"
    )


async def prefill(
    page: Page,
    profile: CandidateProfile,
    assets: dict[str, Any],
    job_title: str,
    job_summary: str,
    llm: LLMClient | None,
) -> int:
    with span("autofill.prefill", url=page.url):
        with contextlib.suppress(PlaywrightError):
            await page.wait_for_load_state("networkidle", timeout=15_000)
        fields = await scan(page)
        by_id = {f.id: f for f in fields}
        plan = plan_fills(fields, profile, assets)
        extra = await answer_unknowns(plan.unresolved, profile, job_summary, llm) if llm else {}
        filled = await execute(plan, by_id, extra)
        with contextlib.suppress(PlaywrightError):
            await page.evaluate(
                BANNER_JS,
                f"JobHunter: {filled} pól wypełnionych — sprawdź (żółte), uzupełnij "
                "(czerwone) i kliknij Wyślij samodzielnie.",
            )
        _summary(job_title, plan, filled, by_id, extra)
        return filled


async def review_loop(
    ctx: BrowserContext, db: Database, profile: CandidateProfile, llm: LLMClient | None, rows: list[Any]
) -> None:
    """Sequential, one application at a time. ENTER=applied · s=skip for now · x=not interested ·
    r=rescan (after you opened the form manually) · q=quit."""
    import json

    page = ctx.pages[0] if ctx.pages else await ctx.new_page()
    for row in rows:
        assets = json.loads(row["assets_json"] or "{}")
        listing = json.loads(row["listing_json"])
        url = listing.get("apply_url") or row["url"]
        title = f"{row['title']} @ {row['company']}"
        console.rule(f"[bold]{title}[/bold]  score={row['final_score']}")
        console.print(f"[link={url}]{url}[/link]")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        except PlaywrightError as e:
            console.print(f"[red]navigation failed: {e.message[:200]}[/red]")
        summary = f"{row['title']} at {row['company']}\n{listing.get('description', '')[:1500]}"
        await prefill(page, profile, assets, title, summary, llm)

        while True:
            console.print(f"\n[bold yellow]{REVIEW_PROMPT}[/bold yellow]")
            choice = (
                (
                    await asyncio.to_thread(
                        input, "  [ENTER]=Applied  s=skip  x=not interested  r=rescan & refill  q=quit > "
                    )
                )
                .strip()
                .lower()
            )
            if choice == "r":
                page = ctx.pages[-1]  # the human may have opened the form in a new tab
                await page.bring_to_front()
                await prefill(page, profile, assets, title, summary, llm)
                continue
            break
        if choice == "q":
            console.print("[dim]stopping; remaining listings stay Prepared[/dim]")
            return
        if choice == "x":
            db.set_status(row["source_key"], ListingStatus.SKIPPED, note="not interested at review")
        elif choice == "":
            db.set_status(row["source_key"], ListingStatus.APPLIED, note=f"submitted manually via {page.url}")
            console.print("[green]✓ marked Applied[/green]")
