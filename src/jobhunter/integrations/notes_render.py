"""Daily digest -> Apple-Notes-compatible HTML.

Notes keeps a conservative HTML subset: h1-h3, b/i/u, ul/ol/li, a, br, div, and tables (Big Sur+).
No CSS classes, no <style>; inline styles are mostly dropped. Keep markup simple and semantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from html import escape
from typing import Any

from jobhunter.agents.interview import InterviewIntel
from jobhunter.agents.linkedin_messaging import InboxMessage, RecruiterTriage
from jobhunter.models import ScoredListing


@dataclass
class DigestItem:
    scored: ScoredListing
    intel: InterviewIntel | None = None
    output_dir: str | None = None
    error: str | None = None


@dataclass
class Digest:
    day: date
    items: list[DigestItem]
    board_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    inbox: list[tuple[InboxMessage, RecruiterTriage]] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"Job Hunt — {self.day.isoformat()}"


def _e(s: object) -> str:
    return escape(str(s)) if s is not None else ""


def _salary_cell(sl: ScoredListing) -> str:
    best = sl.listing.best_salary
    if not best:
        return "—"
    lo, hi = best.b2b_equiv_min, best.b2b_equiv_max
    eq = f"{(lo or 0) / 1000:.0f}–{(hi or lo or 0) / 1000:.0f}k" if (lo or hi) else "?"
    return f"{_e(best.label)}<br><i>≈ {eq} B2B eq.</i>"


def render(d: Digest) -> str:
    h: list[str] = [f"<h1>{_e(d.title)}</h1>"]
    seen = sum(s.get("seen", 0) for s in d.board_stats.values())
    blocked = [b for b, s in d.board_stats.items() if s.get("error")]
    h.append(
        f"<div><b>{len(d.items)}</b> dopasowanych ofert z {seen} przejrzanych"
        + (f" · ⚠️ problemy: {_e(', '.join(blocked))}" if blocked else "")
        + "</div><br>"
    )

    h.append(
        "<table><tr><th>#</th><th>Score</th><th>Rola</th><th>Firma</th><th>Wynagrodzenie</th>"
        "<th>Tryb</th><th>Link</th></tr>"
    )
    for i, it in enumerate(d.items, 1):
        j = it.scored.listing
        h.append(
            f"<tr><td>{i}</td><td><b>{it.scored.final_score:.0f}</b></td><td>{_e(j.title)}</td>"
            f"<td>{_e(j.company)}</td><td>{_salary_cell(it.scored)}</td>"
            f"<td>{_e(j.remote.value)}<br>{_e(', '.join(j.locations[:2]))}</td>"
            f'<td><a href="{_e(j.apply_url or j.url)}">{_e(j.board.value)}</a></td></tr>'
        )
    h.append("</table><br>")

    for i, it in enumerate(d.items, 1):
        sl, j = it.scored, it.scored.listing
        h.append(f"<h2>{i}. {_e(j.title)} — {_e(j.company)} ({sl.final_score:.0f})</h2>")
        if sl.llm:
            h.append(f"<div><b>Pitch:</b> <i>{_e(sl.llm.pitch)}</i></div>")
            if sl.llm.strengths:
                h.append(
                    "<div><b>Mocne strony:</b></div><ul>"
                    + "".join(f"<li>{_e(s)}</li>" for s in sl.llm.strengths)
                    + "</ul>"
                )
            if sl.llm.gaps:
                h.append(
                    "<div><b>Luki:</b></div><ul>"
                    + "".join(f"<li>{_e(g)}</li>" for g in sl.llm.gaps)
                    + "</ul>"
                )
        h.append(f"<div><b>Stack:</b> {_e(', '.join(j.required_skills[:12]))}</div>")
        if it.intel:
            h.append(f"<div><b>Biznes:</b> {_e(it.intel.business_model)}</div>")
            h.append(
                "<div><b>Przygotuj się na:</b></div><ul>"
                + "".join(
                    f"<li>[{_e(q.kind.value)}] {_e(q.question)}</li>" for q in it.intel.likely_questions[:5]
                )
                + "</ul>"
            )
            h.append(
                "<div><b>Zapytaj rekrutera:</b></div><ol>"
                + "".join(f"<li>{_e(q)}</li>" for q in it.intel.reverse_questions)
                + "</ol>"
            )
        if it.output_dir:
            h.append(f"<div><i>Pliki: {_e(it.output_dir)}</i></div>")
        if it.error:
            h.append(f"<div>⚠️ {_e(it.error)}</div>")
        h.append(f'<div><a href="{_e(j.url)}">Ogłoszenie</a></div><br>')

    if d.inbox:
        h.append("<h2>📨 LinkedIn — szkice odpowiedzi</h2>")
        for m, t in d.inbox:
            h.append(
                f"<h3>{_e(m.sender)} — {_e(t.company or '?')} / {_e(t.role or '?')} (fit {t.fit_score})</h3>"
            )
            h.append(f"<div><i>{_e(m.text[:280])}…</i></div>")
            h.append("<div><b>Szkic:</b></div><div>" + _e(t.reply).replace("\n", "<br>") + "</div><br>")
    return "\n".join(h)
