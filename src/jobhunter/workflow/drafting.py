"""Drafting sub-graph: write → adversarial critic → rewrite with its findings → … → approve or hold.

    START ──► draft ──► critique ──┬──► END        (all approved, critic unavailable, or out of revisions)
                ▲                  │
                └──── revise ◄─────┘               (rejected with findings, attempt <= max_revisions)

Runs once per shortlisted job inside the top-level ``prepare`` branch. Each document is drafted and
revised on its own: an approved CV is not rewritten because the letter was rejected.

Outcomes per document (``DraftOutcome``):

* **approved**  — critic found nothing (or ``[critic] enabled = false``): written as the real asset;
* **held**      — rejected, never written as an asset (fail closed). ``quarantined`` when the critic
  still had findings after the last revision — kept for human review and not retried automatically;
  otherwise the critic could not run, so the draft is merely *unverified* and the job is retried on the
  next run like any other failed prepare;
* **error**     — no draft at all (generation failed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from jobhunter.agents import assets as assets_agent
from jobhunter.agents import critic
from jobhunter.agents.assets import CoverLetter, TailoredCV
from jobhunter.agents.critic import CriticReport, DocumentKind, ViolationKind
from jobhunter.config import CriticConfig
from jobhunter.llm import LLMClient
from jobhunter.models import CandidateProfile, ScoredListing
from jobhunter.telemetry import get_logger, traced_node
from jobhunter.workflow.schemas import CritiqueOutput, DraftOutput

log = get_logger("workflow.drafting")

DOCUMENTS: tuple[DocumentKind, ...] = ("cover_letter", "cv")


class DraftState(TypedDict, total=False):
    item: ScoredListing
    todo: list[DocumentKind]  # documents the next ``draft`` step (re)writes
    attempt: int  # draft rounds done
    letter: CoverLetter | None
    cv: TailoredCV | None
    reports: dict[DocumentKind, CriticReport]  # latest verdict per document
    errors: dict[DocumentKind, str]  # generation failures (no draft)


DraftingGraph = CompiledStateGraph[DraftState, None, DraftState, DraftState]


def has_findings(report: CriticReport) -> bool:
    """Rejected for what the draft says — as opposed to the critic failing to run."""
    return any(v.kind is not ViolationKind.CRITIC_UNAVAILABLE for v in report.violations)


def build_drafting_graph(
    profile: CandidateProfile, base_cv: str, llm: LLMClient, cfg: CriticConfig
) -> DraftingGraph:
    @traced_node("draft")
    async def draft(state: DraftState) -> dict[str, Any]:
        item, reports = state["item"], state.get("reports", {})
        errors = dict(state.get("errors", {}))
        out: dict[str, Any] = {}
        # Sequential on purpose — concurrent fallbacks would thrash models in and out of a 16 GB Mac's memory.
        for doc in state.get("todo", []):
            feedback = reports[doc].feedback() if doc in reports else None
            write = assets_agent.draft_cover_letter if doc == "cover_letter" else assets_agent.draft_cv
            key = "letter" if doc == "cover_letter" else "cv"
            try:
                out[key] = await write(item, profile, base_cv, llm, feedback)
                errors.pop(doc, None)
            except Exception as e:  # one document failing must not discard the other
                errors[doc] = repr(e)
                if doc not in reports:  # a failed *revision* keeps the rejected draft held for review
                    out[key] = None
        return DraftOutput(attempt=state.get("attempt", 0) + 1, errors=errors, **out).update()

    @traced_node("critique")
    async def critique(state: DraftState) -> dict[str, Any]:
        if not cfg.enabled:
            return CritiqueOutput(reports={}, todo=[]).update()
        item = state["item"]
        reports = dict(state.get("reports", {}))
        retry: list[DocumentKind] = []
        for doc in state.get("todo", []):
            drafted = state.get("letter") if doc == "cover_letter" else state.get("cv")
            if drafted is None or doc in state.get("errors", {}):  # nothing new to check this round
                continue
            report = await critic.review(
                drafted,
                profile=profile,
                base_cv=base_cv,
                job=item.listing,
                llm=llm if cfg.llm_review else None,
            )
            reports[doc] = report
            if not report.approved and has_findings(report):
                retry.append(doc)
        return CritiqueOutput(reports=reports, todo=retry).update()

    def after_critique(state: DraftState) -> str:
        if state.get("todo") and state.get("attempt", 0) <= cfg.max_revisions:
            log.info(
                "critic: revising {} for {} (round {})",
                state["todo"],
                state["item"].listing.company,
                state.get("attempt", 0) + 1,
            )
            return "draft"
        return END

    g = StateGraph(DraftState)
    g.add_node("draft", draft)
    g.add_node("critique", critique)
    g.add_edge(START, "draft")
    g.add_edge("draft", "critique")
    g.add_conditional_edges("critique", after_critique, ["draft", END])
    return g.compile()


@dataclass
class HeldDraft:
    draft: CoverLetter | TailoredCV
    report: CriticReport

    @property
    def quarantined(self) -> bool:
        return has_findings(self.report)


@dataclass
class DraftOutcome:
    letter: CoverLetter | None = None  # approved (or unchecked with the critic disabled)
    cv: TailoredCV | None = None
    held: dict[DocumentKind, HeldDraft] = field(default_factory=dict)
    errors: dict[DocumentKind, str] = field(default_factory=dict)
    rounds: int = 0


async def run_drafting(graph: DraftingGraph, item: ScoredListing) -> DraftOutcome:
    state = await graph.ainvoke(
        {"item": item, "todo": list(DOCUMENTS), "attempt": 0},
        config={"run_name": "jobhunter-drafting"},
    )
    out = DraftOutcome(errors=dict(state.get("errors", {})), rounds=state.get("attempt", 0))
    reports = state.get("reports", {})
    drafts: dict[DocumentKind, CoverLetter | TailoredCV | None] = {
        "cover_letter": state.get("letter"),
        "cv": state.get("cv"),
    }
    for doc, drafted in drafts.items():
        if drafted is None:
            continue
        report = reports.get(doc)
        if report is not None and not report.approved:
            out.held[doc] = HeldDraft(drafted, report)
        elif isinstance(drafted, CoverLetter):
            out.letter = drafted
        else:
            out.cv = drafted
    return out
