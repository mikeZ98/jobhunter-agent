"""Strict Pydantic v2 contracts for every transition in the daily graph.

LangGraph channels are plain dict updates, so nothing stops a node from returning a misspelled key or a
``list[dict]`` where ``list[JobListing]`` belongs — the mistake only surfaces in a later node, far from its
cause. Here every node builds its update through a ``NodeOutput`` model:

* ``extra="forbid"``  — an unknown key is a ValidationError at the node that produced it;
* ``strict=True``     — no silent coercion (a ``"3"`` is not an ``int``);
* ``InstanceOf[...]`` — domain objects must arrive as instances: strict mode alone still builds a nested
  model from a dict, which would hide a node that forgot to construct (and so validate) its objects;
* ``frozen=True``     — an emitted update cannot be mutated by a later node.

``update()`` returns the channel dict LangGraph merges, preserving the validated objects themselves (no
re-serialisation, so reducers like ``operator.add`` keep working on real instances).

``RunRequest`` validates the graph *input* the same way, so ``--board typo`` fails before a browser is
launched instead of inside the collect node.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, InstanceOf, field_validator

from jobhunter.agents.linkedin_messaging import InboxMessage, RecruiterTriage
from jobhunter.integrations.notes_render import DigestItem
from jobhunter.models import Board, JobListing, ScoredListing

NodeName = Literal["collect", "normalize", "score", "prepare", "inbox", "publish", "autofill"]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class RunRequest(StrictModel):
    """Input to the daily graph — what ``jobhunter run`` was asked to do."""

    run_date: date
    boards: tuple[Board, ...] | None = Field(
        default=None, description="Limit collection to these boards; None means every enabled board."
    )
    skip_collect: bool = False
    with_inbox: bool = False
    apply: bool = False

    @field_validator("boards", mode="before")
    @classmethod
    def _boards(cls, v: object) -> object:
        """Accept board names from the CLI; reject unknown ones with the list of valid names."""
        if v is None or (isinstance(v, list | tuple) and not v):
            return None
        if not isinstance(v, list | tuple):
            return v
        valid = {b.value: b for b in Board}
        unknown = [b for b in v if isinstance(b, str) and b not in valid]
        if unknown:
            raise ValueError(f"unknown board(s) {unknown}; choose from {sorted(valid)}")
        return tuple(valid[b] if isinstance(b, str) else b for b in v)

    def to_state(self) -> dict[str, Any]:
        """The initial ``PipelineState``. Dates and enums cross into the graph as plain values."""
        return {
            "run_date": self.run_date.isoformat(),
            "boards": [b.value for b in self.boards] if self.boards else None,
            "skip_collect": self.skip_collect,
            "with_inbox": self.with_inbox,
            "apply": self.apply,
        }


class BoardStats(StrictModel):
    """Per-board collection counters (see ``agents.collector.collect_board``)."""

    seen: int = Field(0, ge=0)
    known: int = Field(0, ge=0)
    duplicates: int = Field(0, ge=0)
    filtered: int = Field(0, ge=0)
    resumed: int = Field(0, ge=0)
    enriched: int = Field(0, ge=0)
    error: str | None = None


class NodeOutput(StrictModel):
    """Base for a node's state update. Subclasses declare exactly the channels the node may write."""

    @classmethod
    def channels(cls) -> frozenset[str]:
        """State channels this node writes — checked against ``PipelineState`` by the tests."""
        return frozenset(cls.model_fields)

    def update(self) -> dict[str, Any]:
        """The LangGraph update: one entry per declared field, validated instances preserved."""
        return {name: getattr(self, name) for name in type(self).model_fields}


class CollectOutput(NodeOutput):
    listings: list[InstanceOf[JobListing]]
    board_stats: dict[str, BoardStats] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    def update(self) -> dict[str, Any]:
        # Board stats travel as plain dicts: the digest renders them and checkpointers serialise them.
        return {
            "listings": self.listings,
            "board_stats": {b: st.model_dump() for b, st in self.board_stats.items()},
            "errors": self.errors,
        }


class NormalizeOutput(NodeOutput):
    listings: list[InstanceOf[JobListing]]


class ScoreOutput(NodeOutput):
    scored: list[InstanceOf[ScoredListing]]
    shortlist: list[InstanceOf[ScoredListing]]


class PrepareOutput(NodeOutput):
    items: list[InstanceOf[DigestItem]] = Field(
        min_length=1, max_length=1, description="Exactly one per fan-out branch."
    )
    errors: list[str] = Field(default_factory=list)


class InboxOutput(NodeOutput):
    inbox: list[tuple[InstanceOf[InboxMessage], InstanceOf[RecruiterTriage]]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PublishOutput(NodeOutput):
    note_id: str | None = None
    errors: list[str] = Field(default_factory=list)


class AutofillOutput(NodeOutput):
    """Autofill is interactive and writes no channels; the empty model still forbids stray keys."""


NODE_OUTPUTS: dict[NodeName, type[NodeOutput]] = {
    "collect": CollectOutput,
    "normalize": NormalizeOutput,
    "score": ScoreOutput,
    "prepare": PrepareOutput,
    "inbox": InboxOutput,
    "publish": PublishOutput,
    "autofill": AutofillOutput,
}
