"""Transition contracts: every node update is strict, closed and immutable; the graph input is validated."""

from __future__ import annotations

from datetime import date
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from jobhunter.graph import Deps, PipelineState, build_daily_graph
from jobhunter.integrations.notes_render import DigestItem
from jobhunter.models import Board, ScoredListing
from jobhunter.workflow.schemas import (
    NODE_OUTPUTS,
    AutofillOutput,
    BoardStats,
    CollectOutput,
    InboxOutput,
    NormalizeOutput,
    PrepareOutput,
    PublishOutput,
    RunRequest,
    ScoreOutput,
)


def test_every_graph_node_has_a_contract(settings, db):
    deps = Deps(settings, db, None, None, None, "", browser=None, notes=None)  # type: ignore[arg-type]
    nodes = set(build_daily_graph(deps).get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == set(NODE_OUTPUTS)


@pytest.mark.parametrize("name", sorted(NODE_OUTPUTS))
def test_contract_channels_exist_in_pipeline_state(name):
    assert NODE_OUTPUTS[name].channels() <= set(get_type_hints(PipelineState, include_extras=True))


def test_unknown_channel_is_rejected_at_the_producing_node(job):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        NormalizeOutput.model_validate({"listings": [job], "listing": [job]})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AutofillOutput.model_validate({"items": []})


def test_strict_mode_refuses_coercion(job):
    with pytest.raises(ValidationError):
        NormalizeOutput.model_validate({"listings": [job.model_dump()]})  # dict is not a JobListing
    with pytest.raises(ValidationError):
        BoardStats.model_validate({"seen": "3"})
    with pytest.raises(ValidationError):
        PublishOutput.model_validate({"note_id": 42})


def test_board_stats_are_non_negative():
    with pytest.raises(ValidationError):
        BoardStats(seen=-1)


def test_outputs_are_frozen(job):
    out = NormalizeOutput(listings=[job])
    with pytest.raises(ValidationError, match="frozen"):
        out.listings = []  # type: ignore[misc]


def test_update_preserves_validated_instances(job):
    sl = ScoredListing(listing=job, final_score=0.7)
    upd = ScoreOutput(scored=[sl], shortlist=[sl]).update()
    assert set(upd) == {"scored", "shortlist"}
    assert upd["scored"][0] is sl and upd["shortlist"][0] is sl


def test_collect_update_emits_board_stats_as_plain_dicts(job):
    upd = CollectOutput(listings=[job], board_stats={"justjoinit": BoardStats(seen=5, filtered=2)}).update()
    assert upd["board_stats"]["justjoinit"] == {
        "seen": 5,
        "known": 0,
        "duplicates": 0,
        "filtered": 2,
        "resumed": 0,
        "enriched": 0,
        "error": None,
    }
    assert upd["listings"] == [job] and upd["errors"] == []


def test_board_stats_accepts_collector_counters():
    # Mirrors the dict built by ``agents.collector.collect_board`` — a renamed counter must fail here.
    raw = {
        "seen": 3,
        "known": 1,
        "duplicates": 0,
        "filtered": 1,
        "resumed": 0,
        "enriched": 1,
        "error": None,
    }
    assert BoardStats.model_validate(raw).model_dump() == raw


def test_prepare_emits_exactly_one_item(job):
    item = DigestItem(scored=ScoredListing(listing=job))
    assert PrepareOutput(items=[item]).update() == {"items": [item], "errors": [], "quarantined": []}
    with pytest.raises(ValidationError):
        PrepareOutput(items=[])
    with pytest.raises(ValidationError):
        PrepareOutput(items=[item, item])


def test_empty_outputs_still_write_their_channels():
    assert InboxOutput().update() == {"inbox": [], "errors": []}
    assert PublishOutput().update() == {"note_id": None, "errors": []}
    assert AutofillOutput().update() == {}


# ------------------------------------------------------------------ RunRequest


def test_run_request_converts_board_names_and_state_is_plain():
    req = RunRequest.model_validate({"run_date": date(2026, 10, 22), "boards": ["justjoinit", "pracuj"]})
    assert req.boards == (Board.JUSTJOINIT, Board.PRACUJ)
    assert req.to_state() == {
        "run_date": "2026-10-22",
        "boards": ["justjoinit", "pracuj"],
        "skip_collect": False,
        "with_inbox": False,
        "apply": False,
    }


@pytest.mark.parametrize("boards", [None, [], ()])
def test_run_request_no_boards_means_all(boards):
    req = RunRequest.model_validate({"run_date": date(2026, 10, 22), "boards": boards})
    assert req.boards is None and req.to_state()["boards"] is None


def test_run_request_unknown_board_lists_valid_names():
    with pytest.raises(ValidationError) as ei:
        RunRequest.model_validate({"run_date": date(2026, 10, 22), "boards": ["justjoin"]})
    msg = ei.value.errors()[0]["msg"]
    assert "'justjoin'" in msg and "nofluffjobs" in msg


def test_run_request_is_strict():
    with pytest.raises(ValidationError):
        RunRequest.model_validate({"run_date": "2026-10-22"})  # str is not a date in strict mode
    with pytest.raises(ValidationError):
        RunRequest.model_validate({"run_date": date(2026, 10, 22), "apply": "yes"})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RunRequest.model_validate({"run_date": date(2026, 10, 22), "inbox": True})
