import pytest

from jobhunter.db import InvalidTransition
from jobhunter.models import ListingStatus, ScoredListing
from tests.conftest import make_job


def test_upsert_is_idempotent(db, job):
    db.upsert_listings([job])
    db.upsert_listings([job])
    assert db._conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == 1


def test_known_keys_only_counts_scored_rows(db, job):
    db.upsert_listings([job])
    assert db.known_keys([job.source_key]) == set()  # stored by a crashed run -> retry
    db.save_scores([ScoredListing(listing=job, final_score=70)])
    assert db.known_keys([job.source_key]) == {job.source_key}
    assert db.known_fingerprints([job.fingerprint]) == {job.fingerprint}


def test_cross_board_fingerprint_matches_reposts():
    a = make_job(company="Acme Sp. z o.o.", title="Python Developer (k/m)")
    b = make_job(board="nofluffjobs", external_id="x", company="ACME", title="Python  Developer")
    assert a.source_key != b.source_key
    assert a.fingerprint == b.fingerprint


def test_status_transitions_are_enforced(db, job):
    db.upsert_listings([job])
    with pytest.raises(InvalidTransition):
        db.set_status(job.source_key, ListingStatus.APPLIED)  # can't apply before preparing
    db.set_status(job.source_key, ListingStatus.PREPARED)
    db.set_status(job.source_key, ListingStatus.APPLIED, note="manual submit")
    db.set_status(job.source_key, ListingStatus.INTERVIEW)
    history = db._conn.execute("SELECT from_status, to_status FROM status_history ORDER BY id").fetchall()
    assert [tuple(r) for r in history] == [
        ("scraped", "prepared"),
        ("prepared", "applied"),
        ("applied", "interview"),
    ]


def test_force_overrides_transition(db, job):
    db.upsert_listings([job])
    db.set_status(job.source_key, ListingStatus.INTERVIEW, force=True)
    assert db.get(job.source_key)["status"] == "interview"


def test_stats_shape(db, job):
    db.upsert_listings([job])
    db.log_llm_call(
        task="score",
        provider="ollama",
        model="qwen",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=12.5,
        success=True,
    )
    db.log_antibot("pracuj", "cloudflare", 403, "https://it.pracuj.pl")
    st = db.stats()
    assert st["funnel"] == {"scraped": 1}
    assert st["llm"][0]["calls"] == 1 and st["antibot"][0]["kind"] == "cloudflare"
