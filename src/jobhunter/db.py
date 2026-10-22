"""SQLite persistence (``job_hunter.db``).

Design notes
* ``listings.source_key`` (board+external id) is the PK -> re-scrapes are idempotent upserts.
* ``listings.fingerprint`` (company+title slug) collapses the same role cross-posted on several boards.
* Status changes are append-only in ``status_history`` so the funnel can be reconstructed later.
* stdlib ``sqlite3`` + WAL: every write here is sub-millisecond, an async driver would add nothing.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from jobhunter.models import STATUS_TRANSITIONS, JobListing, ListingStatus, ScoredListing

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS listings (
    source_key    TEXT PRIMARY KEY,
    fingerprint   TEXT NOT NULL,
    board         TEXT NOT NULL,
    external_id   TEXT NOT NULL,
    url           TEXT NOT NULL,
    title         TEXT NOT NULL,
    company       TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'scraped',
    final_score   REAL,
    listing_json  TEXT NOT NULL,
    scored_json   TEXT,
    assets_json   TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_listings_fingerprint ON listings(fingerprint);
CREATE INDEX IF NOT EXISTS ix_listings_status_score ON listings(status, final_score DESC);

CREATE TABLE IF NOT EXISTS status_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key  TEXT NOT NULL REFERENCES listings(source_key) ON DELETE CASCADE,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    note        TEXT,
    at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interviews (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key   TEXT NOT NULL REFERENCES listings(source_key) ON DELETE CASCADE,
    stage        TEXT NOT NULL,          -- hr_screen | technical | system_design | final | offer
    scheduled_at TEXT,
    outcome      TEXT,                   -- passed | failed | pending | withdrawn
    notes        TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS antibot_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    board       TEXT NOT NULL,
    kind        TEXT NOT NULL,           -- cloudflare | captcha | rate_limit | blocked | circuit_open
    status_code INTEGER,
    url         TEXT,
    at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    task              TEXT NOT NULL,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency_ms        REAL,
    success           INTEGER NOT NULL,
    retries           INTEGER NOT NULL DEFAULT 0,
    error             TEXT,
    at                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recruiter_messages (
    conversation_id TEXT PRIMARY KEY,
    sender          TEXT,
    last_message    TEXT,
    analysis_json   TEXT,
    draft_reply     TEXT,
    status          TEXT NOT NULL DEFAULT 'drafted',   -- drafted | sent_manually | ignored
    updated_at      TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class InvalidTransition(ValueError):
    pass


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    # ---------- listings ----------

    def known_keys(self, keys: Iterable[str]) -> set[str]:
        """Keys already *scored* — a listing stored by a crashed run gets another chance."""
        keys = list(keys)
        if not keys:
            return set()
        found: set[str] = set()
        for i in range(0, len(keys), 500):  # stay under SQLITE_MAX_VARIABLE_NUMBER
            chunk = keys[i : i + 500]
            q = (
                "SELECT source_key FROM listings WHERE scored_json IS NOT NULL AND source_key IN"
                f" ({','.join('?' * len(chunk))})"
            )
            found |= {r[0] for r in self._conn.execute(q, chunk)}
        return found

    def known_fingerprints(self, fps: Iterable[str]) -> set[str]:
        fps = list(fps)
        found: set[str] = set()
        for i in range(0, len(fps), 500):
            chunk = fps[i : i + 500]
            q = f"SELECT DISTINCT fingerprint FROM listings WHERE scored_json IS NOT NULL AND fingerprint IN ({','.join('?' * len(chunk))})"
            found |= {r[0] for r in self._conn.execute(q, chunk)}
        return found

    def pending_listings(self, keys: Iterable[str]) -> dict[str, JobListing]:
        """Listings fetched by an earlier run that never got scored (e.g. interrupted with Ctrl-C).
        The collector reuses them instead of fetching their detail pages again."""
        keys = list(keys)
        out: dict[str, JobListing] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            q = (
                "SELECT source_key, listing_json FROM listings WHERE scored_json IS NULL AND source_key IN"
                f" ({','.join('?' * len(chunk))})"
            )
            for key, js in self._conn.execute(q, chunk):
                try:
                    out[key] = JobListing.model_validate_json(js)
                except ValueError:
                    continue  # stored by an older schema -> just fetch it again
        return out

    def failed_prepares(self, days: int = 7, limit: int = 15) -> list[ScoredListing]:
        """Shortlisted listings whose cover letter was never produced (LLM failure, critic unavailable,
        Ctrl-C during ``prepare``). They are already "known", so collection skips them; the score node
        re-queues them. Drafts the critic *quarantined* are held for a human and never re-queued."""
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
        rows = self._conn.execute(
            "SELECT scored_json FROM listings WHERE status=? AND scored_json IS NOT NULL"
            " AND assets_json IS NOT NULL AND assets_json NOT LIKE '%cover_letter_md%'"
            " AND assets_json NOT LIKE '%\"quarantined\"%'"
            " AND last_seen_at >= ? ORDER BY final_score DESC LIMIT ?",
            (ListingStatus.SCRAPED.value, since, limit),
        ).fetchall()
        out: list[ScoredListing] = []
        for (js,) in rows:
            try:
                out.append(ScoredListing.model_validate_json(js))
            except ValueError:
                continue
        return out

    def upsert_listings(self, listings: Iterable[JobListing]) -> int:
        now = _now()
        rows = [
            (
                job.source_key,
                job.fingerprint,
                job.board.value,
                job.external_id,
                job.url,
                job.title,
                job.company,
                job.model_dump_json(),
                now,
                now,
                now,
            )
            for job in listings
        ]
        with self.tx() as c:
            c.executemany(
                """INSERT INTO listings (source_key, fingerprint, board, external_id, url, title, company,
                                         listing_json, first_seen_at, last_seen_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(source_key) DO UPDATE SET
                       last_seen_at = excluded.last_seen_at,
                       listing_json = excluded.listing_json""",
                rows,
            )
        return len(rows)

    def save_scores(self, scored: Iterable[ScoredListing]) -> None:
        with self.tx() as c:
            c.executemany(
                "UPDATE listings SET final_score=?, scored_json=?, updated_at=? WHERE source_key=?",
                [(s.final_score, s.model_dump_json(), _now(), s.listing.source_key) for s in scored],
            )

    def save_assets(self, source_key: str, assets: dict[str, Any]) -> None:
        self._conn.execute(
            "UPDATE listings SET assets_json=?, updated_at=? WHERE source_key=?",
            (json.dumps(assets, ensure_ascii=False, default=str), _now(), source_key),
        )

    def get(self, source_key: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM listings WHERE source_key=?", (source_key,)
        ).fetchone()
        return row

    def by_status(self, status: ListingStatus, limit: int = 100) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM listings WHERE status=? ORDER BY final_score DESC NULLS LAST LIMIT ?",
            (status.value, limit),
        ).fetchall()

    def set_status(
        self, source_key: str, to: ListingStatus, note: str | None = None, *, force: bool = False
    ) -> None:
        with self.tx() as c:
            row = c.execute("SELECT status FROM listings WHERE source_key=?", (source_key,)).fetchone()
            if row is None:
                raise KeyError(source_key)
            current = ListingStatus(row["status"])
            if not force and to not in STATUS_TRANSITIONS[current]:
                raise InvalidTransition(f"{current} -> {to} not allowed (use force=True to override)")
            c.execute(
                "UPDATE listings SET status=?, updated_at=? WHERE source_key=?",
                (to.value, _now(), source_key),
            )
            c.execute(
                "INSERT INTO status_history (source_key, from_status, to_status, note, at) VALUES (?,?,?,?,?)",
                (source_key, current.value, to.value, note, _now()),
            )

    def log_interview(
        self,
        source_key: str,
        stage: str,
        scheduled_at: str | None = None,
        outcome: str = "pending",
        notes: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO interviews (source_key, stage, scheduled_at, outcome, notes, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (source_key, stage, scheduled_at, outcome, notes, _now()),
        )

    # ---------- telemetry sinks ----------

    def log_antibot(self, board: str, kind: str, status_code: int | None, url: str | None) -> None:
        self._conn.execute(
            "INSERT INTO antibot_events (board, kind, status_code, url, at) VALUES (?,?,?,?,?)",
            (board, kind, status_code, url, _now()),
        )

    def log_llm_call(self, **kw: Any) -> None:
        kw.setdefault("at", _now())
        kw["success"] = int(kw.get("success", True))
        cols = ",".join(kw)
        self._conn.execute(
            f"INSERT INTO llm_calls ({cols}) VALUES ({','.join('?' * len(kw))})", tuple(kw.values())
        )

    def upsert_recruiter_message(self, conversation_id: str, **kw: Any) -> None:
        kw["updated_at"] = _now()
        cols = ["conversation_id", *kw]
        updates = ",".join(f"{k}=excluded.{k}" for k in kw)
        self._conn.execute(
            f"INSERT INTO recruiter_messages ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
            f" ON CONFLICT(conversation_id) DO UPDATE SET {updates}",
            (conversation_id, *kw.values()),
        )

    def stats(self) -> dict[str, Any]:
        q = self._conn.execute
        return {
            "funnel": dict(q("SELECT status, COUNT(*) FROM listings GROUP BY status").fetchall()),
            "by_board": dict(q("SELECT board, COUNT(*) FROM listings GROUP BY board").fetchall()),
            "llm": [
                dict(r)
                for r in q(
                    """SELECT provider, model, task, COUNT(*) calls, SUM(prompt_tokens) prompt_tokens,
                          SUM(completion_tokens) completion_tokens, ROUND(AVG(latency_ms)) avg_ms,
                          ROUND(100.0 * SUM(success) / COUNT(*), 1) success_pct, SUM(retries) retries
                   FROM llm_calls GROUP BY provider, model, task ORDER BY calls DESC"""
                )
            ],
            "antibot": [
                dict(r)
                for r in q(
                    """SELECT board, kind, COUNT(*) n, MAX(at) last_at FROM antibot_events
                   WHERE at >= datetime('now', '-14 days') GROUP BY board, kind ORDER BY n DESC"""
                )
            ],
        }
