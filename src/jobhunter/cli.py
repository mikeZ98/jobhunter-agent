"""``jobhunter`` CLI."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sqlite3
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from rich.console import Console
from rich.table import Table

from jobhunter.config import ROOT, Settings, get_settings
from jobhunter.db import Database
from jobhunter.models import ListingStatus
from jobhunter.telemetry import setup_telemetry

if TYPE_CHECKING:
    from jobhunter.graph import Deps, PipelineState

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="JobHunter-Agent — HITL job search pipeline."
)
console = Console()
EXAMPLES = Path(__file__).parent / "examples"


def _boot() -> tuple[Settings, Database]:
    s = get_settings()
    setup_telemetry(s.log_level, s.otel_endpoint, ROOT / "logs")
    return s, Database(s.db_path)


async def _deps(s: Settings, db: Database) -> Deps:
    from jobhunter.graph import Deps, load_profile
    from jobhunter.integrations.apple_notes import AppleNotes
    from jobhunter.llm import Embedder, LLMClient
    from jobhunter.scrapers.browser import BrowserManager

    profile, base_cv = load_profile(s)
    browser = await BrowserManager(s).__aenter__()
    notes = AppleNotes(s.notes.folder, s.notes.account) if AppleNotes.available() else None
    return Deps(s, db, LLMClient(s, db), Embedder(s), profile, base_cv, browser, notes)


@app.command()
def init() -> None:
    """Create data/profile.json, data/base_cv.md, jobhunter.toml and .env from templates."""
    s = get_settings()
    for src, dst in [
        ("profile.example.json", s.profile_path),
        ("base_cv.example.md", s.base_cv_path),
        ("jobhunter.example.toml", ROOT / "jobhunter.toml"),
        ("env.example", ROOT / ".env"),
    ]:
        if dst.exists():
            console.print(f"[dim]exists:[/dim] {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(EXAMPLES / src, dst)
            console.print(f"[green]created:[/green] {dst}")
    console.print("\nNext: edit data/profile.json + data/base_cv.md, then `jobhunter doctor`.")


@app.command()
def run(
    boards: Annotated[
        list[str] | None, typer.Option("--board", "-b", help="Limit to boards (repeatable)")
    ] = None,
    apply: Annotated[bool, typer.Option(help="After the digest, open the browser for HITL autofill")] = False,
    inbox: Annotated[bool, typer.Option(help="Also triage LinkedIn recruiter messages")] = False,
    notes: Annotated[bool, typer.Option(help="Write the Apple Notes digest")] = True,
    skip_collect: Annotated[
        bool, typer.Option(help="Don't scrape: only re-prepare jobs whose cover letter failed, then publish")
    ] = False,
) -> None:
    """Daily pipeline: scrape → normalize → score → prepare assets → digest (→ autofill)."""
    from pydantic import ValidationError

    from jobhunter.workflow.schemas import RunRequest

    try:  # before _boot(): a typo in --board must not cost a browser launch
        request = RunRequest.model_validate(
            {
                "run_date": date.today(),
                "boards": boards,
                "skip_collect": skip_collect,
                "with_inbox": inbox,
                "apply": apply,
            }
        )
    except ValidationError as e:
        raise typer.BadParameter(e.errors()[0]["msg"], param_hint="--board") from None
    s, db = _boot()
    s.notes.enabled = notes

    async def main() -> dict[str, Any]:
        from jobhunter.graph import build_daily_graph

        deps = await _deps(s, db)
        try:
            if s.llm.local_only:  # fail before 6 minutes of scraping, not after
                probe = deps.llm.router.probe
                await asyncio.to_thread(probe.refresh, True)
                if not probe._ollama_up:
                    console.print(
                        f"[red]Ollama is not reachable at {s.llm.ollama_base_url}.[/red] "
                        "Start it (open -a Ollama) and run again."
                    )
                    raise typer.Exit(1)
            graph = build_daily_graph(deps)
            result: dict[str, Any] = await graph.ainvoke(
                cast("PipelineState", request.to_state()),
                config={"recursion_limit": 50, "run_name": "jobhunter-daily"},
            )
        finally:
            with contextlib.suppress(Exception):  # driver may already be gone after Ctrl-C
                await deps.browser.__aexit__(None, None, None)
        return result

    try:
        result = asyncio.run(main())
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted.[/yellow] Listings fetched so far are saved in job_hunter.db — "
            "the next `jobhunter run` reuses them instead of downloading them again."
        )
        raise typer.Exit(130) from None
    _print_digest(result)


def _print_digest(result: dict[str, Any]) -> None:
    t = Table(title=f"Daily digest — {len(result.get('items', []))} prepared")
    for col in ("Score", "Role", "Company", "Salary (B2B eq.)", "Board", "Files"):
        t.add_column(col, overflow="fold")
    for it in sorted(result.get("items", []), key=lambda i: i.scored.final_score, reverse=True):
        j = it.scored.listing
        best = j.best_salary
        sal = (
            f"{(best.b2b_equiv_min or 0) / 1000:.0f}–{(best.b2b_equiv_max or 0) / 1000:.0f}k" if best else "—"
        )
        t.add_row(
            f"{it.scored.final_score:.0f}",
            j.title,
            j.company,
            sal,
            j.board.value,
            "⚠️ " + it.error if it.error else (it.output_dir or ""),
        )
    console.print(t)
    for e in result.get("errors", []):
        console.print(f"[yellow]⚠ {e}[/yellow]")
    if result.get("note_id"):
        console.print("[green]Apple Note updated[/green] (folder: Job Search)")


@app.command()
def publish(day: Annotated[str | None, typer.Argument(help="YYYY-MM-DD (default: today)")] = None) -> None:
    """Send an already-built digest (output/<day>/digest.html) to Apple Notes again."""
    from jobhunter.integrations.apple_notes import AppleNotes, AppleNotesError

    s, _db = _boot()
    day = day or date.today().isoformat()
    path = s.output_dir / day / "digest.html"
    if not path.exists():
        console.print(f"[red]{path} not found[/red] — run `jobhunter run` first.")
        raise typer.Exit(1)
    try:
        res = AppleNotes(s.notes.folder, s.notes.account).upsert(f"Job Hunt — {day}", path.read_text())
    except AppleNotesError as e:
        console.print(f"[red]Apple Notes failed:[/red] {e}")
        raise typer.Exit(1) from None
    console.print(f"[green]Note {res.action}[/green] in “{s.notes.folder}”: Job Hunt — {day}")


@app.command("apply")
def apply_cmd(limit: Annotated[int, typer.Option(help="Max listings to review")] = 15) -> None:
    """Open the authenticated browser and pre-fill Prepared applications one by one (you submit)."""
    s, db = _boot()

    async def main() -> None:
        from jobhunter.agents.autofill import review_loop

        deps = await _deps(s, db)
        try:
            rows = db.by_status(ListingStatus.PREPARED, limit=limit)
            if not rows:
                console.print("Nothing in Prepared. Run `jobhunter run` first.")
                return
            async with deps.browser.persistent_profile() as ctx:
                await review_loop(ctx, db, deps.profile, deps.llm, rows)
        finally:
            with contextlib.suppress(Exception):  # driver may already be gone after Ctrl-C
                await deps.browser.__aexit__(None, None, None)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow] Applications you confirmed with ENTER are saved.")
        raise typer.Exit(130) from None


@app.command("inbox")
def inbox_cmd(limit: int = 20) -> None:
    """Draft replies to LinkedIn recruiter messages (never sends)."""
    s, db = _boot()

    async def main() -> None:
        from jobhunter.agents.linkedin_messaging import draft_replies, fetch_inbox
        from jobhunter.graph import load_profile
        from jobhunter.llm import LLMClient

        profile, _ = load_profile(s)
        msgs = await asyncio.to_thread(fetch_inbox, s, limit)
        for m, t in await draft_replies(msgs, profile, LLMClient(s, db), db, s.llm.max_concurrency):
            console.rule(f"{m.sender} — {t.company or '?'} / {t.role or '?'}  fit={t.fit_score} [{t.intent}]")
            console.print(f"[dim]{m.text[:400]}[/dim]\n")
            console.print(t.reply)

    asyncio.run(main())


@app.command()
def login(target: Annotated[str, typer.Argument(help="pracuj | rocketjobs | linkedin | <url>")]) -> None:
    """Open a headed browser to log in / clear a WAF challenge once; the session is persisted."""
    s, _ = _boot()
    urls = {
        "pracuj": "https://it.pracuj.pl/praca",
        "rocketjobs": "https://rocketjobs.pl/",
        "linkedin": "https://www.linkedin.com/login",
    }

    async def main() -> None:
        from jobhunter.scrapers.browser import BrowserManager

        async with BrowserManager(s) as bm:
            if target in ("pracuj", "rocketjobs"):  # scraping context -> storage_state for that board
                cm = bm.scraping_context(target, headless=False)
            else:  # persistent profile -> used by autofill (LinkedIn, ATS logins...)
                cm = bm.persistent_profile()
            async with cm as ctx:
                page = ctx.pages[0] if ctx.pages else await ctx.new_page()
                await page.goto(urls.get(target, target))
                await asyncio.to_thread(
                    input, "Log in / pass the challenge in the browser, then press ENTER… "
                )

    asyncio.run(main())


@app.command()
def status(
    key: Annotated[str, typer.Argument(help="source_key (prefix ok)")],
    new_status: Annotated[ListingStatus, typer.Argument()],
    note: str | None = None,
    force: bool = False,
) -> None:
    """Move a listing through the funnel (e.g. applied → interview)."""
    _, db = _boot()
    row = _find(db, key)
    db.set_status(row["source_key"], new_status, note, force=force)
    console.print(f"{row['title']} @ {row['company']}: → [bold]{new_status.value}[/bold]")


@app.command()
def interview(
    key: str,
    stage: Annotated[str, typer.Option(help="hr_screen | technical | system_design | final | offer")],
    outcome: str = "pending",
    at: Annotated[str | None, typer.Option(help="ISO datetime")] = None,
    notes: str | None = None,
) -> None:
    """Log an interview round (moves the listing to Interview)."""
    _, db = _boot()
    row = _find(db, key)
    db.log_interview(row["source_key"], stage, at, outcome, notes)
    if row["status"] != ListingStatus.INTERVIEW.value:
        db.set_status(row["source_key"], ListingStatus.INTERVIEW, note=stage, force=True)
    console.print(f"logged {stage} ({outcome}) for {row['title']} @ {row['company']}")


@app.command("list")
def list_cmd(
    status_: Annotated[ListingStatus, typer.Option("--status")] = ListingStatus.PREPARED, limit: int = 30
) -> None:
    """List listings by status."""
    _, db = _boot()
    t = Table()
    for c in ("key", "score", "title", "company", "board", "updated"):
        t.add_column(c, overflow="fold")
    for r in db.by_status(status_, limit):
        t.add_row(
            r["source_key"][:10],
            f"{r['final_score'] or 0:.0f}",
            r["title"],
            r["company"],
            r["board"],
            r["updated_at"][:16],
        )
    console.print(t)


@app.command()
def stats() -> None:
    """Funnel, LLM token/latency/parse metrics and anti-bot events."""
    _, db = _boot()
    st = db.stats()
    console.print("[bold]Funnel[/bold]", st["funnel"])
    console.print("[bold]By board[/bold]", st["by_board"])
    t = Table(title="LLM calls")
    cols = (
        "provider",
        "model",
        "task",
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "avg_ms",
        "success_pct",
        "retries",
    )
    for c in cols:
        t.add_column(c)
    for r in st["llm"]:
        t.add_row(*(str(r[c]) for c in cols))
    console.print(t)
    a = Table(title="Anti-bot events (14 days)")
    for c in ("board", "kind", "n", "last_at"):
        a.add_column(c)
    for r in st["antibot"]:
        a.add_row(*(str(r[c]) for c in ("board", "kind", "n", "last_at")))
    console.print(a)


@app.command()
def doctor() -> None:
    """Check Ollama models, API keys, Playwright, Apple Notes access and profile validity."""
    s, _ = _boot()
    from jobhunter.integrations.apple_notes import AppleNotes
    from jobhunter.llm.router import ModelRouter, Task

    router = ModelRouter(s)
    router.probe.refresh(force=True)

    def ok(passed: bool) -> str:
        return "[green]✓[/green]" if passed else "[red]✗[/red]"

    console.print(f"{ok(router.probe._ollama_up)} Ollama at {s.llm.ollama_base_url}")
    seen = set()
    for models in router.matrix.values():
        for m in map(router.probe.resolve, models):
            if m.local and m.key not in seen:
                seen.add(m.key)
                console.print(
                    f"   {ok(router.probe.installed(m))} {m.name} (~{m.mem_gb} GB)"
                    + ("" if router.probe.installed(m) else f"  → ollama pull {m.name}")
                )
    if s.llm.local_only:
        console.print("[green]✓[/green] local_only = true — cloud models disabled (API keys ignored)")
    else:
        console.print(
            f"{ok(bool(s.openrouter_api_key))} OPENROUTER_API_KEY   "
            f"{ok(bool(s.anthropic_api_key))} ANTHROPIC_API_KEY   {ok(bool(s.openai_api_key))} OPENAI_API_KEY"
        )
    console.print(f"   free memory for local models: {router.probe.free_memory_gb():.1f} GB")
    for task in Task:
        try:
            d = router.route(task, "x" * 12000, 1500)
            console.print(f"   route {task.value:<12} → {d.model.key}")
        except Exception as e:
            console.print(f"   route {task.value:<12} → [red]{e}[/red]")
    # Honours PLAYWRIGHT_BROWSERS_PATH (pinned to the project's .cache/ by .envrc), like Playwright does.
    chromium = Path(
        os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or Path.home() / "Library/Caches/ms-playwright"
    )
    console.print(f"{ok(chromium.exists())} Playwright browsers (`uv run playwright install chromium`)")
    console.print(f"{ok(AppleNotes.available())} osascript / Apple Notes")
    try:
        from jobhunter.graph import load_profile

        p, cv = load_profile(s)
        console.print(
            f"{ok(True)} profile: {p.full_name}, {len(p.skills)} skills, {len(p.projects)} projects, "
            f"base CV {len(cv)} chars"
        )
    except Exception as e:
        console.print(f"{ok(False)} profile: {e}")
    console.print(f"{ok(bool(s.linkedin_li_at))} LinkedIn cookies (optional, for `inbox`)")


def _find(db: Database, key: str) -> sqlite3.Row:
    rows: list[sqlite3.Row] = db._conn.execute(
        "SELECT * FROM listings WHERE source_key LIKE ? LIMIT 2", (key + "%",)
    ).fetchall()
    if len(rows) != 1:
        raise typer.BadParameter(f"{'no' if not rows else 'ambiguous'} listing for key {key!r}")
    return rows[0]


@app.command()
def show(key: str) -> None:
    """Show one listing with scores and generated asset paths."""
    _, db = _boot()
    r = _find(db, key)
    console.print_json(
        json.dumps(
            {
                k: (json.loads(r[k]) if k.endswith("_json") and r[k] else r[k])
                for k in r
                if k != "listing_json"
            },
            default=str,
        )
    )


if __name__ == "__main__":
    app()
