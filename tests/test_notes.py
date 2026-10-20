import subprocess
import sys
from datetime import date

import pytest

from jobhunter.integrations.apple_notes import JXA_UPSERT
from jobhunter.integrations.notes_render import Digest, DigestItem, render
from jobhunter.models import MatchAssessment, ScoredListing
from tests.conftest import make_job


def test_render_starts_with_title_and_escapes_html():
    sl = ScoredListing(
        listing=make_job(title="<script>alert(1)</script> Dev"),
        final_score=81.4,
        llm=MatchAssessment(score=80, strengths=["FastAPI & RAG"], gaps=[], pitch="DocuMind → RAG"),
    )
    d = Digest(
        day=date(2026, 10, 19), items=[DigestItem(scored=sl)], board_stats={"justjoinit": {"seen": 120}}
    )
    html = render(d)
    assert html.startswith("<h1>Job Hunt — 2026-10-19</h1>")  # Notes derives the title from line 1
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "FastAPI &amp; RAG" in html and "<table>" in html and "81" in html


@pytest.mark.skipif(sys.platform != "darwin", reason="osacompile is macOS-only")
def test_jxa_script_compiles(tmp_path):
    src = tmp_path / "upsert.js"
    src.write_text(JXA_UPSERT)
    r = subprocess.run(
        ["osacompile", "-l", "JavaScript", "-o", str(tmp_path / "out.scpt"), str(src)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
