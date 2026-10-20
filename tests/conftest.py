from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobhunter.config import Settings
from jobhunter.db import Database
from jobhunter.models import Board, CandidateProfile, JobListing, RemoteMode, SalaryRange, Seniority

EXAMPLES = Path(__file__).parents[1] / "src" / "jobhunter" / "examples"


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    # Settings reads ./jobhunter.toml and ./.env — run from an empty dir so the user's real config never leaks in.
    monkeypatch.chdir(tmp_path)
    s = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        state_dir=tmp_path / ".state",
        db_path=tmp_path / "test.db",
        profile_path=EXAMPLES / "profile.example.json",
        base_cv_path=EXAMPLES / "base_cv.example.md",
    )
    s.ensure_dirs()
    return s


@pytest.fixture
def db(settings: Settings) -> Database:
    d = Database(settings.db_path)
    yield d
    d.close()


@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile.model_validate(json.loads((EXAMPLES / "profile.example.json").read_text()))


def make_job(**kw) -> JobListing:
    base = dict(
        board=Board.JUSTJOINIT,
        external_id="acme-python-dev",
        url="https://justjoin.it/job-offer/acme",
        title="Python Developer",
        company="Acme",
        description="Build FastAPI services with PostgreSQL.",
        seniority=Seniority.MID,
        remote=RemoteMode.REMOTE,
        locations=["Warszawa"],
        required_skills=["Python", "FastAPI", "PostgreSQL"],
        nice_to_have_skills=["Kubernetes"],
        salaries=[
            SalaryRange(
                min=15000, max=20000, contract="b2b", is_gross=False, b2b_equiv_min=15000, b2b_equiv_max=20000
            )
        ],
    )
    base.update(kw)
    return JobListing(**base)


@pytest.fixture
def job() -> JobListing:
    return make_job()
