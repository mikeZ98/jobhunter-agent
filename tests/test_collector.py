"""Collector bookkeeping: title filter before detail fetches, resume after an interrupted run."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import ClassVar

import pytest

from jobhunter.agents.collector import _collect_board, title_prefilter
from jobhunter.graph import load_profile
from jobhunter.models import Board, RawListing, Seniority
from jobhunter.scrapers.base import Scraper
from tests.conftest import make_job

TITLES = ["Python Developer", "Senior Python Developer", "Java Developer", "LLM Engineer", "Programista PLC"]


class CountingBoard(Scraper):
    board: ClassVar[Board] = Board.JUSTJOINIT
    strategy: ClassVar[str] = "api"

    def __init__(self, search, fail_after: int | None = None) -> None:
        super().__init__(search)
        self.enriched: list[str] = []
        self.fail_after = fail_after

    async def search(self) -> AsyncIterator[RawListing]:
        for i, t in enumerate(TITLES):
            yield RawListing(
                board=self.board, external_id=f"j{i}", url=f"https://x/{i}", title=t, company="Acme"
            )

    async def enrich(self, raw: RawListing) -> RawListing:
        if self.fail_after is not None and len(self.enriched) >= self.fail_after:
            raise KeyboardInterrupt  # what Ctrl-C looks like mid-collect
        self.enriched.append(raw.external_id)
        return raw

    def parse(self, raw: RawListing):
        return make_job(external_id=raw.external_id, url=raw.url, title=raw.title, company=raw.company,
                        seniority=Seniority.MID)  # fmt: skip


async def test_title_filter_skips_detail_requests(settings, db):
    profile, _ = load_profile(settings)
    skip = lambda r: title_prefilter(r, profile)  # noqa: E731
    board = CountingBoard(settings.search)
    listings, stats = await _collect_board(board, db, 60, skip)
    assert board.enriched == ["j0", "j3", "j4"]  # senior + java never fetched
    assert stats["filtered"] == 2 and len(listings) == 3


async def test_interrupted_run_resumes_without_refetching(settings, db):
    first = CountingBoard(settings.search, fail_after=2)
    with pytest.raises(KeyboardInterrupt):
        await _collect_board(first, db, 60)
    assert first.enriched == ["j0", "j1"]  # saved to the DB before the interrupt

    second = CountingBoard(settings.search)
    listings, stats = await _collect_board(second, db, 60)
    assert stats["resumed"] == 2 and second.enriched == ["j2", "j3", "j4"]
    assert sorted(j.external_id for j in listings) == ["j0", "j1", "j2", "j3", "j4"]


def test_title_prefilter_reasons(settings):
    profile, _ = load_profile(settings)
    raw = lambda t: RawListing(board=Board.PRACUJ, external_id="1", url="https://x", title=t, company="X")  # noqa: E731
    assert title_prefilter(raw("Senior Data Engineer"), profile) == "seniority senior"
    assert title_prefilter(raw("Java Developer"), profile) == "excluded keyword 'java'"
    assert title_prefilter(raw("Mid Python Developer"), profile) is None


def test_rocketjobs_card_level_and_pracuj_keyword_urls(settings):
    from jobhunter.scrapers.browser_boards import PracujScraper, rocket_card_level

    card = "Kodland\n\nWarszawa\nZdalnie\nKorepetytor online - Python\n3 500 - 4 500\nSpecjalista / Mid\nB2B"
    assert rocket_card_level(card) == "mid"
    assert rocket_card_level("Acme\nData Engineer\nSenior\nUoP") == "senior"
    assert rocket_card_level("Acme\nData Engineer") is None

    settings.search.keywords = ["Python Developer", "Inżynier Automatyk", "Programista PLC"]
    urls = PracujScraper(settings.search, None, None, None)._search_urls()  # type: ignore[arg-type]
    assert "itth=37" in urls[0] and len(urls) == 3  # python keywords folded into the itth=37 filter
    assert "/praca/in%C5%BCynier%20automatyk;kw?et=" in urls[1]
