from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from jobhunter.config import SearchConfig
from jobhunter.models import Board, JobListing, RawListing
from jobhunter.telemetry import get_logger

if TYPE_CHECKING:
    from jobhunter.scrapers.http_client import StealthHttpClient


class Scraper(abc.ABC):
    """A board adapter. ``search`` yields raw listings; ``parse`` maps a board payload to a
    ``JobListing`` with *as-advertised* salary values (the normalizer agent makes them comparable)."""

    board: ClassVar[Board]
    strategy: ClassVar[str]  # "api" | "browser"

    def __init__(self, search: SearchConfig) -> None:
        self.search_cfg = search
        self.log = get_logger(f"scraper.{self.board.value}")

    @abc.abstractmethod
    def search(self) -> AsyncIterator[RawListing]: ...

    @abc.abstractmethod
    def parse(self, raw: RawListing) -> JobListing: ...

    async def enrich(self, raw: RawListing) -> RawListing:
        """Optional second request for the full description (only called for unseen listings)."""
        return raw

    async def __aenter__(self) -> Scraper:
        """Acquire long-lived resources (browser context) shared by ``search`` and ``enrich``."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class ApiScraper(Scraper):
    """Strategy 1 boards: owns a ``StealthHttpClient`` for its whole run (one TLS session, warm cookies)."""

    strategy: ClassVar[str] = "api"

    def __init__(self, search: SearchConfig, http: StealthHttpClient) -> None:
        super().__init__(search)
        self.http = http

    async def __aenter__(self) -> ApiScraper:
        await self.http.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.http.__aexit__(*exc)
