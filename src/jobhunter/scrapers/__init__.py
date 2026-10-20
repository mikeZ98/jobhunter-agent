"""Board registry: wires each board to its acquisition strategy and shared anti-ban infrastructure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jobhunter.config import Settings
from jobhunter.models import Board
from jobhunter.scrapers.antiban import CircuitBreaker, HostRateLimiter, HumanDelay
from jobhunter.scrapers.base import Scraper
from jobhunter.scrapers.browser import BrowserManager
from jobhunter.scrapers.browser_boards import PracujScraper, RocketJobsScraper
from jobhunter.scrapers.http_client import StealthHttpClient
from jobhunter.scrapers.justjoinit import JustJoinItScraper
from jobhunter.scrapers.linkedin import LinkedInGuestScraper
from jobhunter.scrapers.nofluffjobs import NoFluffJobsScraper

AntibotSink = Callable[[str, str, int | None, str | None], None]

API_BOARDS: dict[Board, type[JustJoinItScraper | NoFluffJobsScraper | LinkedInGuestScraper]] = {
    Board.JUSTJOINIT: JustJoinItScraper,
    Board.NOFLUFFJOBS: NoFluffJobsScraper,
    Board.LINKEDIN: LinkedInGuestScraper,
}
BROWSER_BOARDS: dict[Board, type[PracujScraper | RocketJobsScraper]] = {
    Board.PRACUJ: PracujScraper,
    Board.ROCKETJOBS: RocketJobsScraper,
}


@dataclass
class ScrapeInfra:
    settings: Settings
    delay: HumanDelay
    limiter: HostRateLimiter
    breaker: CircuitBreaker
    browser: BrowserManager
    on_antibot: AntibotSink | None = None

    @classmethod
    def create(
        cls, settings: Settings, browser: BrowserManager, on_antibot: AntibotSink | None = None
    ) -> ScrapeInfra:
        delay = HumanDelay(settings.scraper)
        return cls(
            settings,
            delay,
            HostRateLimiter(delay),
            CircuitBreaker(settings.scraper.challenge_circuit_threshold),
            browser,
            on_antibot,
        )


def build_scrapers(infra: ScrapeInfra, boards: list[str] | None = None) -> list[Scraper]:
    s = infra.settings
    enabled = [Board(b) for b in (boards or s.scraper.enabled_boards)]
    scrapers: list[Scraper] = []
    for board in enabled:
        if board in API_BOARDS:
            http = StealthHttpClient(
                board.value, s.scraper, s.cookies_dir, infra.limiter, infra.breaker, infra.on_antibot
            )
            scrapers.append(API_BOARDS[board](s.search, http))
        elif board in BROWSER_BOARDS:
            scrapers.append(
                BROWSER_BOARDS[board](s.search, infra.browser, infra.breaker, infra.delay, infra.on_antibot)
            )
    return scrapers


__all__ = ["ScrapeInfra", "Scraper", "build_scrapers"]
