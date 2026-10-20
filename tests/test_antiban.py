import re

from jobhunter.config import ScraperConfig
from jobhunter.scrapers.antiban import (
    PROFILES,
    ChallengeKind,
    CircuitBreaker,
    HumanDelay,
    backoff_delay,
    detect_challenge,
    pick_profile,
)


def test_profiles_are_internally_consistent():
    for p in PROFILES:
        if p.impersonate.startswith("chrome"):
            major = p.impersonate.removeprefix("chrome")
            assert f"Chrome/{major}." in p.user_agent
            assert f'v="{major}"' in p.sec_ch_ua
            assert ("Macintosh" in p.user_agent) == (p.platform == '"macOS"')
        else:  # Safari / Firefox never send client hints
            assert p.sec_ch_ua is None
            assert "Chrome/" not in p.user_agent


def test_pick_profile_is_stable_per_seed():
    assert pick_profile("jjit:2026-W42") == pick_profile("jjit:2026-W42")


def test_human_delay_is_bounded():
    cfg = ScraperConfig(long_pause_probability=0.0)
    d = HumanDelay(cfg)
    samples = [d.sample() for _ in range(2000)]
    assert min(samples) >= cfg.min_delay and max(samples) <= cfg.max_delay
    assert len({round(s, 2) for s in samples}) > 200  # actually jittered


def test_detect_cloudflare_challenge():
    body = "<html><title>Just a moment...</title><script src='/cdn-cgi/challenge-platform/h/b'>"
    assert detect_challenge(403, {"Server": "cloudflare", "cf-ray": "x"}, body) == ChallengeKind.CLOUDFLARE
    assert detect_challenge(503, {"server": "cloudflare"}, "") == ChallengeKind.CLOUDFLARE


def test_detect_rate_limit_and_linkedin_999():
    assert detect_challenge(429, {}, "") == ChallengeKind.RATE_LIMIT
    assert detect_challenge(999, {}, "") == ChallengeKind.BLOCKED


def test_plain_pages_are_not_challenges():
    assert detect_challenge(200, {"server": "nginx"}, "<html>offers</html>") is None
    assert detect_challenge(404, {}, "not found") is None


def test_backoff_honours_retry_after_and_cap():
    assert backoff_delay(0, 4, 120, retry_after="7") == 7
    assert backoff_delay(0, 4, 120, retry_after="999") == 120
    assert all(0 <= backoff_delay(10, 4, 60) <= 60 for _ in range(100))


def test_circuit_breaker_opens_once_at_threshold():
    cb = CircuitBreaker(threshold=2)
    assert cb.record("pracuj") is False and not cb.is_open("pracuj")
    assert cb.record("pracuj") is True and cb.is_open("pracuj")
    assert cb.record("pracuj") is False  # notify only on the transition
    assert not cb.is_open("nofluffjobs")


def test_safari_ua_has_no_chrome_tokens():
    safari = next(p for p in PROFILES if p.impersonate.startswith("safari"))
    assert re.search(r"Version/\d+\.\d+ Safari/", safari.user_agent)


def test_referer_with_polish_keyword_is_header_safe():
    from jobhunter.scrapers.http_client import header_safe_url

    url = header_safe_url("https://www.linkedin.com/jobs/search?keywords=Inżynier Automatyk&a=1%20b")
    url.encode("latin-1")  # would raise UnicodeEncodeError before the fix
    assert url == "https://www.linkedin.com/jobs/search?keywords=In%C5%BCynier%20Automatyk&a=1%20b"
