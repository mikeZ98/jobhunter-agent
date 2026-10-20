"""Parsers against payload shapes captured from the live APIs (2026-10)."""

from jobhunter.config import SearchConfig
from jobhunter.models import Board, ContractType, RawListing, RemoteMode, Seniority
from jobhunter.scrapers.browser_boards import PracujScraper, _find_offer_dicts
from jobhunter.scrapers.justjoinit import parse_candidate_offer
from jobhunter.scrapers.linkedin import LinkedInGuestScraper
from jobhunter.scrapers.nofluffjobs import NoFluffJobsScraper

JJIT_OFFER = {
    "slug": "acme-python-developer-warszawa-python",
    "title": "Python Developer",
    "companyName": "Acme",
    "workplaceType": "hybrid",
    "experienceLevel": "mid",
    "city": "Warszawa",
    "locations": [{"city": "Warszawa"}, {"city": "Kraków"}],
    "requiredSkills": [{"name": "Python", "level": 4}, {"name": "FastAPI", "level": 3}],
    "niceToHaveSkills": [{"name": "Kubernetes", "level": 1}],
    "employmentTypes": [
        {
            "from": 16000,
            "to": 22000,
            "currency": "pln",
            "currencySource": "original",
            "type": "b2b",
            "unit": "month",
            "gross": False,
        },
        {
            "from": 3700,
            "to": 5100,
            "currency": "EUR",
            "currencySource": "conversion",
            "type": "b2b",
            "unit": "month",
            "gross": False,
        },
        {
            "from": 13000,
            "to": 18000,
            "currency": "PLN",
            "currencySource": "original",
            "type": "permanent",
            "unit": "month",
            "gross": True,
        },
    ],
    "publishedAt": "2026-10-19T08:56:44.065865Z",
    "applyUrl": "https://acme.teamtailor.com/jobs/1",
}


def test_jjit_parse_keeps_original_currency_only():
    raw = RawListing(
        board=Board.JUSTJOINIT,
        external_id=JJIT_OFFER["slug"],
        url="https://justjoin.it/x",
        title="Python Developer",
        company="Acme",
        payload=JJIT_OFFER,
    )
    job = parse_candidate_offer(Board.JUSTJOINIT, raw)
    assert [s.contract for s in job.salaries] == [ContractType.B2B, ContractType.UOP]
    assert job.salaries[0].currency == "PLN" and job.salaries[1].is_gross is True
    assert job.remote == RemoteMode.HYBRID and job.seniority == Seniority.MID
    assert job.locations == ["Warszawa", "Kraków"]
    assert job.required_skills == ["Python", "FastAPI"] and job.apply_url.startswith("https://acme")


NFJ_POSTING = {
    "id": "mid-python-developer-example-corp-Remote-4",
    "name": "Example Corp",
    "title": "Mid Python Developer",
    "posted": 1792274444388,
    "technology": "Python",
    "seniority": ["Mid"],
    "url": "mid-python-developer-remote-5",
    "location": {
        "places": [{"city": "Remote"}, {"province": "lodz", "provinceOnly": True}],
        "fullyRemote": False,
        "type": "remote",
    },
    "salary": {"from": 15000.0, "to": 20000.0, "type": "b2b", "currency": "PLN", "period": "Month"},
}


def test_nfj_parse():
    scraper = NoFluffJobsScraper(SearchConfig(), http=None)  # type: ignore[arg-type]
    raw = RawListing(
        board=Board.NOFLUFFJOBS,
        external_id=NFJ_POSTING["id"],
        url="https://nofluffjobs.com/pl/job/x",
        title=NFJ_POSTING["title"],
        company="Example Corp",
        payload={**NFJ_POSTING, "musts": ["Python", "Django"], "nices": ["Docker"]},
    )
    job = scraper.parse(raw)
    assert job.remote == RemoteMode.REMOTE and job.seniority == Seniority.MID
    assert job.salaries[0].contract == ContractType.B2B and job.salaries[0].is_gross is False
    assert job.salaries[0].period == "month"
    assert job.required_skills == ["Python", "Django"] and job.locations == ["Remote"]
    assert job.published_at.year >= 2026


LINKEDIN_HTML = """
<li><div class="base-card base-search-card job-search-card" data-entity-urn="urn:li:jobPosting:4470805084">
  <a class="base-card__full-link" href="https://pl.linkedin.com/jobs/view/python-engineer-at-example-software-4470805084?position=1&refId=x"></a>
  <div class="base-search-card__info">
    <h3 class="base-search-card__title">  Python Engineer (Remote) </h3>
    <h4 class="base-search-card__subtitle"><a>Example Software</a></h4>
    <span class="job-search-card__location">Cracow, Małopolskie, Poland</span>
    <time datetime="2026-10-18">22 hours ago</time>
  </div></div></li>
<li><div class="base-card base-search-card" data-entity-urn="urn:li:jobPosting:1"><h3>broken card</h3></div></li>
"""


def test_linkedin_card_parse_strips_tracking_and_skips_broken_cards():
    s = LinkedInGuestScraper(SearchConfig(), http=None)  # type: ignore[arg-type]
    cards = s._parse_cards(LINKEDIN_HTML)
    assert len(cards) == 1
    c = cards[0]
    assert c.external_id == "4470805084" and "?" not in c.url
    assert (c.title, c.company) == ("Python Engineer (Remote)", "Example Software")
    job = s.parse(c)
    assert job.remote == RemoteMode.REMOTE and job.locations == ["Cracow"]


def test_pracuj_next_data_walker_and_parse():
    next_data = {
        "props": {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "state": {
                                "data": {
                                    "groupedOffers": [
                                        {
                                            "groupId": "g1",
                                            "jobTitle": "Python Developer",
                                            "companyName": "Bank X",
                                            "salaryDisplayText": "14 000–19 000 zł netto (+ VAT) / mies.",
                                            "typesOfContract": ["kontrakt B2B"],
                                            "workModes": ["praca hybrydowa"],
                                            "positionLevels": ["specjalista (Mid / Regular)"],
                                            "offers": [
                                                {
                                                    "partitionId": 1001,
                                                    "offerAbsoluteUri": "https://www.pracuj.pl/praca/x,oferta,1001",
                                                    "displayWorkplace": "Warszawa",
                                                }
                                            ],
                                        }
                                    ]
                                }
                            }
                        }
                    ]
                }
            }
        }
    }
    groups: list = []
    _find_offer_dicts(next_data, groups)
    assert len(groups) == 1
    scraper = PracujScraper(SearchConfig(), browser=None, breaker=None, delay=None)  # type: ignore[arg-type]
    g = groups[0]
    o = g["offers"][0]
    raw = RawListing(
        board=Board.PRACUJ,
        external_id=str(o["partitionId"]),
        url=o["offerAbsoluteUri"],
        title=g["jobTitle"],
        company=g["companyName"],
        payload={
            "salary_text": g["salaryDisplayText"],
            "contracts": g["typesOfContract"],
            "work_modes": g["workModes"],
            "levels": g["positionLevels"],
            "location": "Warszawa",
        },
    )
    job = scraper.parse(raw)
    assert job.remote == RemoteMode.HYBRID and job.seniority == Seniority.MID
    assert job.salaries[0].contract == ContractType.B2B and job.salaries[0].max == 19000


def test_jjit_hourly_offer_uses_precomputed_monthly_amount():
    payload = {
        **JJIT_OFFER,
        "employmentTypes": [
            {
                "from": 15120.0,
                "fromPerUnit": 90.0,
                "to": 18480.0,
                "toPerUnit": 110.0,
                "unit": "Hour",
                "type": "b2b",
                "gross": False,
                "currency": "PLN",
                "currencySource": "original",
            }
        ],
    }
    raw = RawListing(
        board=Board.JUSTJOINIT, external_id="x", url="u", title="t", company="c", payload=payload
    )
    s = parse_candidate_offer(Board.JUSTJOINIT, raw).salaries[0]
    assert (s.min, s.max, s.period) == (15120.0, 18480.0, "month")  # NOT 15120 * 168
