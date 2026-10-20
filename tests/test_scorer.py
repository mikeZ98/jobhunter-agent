from jobhunter.agents.collector import infer_seniority, normalize_listing
from jobhunter.agents.scorer import city_key, prefilter, select_daily, skill_score
from jobhunter.config import ScoringConfig
from jobhunter.models import MatchAssessment, RemoteMode, SalaryRange, ScoredListing, Seniority
from tests.conftest import make_job


def test_title_overrides_board_seniority():
    assert infer_seniority("Senior Python Developer", Seniority.MID) == Seniority.SENIOR
    assert infer_seniority("Python Developer", Seniority.MID) == Seniority.MID
    assert infer_seniority("Tech Lead (Python)", Seniority.UNKNOWN) == Seniority.LEAD


def test_prefilter_rejects_seniority_and_exclusions(profile):
    assert prefilter(make_job(seniority=Seniority.SENIOR), profile).startswith("seniority")
    assert "excluded keyword" in prefilter(make_job(title="Java Developer"), profile)
    assert prefilter(make_job(), profile) is None


def test_prefilter_location_rules(profile):
    # profile: remote/hybrid, Warszawa only
    assert prefilter(make_job(remote=RemoteMode.HYBRID, locations=["Kraków"]), profile).startswith("hybrid")
    assert prefilter(make_job(remote=RemoteMode.HYBRID, locations=["Warsaw"]), profile) is None  # EN alias
    assert prefilter(make_job(remote=RemoteMode.ONSITE, locations=["Warszawa"]), profile).startswith("onsite")
    assert prefilter(make_job(remote=RemoteMode.REMOTE, locations=["Gdańsk"]), profile) is None


def test_prefilter_salary_floor(profile):
    low = SalaryRange(min=8000, max=10000, contract="b2b", b2b_equiv_min=8000, b2b_equiv_max=10000)
    assert prefilter(make_job(salaries=[low]), profile).startswith("salary")


def test_city_key():
    assert city_key("Cracow, Małopolskie, Poland") == city_key("Kraków") == "krakow"
    assert city_key("Łódź") == "lodz"


def test_skill_score_uses_aliases(profile):
    s, hits, missing = skill_score(make_job(required_skills=["python3", "postgres", "k8s", "Go"]), profile)
    assert {"python", "postgresql", "kubernetes"} <= set(hits)
    assert missing == ["go"]
    assert s == 80.0  # 0.8 * 3/4 required + 0.2 * 1/1 nice-to-have (Kubernetes)


def test_normalize_listing_fills_b2b_equivalent():
    job = make_job(
        salaries=[SalaryRange(min=12000, max=16000, contract="uop", is_gross=True)],
        remote=RemoteMode.UNKNOWN,
        description="Praca w pełni zdalna, fully remote.",
    )
    n = normalize_listing(job, uop_to_b2b=1.25, hours_per_month=168)
    assert n.salaries[0].b2b_equiv_max == 20000
    assert n.remote == RemoteMode.REMOTE


def _sl(score: float, judged: bool = True) -> ScoredListing:
    llm = MatchAssessment(score=int(score), strengths=[], gaps=[], pitch="x") if judged else None
    return ScoredListing(
        listing=make_job(external_id=str(score)),
        final_score=score,
        llm=llm,
        rejected_reason=None if judged else "below LLM shortlist cut",
    )


def test_select_daily_caps_and_tops_up():
    cfg = ScoringConfig(min_daily=3, max_daily=4, min_score=60)
    assert len(select_daily([_sl(s) for s in (90, 85, 80, 75, 70, 65)], cfg)) == 4
    picked = select_daily([_sl(s) for s in (70, 55, 52, 40)], cfg)
    assert [p.final_score for p in picked] == [70, 55, 52]  # topped up within threshold-10, never 40
    assert select_daily([_sl(95, judged=False)], cfg) == []  # unjudged never makes the digest
