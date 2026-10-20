import pytest

from jobhunter.models import ContractType, SalaryRange
from jobhunter.salary import FX_PLN, normalize, parse_salary_text


@pytest.mark.parametrize(
    ("text", "lo", "hi", "contract", "period", "gross"),
    [
        ("12 000–18 000 zł netto (+ VAT) / mies.", 12000, 18000, ContractType.B2B, "month", False),
        ("10 000 - 14 000 zł brutto / mies. umowa o pracę", 10000, 14000, ContractType.UOP, "month", True),
        ("120–160 zł netto (+ VAT) / godz.", 120, 160, ContractType.B2B, "hour", False),
        ("15k-20k PLN B2B", 15000, 20000, ContractType.B2B, "month", False),
        ("od 9 000 zł brutto / mies.", 9000, None, ContractType.OTHER, "month", True),
        ("€4,000 - €5,500 per month gross", 4000, 5500, ContractType.OTHER, "month", True),
        ("150-180 zł", 150, 180, ContractType.OTHER, "hour", None),  # tiny monthly number => hourly
    ],
)
def test_parse_salary_text(text, lo, hi, contract, period, gross):
    s = parse_salary_text(text)
    assert s is not None
    assert (s.min, s.max, s.contract, s.period, s.is_gross) == (lo, hi, contract, period, gross)


def test_parse_salary_text_currency():
    assert parse_salary_text("€4,000 - €5,500 per month").currency == "EUR"
    assert parse_salary_text("no salary here") is None


def test_month_is_not_mistaken_for_hour():
    assert parse_salary_text("18 000 - 22 000 PLN / month").period == "month"


def test_internet_does_not_mean_net():
    assert parse_salary_text("10 000 - 12 000 zł, internet provider").is_gross is None


def test_normalize_uop_gross_to_b2b_equivalent():
    s = normalize(
        SalaryRange(min=10000, max=14000, contract=ContractType.UOP, is_gross=True), uop_to_b2b=1.25
    )
    assert s.monthly_pln_max == 14000
    assert s.b2b_equiv_min == 12500 and s.b2b_equiv_max == 17500


def test_normalize_hourly_b2b():
    s = normalize(
        SalaryRange(min=100, max=150, period="hour", contract=ContractType.B2B), hours_per_month=168
    )
    assert s.b2b_equiv_min == 16800 and s.b2b_equiv_max == 25200


def test_normalize_foreign_currency():
    s = normalize(SalaryRange(min=4000, max=None, currency="EUR", contract=ContractType.B2B))
    assert s.monthly_pln_min == round(4000 * FX_PLN["EUR"], -1)
    assert s.b2b_equiv_max is None


def test_hourly_rate_mistyped_as_monthly_is_detected():
    s = normalize(
        SalaryRange(min=140, max=150, period="month", contract=ContractType.B2B), hours_per_month=168
    )
    assert s.period == "hour" and s.b2b_equiv_max == 25200
