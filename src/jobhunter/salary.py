"""Deterministic salary parsing + normalization (no LLM needed for ~95% of listings).

Normalized target: **monthly PLN, B2B-net-equivalent**, so a "15k brutto UoP" and an "18k netto B2B"
land on the same axis. UoP gross -> B2B equivalence uses a configurable rule-of-thumb factor
(employer-side costs, paid leave, ZUS/tax on B2B are folded into one number on purpose — it's for
ranking, not for tax advice).
"""

from __future__ import annotations

import re

from jobhunter.models import ContractType, SalaryRange

# Fallback FX (PLN per unit) used when the NBP API is unreachable. Refreshed by ``refresh_fx_rates``.
FX_PLN: dict[str, float] = {"PLN": 1.0, "EUR": 4.27, "USD": 3.70, "GBP": 4.95, "CHF": 4.55}

PERIOD_MULTIPLIER = {"hour": None, "day": 21.0, "month": 1.0, "year": 1 / 12}


def refresh_fx_rates(timeout: float = 5.0) -> None:
    """Pull today's mid rates from the National Bank of Poland (table A). Best effort."""
    import httpx

    try:
        r = httpx.get("https://api.nbp.pl/api/exchangerates/tables/A/?format=json", timeout=timeout)
        r.raise_for_status()
        for rate in r.json()[0]["rates"]:
            if rate["code"] in FX_PLN:
                FX_PLN[rate["code"]] = float(rate["mid"])
    except Exception:
        pass


def normalize(s: SalaryRange, *, uop_to_b2b: float = 1.25, hours_per_month: int = 168) -> SalaryRange:
    period = s.period.lower().rstrip("s")
    period = {
        "hourly": "hour",
        "monthly": "month",
        "yearly": "year",
        "annual": "year",
        "godz": "hour",
        "mies": "month",
        "rok": "year",
        "dzień": "day",
        "day": "day",
    }.get(period, period)
    if period == "month" and (s.max or s.min or 0) < 400:  # hourly rate entered as monthly (seen on JJIT)
        period = "hour"
    mult = PERIOD_MULTIPLIER.get(period, 1.0)
    if mult is None:
        mult = float(hours_per_month)
    fx = FX_PLN.get(s.currency.upper(), 1.0)

    def monthly(v: float | None) -> float | None:
        return round(v * mult * fx, -1) if v else None

    lo, hi = monthly(s.min), monthly(s.max)
    if s.contract == ContractType.B2B:
        factor = 1.0
    elif s.contract in (ContractType.UOP, ContractType.UZ, ContractType.UOD) or s.is_gross:
        factor = uop_to_b2b
    else:
        factor = 1.0
    return s.model_copy(
        update={
            "period": period,
            "monthly_pln_min": lo,
            "monthly_pln_max": hi,
            "b2b_equiv_min": round(lo * factor, -1) if lo else None,
            "b2b_equiv_max": round(hi * factor, -1) if hi else None,
        }
    )


# ---------- free-text parsing ("12 000–18 000 zł netto (+ VAT) / mies.") ----------

_NUM = r"\d{1,3}(?:[  .,]\d{3})*(?:[.,]\d+)?\s*[kK]?"
_RANGE_RE = re.compile(
    rf"(?P<lo>{_NUM})\s*(?:zł|pln|eur|usd)?\s*(?:[-–—]|do|to)\s*[€$£]?\s*(?P<hi>{_NUM})"
    rf"|(?:od|from)\s*[€$£]?\s*(?P<only>{_NUM})",
    re.I,
)
_CURRENCY_RE = re.compile(r"(zł|pln|eur|€|usd|\$|gbp|£|chf)", re.I)
_PERIOD_RE = re.compile(r"(/\s*h\b|godz|hour|/\s*d\b|dzie|day|mies|month|mth|rok|year|annual)", re.I)


def _to_float(token: str) -> float:
    t = token.strip().lower().replace(" ", " ")
    k = t.endswith("k")
    t = t.rstrip("k").strip()
    # "12 000,50" / "12.000" / "12,000" — treat a 3-digit tail after a separator as thousands.
    t = re.sub(r"[ .,](?=\d{3}\b)", "", t).replace(",", ".")
    v = float(t)
    return v * 1000 if k else v


def parse_salary_text(text: str, contract_hint: str | None = None) -> SalaryRange | None:
    if not text:
        return None
    m = _RANGE_RE.search(text)
    if not m:
        return None
    try:
        if m.group("only"):
            lo, hi = _to_float(m.group("only")), None
        else:
            lo, hi = _to_float(m.group("lo")), _to_float(m.group("hi"))
    except ValueError:
        return None

    cur_m = _CURRENCY_RE.search(text)
    cur = (
        {"zł": "PLN", "€": "EUR", "$": "USD", "£": "GBP"}.get(cur_m.group(1).lower(), cur_m.group(1).upper())
        if cur_m
        else "PLN"
    )
    per_m = _PERIOD_RE.search(text)
    per_raw = re.sub(r"\s+", "", per_m.group(1).lower()) if per_m else ""
    period = {
        "/h": "hour",
        "godz": "hour",
        "hour": "hour",
        "/d": "day",
        "dzie": "day",
        "day": "day",
        "rok": "year",
        "year": "year",
        "annual": "year",
    }.get(per_raw, "month")
    if period == "month" and lo and lo < 400:  # "150-180 zł" with no unit is almost surely hourly B2B
        period = "hour"

    low = f"{text} {contract_hint or ''}".lower()
    if "b2b" in low or "kontrakt" in low or "+ vat" in low or "+vat" in low:
        contract = ContractType.B2B
    elif "o pracę" in low or "o prace" in low or "uop" in low or "permanent" in low or "employment" in low:
        contract = ContractType.UOP
    elif "zlecen" in low:
        contract = ContractType.UZ
    elif "dzieło" in low or "dzielo" in low:
        contract = ContractType.UOD
    else:
        contract = ContractType.OTHER

    is_gross = (
        True
        if re.search(r"\b(brutto|gross)\b", low)
        else False
        if re.search(r"\b(netto|net)\b", low)
        else None
    )
    if is_gross is None:
        is_gross = contract != ContractType.B2B if contract != ContractType.OTHER else None
    return SalaryRange(min=lo, max=hi, currency=cur, period=period, contract=contract, is_gross=is_gross)
