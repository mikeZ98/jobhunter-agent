"""Autofill: unit tests for field mapping + a real-browser test of the never-submit contract."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from jobhunter.agents import autofill
from jobhunter.agents.autofill import FieldDesc, plan_fills


def fd(id_, label="", type_="text", tag="input", value="", required=False, options=None, name="", auto=""):
    return FieldDesc(
        id=id_,
        tag=tag,
        type=type_,
        name=name,
        el_id="",
        placeholder="",
        autocomplete=auto,
        label=label,
        required=required,
        value=value,
        options=options or [],
        accept="",
        maxlength=None,
    )


def test_plan_maps_polish_and_english_labels(profile):
    fields = [
        fd("1", "Imię"),
        fd("2", "Nazwisko"),
        fd("3", "E-mail", "email"),
        fd("4", "Numer telefonu", "tel"),
        fd("5", "LinkedIn profile URL"),
        fd("6", "Oczekiwania finansowe"),
        fd("7", "Okres wypowiedzenia"),
        fd("8", "Full name"),
        fd("9", auto="given-name"),
    ]
    plan = plan_fills(fields, profile, {})
    assert plan.text["1"] == "Jan" and plan.text["2"] == "Kowalski" and plan.text["8"] == "Jan Kowalski"
    assert plan.text["3"] == profile.email and plan.text["4"] == profile.phone
    assert "linkedin.com" in plan.text["5"] and "18 000 PLN" in plan.text["6"]
    assert plan.text["7"] == profile.notice_period and plan.text["9"] == "Jan"


def test_plan_never_overwrites_and_never_ticks_consents(profile, tmp_path):
    cv = tmp_path / "cv.pdf"
    cv.write_bytes(b"%PDF-1.4")
    fields = [
        fd("1", "E-mail", "email", value="already@there.pl"),
        fd("2", "Wyrażam zgodę na przetwarzanie danych (RODO)", "checkbox", required=True),
        fd("3", "Upload your CV", "file"),
        fd("4", "Why do you want to join us?", tag="textarea", required=True),
    ]
    plan = plan_fills(fields, profile, {"cv_pdf": str(cv)})
    assert "1" not in plan.text and plan.skipped_prefilled == 1
    assert [c.id for c in plan.consents] == ["2"]
    assert plan.files == {"3": cv}
    assert [u.id for u in plan.unresolved] == ["4"]  # goes to the LLM / human, not guessed by regex


def test_extra_answers_override_heuristics(profile):
    plan = plan_fills([fd("1", "Do you have a valid work permit for Poland?", required=True)], profile, {})
    assert plan.text["1"].startswith("Tak")


def test_module_never_clicks_or_submits():
    """Static guarantee: no call in the autofill module can submit a form."""
    tree = ast.parse(inspect.getsource(autofill))
    forbidden = {"click", "dblclick", "tap", "press", "submit", "dispatch_event", "request_submit", "check"}
    calls = {
        n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert not (calls & forbidden), calls & forbidden
    js = autofill.SCAN_JS + autofill.BANNER_JS
    assert ".submit(" not in js and ".click(" not in js and "requestSubmit" not in js


FORM = """<!doctype html><html><body>
<form id="f" onsubmit="window.__submitted = true; return false;">
  <label for="fn">Imię *</label><input id="fn" name="first_name" required>
  <label for="ln">Nazwisko *</label><input id="ln" name="last_name" required>
  <label>E-mail <input type="email" name="email" required></label>
  <input type="tel" name="phone" placeholder="Telefon">
  <label for="pre">LinkedIn</label><input id="pre" name="li" value="https://linkedin.com/in/someone-else">
  <label for="cv">Załącz CV</label><input type="file" id="cv" accept=".pdf">
  <label for="msg">Wiadomość do rekrutera</label><textarea id="msg"></textarea>
  <label><input type="checkbox" name="rodo" required> Wyrażam zgodę na przetwarzanie danych osobowych</label>
  <button type="submit">Wyślij aplikację</button>
</form></body></html>"""


@pytest.mark.asyncio
async def test_prefill_real_browser_fills_but_never_submits(profile, tmp_path: Path):
    pw_mod = pytest.importorskip("playwright.async_api")
    cv = tmp_path / "cv.pdf"
    cv.write_bytes(b"%PDF-1.4 test")
    async with pw_mod.async_playwright() as pw:
        try:
            browser = await pw.chromium.launch()
        except Exception as e:
            pytest.skip(f"chromium not installed: {e}")
        page = await browser.new_page()
        await page.set_content(FORM)
        assets = {"cv_pdf": str(cv), "cover_letter_text": "Dzień dobry,\nw projekcie DocuMind...\nPozdrawiam"}
        filled = await autofill.prefill(page, profile, assets, "Test @ Acme", "summary", llm=None)

        assert await page.input_value("#fn") == "Jan"
        assert await page.input_value("#ln") == "Kowalski"
        assert await page.input_value("input[name=email]") == profile.email
        assert await page.input_value("input[name=phone]") == profile.phone
        assert await page.input_value("#pre") == "https://linkedin.com/in/someone-else"  # untouched
        assert "DocuMind" in await page.input_value("#msg")
        assert await page.evaluate("document.getElementById('cv').files[0].name") == "cv.pdf"
        assert await page.is_checked("input[name=rodo]") is False  # consent left to the human
        assert await page.evaluate("window.__submitted === true") is False  # the contract
        assert await page.locator("#jh-banner").count() == 1
        assert filled == 6
        await browser.close()
