from __future__ import annotations

import re
from datetime import UTC, datetime

from bs4 import BeautifulSoup


def html_to_text(html: str, max_chars: int = 12000) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for li in soup.find_all("li"):
        li.insert_before("\n• ")
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()[:max_chars]


def parse_dt(value: str | int | float | None) -> datetime | None:
    """ISO strings, ``Z`` suffix, or epoch millis (NoFluffJobs)."""
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value / 1000 if value > 1e11 else value, tz=UTC)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
