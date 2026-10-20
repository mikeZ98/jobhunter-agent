"""Agent 6 — LinkedIn Messaging Assistant (read + draft only; never sends).

Source: the unofficial ``linkedin-api`` (Voyager) authenticated with your *existing browser cookies*
(``li_at`` + ``JSESSIONID`` in ``.env``) — no password login, which is what trips LinkedIn's
checkpoint challenges. This is against LinkedIn's User Agreement in spirit; volume is one inbox read
per run, but the account risk is yours. Drafts land in SQLite + the daily Apple Note; you send them.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from jobhunter.config import Settings
from jobhunter.db import Database
from jobhunter.llm import LLMClient, Task
from jobhunter.models import CandidateProfile
from jobhunter.telemetry import get_logger

log = get_logger("agent.linkedin")


class InboxMessage(BaseModel):
    conversation_id: str
    sender: str
    sender_headline: str = ""
    text: str
    from_me: bool = False


class Intent(StrEnum):
    JOB_OFFER = "job_offer"
    FOLLOW_UP = "follow_up"
    SCHEDULING = "scheduling"
    NETWORKING = "networking"
    SPAM = "spam"
    OTHER = "other"


class RecruiterTriage(BaseModel):
    is_recruiter_outreach: bool
    intent: Intent
    company: str | None = None
    role: str | None = None
    salary_mentioned: str | None = None
    fit_score: int = Field(ge=0, le=100, description="Fit vs candidate targets (role, level, remote, rate).")
    missing_info: list[str] = Field(default_factory=list, description="What the candidate needs to ask.")
    language: str = Field(pattern="^(pl|en)$")
    reply: str = Field(max_length=1200, description="Draft reply, same language as the incoming message.")


SYSTEM = """You draft replies to recruiter messages on LinkedIn for a software engineer.
Rules:
- Reply in the language of the incoming message. Polite, concise (60-140 words), human, no flattery.
- If the role fits: express interest, state availability and rate expectations exactly as in the profile
  (B2B net monthly PLN, or UoP gross if the recruiter mentions employment contract), and ask for any
  missing essentials: salary range, stack, remote policy, contract type, recruitment process.
- If it doesn't fit (seniority, stack, onsite outside acceptable cities, rate far below): decline kindly,
  say what *would* fit, keep the door open.
- Spam / unrelated: intent=spam and a one-line polite decline.
Never promise anything not in the profile. Never include placeholders like [Name] — use the sender's name."""


def _client(settings: Settings) -> Any:
    try:
        from linkedin_api import Linkedin
        from requests.cookies import RequestsCookieJar
    except ImportError as e:
        raise RuntimeError("pip install 'jobhunter-agent[linkedin]'") from e
    if not (settings.linkedin_li_at and settings.linkedin_jsessionid):
        raise RuntimeError("Set LINKEDIN_LI_AT and LINKEDIN_JSESSIONID in .env (copy from browser devtools).")
    jar = RequestsCookieJar()
    jsid = settings.linkedin_jsessionid.get_secret_value().strip('"')
    jar.set("li_at", settings.linkedin_li_at.get_secret_value(), domain=".linkedin.com", path="/")
    jar.set("JSESSIONID", f'"{jsid}"', domain=".linkedin.com", path="/")
    return Linkedin("", "", cookies=jar)


def _dig(d: Any, *path: str, default: Any = None) -> Any:
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p)
    return d if d is not None else default


def _parse_conversations(raw: dict[str, Any], me_urn: str | None) -> list[InboxMessage]:
    """Voyager payloads drift; parse the legacy ``elements[].events[]`` shape defensively."""
    out = []
    for conv in raw.get("elements", []):
        conv_id = (conv.get("entityUrn") or "").rsplit(":", 1)[-1]
        event = (conv.get("events") or [{}])[0]
        text = _dig(
            event,
            "eventContent",
            "com.linkedin.voyager.messaging.event.MessageEvent",
            "attributedBody",
            "text",
            default="",
        ) or _dig(
            event, "eventContent", "com.linkedin.voyager.messaging.event.MessageEvent", "body", default=""
        )
        member = _dig(
            event, "from", "com.linkedin.voyager.messaging.MessagingMember", "miniProfile", default={}
        )
        sender = f"{member.get('firstName', '')} {member.get('lastName', '')}".strip() or "Unknown"
        from_me = bool(me_urn and member.get("entityUrn") == me_urn)
        if conv_id and text:
            out.append(
                InboxMessage(
                    conversation_id=conv_id,
                    sender=sender,
                    text=text,
                    sender_headline=member.get("occupation", ""),
                    from_me=from_me,
                )
            )
    return out


def fetch_inbox(settings: Settings, limit: int = 20) -> list[InboxMessage]:
    api = _client(settings)
    try:
        me = api.get_user_profile()
        me_urn = _dig(me, "miniProfile", "entityUrn")
    except Exception:
        me_urn = None
    raw = api.get_conversations()
    msgs = _parse_conversations(raw, me_urn)
    return [m for m in msgs if not m.from_me][:limit]  # waiting on *my* reply only


def _profile_brief(p: CandidateProfile) -> str:
    return (
        f"Name: {p.full_name}. Headline: {p.headline}. Level: {p.seniority.value}. Targets: {p.target_roles}. "
        f"Expected B2B: {p.expected_b2b_monthly_pln} PLN net/month; min UoP: {p.min_uop_monthly_pln} PLN gross. "
        f"Remote: {[r.value for r in p.remote_preference]}; cities: {p.acceptable_cities}. "
        f"Notice: {p.notice_period}. Availability: {p.availability}. Languages: {p.languages}. "
        f"Key skills: {', '.join(s.name for s in p.skills[:15])}."
    )


async def draft_replies(
    messages: list[InboxMessage],
    profile: CandidateProfile,
    llm: LLMClient,
    db: Database,
    concurrency: int = 2,
) -> list[tuple[InboxMessage, RecruiterTriage]]:
    sem = asyncio.Semaphore(concurrency)
    brief = _profile_brief(profile)

    async def one(m: InboxMessage) -> tuple[InboxMessage, RecruiterTriage] | None:
        async with sem:
            try:
                t = await llm.structured(
                    Task.MESSAGING,
                    RecruiterTriage,
                    SYSTEM,
                    f"CANDIDATE:\n{brief}\n\nMESSAGE from {m.sender} ({m.sender_headline}):\n{m.text[:4000]}",
                    max_tokens=900,
                )
            except Exception as e:
                log.warning("draft failed for {}: {!r}", m.conversation_id, e)
                return None
        db.upsert_recruiter_message(
            m.conversation_id,
            sender=m.sender,
            last_message=m.text,
            analysis_json=t.model_dump_json(),
            draft_reply=t.reply,
            status="drafted",
        )
        return m, t

    results = await asyncio.gather(*(one(m) for m in messages))
    return [r for r in results if r]
