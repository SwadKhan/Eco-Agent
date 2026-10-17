"""
Skopje Eco-Agent — Civic Air Quality Platform
Hackathon MVP — Hacktoberfest 2026, Base42, Skopje

Not just a monitor — a civic action platform.
When the air is bad, citizens can: report sources with evidence,
discuss anonymously, organise events, and generate + publish
AI-backed evidence briefs for municipal authorities.

Data: Pulse.eco (PM2.5/PM10) + Open-Meteo (weather)
AI: gpt-oss-20b via Groq (health alerts + municipal briefs)
"""

from __future__ import annotations

import os
import json
import re
import uuid
import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("eco-agent")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
PULSE_ECO_BASE = "https://skopje.pulse.eco/rest"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

SKP_LAT, SKP_LON = 41.9981, 21.4254

# ---------------------------------------------------------------------------
# In-memory stores (SQLite in production — fine for hackathon demo)
# ---------------------------------------------------------------------------
REPORTS_DB: list[dict] = []
COMMENTS_DB: list[dict] = []      # anonymous discussion threads
EVENTS_DB: list[dict] = []
BRIEFS_DB: list[dict] = []        # published municipal briefs
UPVOTES_DB: dict[str, int] = {}   # report_id -> support count

# ---------------------------------------------------------------------------
# Pydantic models — Air Quality
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"

class SensorReading(BaseModel):
    sensor_id: str = ""
    type: str = ""
    value: float = 0.0
    unit: str = "ug/m3"
    stamp: str = ""

class WeatherSnapshot(BaseModel):
    temperature_2m: list[float] = Field(default_factory=list)
    windspeed_10m: list[float] = Field(default_factory=list)
    relative_humidity_2m: list[float] = Field(default_factory=list)
    time: list[str] = Field(default_factory=list)

class HealthAlert(BaseModel):
    risk_level: RiskLevel = RiskLevel.LOW
    headline: str = ""
    summary: str = ""
    advice_general: str = ""
    advice_vulnerable: str = ""
    pm25_avg: float = 0.0
    pm10_avg: float = 0.0
    wind_speed_avg: float = 0.0
    inversion_risk: bool = False
    generated_at: str = ""

# ---------------------------------------------------------------------------
# Pydantic models — Civic Reports (with evidence)
# ---------------------------------------------------------------------------

REPORT_CATEGORIES = [
    "illegal_burning",
    "construction_dust",
    "factory_emissions",
    "vehicle_exhaust",
    "waste_dumping",
    "municipal_negligence",
    "other",
]

class CivicReportIn(BaseModel):
    """What a citizen submits — no identity required."""
    category: str = Field(..., description="One of the report categories")
    description: str = Field(..., min_length=5, max_length=1000)
    neighbourhood: str = Field(default="", max_length=100)
    urgency: int = Field(default=3, ge=1, le=5, description="1=minor, 5=emergency")
    # Evidence fields — all optional
    evidence_links: list[str] = Field(default_factory=list, description="URLs to photos/videos")
    evidence_text: str = Field(default="", max_length=2000, description="Detailed evidence description")

class CivicReport(CivicReportIn):
    """Stored report with metadata."""
    id: str = ""
    timestamp: str = ""
    pm25_at_time: float = 0.0
    pm10_at_time: float = 0.0
    supports: int = 0  # anonymous upvote count

# ---------------------------------------------------------------------------
# Pydantic models — Anonymous Comments
# ---------------------------------------------------------------------------

class CommentIn(BaseModel):
    """Anonymous comment on a report."""
    text: str = Field(..., min_length=1, max_length=1000)

class Comment(CommentIn):
    id: str = ""
    report_id: str = ""
    timestamp: str = ""
    anon_label: str = ""  # e.g. "Citizen #3"

# ---------------------------------------------------------------------------
# Pydantic models — Community Events
# ---------------------------------------------------------------------------

class CommunityEventIn(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(default="", max_length=1000)
    event_type: str = Field(default="cleanup", description="cleanup | meetup | petition | protest | awareness")
    neighbourhood: str = Field(default="", max_length=100)
    date: str = Field(default="", description="ISO date or 'TBD'")

class CommunityEvent(CommunityEventIn):
    id: str = ""
    created_at: str = ""
    attendees: int = 0

# ---------------------------------------------------------------------------
# Pydantic models — Municipal Brief
# ---------------------------------------------------------------------------

class MunicipalBrief(BaseModel):
    id: str = ""
    title: str = ""
    date: str = ""
    air_quality_summary: str = ""
    citizen_reports_summary: str = ""
    demands: list[str] = Field(default_factory=list)
    evidence: str = ""
    full_text: str = ""
    published: bool = False
    email_sent_to: list[str] = Field(default_factory=list)

class EmailRequest(BaseModel):
    """Send a brief anonymously via email."""
    brief_id: str
    recipient_emails: list[str] = Field(..., min_length=1, description="e.g. ['info@skopje.gov.mk']")
    sender_alias: str = Field(default="Concerned Citizens of Skopje", max_length=100)

# ---------------------------------------------------------------------------
# Data ingestion — Pulse.eco
# ---------------------------------------------------------------------------

async def fetch_pulse_eco(measure: str = "pm25") -> list[SensorReading]:
    # /current returns every pollutant regardless of the type param, so filter here
    url = f"{PULSE_ECO_BASE}/current"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            raw = resp.json()
            readings = [
                SensorReading(
                    sensor_id=item.get("sensorId", ""),
                    type=item.get("type", measure),
                    value=float(item.get("value", 0)),
                    unit=item.get("unit", "ug/m3"),
                    stamp=item.get("stamp", ""),
                )
                for item in raw
                if item.get("type") == measure
            ]
            log.info("Pulse.eco -> %d %s readings", len(readings), measure)
            return readings
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Pulse.eco request failed: %s", exc)
        return []

# ---------------------------------------------------------------------------
# Data ingestion — Open-Meteo
# ---------------------------------------------------------------------------

async def fetch_weather() -> WeatherSnapshot:
    params = {
        "latitude": SKP_LAT,
        "longitude": SKP_LON,
        "hourly": "temperature_2m,windspeed_10m,relative_humidity_2m",
        "forecast_days": 1,
        "timezone": "Europe/Skopje",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(OPEN_METEO_BASE, params=params)
            resp.raise_for_status()
            data = resp.json().get("hourly", {})
            snap = WeatherSnapshot(
                temperature_2m=data.get("temperature_2m", []),
                windspeed_10m=data.get("windspeed_10m", []),
                relative_humidity_2m=data.get("relative_humidity_2m", []),
                time=data.get("time", []),
            )
            log.info("Open-Meteo -> %d hourly data points", len(snap.time))
            return snap
    except httpx.HTTPError as exc:
        log.warning("Open-Meteo request failed: %s", exc)
        return WeatherSnapshot()

async def _fetch_all_data():
    import asyncio
    pm25_task = asyncio.create_task(fetch_pulse_eco("pm25"))
    pm10_task = asyncio.create_task(fetch_pulse_eco("pm10"))
    weather_task = asyncio.create_task(fetch_weather())
    return await pm25_task, await pm10_task, await weather_task

def _make_llm(temperature: float) -> ChatGroq:
    # JSON mode makes the model return a bare JSON object
    # gpt-oss reasons before answering; keep that short so the JSON isn't cut off
    return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=GROQ_API_KEY, max_tokens=4096,
                    reasoning_effort="low" if "gpt-oss" in GROQ_MODEL else None,
                    model_kwargs={"response_format": {"type": "json_object"}})


def _parse_json(text: str) -> dict:
    """Parse the model's JSON, tolerating markdown fences or text around it."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])

# ---------------------------------------------------------------------------
# AI Agent — Health Alert
# ---------------------------------------------------------------------------

ALERT_SYSTEM_PROMPT = """\
You are Eco-Agent, an environmental-health AI for Skopje, Macedonia.
You receive real-time air-quality sensor data and weather forecasts.

Your job:
1. Analyse PM2.5 and PM10 against WHO guidelines (PM2.5 24h <= 15 ug/m3, PM10 24h <= 45 ug/m3).
2. Check for temperature-inversion conditions (low wind < 5 km/h, high humidity > 75%).
3. Predict whether a smog event is likely in the next 12 hours.
4. Output JSON with these exact keys:
   - risk_level: "low" | "moderate" | "high" | "severe"
   - headline: one-sentence alert title
   - summary: 2-3 sentence situation overview
   - advice_general: one-sentence advice for general population
   - advice_vulnerable: one-sentence advice for children, elderly, asthmatics
   - inversion_risk: true/false

Respond with ONLY valid JSON, no markdown fences, no extra text.
"""

ALERT_USER_TEMPLATE = """\
=== LIVE SENSOR DATA (Skopje) ===
PM2.5 readings (ug/m3): {pm25_values}
PM2.5 average: {pm25_avg:.1f} ug/m3
PM10 readings (ug/m3): {pm10_values}
PM10 average: {pm10_avg:.1f} ug/m3

=== WEATHER FORECAST (next 24h) ===
Temperatures (C): {temperatures}
Wind speeds (km/h): {wind_speeds}
Humidity (%): {humidity}
Average wind: {wind_avg:.1f} km/h | Average humidity: {humidity_avg:.1f}%

Current time: {now}
Analyse this data and produce the JSON health alert.
"""

async def run_alert_agent(pm25, pm10, weather) -> HealthAlert:
    pm25_vals = [r.value for r in pm25] or [0.0]
    pm10_vals = [r.value for r in pm10] or [0.0]
    pm25_avg = sum(pm25_vals) / len(pm25_vals)
    pm10_avg = sum(pm10_vals) / len(pm10_vals)
    wind_avg = (sum(weather.windspeed_10m) / len(weather.windspeed_10m)) if weather.windspeed_10m else 0.0
    hum_avg = (sum(weather.relative_humidity_2m) / len(weather.relative_humidity_2m)) if weather.relative_humidity_2m else 0.0

    prompt = ChatPromptTemplate.from_messages([
        ("system", ALERT_SYSTEM_PROMPT),
        ("human", ALERT_USER_TEMPLATE),
    ])
    now = datetime.now(timezone.utc).isoformat()

    try:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")
        llm = _make_llm(0.2)
        chain = prompt | llm
        result = await chain.ainvoke({
            "pm25_values": pm25_vals[:20],
            "pm25_avg": pm25_avg,
            "pm10_values": pm10_vals[:20],
            "pm10_avg": pm10_avg,
            "temperatures": weather.temperature_2m[:24],
            "wind_speeds": weather.windspeed_10m[:24],
            "humidity": weather.relative_humidity_2m[:24],
            "wind_avg": wind_avg,
            "humidity_avg": hum_avg,
            "now": now,
        })
        parsed = _parse_json(result.content)
        return HealthAlert(
            risk_level=parsed.get("risk_level", "low"),
            headline=parsed.get("headline", "Air quality data processed"),
            summary=parsed.get("summary", ""),
            advice_general=parsed.get("advice_general", ""),
            advice_vulnerable=parsed.get("advice_vulnerable", ""),
            pm25_avg=round(pm25_avg, 1),
            pm10_avg=round(pm10_avg, 1),
            wind_speed_avg=round(wind_avg, 1),
            inversion_risk=parsed.get("inversion_risk", False),
            generated_at=now,
        )
    except Exception as exc:
        log.error("Alert agent error: %s", exc)
        return _fallback_alert(pm25_avg, pm10_avg, wind_avg, hum_avg)


def _fallback_alert(pm25, pm10, wind, hum):
    if pm25 > 55 or pm10 > 150:
        level = RiskLevel.SEVERE
    elif pm25 > 35 or pm10 > 100:
        level = RiskLevel.HIGH
    elif pm25 > 15 or pm10 > 45:
        level = RiskLevel.MODERATE
    else:
        level = RiskLevel.LOW
    inversion = wind < 5 and hum > 75
    return HealthAlert(
        risk_level=level,
        headline=f"Skopje air quality: {level.value} risk (fallback)",
        summary=f"PM2.5 avg {pm25:.0f} ug/m3, PM10 avg {pm10:.0f} ug/m3. Wind {wind:.0f} km/h, humidity {hum:.0f}%.",
        advice_general="Reduce outdoor exercise and close windows." if level != RiskLevel.LOW else "No action needed.",
        advice_vulnerable="Stay indoors. Use air purifier if available." if level != RiskLevel.LOW else "No special precautions.",
        pm25_avg=round(pm25, 1), pm10_avg=round(pm10, 1),
        wind_speed_avg=round(wind, 1), inversion_risk=inversion,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

# ---------------------------------------------------------------------------
# AI Agent — Municipal Brief Generator
# ---------------------------------------------------------------------------

BRIEF_SYSTEM_PROMPT = """\
You are a civic advocacy AI writing a formal brief to the Municipality of Skopje.
You receive: current air quality data, weather conditions, and anonymous citizen reports
about pollution sources.

Write a professional 300-word brief that:
1. States the current air quality emergency with exact numbers
2. Summarises citizen reports by category (how many reports of each type)
3. Lists 3-5 specific, actionable demands
4. Cites sensor data timestamps as evidence

Tone: firm but professional. This is a public document, not a complaint letter.

Output JSON with these keys:
- title: brief title
- air_quality_summary: 2-sentence AQ situation
- citizen_reports_summary: summary of citizen reports
- demands: list of 3-5 actionable demands as strings
- evidence: key data points cited
- full_text: the complete 300-word brief as formatted text

Respond with ONLY valid JSON.
"""

BRIEF_USER_TEMPLATE = """\
=== AIR QUALITY RIGHT NOW ===
PM2.5 average: {pm25_avg:.1f} ug/m3 (WHO limit: 15)
PM10 average: {pm10_avg:.1f} ug/m3 (WHO limit: 45)
Wind: {wind_avg:.1f} km/h | Humidity: {humidity_avg:.1f}%
Inversion risk: {inversion}

=== CITIZEN REPORTS ({report_count} total) ===
{reports_text}

=== EVIDENCE SUBMITTED ===
{evidence_text}

=== DATE ===
{now}

Generate the municipal brief.
"""

async def generate_municipal_brief(pm25_avg, pm10_avg, wind_avg, hum_avg, inversion) -> MunicipalBrief:
    """Use LLM to generate an evidence-backed brief for city authorities."""
    category_counts: dict[str, int] = {}
    report_excerpts = []
    evidence_lines = []
    for r in REPORTS_DB[-50:]:
        cat = r.get("category", "other")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        report_excerpts.append(f"- [{cat}] {r.get('description', '')[:100]}")
        # Include evidence if present
        if r.get("evidence_text"):
            evidence_lines.append(f"  Evidence: {r['evidence_text'][:200]}")
        if r.get("evidence_links"):
            evidence_lines.append(f"  Links: {', '.join(r['evidence_links'][:3])}")

    reports_text = "\n".join(report_excerpts) if report_excerpts else "No citizen reports yet."
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "No evidence attachments."

    prompt = ChatPromptTemplate.from_messages([
        ("system", BRIEF_SYSTEM_PROMPT),
        ("human", BRIEF_USER_TEMPLATE),
    ])
    now = datetime.now(timezone.utc).isoformat()

    try:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")
        llm = _make_llm(0.3)
        chain = prompt | llm
        result = await chain.ainvoke({
            "pm25_avg": pm25_avg,
            "pm10_avg": pm10_avg,
            "wind_avg": wind_avg,
            "humidity_avg": hum_avg,
            "inversion": inversion,
            "report_count": len(REPORTS_DB),
            "reports_text": reports_text,
            "evidence_text": evidence_text,
            "now": now,
        })
        parsed = _parse_json(result.content)

        def text(key, default=""):
            # The model sometimes returns lists/objects for text fields
            val = parsed.get(key, default)
            if isinstance(val, list):
                return "\n".join(str(v) for v in val)
            return val if isinstance(val, str) else json.dumps(val)

        demands = parsed.get("demands", [])
        if isinstance(demands, str):
            demands = [demands]
        brief_id = uuid.uuid4().hex[:12]
        return MunicipalBrief(
            id=brief_id,
            title=text("title", "Municipal Air Quality Brief -- Skopje"),
            date=now,
            air_quality_summary=text("air_quality_summary"),
            citizen_reports_summary=text("citizen_reports_summary"),
            demands=[re.sub(r"^\s*\d+[.)]\s*", "", str(d)) for d in demands],
            evidence=text("evidence"),
            full_text=text("full_text"),
        )
    except Exception as exc:
        log.error("Brief agent error: %s", exc)
        # Template brief so the publish/email flow still works without the LLM
        brief_id = uuid.uuid4().hex[:12]
        aq_summary = f"PM2.5: {pm25_avg:.0f} ug/m3 ({pm25_avg/15:.1f}x WHO limit). PM10: {pm10_avg:.0f} ug/m3."
        cat_summary = ", ".join(f"{n} x {c}" for c, n in category_counts.items()) or "none yet"
        reports_summary = f"{len(REPORTS_DB)} citizen reports filed ({cat_summary})."
        demands = [
            "Enforce existing burning bans",
            "Publish real-time AQ on municipal website",
            "Increase inspections in reported neighbourhoods",
        ]
        full_text = (
            "To the Municipality of Skopje,\n\n"
            f"Air quality: {aq_summary} Wind {wind_avg:.1f} km/h, humidity {hum_avg:.0f}%"
            f"{', temperature inversion likely' if inversion else ''}.\n\n"
            f"Citizen reports: {reports_summary}\n\n"
            "We request that the Municipality:\n"
            + "\n".join(f"{i}. {d}" for i, d in enumerate(demands, 1))
            + f"\n\nEvidence: Pulse.eco sensor network data as of {now}; anonymous citizen reports via Eco-Agent."
        )
        return MunicipalBrief(
            id=brief_id,
            title="Municipal Air Quality Brief -- Skopje",
            date=now,
            air_quality_summary=aq_summary,
            citizen_reports_summary=reports_summary,
            demands=demands,
            evidence="Pulse.eco sensor network data.",
            full_text=full_text,
        )

# ---------------------------------------------------------------------------
# Anonymous email sender
# ---------------------------------------------------------------------------

def send_anonymous_email(brief: dict, recipients: list[str], sender_alias: str) -> dict:
    """
    Send municipal brief via email anonymously.
    For hackathon demo: uses env vars SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS.
    Falls back to a 'queued' response if SMTP is not configured.
    """
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    subject = brief.get("title", "Municipal Air Quality Brief -- Skopje")
    body = brief.get("full_text", "")
    body += f"\n\n---\nSent anonymously via Eco-Agent on behalf of: {sender_alias}"
    body += "\nThis brief contains verified sensor data from pulse.eco and citizen reports."
    body += "\n\nGenerated: " + brief.get("date", "")

    # Treat the .env.example placeholders as "not configured"
    if not smtp_host or not smtp_user or not smtp_pass or smtp_pass == "your_app_password":
        # Demo mode — queue the email for display
        log.info("SMTP not configured — email queued for demo")
        return {
            "status": "queued",
            "message": "Email queued (SMTP not configured — in production this sends automatically)",
            "recipients": recipients,
            "subject": subject,
            "preview": body[:500],
        }

    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = f"{sender_alias} <{smtp_user}>"
        msg["To"] = ", ".join(recipients)

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipients, msg.as_string())

        log.info("Email sent to %s", recipients)
        return {"status": "sent", "recipients": recipients, "subject": subject}
    except Exception as exc:
        log.error("Email send failed: %s", exc)
        return {"status": "failed", "error": str(exc), "recipients": recipients}

# ---------------------------------------------------------------------------
# Sample history (DEMO_SEED=1) — data is in-memory, so without this the
# public dashboard is empty after every restart. Seeded rows are flagged
# "sample" and the UI labels them.
# ---------------------------------------------------------------------------

SAMPLE_REPORTS = [
    # (hours ago, category, neighbourhood, urgency, pm25, pm10, supports, description, evidence)
    (70, "illegal_burning", "Aerodrom", 4, 48.2, 71.5, 9,
     "Thick smoke from several backyard stoves on the east side of Aerodrom every evening after 6pm.",
     "Smell of burning plastic and tyres, visible haze over the blocks near the market."),
    (52, "construction_dust", "Karposh", 3, 31.7, 88.0, 4,
     "Large building site on bul. Partizanski Odredi with no dust covers, trucks leave mud and dust on the road.",
     ""),
    (40, "factory_emissions", "Butel", 5, 57.9, 92.3, 12,
     "Dark grey plume from the industrial zone stacks early in the morning, strongest between 5 and 7am.",
     "Plume photographed from the Butel roundabout at 06:10, drifting towards the city centre."),
    (26, "waste_dumping", "Gazi Baba", 3, 36.4, 59.8, 6,
     "Illegal dump by the river bank keeps getting set on fire at night.",
     "Burnt waste pile roughly 20 metres long, still smouldering in the morning."),
    (14, "vehicle_exhaust", "Centar", 2, 29.3, 47.1, 3,
     "Idling buses and trucks around the main bus station, black exhaust at rush hour.",
     ""),
    (5, "municipal_negligence", "Kisela Voda", 4, 44.0, 66.2, 7,
     "Promised street-cleaning and watering of dusty roads has not happened in weeks.",
     "Dust on parked cars is visibly thicker every day; no cleaning vehicles seen since September."),
    (64, "illegal_burning", "Aerodrom", 3, 51.0, 74.8, 5,
     "Someone burns garden waste and plastic in the lot behind the school on Jane Sandanski.", ""),
    (45, "illegal_burning", "Karposh", 4, 46.3, 69.0, 8,
     "Wood and coal stoves in the older houses, the whole street is foggy with smoke at night.", ""),
    (33, "factory_emissions", "Butel", 4, 53.1, 85.6, 6,
     "Strong chemical smell near the industrial zone, worse when there is no wind.", ""),
    (9, "illegal_burning", "Gazi Baba", 3, 41.8, 63.4, 4,
     "Cable burning near the scrapyard to recover copper, black smoke for over an hour.",
     "Black smoke column visible from the main road around 20:30."),
]

SAMPLE_COMMENTS = [
    # (report index, hours ago, text)
    (0, 68, "Same in my building, we have to keep windows shut every night."),
    (0, 60, "It's worst on weekends. My kid's asthma gets noticeably worse."),
    (0, 30, "Has anyone reported this to the inspectorate? Happy to sign something."),
    (2, 39, "I drive past at 6am every day, can confirm the plume."),
    (2, 37, "The PM2.5 sensor in Butel jumps every morning at the same time, matches this."),
    (2, 20, "We should bring this report to the next council session."),
    (3, 25, "Called the communal service twice, no response."),
    (5, 4, "Same on my street near the Kisela Voda market."),
    (5, 3, "The municipality's website still says weekly cleaning."),
]

SAMPLE_EVENTS = [
    # (hours ago, title, type, neighbourhood, date, attendees, description)
    (48, "Gradski Park clean-air cleanup", "cleanup", "Centar", "Saturday 10:00", 23,
     "Bring gloves and bags. Meeting at the main entrance."),
    (20, "Petition: enforce the burning ban in Aerodrom", "petition", "Aerodrom", "Ongoing", 41,
     "Collecting signatures to present at the next municipal council session."),
]


def _seed_demo_data():
    now = datetime.now(timezone.utc)
    ago = lambda h: (now - timedelta(hours=h)).isoformat()
    ids = {}  # SAMPLE_REPORTS index -> report id, for attaching comments
    for i in sorted(range(len(SAMPLE_REPORTS)), key=lambda i: -SAMPLE_REPORTS[i][0]):
        hours, cat, hood, urg, pm25, pm10, supports, desc, ev = SAMPLE_REPORTS[i]
        rid = ids[i] = uuid.uuid4().hex[:12]
        REPORTS_DB.append({
            "id": rid, "category": cat, "description": desc, "neighbourhood": hood, "urgency": urg,
            "evidence_links": [], "evidence_text": ev, "timestamp": ago(hours),
            "pm25_at_time": pm25, "pm10_at_time": pm10, "supports": supports, "sample": True,
        })
        UPVOTES_DB[rid] = supports
    for idx, hours, text in SAMPLE_COMMENTS:
        rid = ids[idx]
        n = sum(1 for c in COMMENTS_DB if c["report_id"] == rid) + 1
        COMMENTS_DB.append({"id": uuid.uuid4().hex[:12], "report_id": rid, "text": text,
                            "timestamp": ago(hours), "anon_label": f"Citizen #{n}", "sample": True})
    COMMENTS_DB.sort(key=lambda c: c["timestamp"])
    for hours, title, etype, hood, date, attendees, desc in SAMPLE_EVENTS:
        EVENTS_DB.append({"id": uuid.uuid4().hex[:12], "title": title, "description": desc,
                          "event_type": etype, "neighbourhood": hood, "date": date,
                          "created_at": ago(hours), "attendees": attendees, "sample": True})
    BRIEFS_DB.append({
        "id": uuid.uuid4().hex[:12], "title": "Air Quality Brief: Burning and Industrial Emissions",
        "date": ago(30), "published": True, "published_at": ago(30), "email_sent_to": [], "sample": True,
        "air_quality_summary": "PM2.5 averaged 52 ug/m3 across Skopje, 3.5x the WHO limit, with evening peaks in Aerodrom and morning peaks in Butel.",
        "citizen_reports_summary": "3 citizen reports: illegal burning (Aerodrom), factory emissions (Butel), construction dust (Karposh).",
        "demands": ["Enforce the residential burning ban in Aerodrom with evening inspections",
                    "Inspect the Butel industrial zone stacks between 5 and 7am",
                    "Require dust covers and road cleaning at large construction sites"],
        "evidence": "Pulse.eco sensor readings at the time of each report; 25 community supports.",
        "full_text": "To the Municipality of Skopje,\n\nOver the past three days citizens filed reports of "
                     "illegal burning in Aerodrom, industrial emissions in Butel and construction dust in Karposh, "
                     "each matching PM2.5 readings well above the WHO limit.\n\nWe request that the Municipality:\n"
                     "1. Enforce the residential burning ban in Aerodrom with evening inspections\n"
                     "2. Inspect the Butel industrial zone stacks between 5 and 7am\n"
                     "3. Require dust covers and road cleaning at large construction sites\n\n"
                     "Concerned Citizens of Skopje",
    })
    log.info("Seeded sample history: %d reports, %d comments", len(REPORTS_DB), len(COMMENTS_DB))


if os.getenv("DEMO_SEED") == "1":
    _seed_demo_data()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Skopje Eco-Agent",
    version="0.3.0",
    description="Civic air quality platform -- monitor, report, discuss, organise, demand",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health ---
@app.get("/")
async def root():
    return {"status": "ok", "service": "Skopje Eco-Agent", "version": "0.3.0"}

# --- Air quality ---
@app.get("/alert", response_model=HealthAlert)
async def get_alert():
    pm25, pm10, weather = await _fetch_all_data()
    return await run_alert_agent(pm25, pm10, weather)

@app.get("/data/sensors")
async def get_sensors():
    pm25 = await fetch_pulse_eco("pm25")
    pm10 = await fetch_pulse_eco("pm10")
    return {"pm25": [r.model_dump() for r in pm25], "pm10": [r.model_dump() for r in pm10]}

@app.get("/data/weather")
async def get_weather():
    w = await fetch_weather()
    return w.model_dump()

# --- Civic Reports (anonymous, with evidence) ---
@app.post("/reports", response_model=CivicReport)
async def submit_report(report: CivicReportIn):
    """Submit an anonymous pollution report with optional evidence. No login, no identity."""
    pm25 = await fetch_pulse_eco("pm25")
    pm10 = await fetch_pulse_eco("pm10")
    pm25_avg = sum(r.value for r in pm25) / len(pm25) if pm25 else 0.0
    pm10_avg = sum(r.value for r in pm10) / len(pm10) if pm10 else 0.0

    report_id = uuid.uuid4().hex[:12]
    stored = CivicReport(
        **report.model_dump(),
        id=report_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        pm25_at_time=round(pm25_avg, 1),
        pm10_at_time=round(pm10_avg, 1),
        supports=0,
    )
    REPORTS_DB.append(stored.model_dump())
    UPVOTES_DB[report_id] = 0
    log.info("Report filed: [%s] %s", stored.category, stored.description[:60])
    return stored

@app.get("/reports")
async def list_reports(limit: int = 50):
    """List recent anonymous reports (newest first) with support counts."""
    reports = list(reversed(REPORTS_DB[-limit:]))
    # Attach latest support counts
    for r in reports:
        r["supports"] = UPVOTES_DB.get(r["id"], 0)
    return {"reports": reports, "total": len(REPORTS_DB)}

@app.post("/reports/{report_id}/support")
async def support_report(report_id: str):
    """Anonymous upvote — 'I've seen this too'."""
    for r in REPORTS_DB:
        if r["id"] == report_id:
            UPVOTES_DB[report_id] = UPVOTES_DB.get(report_id, 0) + 1
            r["supports"] = UPVOTES_DB[report_id]
            return {"report_id": report_id, "supports": UPVOTES_DB[report_id]}
    raise HTTPException(404, "Report not found")

@app.get("/reports/stats")
async def report_stats():
    """Aggregate report statistics by category."""
    cats: dict[str, int] = {}
    hoods: dict[str, int] = {}
    for r in REPORTS_DB:
        cat = r.get("category", "other")
        cats[cat] = cats.get(cat, 0) + 1
        hood = r.get("neighbourhood", "")
        if hood:
            hoods[hood] = hoods.get(hood, 0) + 1
    return {
        "total": len(REPORTS_DB),
        "by_category": cats,
        "by_neighbourhood": hoods,
        "total_supports": sum(UPVOTES_DB.values()),
        "reports_with_evidence": sum(1 for r in REPORTS_DB if r.get("evidence_links") or r.get("evidence_text")),
    }

# --- Anonymous Discussion on Reports ---
@app.post("/reports/{report_id}/comments", response_model=Comment)
async def add_comment(report_id: str, comment: CommentIn):
    """Add an anonymous comment to a report discussion thread."""
    # Check report exists
    if not any(r["id"] == report_id for r in REPORTS_DB):
        raise HTTPException(404, "Report not found")

    # Count existing comments on this report to assign anon label
    existing = sum(1 for c in COMMENTS_DB if c["report_id"] == report_id)
    anon_num = existing + 1

    stored = Comment(
        id=uuid.uuid4().hex[:12],
        report_id=report_id,
        text=comment.text,
        timestamp=datetime.now(timezone.utc).isoformat(),
        anon_label=f"Citizen #{anon_num}",
    )
    COMMENTS_DB.append(stored.model_dump())
    log.info("Comment on report %s by %s", report_id, stored.anon_label)
    return stored

@app.get("/reports/{report_id}/comments")
async def list_comments(report_id: str):
    """List all anonymous comments on a report."""
    comments = [c for c in COMMENTS_DB if c["report_id"] == report_id]
    return {"comments": comments, "total": len(comments)}

# --- Community Events ---
@app.post("/events", response_model=CommunityEvent)
async def create_event(event: CommunityEventIn):
    stored = CommunityEvent(
        **event.model_dump(),
        id=uuid.uuid4().hex[:12],
        created_at=datetime.now(timezone.utc).isoformat(),
        attendees=1,
    )
    EVENTS_DB.append(stored.model_dump())
    log.info("Event created: %s", stored.title)
    return stored

@app.get("/events")
async def list_events():
    return {"events": list(reversed(EVENTS_DB)), "total": len(EVENTS_DB)}

@app.post("/events/{event_id}/join")
async def join_event(event_id: str):
    for ev in EVENTS_DB:
        if ev["id"] == event_id:
            ev["attendees"] = ev.get("attendees", 0) + 1
            return {"event_id": event_id, "attendees": ev["attendees"]}
    raise HTTPException(404, "Event not found")

# --- Municipal Brief (generate, publish, email) ---
@app.get("/brief", response_model=MunicipalBrief)
async def get_municipal_brief():
    """Generate an AI-backed evidence brief."""
    pm25, pm10, weather = await _fetch_all_data()
    pm25_avg = sum(r.value for r in pm25) / len(pm25) if pm25 else 0.0
    pm10_avg = sum(r.value for r in pm10) / len(pm10) if pm10 else 0.0
    wind_avg = sum(weather.windspeed_10m) / len(weather.windspeed_10m) if weather.windspeed_10m else 0.0
    hum_avg = sum(weather.relative_humidity_2m) / len(weather.relative_humidity_2m) if weather.relative_humidity_2m else 0.0
    inversion = wind_avg < 5 and hum_avg > 75
    return await generate_municipal_brief(pm25_avg, pm10_avg, wind_avg, hum_avg, inversion)

@app.post("/briefs/publish")
async def publish_brief(brief: MunicipalBrief):
    """Publish a generated brief to the public board."""
    for existing in BRIEFS_DB:
        if existing.get("id") == brief.id:
            return {"status": "published", "brief_id": brief.id, "total_published": len(BRIEFS_DB)}
    brief_data = brief.model_dump()
    brief_data["published"] = True
    brief_data["published_at"] = datetime.now(timezone.utc).isoformat()
    BRIEFS_DB.append(brief_data)
    log.info("Brief published: %s", brief.title)
    return {"status": "published", "brief_id": brief.id, "total_published": len(BRIEFS_DB)}

@app.get("/briefs")
async def list_published_briefs():
    """List all publicly published municipal briefs."""
    return {"briefs": list(reversed(BRIEFS_DB)), "total": len(BRIEFS_DB)}

@app.post("/briefs/email")
async def email_brief(req: EmailRequest):
    """Send a published brief via anonymous email."""
    # Find the brief
    brief = None
    for b in BRIEFS_DB:
        if b.get("id") == req.brief_id:
            brief = b
            break
    if not brief:
        # Maybe it's not published yet — try generating fresh
        raise HTTPException(404, "Brief not found. Publish it first.")

    result = send_anonymous_email(brief, req.recipient_emails, req.sender_alias)

    # Track that email was sent
    if result["status"] in ("sent", "queued"):
        brief.setdefault("email_sent_to", []).extend(req.recipient_emails)

    return result

# --- Public Dashboard Data ---
@app.get("/dashboard")
async def public_dashboard():
    """Aggregated data for the public evidence dashboard."""
    stats = {
        "total_reports": len(REPORTS_DB),
        "total_supports": sum(UPVOTES_DB.values()),
        "total_comments": len(COMMENTS_DB),
        "total_events": len(EVENTS_DB),
        "total_briefs_published": len(BRIEFS_DB),
        "reports_with_evidence": sum(1 for r in REPORTS_DB if r.get("evidence_links") or r.get("evidence_text")),
    }
    # Category breakdown
    cats: dict[str, int] = {}
    for r in REPORTS_DB:
        cat = r.get("category", "other")
        cats[cat] = cats.get(cat, 0) + 1
    stats["by_category"] = cats

    # Neighbourhood breakdown
    hoods: dict[str, int] = {}
    for r in REPORTS_DB:
        hood = r.get("neighbourhood", "")
        if hood:
            hoods[hood] = hoods.get(hood, 0) + 1
    stats["by_neighbourhood"] = hoods

    # Recent reports with comment counts
    recent = []
    for r in list(reversed(REPORTS_DB[-15:])):
        r_copy = dict(r)
        # Include the thread so the UI needs one request, not one per report
        r_copy["comments"] = [c for c in COMMENTS_DB if c["report_id"] == r["id"]]
        r_copy["comment_count"] = len(r_copy["comments"])
        r_copy["supports"] = UPVOTES_DB.get(r["id"], 0)
        recent.append(r_copy)
    stats["recent_reports"] = recent
    stats["has_sample_data"] = any(r.get("sample") for r in REPORTS_DB)

    return stats

@app.get("/comments/recent")
async def recent_comments(limit: int = 8):
    """Newest comments across all reports, for the live discussion feed."""
    reports = {r["id"]: r for r in REPORTS_DB}
    feed = []
    for c in reversed(COMMENTS_DB[-limit:]):
        r = reports.get(c["report_id"], {})
        feed.append({**c, "category": r.get("category", ""), "neighbourhood": r.get("neighbourhood", "")})
    return {"comments": feed, "total": len(COMMENTS_DB)}

# --- Meta ---
@app.get("/categories")
async def get_categories():
    return {"categories": REPORT_CATEGORIES}


if __name__ == "__main__":
    import uvicorn
    # access_log=False: don't record citizens' IP addresses
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), access_log=False)
