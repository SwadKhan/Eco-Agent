"""
Streamlit dashboard for Skopje Eco-Agent — Civic Air Quality Platform.
Run: streamlit run app.py
Requires the FastAPI backend on localhost:8000.
"""

import os
from datetime import datetime, timezone

import streamlit as st
import httpx

# Backend URL — override with ECO_API_URL when deployed
API = os.getenv("ECO_API_URL", "http://localhost:8000").rstrip("/")

RISK_COLORS = {"low": "#22c55e", "moderate": "#f59e0b", "high": "#ef4444", "severe": "#7c1d1d"}
RISK_EMOJIS = {"low": "🟢", "moderate": "🟡", "high": "🔴", "severe": "🟣"}

CATEGORY_LABELS = {
    "illegal_burning": "🔥 Illegal burning",
    "construction_dust": "🏗️ Construction dust",
    "factory_emissions": "🏭 Factory emissions",
    "vehicle_exhaust": "🚗 Vehicle exhaust",
    "waste_dumping": "🗑️ Waste dumping",
    "municipal_negligence": "🏛️ Municipal negligence",
    "other": "📋 Other",
}

EVENT_TYPES = {
    "cleanup": "🧹 Community cleanup",
    "meetup": "🤝 Indoor meetup",
    "petition": "📝 Petition campaign",
    "protest": "📢 Protest / demonstration",
    "awareness": "📣 Awareness campaign",
}

st.set_page_config(page_title="Eco-Agent Skopje", page_icon="🌍", layout="wide")

# -- helpers ----------------------------------------------------------------

def api_get(path, timeout=30):
    try:
        r = httpx.get(f"{API}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None

def api_post(path, data, timeout=30):
    try:
        r = httpx.post(f"{API}{path}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None

def ago(ts):
    """ISO timestamp -> '5 min ago' style."""
    try:
        secs = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
    except (TypeError, ValueError):
        return ""
    if secs < 60:
        return "just now"
    for unit, size in (("day", 86400), ("h", 3600), ("min", 60)):
        if secs >= size:
            n = int(secs // size)
            return f"{n} {unit}{'s' if unit == 'day' and n > 1 else ''} ago"

def post_comment(report_id):
    # Runs as a button callback so the input can be cleared afterwards
    key = f"comment_input_{report_id}"
    text = st.session_state.get(key, "").strip()
    if text:
        api_post(f"/reports/{report_id}/comments", {"text": text})
        st.session_state[key] = ""

@st.fragment(run_every=4)
def live_feed():
    # Re-runs on its own every few seconds, so new comments show up without a reload
    st.markdown("### 💬 Live discussion")
    feed = api_get("/comments/recent?limit=6")
    if not feed or not feed.get("comments"):
        st.caption("No comments yet.")
        return
    for c in feed["comments"]:
        cat = CATEGORY_LABELS.get(c.get("category", ""), c.get("category", ""))
        st.markdown(
            f"**{c.get('anon_label', 'Citizen')}** on {cat} · {c.get('neighbourhood', '')} "
            f"<span style='color:#888;font-size:0.85em'>· {ago(c.get('timestamp'))}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"“{c.get('text', '')}”")

@st.cache_data(ttl=300, show_spinner=False)
def cached_alert():
    # The alert runs the LLM — cache it so every button click doesn't re-run it
    return api_get("/alert", timeout=60)

# -- header -----------------------------------------------------------------

st.title("🌍 Eco-Agent Skopje")
st.caption("Civic air quality platform — monitor · report · discuss · organise · demand")

# -- tabs -------------------------------------------------------------------

tab_air, tab_report, tab_dashboard, tab_events, tab_brief = st.tabs([
    "🌫️ Air Quality",
    "📢 Report Pollution",
    "📊 Evidence Dashboard",
    "🤝 Community Events",
    "📄 Municipal Brief",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Air Quality Monitor
# ══════════════════════════════════════════════════════════════════════════════

with tab_air:
    if st.button("🔄 Refresh", key="refresh_air", type="primary", use_container_width=True):
        st.cache_data.clear()

    with st.spinner("Fetching live data and running AI analysis..."):
        alert = cached_alert()

    if alert:
        risk = alert.get("risk_level", "low")
        emoji = RISK_EMOJIS.get(risk, "⚪")
        color = RISK_COLORS.get(risk, "#888")

        st.markdown(
            f"""<div style="background:{color}22;border-left:6px solid {color};
            padding:1.2rem 1.5rem;border-radius:8px;margin-bottom:1rem;">
            <h2 style="margin:0;color:{color};">{emoji} {alert.get('headline','')}</h2>
            <p style="margin:0.5rem 0 0 0;font-size:1.1rem;">{alert.get('summary','')}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PM2.5", f"{alert.get('pm25_avg',0):.1f} ug/m3", delta="WHO: 15", delta_color="inverse")
        c2.metric("PM10", f"{alert.get('pm10_avg',0):.1f} ug/m3", delta="WHO: 45", delta_color="inverse")
        c3.metric("Wind", f"{alert.get('wind_speed_avg',0):.1f} km/h")
        c4.metric("Inversion", "⚠️ YES" if alert.get("inversion_risk") else "✅ NO")

        a1, a2 = st.columns(2)
        with a1:
            st.info(f"**General:** {alert.get('advice_general','')}")
        with a2:
            st.warning(f"**Vulnerable groups:** {alert.get('advice_vulnerable','')}")

        weather = api_get("/data/weather")
        if weather and weather.get("time"):
            import pandas as pd
            with st.expander("📊 24h weather forecast"):
                df = pd.DataFrame({
                    "Time": weather["time"],
                    "Temp (C)": weather.get("temperature_2m", []),
                    "Wind (km/h)": weather.get("windspeed_10m", []),
                    "Humidity (%)": weather.get("relative_humidity_2m", []),
                })
                st.line_chart(df.set_index("Time"))

        with st.expander("🔍 Raw alert JSON"):
            st.json(alert)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Anonymous Pollution Reports (with evidence)
# ══════════════════════════════════════════════════════════════════════════════

with tab_report:
    st.subheader("📢 Report a pollution source")
    st.caption("100% anonymous — no login, no tracking, no identity stored.")

    with st.form("report_form"):
        cat = st.selectbox(
            "What are you reporting?",
            options=list(CATEGORY_LABELS.keys()),
            format_func=lambda x: CATEGORY_LABELS[x],
        )
        desc = st.text_area(
            "Describe what you see/smell/hear",
            placeholder="e.g. 'Thick smoke from a backyard in Aerodrom, smells like burning plastic'",
            max_chars=1000,
        )
        hood = st.text_input("Neighbourhood (optional)", placeholder="e.g. Centar, Aerodrom, Kisela Voda")
        urgency = st.slider("Urgency", 1, 5, 3, help="1 = minor nuisance, 5 = immediate health hazard")

        st.markdown("---")
        st.markdown("**📎 Evidence (optional — strengthens your report)**")
        evidence_text = st.text_area(
            "Describe your evidence in detail",
            placeholder="e.g. 'I can see thick black smoke from the chimney at the corner of ul. Partizanski Odredi and Leninova. It's been going on for 2 hours.'",
            max_chars=2000,
            key="evidence_text",
        )
        evidence_links_raw = st.text_input(
            "Photo/video links (comma-separated)",
            placeholder="e.g. https://imgur.com/abc123, https://photos.google.com/xyz",
            key="evidence_links",
        )
        submitted = st.form_submit_button("Submit report", type="primary", use_container_width=True)

    if submitted and desc:
        evidence_links = [l.strip() for l in evidence_links_raw.split(",") if l.strip()] if evidence_links_raw else []
        result = api_post("/reports", {
            "category": cat,
            "description": desc,
            "neighbourhood": hood,
            "urgency": urgency,
            "evidence_text": evidence_text,
            "evidence_links": evidence_links,
        })
        if result:
            st.success(f"Report filed. PM2.5 at time of report: **{result.get('pm25_at_time', 0)} ug/m3**")
            if evidence_text or evidence_links:
                st.info("📎 Evidence attached — this strengthens the municipal brief.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Public Evidence Dashboard + Discussion
# ══════════════════════════════════════════════════════════════════════════════

with tab_dashboard:
    st.subheader("📊 Public Evidence Dashboard")
    st.caption("All reports, evidence, and discussions — visible to everyone, anonymous for everyone.")

    # Fetch dashboard data
    dashboard = api_get("/dashboard")

    if dashboard:
        # Top-level stats
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Reports", dashboard.get("total_reports", 0))
        m2.metric("Community Support", dashboard.get("total_supports", 0))
        m3.metric("Discussions", dashboard.get("total_comments", 0))
        m4.metric("With Evidence", dashboard.get("reports_with_evidence", 0))
        m5.metric("Briefs Published", dashboard.get("total_briefs_published", 0))
        if dashboard.get("has_sample_data"):
            st.caption("ℹ️ Includes sample history so the dashboard shows how it looks in use.")

        col_charts, col_feed = st.columns([3, 2], gap="large")
        with col_charts:
            # Category breakdown
            by_cat = dashboard.get("by_category", {})
            if by_cat:
                st.markdown("### Reports by category")
                st.bar_chart({CATEGORY_LABELS.get(k, k): v for k, v in by_cat.items()}, horizontal=True)

            # Neighbourhood breakdown
            by_hood = dashboard.get("by_neighbourhood", {})
            if by_hood:
                st.markdown("### Reports by neighbourhood")
                st.bar_chart(by_hood, horizontal=True)
        with col_feed:
            live_feed()

        # Recent reports with discussion
        st.markdown("---")
        st.markdown("### Recent reports & discussions")

        recent = dashboard.get("recent_reports", [])
        if recent:
            for r in recent:
                report_id = r.get("id", "")
                cat_label = CATEGORY_LABELS.get(r.get("category", ""), r.get("category", ""))
                supports = r.get("supports", 0)
                comment_count = r.get("comment_count", 0)
                has_evidence = bool(r.get("evidence_links") or r.get("evidence_text"))

                with st.container():
                    # Report header
                    col_info, col_actions = st.columns([4, 1])
                    with col_info:
                        evidence_badge = " 📎" if has_evidence else ""
                        st.markdown(
                            f"**{cat_label}{evidence_badge}** · "
                            f"{r.get('neighbourhood', 'Unknown area')} · "
                            f"PM2.5: {r.get('pm25_at_time', '?')} ug/m3 · "
                            f"Urgency: {'🔴' * r.get('urgency', 1)} · "
                            f"_{ago(r.get('timestamp'))}_"
                        )
                        st.caption(r.get("description", ""))

                    with col_actions:
                        if st.button(f"👍 {supports}", key=f"support_{report_id}", use_container_width=True):
                            api_post(f"/reports/{report_id}/support", {})
                            st.rerun()

                    # Evidence section
                    if has_evidence:
                        with st.expander(f"📎 View evidence"):
                            if r.get("evidence_text"):
                                st.write(r["evidence_text"])
                            if r.get("evidence_links"):
                                for link in r["evidence_links"]:
                                    st.markdown(f"- [{link}]({link})")

                    # Discussion thread
                    with st.expander(f"💬 Discussion ({comment_count} comments)"):
                        for c in r.get("comments", []):
                            st.markdown(
                                f"**{c.get('anon_label', 'Citizen')}** "
                                f"<span style='color:#888;font-size:0.85em'>· {ago(c.get('timestamp'))}</span><br>"
                                f"{c.get('text', '')}",
                                unsafe_allow_html=True,
                            )

                        # Add comment form
                        st.text_input(
                            "Add to discussion (anonymous)",
                            placeholder="Share your observation or additional info...",
                            key=f"comment_input_{report_id}",
                        )
                        st.button("Post comment", key=f"comment_btn_{report_id}",
                                  on_click=post_comment, args=(report_id,))

                    st.divider()
        else:
            st.info("No reports yet. Go to the Report tab to file the first one.")

        # Published briefs section
        st.markdown("---")
        st.markdown("### Published municipal briefs")
        briefs_data = api_get("/briefs")
        if briefs_data and briefs_data.get("briefs"):
            for b in briefs_data["briefs"]:
                with st.expander(f"📄 {b.get('title', 'Brief')} — {b.get('date', '')[:10]}"):
                    st.markdown(f"**Air quality:** {b.get('air_quality_summary', '')}")
                    st.markdown(f"**Citizen reports:** {b.get('citizen_reports_summary', '')}")
                    if b.get("demands"):
                        st.markdown("**Demands:**")
                        for i, d in enumerate(b["demands"], 1):
                            st.markdown(f"{i}. {d}")
                    if b.get("email_sent_to"):
                        st.success(f"✉️ Emailed to: {', '.join(b['email_sent_to'])}")
                    st.code(b.get("full_text", ""), language=None)
        else:
            st.info("No briefs published yet. Generate one in the Municipal Brief tab.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Community Events
# ══════════════════════════════════════════════════════════════════════════════

with tab_events:
    st.subheader("🤝 Organise community action")
    st.caption("Create or join events — cleanups, meetups, petitions, awareness campaigns.")

    with st.form("event_form"):
        title = st.text_input("Event title", placeholder="e.g. 'Park cleanup in Gradski Park'")
        etype = st.selectbox(
            "Event type",
            options=list(EVENT_TYPES.keys()),
            format_func=lambda x: EVENT_TYPES[x],
        )
        edesc = st.text_area("Details", placeholder="What, where, and when", max_chars=1000)
        ehood = st.text_input("Neighbourhood", placeholder="e.g. Centar")
        edate = st.text_input("Date", placeholder="e.g. 2026-10-24 or 'This Saturday'")
        create = st.form_submit_button("Create event", type="primary", use_container_width=True)

    if create and title:
        result = api_post("/events", {
            "title": title,
            "event_type": etype,
            "description": edesc,
            "neighbourhood": ehood,
            "date": edate,
        })
        if result:
            st.success(f"Event **{title}** created.")

    st.divider()
    st.subheader("Upcoming events")
    events_data = api_get("/events")
    if events_data and events_data.get("events"):
        for ev in events_data["events"]:
            etype_label = EVENT_TYPES.get(ev.get("event_type", ""), ev.get("event_type", ""))
            with st.container():
                col_info, col_join = st.columns([4, 1])
                with col_info:
                    st.markdown(f"### {ev.get('title', 'Untitled')}")
                    st.caption(f"{etype_label} · {ev.get('neighbourhood', '')} · {ev.get('date', 'TBD')}")
                    if ev.get("description"):
                        st.write(ev["description"])
                    st.caption(f"👥 {ev.get('attendees', 0)} attending")
                with col_join:
                    if st.button("Join", key=f"join_{ev['id']}", use_container_width=True):
                        api_post(f"/events/{ev['id']}/join", {})
                        st.rerun()
                st.divider()
    else:
        st.info("No events yet. Create the first one above.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI Municipal Brief (generate, publish, email)
# ══════════════════════════════════════════════════════════════════════════════

with tab_brief:
    st.subheader("📄 Generate & publish municipal brief")
    st.caption(
        "AI generates a formal evidence brief combining live sensor data + citizen reports + evidence. "
        "Publish it publicly, then email it anonymously to the municipality."
    )

    # Step 1: Generate
    if st.button("🤖 Generate brief", type="primary", use_container_width=True):
        with st.spinner("AI is drafting the brief..."):
            brief = api_get("/brief", timeout=60)
        if brief:
            # Store in session state for publish/email
            st.session_state["current_brief"] = brief

    # Keep showing the brief across reruns (publish/email clicks)
    brief = st.session_state.get("current_brief")
    if brief:
        st.markdown(f"## {brief.get('title', 'Municipal Brief')}")
        st.caption(brief.get("date", ""))

        st.markdown("### Air quality situation")
        st.write(brief.get("air_quality_summary", ""))

        st.markdown("### Citizen reports")
        st.write(brief.get("citizen_reports_summary", ""))

        st.markdown("### Demands")
        for i, d in enumerate(brief.get("demands", []), 1):
            st.markdown(f"**{i}.** {d}")

        st.markdown("### Evidence cited")
        st.write(brief.get("evidence", ""))

        st.divider()
        st.markdown("### Full brief (copy-paste ready)")
        st.code(brief.get("full_text", ""), language=None)

    # Step 2: Publish to public board
    if "current_brief" in st.session_state:
        st.divider()
        st.markdown("### Publish & send")

        col_pub, col_email = st.columns(2)

        with col_pub:
            if st.button("📢 Publish to public board", use_container_width=True):
                result = api_post("/briefs/publish", st.session_state["current_brief"])
                if result and result.get("status") == "published":
                    st.success(f"Brief published! Now visible on the Evidence Dashboard.")
                    st.balloons()

        # Step 3: Email anonymously
        with col_email:
            st.markdown("**Send anonymously via email**")

        with st.form("email_form"):
            recipients = st.text_input(
                "Recipient emails (comma-separated)",
                placeholder="e.g. info@skopje.gov.mk, sovet@skopje.gov.mk",
            )
            sender_alias = st.text_input(
                "Sender name (anonymous alias)",
                value="Concerned Citizens of Skopje",
                help="This is what appears as the sender — no real identity is revealed.",
            )
            send_btn = st.form_submit_button("✉️ Send anonymous email", type="primary", use_container_width=True)

        if send_btn and recipients:
            brief = st.session_state.get("current_brief", {})
            if not brief.get("id"):
                st.error("Generate a brief first.")
            else:
                # First publish if not already
                api_post("/briefs/publish", brief)
                # Then email
                recipient_list = [r.strip() for r in recipients.split(",") if r.strip()]
                result = api_post("/briefs/email", {
                    "brief_id": brief["id"],
                    "recipient_emails": recipient_list,
                    "sender_alias": sender_alias,
                })
                if result:
                    status = result.get("status", "")
                    if status == "sent":
                        st.success(f"Email sent to {', '.join(recipient_list)}!")
                    elif status == "queued":
                        st.info(
                            f"Email queued for delivery. (In production, this sends immediately.)\n\n"
                            f"**Preview:**\n{result.get('preview', '')[:300]}..."
                        )
                    else:
                        st.error(f"Email failed: {result.get('error', 'Unknown error')}")
    else:
        st.info("Click the button above to generate a brief based on current data and citizen reports.")

# -- footer -----------------------------------------------------------------

st.divider()
st.caption(
    "Built at Hacktoberfest 2026 · Base42, Skopje 🇲🇰 · "
    "Data: [pulse.eco](https://pulse.eco) + [Open-Meteo](https://open-meteo.com) · "
    "AI: gpt-oss-20b via [Groq](https://groq.com) · 100% open-source · 100% anonymous"
)
