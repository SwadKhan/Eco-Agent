# Eco-Agent

**Civic air quality action platform for Skopje.** Eco-Agent turns live air quality data into organised civic pressure, and keeps everyone anonymous the whole time.

Built at Hacktoberfest 2026, Base42, Skopje 🇲🇰

## Features

| Tab | What it does |
|---|---|
| 🌫️ **Air Quality** | Live PM2.5 / PM10 from Pulse.eco sensors + Open-Meteo weather, with an AI health alert and temperature-inversion detection |
| 📢 **Report Pollution** | Anonymous reports with evidence (photo/video links + description). Each one is stamped with the PM2.5 reading at submission time |
| 📊 **Evidence Dashboard** | Public board of every report, with category and neighbourhood breakdowns, 👍 "I've seen this too" support, anonymous discussion threads (`Citizen #1`, `Citizen #2`…) and published briefs |
| 🤝 **Community Events** | Create and join cleanups, meetups, petitions and awareness campaigns |
| 📄 **Municipal Brief** | AI writes a formal evidence brief from sensor data + reports + evidence, which you can publish to the public board and email anonymously to the municipality |

**Anonymity:** no login, no accounts, no identity fields. The backend turns off request (IP) logging, and emails go out under an alias (e.g. "Concerned Citizens of Skopje") from a shared sender account.

## Stack

- **AI:** Llama 3.1 8B via [Groq](https://console.groq.com) (free tier) + LangChain. If no key is set, it falls back to rule-based alerts and template briefs
- **Open data:** [Pulse.eco](https://pulse.eco) REST API and [Open-Meteo](https://open-meteo.com), neither needs an API key
- **Backend:** FastAPI + Pydantic + httpx, email via Python's built-in `smtplib`
- **Frontend:** Streamlit

## Run locally

```bash
git clone https://github.com/SwadKhan/Eco-Agent.git
cd Eco-Agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then paste your GROQ_API_KEY into .env

# Terminal 1 — backend (http://localhost:8000, API docs at /docs)
python main.py

# Terminal 2 — frontend (http://localhost:8501)
streamlit run app.py
```

### Configuration (`.env`)

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Recommended | AI alerts and briefs. Free at console.groq.com. Without it, rule-based fallback is used |
| `GROQ_MODEL` | No | Defaults to `llama-3.1-8b-instant` |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASS` | No | Real anonymous email sending. Use a throwaway Gmail + [App Password](https://myaccount.google.com/apppasswords). Without these, emails run in demo mode and show a preview |
| `ECO_API_URL` | No (frontend) | Backend URL if it isn't `http://localhost:8000` |

## Deploy

The backend and frontend are separate services:

1. **Backend → [Render](https://render.com)** (free web service): connect this repo, set the build command to `pip install -r requirements.txt` and the start command to `python main.py`, then add `GROQ_API_KEY` (and SMTP vars if you want them) as environment variables.
2. **Frontend → [Streamlit Community Cloud](https://share.streamlit.io)**: deploy `app.py` from this repo and set the environment variable / secret `ECO_API_URL` to your Render URL.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/alert` | AI health alert from live data |
| GET | `/data/sensors`, `/data/weather` | Raw sensor and weather data |
| POST / GET | `/reports` | File / list anonymous reports |
| POST | `/reports/{id}/support` | Anonymous upvote |
| POST / GET | `/reports/{id}/comments` | Anonymous discussion |
| GET | `/reports/stats`, `/dashboard` | Aggregates for the public dashboard |
| POST / GET | `/events`, POST `/events/{id}/join` | Community events |
| GET | `/brief` | Generate a municipal brief |
| POST | `/briefs/publish`, GET `/briefs` | Public brief board |
| POST | `/briefs/email` | Send a published brief anonymously |

Data is kept in memory (hackathon MVP), so it resets when the backend restarts.
