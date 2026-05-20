# chipLLM – Restaurant Tech Support Chatbot
> Mobile-focused Streamlit PoC · Vertex AI · Cloud Run Serverless

---

## Architecture

```
User (Mobile Browser)
      │
      ▼
┌─────────────────────────────────────────────┐
│               Streamlit (app.py)             │
│                                             │
│  ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │Guardrails│──▶│  RAG KB  │──▶│  LLM    │ │
│  │Layer 1   │   │(in-mem)  │   │ Client  │ │
│  └──────────┘   └──────────┘   └─────────┘ │
│                                      │      │
│  ┌─────────────────────────────────┐ │      │
│  │   Ticket Metadata Extractor     │◀┘      │
│  │   ServiceNow / Genesys Payload  │        │
│  └─────────────────────────────────┘        │
└─────────────────────────────────────────────┘
      │
      ▼
Vertex AI (gemini-2.0-flash-001)
```

## Project Structure

```
chipllm/
├── app.py              # Main Streamlit UI + orchestration
├── guardrails.py       # Layer 1: local keyword guardrails & intent classification
├── knowledge_base.py   # In-memory RAG playbooks (POS, Printer, Kiosk, KDS)
├── llm_client.py       # Google Gen AI SDK wrapper + ticket metadata extractor
├── requirements.txt
├── Dockerfile          # Multi-stage, non-root, Cloud Run ready
├── .streamlit/
│   └── config.toml    # Headless server + dark theme config
└── .env.example        # Environment variable template
```

## Local Development

### Prerequisites
- Python 3.12+
- A Google Cloud project with Vertex AI API enabled, **or** a Google AI Studio API key

### Setup

```bash
cd chipllm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — fill in GOOGLE_API_KEY (local) or GOOGLE_CLOUD_PROJECT (Vertex)
```

### Run

```bash
# Using API key (local dev)
GOOGLE_API_KEY=your_key_here streamlit run app.py

# Using Vertex AI (ADC — run `gcloud auth application-default login` first)
GOOGLE_CLOUD_PROJECT=your-project GOOGLE_CLOUD_LOCATION=us-central1 streamlit run app.py
```

Open: http://localhost:8501

---

## Cloud Run Deployment

### 1. Build & push

```bash
export PROJECT_ID=your-gcp-project
export REGION=us-central1
export IMAGE=gcr.io/$PROJECT_ID/chipllm:latest

gcloud builds submit --tag $IMAGE
```

### 2. Deploy

```bash
gcloud run deploy chipllm \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION \
  --service-account chipllm-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --port 8080
```

### 3. Service account permissions

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:chipllm-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

---

## Features

| Feature | Implementation |
|---|---|
| Mobile viewport (375px) | Custom CSS `max-width: 420px` |
| Persistent conversation | `st.session_state.messages` |
| Domain guardrails | `guardrails.py` — keyword blocklist, zero LLM calls |
| In-memory RAG | `knowledge_base.py` — 4 detailed playbooks, keyword retrieval |
| Streaming responses | `client.models.generate_content_stream()` |
| Ticket metadata | `llm_client.extract_ticket_metadata()` — ServiceNow/Genesys payload |
| Structured sidebar | JSON rendered live, severity-color coded |
| Dark kitchen theme | HSL-tuned CSS, Inter font, orange accent |

## Guardrail-Blocked Topics

| Category | Sample Keywords |
|---|---|
| Sports | baseball, nfl, nba, soccer, espn |
| Weather | forecast, temperature, rain, storm |
| Politics | democrat, republican, election, policy |
| Trivia | recipe, movie, music, travel, crypto |

## Escalation Triggers

Say any of: `agent`, `ticket`, `escalate`, `live support`, `totally down`, `urgent` → sidebar populates with a live `ServiceNow / Genesys` JSON payload instantly.
