"""
llm_client.py
-------------
Vertex AI / Google Gen AI SDK wrapper for chipLLM.

Authentication: Application Default Credentials (ADC) exclusively.
  - Local dev:   `gcloud auth application-default login`
  - Cloud Run:   Workload Identity (service account attached to the revision)
  No API keys. No service account JSON paths. No credential overrides.
"""

from __future__ import annotations

import os
import re
import random
import string
from datetime import datetime

import streamlit as st
from google import genai
from google.genai import types as genai_types

# ---------------------------------------------------------------------------
# Runtime context — sourced from environment, never hardcoded
# ---------------------------------------------------------------------------

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
LOCATION   = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")


@st.cache_resource
def get_genai_client() -> genai.Client:
    """
    Initializes the Google Gen AI Client using native Application Default
    Credentials (ADC). Decorated with @st.cache_resource so the underlying
    HTTP connection pool is created exactly once and shared across all
    Streamlit sessions for the lifetime of the server process.

    Local:      binds to the active developer token from
                `gcloud auth application-default login`.
    Cloud Run:  seamlessly inherits the runtime service identity via
                Workload Identity — zero credential management required.
    """
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )

# ---------------------------------------------------------------------------
# System Prompt – strict persona & domain boundary definition
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are chipLLM, the Technical Support Assistant for store managers at a global restaurant chain.

## Your Mission
You exclusively triage and resolve restaurant technology issues. You are the first line of defense before a live support ticket is created. You are calm, professional, and efficient.

## In-Scope Technologies
- Point-of-Sale (POS) terminals and software (Aloha, Toast, Square, Brink, MICROS)
- Receipt and kitchen printers (Epson, Star, Bixolon)
- Self-Order Kiosks (Tillster, ACRELEC, Elo)
- Kitchen Video Systems (KVS) and Kitchen Display Systems (KDS/Waystation)
- Store network infrastructure as it relates to the above

## Strict Rules
1. NEVER discuss topics outside restaurant technology: no sports, weather, politics, cooking recipes, personal advice, trivia, or general knowledge.
2. If asked an out-of-scope question, politely but firmly redirect: "I'm chipLLM — I only handle restaurant technology issues. What tech problem can I help you troubleshoot?"
3. ALWAYS ask clarifying questions to gather: Store ID, affected asset type, asset serial number, and symptom description.
4. When a user confirms an issue is RESOLVED, congratulate them and remind them to log the resolution in their shift notes.
5. If a user requests a human agent or the issue cannot be resolved remotely, acknowledge the escalation and confirm a ticket will be generated.
6. Keep responses concise and action-oriented. Use numbered steps for troubleshooting. Use **bold** for critical warnings.

## Response Format
- Use markdown formatting for clarity
- Number all troubleshooting steps
- Put critical warnings in **bold**
- Aim for responses under 250 words unless a detailed playbook is provided in context

## Tone
Professional, direct, and empathetic. You understand the manager is stressed — help them fast."""


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class ChipLLMClient:
    """
    Thin wrapper around the Google Gen AI SDK for Vertex AI.

    The underlying genai.Client is obtained from get_genai_client(), which is
    decorated with @st.cache_resource — one connection pool per server process,
    shared across all user sessions. ChipLLMClient itself is lightweight and
    may be instantiated per-session via st.session_state.
    """

    def __init__(self) -> None:
        # Reuse the process-level cached client — no new connection on each call
        self._client = get_genai_client()
        self._model  = os.environ.get("CHIPLLM_MODEL", "gemini-2.0-flash-001")

    def build_contents(
        self,
        messages: list[dict],
        rag_context: str | None = None,
    ) -> list[genai_types.Content]:
        """
        Convert the Streamlit session message list into the SDK Content format.
        Optionally injects RAG context into the final user turn.
        """
        contents: list[genai_types.Content] = []

        for i, msg in enumerate(messages):
            role = "user" if msg["role"] == "user" else "model"
            text = msg["content"]

            # Inject RAG context into the most recent user message
            if rag_context and i == len(messages) - 1 and role == "user":
                text = (
                    f"[RELEVANT KNOWLEDGE BASE CONTEXT — use this to answer]\n"
                    f"{rag_context}\n"
                    f"[END CONTEXT]\n\n"
                    f"User question: {text}"
                )

            contents.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part(text=text)],
                )
            )

        return contents

    def stream_response(
        self,
        messages: list[dict],
        rag_context: str | None = None,
    ):
        """
        Generator that yields text chunks from a streaming Vertex AI response.
        """
        contents = self.build_contents(messages, rag_context)

        config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=1024,
            top_p=0.9,
        )

        response_stream = self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        )

        for chunk in response_stream:
            if chunk.text:
                yield chunk.text


# ---------------------------------------------------------------------------
# Structured metadata extraction for ServiceNow / Genesys routing
# ---------------------------------------------------------------------------

def _rand_id(prefix: str, length: int = 8) -> str:
    return prefix + "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def extract_ticket_metadata(conversation_history: list[dict]) -> dict:
    """
    Parse the conversation history to extract structured ticket metadata
    that would be passed to ServiceNow or Genesys for zero-touch routing.
    """
    full_text = " ".join(m["content"] for m in conversation_history).lower()

    # --- Asset type detection ------------------------------------------------
    asset_type = "UNKNOWN"
    asset_map = {
        "POS_TERMINAL": ["pos", "terminal", "register", "aloha", "toast", "square", "brink"],
        "PRINTER": ["printer", "receipt", "print", "paper", "jam"],
        "KIOSK": ["kiosk", "self order", "self-order", "self service"],
        "KDS_KVS": ["kds", "kvs", "waystation", "kitchen display", "kitchen video"],
    }
    for asset, keywords in asset_map.items():
        if any(kw in full_text for kw in keywords):
            asset_type = asset
            break

    # --- Severity inference --------------------------------------------------
    severity = "MEDIUM"
    if any(kw in full_text for kw in ["critical", "down completely", "totally down", "urgent", "30 minutes", "hour"]):
        severity = "CRITICAL"
    elif any(kw in full_text for kw in ["slow", "intermittent", "occasionally", "sometimes"]):
        severity = "LOW"
    elif any(kw in full_text for kw in ["offline", "not working", "broken", "frozen", "crash"]):
        severity = "HIGH"

    # --- Store ID extraction (look for numeric patterns) ---------------------
    store_id_match = re.search(r"\b(?:store|location|site)[\s#-]*(\d{3,6})\b", full_text)
    store_id = f"STORE-{store_id_match.group(1)}" if store_id_match else "STORE-UNCONFIRMED"

    # --- Serial number extraction -------------------------------------------
    serial_match = re.search(r"\b([A-Z]{2,4}[\d]{5,12})\b", " ".join(m["content"] for m in conversation_history))
    serial_number = serial_match.group(1) if serial_match else "SN-PENDING"

    # --- Symptom summary from last few assistant turns -----------------------
    assistant_msgs = [m["content"] for m in conversation_history if m["role"] == "assistant"]
    description = assistant_msgs[-1][:200] + "..." if assistant_msgs else "No description captured."

    # --- Routing target based on severity -----------------------------------
    routing_target = {
        "CRITICAL": "GENESYS_TIER2_IMMEDIATE",
        "HIGH": "GENESYS_TIER1_PRIORITY",
        "MEDIUM": "SERVICENOW_AUTO_TICKET",
        "LOW": "SERVICENOW_STANDARD_QUEUE",
    }.get(severity, "SERVICENOW_AUTO_TICKET")

    return {
        "ticket_id": _rand_id("CHK-"),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "store_id": store_id,
        "asset_type": asset_type,
        "asset_serial_number": serial_number,
        "severity": severity,
        "description": description,
        "routing_target": routing_target,
        "channel": "chipLLM_CHATBOT",
        "sla_breach_minutes": {"CRITICAL": 30, "HIGH": 60, "MEDIUM": 120, "LOW": 480}.get(severity, 120),
        "auto_dispatch": severity in ("CRITICAL", "HIGH"),
    }
