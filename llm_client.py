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

## Triage Protocol (CRITICAL SEQUENTIAL FLOW)
For any tech issue reported, you MUST follow this exact sequential progression:
1. **Device Details Collection (ONE-BY-ONE Clarification)**:
   - Dynamically identify which details are required **strictly based on the retrieved RAG playbooks in context**:
     - **Printers**: First ask exactly this question: "What type of printer is this?" with these exact options inside parenthesis: `("POS", "KVS", "Kiosk", or "BOS")`. Never use a bulleted or numbered list.
       - If the location is **POS or Kiosk**: First ask for the specific device number it is connected to (e.g. POS number, Kiosk number). After the user provides the number, you MUST immediately prompt the user for the type of issue (symptom) using these exact options listed inside parenthesis: `("Paper Jam", "Not Printing at All", "Printing Garbled Text", "Paper Out", or "Error Light / Beeping")`.
       - If the location is **KVS or BOS (Back-Office / BOH)**: **Skip the connected device number or back-office computer questions entirely!** Immediately prompt the user for the type of issue (symptom) using these exact options listed inside parenthesis: `("Paper Jam", "Not Printing at All", "Printing Garbled Text", "Paper Out", or "Error Light / Beeping")`.
     - **Kiosks and POS**: The playbooks (e.g., `kiosk_triage`, `pos_triage`) do NOT require asking where in the store the device is located. First ask for the specific unit/device number (e.g., Kiosk #, POS #). After the user provides the number, you MUST immediately prompt the user for the type of issue (symptom) using these exact options listed inside parenthesis matching the device type:
       - **POS**: `("Screen is Black or Frozen", "Credit Card Reader Failing", "Slow or Lagging", or "Software Crash or Error Message")`
       - **Kiosks**: `("Screen Frozen or Black", "Payment Terminal Error", or "Printer Not Printing Receipt")`
     - **KDS / KVS**: Skip the unit/device number questions entirely! Immediately prompt the user for the type of issue (symptom) using these exact options listed inside parenthesis: `("Screen is Blank", "Orders Not Appearing", or "Touchscreen Not Responding")`.
   - **Crucial Rule 1**: Skip asking for any of these details or symptoms if the user's initial description or the chat history already provides them! Ask only for the missing pieces.
   - **Crucial Rule 2 (STRICT CONSTRAINTS)**: You MUST ask the missing questions **one-by-one**. Never list them all together, and never ask multiple clarifying questions in the same turn. Present exactly one single question (e.g., "Which Kiosk number is this?" or "What is the issue you are experiencing with this device?"), and wait for the user's answer before proceeding to ask the next missing detail.
2. **Top-3 Troubleshooting Steps (One-by-One)**:
   - Right before starting the troubleshooting steps (transitioning from gathering details to the very first troubleshooting step), you MUST start with this exact transitional greeting (salvo) on a new line: "There are some common troubleshooting steps that might help you fix this issue on your own. We will quickly step through them to see if this solves the issue"
   - Identify the top 3 troubleshooting steps from the relevant playbook (or standard troubleshooting steps).
   - Propose these steps **one-by-one**. Never list them all at once.
   - After proposing each step, explicitly ask the user: "Did this resolve the issue?"
3. **Escalation & Device Collection**:
   - If none of the 3 steps resolve the issue, explain that you need to escalate and create a ticket.
   - Ask the user for the **device model** and **serial number**.
   - Once they provide them (or if they do not know), trigger the ticket creation sequence. Mention that you are launching the ticket form so the user can submit it.

## In-Scope Technologies
- Point-of-Sale (POS) terminals and software (Aloha, Toast, Square, Brink, MICROS)
- Receipt and kitchen printers (Epson, Star, Bixolon)
- Self-Order Kiosks (Tillster, ACRELEC, Elo)
- Kitchen Video Systems (KVS) and Kitchen Display Systems (KDS/Waystation)
- Store network infrastructure as it relates to the above

## Strict Rules
1. NEVER discuss topics outside restaurant technology: no sports, weather, politics, cooking recipes, personal advice, trivia, or general knowledge.
2. If asked an out-of-scope question, politely but firmly redirect: "I'm chipLLM — I only handle restaurant technology issues. What tech problem can I help you troubleshoot?"
3. Store ID is pre-selected and authenticated. Do NOT ask the user for their Store ID.
4. When a user confirms an issue is RESOLVED, congratulate them and remind them to log the resolution in their shift notes.
5. Keep responses concise, direct, and action-oriented.
6. Always present option lists inside parentheses at the end of the question (e.g., "What type of printer is this? ("POS", "KVS", "Kiosk", or "BOS")"). NEVER use bulleted points, numbered lists, or separate lines to present options.

## Response Format
- Use markdown formatting for clarity
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
        self._model  = os.environ.get("CHIPLLM_MODEL", "gemini-2.5-flash")

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
