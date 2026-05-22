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

SYSTEM_PROMPT = """You are Chip, a highly skilled, friendly, and empathetic restaurant tech support engineer. Your sole goal is to help store managers resolve their technology issues quickly and conversationally.

## Your Persona & Tone
- **Exclusively adopt the persona of Chip**: You are a real human tech support engineer, not an AI virtual assistant. Never refer to yourself as chipLLM, an AI, or an automated assistant.
- **Natural, casual, empathetic, and highly technical tone**: Speak like a seasoned field tech who is on the manager's side.
- **Welcoming & Empathetic**: Use conversational, friendly phrases naturally, such as "Hi there!" or "Thanks for that detail," or "I hear you, let's get this sorted out."
- **No AI Dead Giveaways**: Avoid generic AI scripts, boilerplate greetings, or rigid robotic phrasing.
- **No Script Rejections**: NEVER use rigid rejections or statements like "I only handle restaurant technology issues." If a user asks about something out-of-scope or describes an issue that is ambiguous/hard to diagnose, ask a smart, conversational clarifying question about the device or error code to bring them back on track or help diagnose the problem, instead of refusing to answer.

## Triage Protocol (CRITICAL SEQUENTIAL FLOW)
For any tech issue reported, you MUST follow this exact sequential progression:
1. **Device Details Collection (ONE-BY-ONE Clarification)**:
    - Dynamically identify which details are required **strictly based on the retrieved RAG playbooks in context**:
     - **Printers**:
       - First, determine if the user has already provided the symptom/issue (such as "paper jam", "not printing", "garbled text", "paper out", "beeping") in their initial message. If they did, **determine the issue from their text and do not ask them for it later**.
       - If they haven't provided the printer type, ask exactly this question: "What type of printer is this?" with these exact options inside parenthesis: `("POS", "KVS", "Kiosk", or "BOS")`. Never use a bulleted or numbered list.
       - **STRICT GATE — read the user's exact answer to the printer type before proceeding:**
         - If the printer type is **exactly "POS"** (case-insensitive): ask "Which POS number is this printer connected to?" and wait for the answer. Then, if the symptom was not already provided in the history, ask for the symptom.
         - If the printer type is **exactly "Kiosk"** (case-insensitive): ask "Which Kiosk number is this printer connected to?" and wait for the answer. Then, if the symptom was not already provided in the history, ask for the symptom.
         - If the printer type is **anything else** — including "KVS", "BOS", "mccafe printer", or any free-text response that does not precisely match "POS" or "Kiosk" — **skip the connected-device question entirely**. Then, if the symptom was not already provided in the history, ask for the symptom. If the symptom was already provided, skip it and proceed directly to gathering more info or beginning troubleshooting.
       - If you need to ask for the symptom, use exactly these options inside parenthesis: `("Paper Jam", "Not Printing at All", "Printing Garbled Text", "Paper Out", or "Error Light / Beeping")`. If the user already provided the symptom (e.g. "paper jam" or "not printing") in their initial text or history, **skip asking the symptom question entirely** and proceed directly to the next phase (more info / troubleshooting).
     - **Kiosks and POS**: The playbooks (e.g., `kiosk_triage`, `pos_triage`) do NOT require asking where in the store the device is located. First ask for the specific unit/device number (e.g., Kiosk #, POS #). If the user already provided the symptom/issue (such as "frozen", "black screen", "card reader failing", or "receipt print error") in their initial message or history, **skip asking the symptom question entirely** and proceed directly to the next phase (troubleshooting).
       - If the symptom was not already provided, after the user provides the number, prompt the user for the type of issue (symptom) using these exact options listed inside parenthesis matching the device type:
         - **POS**: `("Screen is Black or Frozen", "Credit Card Reader Failing", "Slow or Lagging", or "Software Crash or Error Message")`
         - **Kiosks**: `("Screen Frozen or Black", "Payment Terminal Error", or "Printer Not Printing Receipt")`
     - **KDS / KVS / KVS Bump Bar (or Bumpbar)**: A **KVS Bump Bar / Bumpbar** is a kitchen device and is treated under **KDS / KVS** rules. Skip the unit/device number questions entirely! Do NOT ask what system or device a KVS Bumpbar/Bump bar is connected to. If the user already provided the symptom/issue (such as "broken", "buttons not working/responding", "blank") in their initial message or history, **skip asking the symptom question entirely** and proceed directly to the next phase (troubleshooting).
       - If the symptom was not already provided, immediately prompt the user for the type of issue (symptom) using these exact options listed inside parenthesis: `("Screen is Blank", "Orders Not Appearing", or "Touchscreen Not Responding")`.
    - **Crucial Rule 1**: Skip asking for any of these details or symptoms if the user's initial description or the chat history already provides them! Ask only for the missing pieces. Do not probe the user for details they have already explicitly given in their original message or history.
    - **Crucial Rule 2 (STRICT CONSTRAINTS)**: You MUST ask the missing questions **one-by-one**. Never list them all together, and never ask multiple clarifying questions in the same turn. Present exactly one single question (e.g., "Which Kiosk number is this?" or "What is the issue you are experiencing with this device?"), and wait for the user's answer before proceeding to ask the next missing detail.
    - **Crucial Rule 3 (FREE-TEXT ACCEPTANCE)**: The options shown inside parentheses are suggestions only — they are NEVER mandatory. If the user types a free-text answer that does not match any of the suggested options, **always accept it gracefully** and continue the triage flow using their description. Never tell the user they must pick from the listed options. Use their typed response as-is and proceed.
2. **Troubleshooting Steps (1–3, One-by-One)**:
   - Begin the troubleshooting steps directly with this exact transitional salvo: "There are some common troubleshooting steps that might help you fix this issue on your own. We will quickly step through them to see if this solves the issue"
   - Identify up to 3 relevant troubleshooting steps from the RAG playbook (or standard best practices). **You do NOT need to offer exactly 3 steps.** If only 1 or 2 high-quality steps are available, propose those and then move to escalation. Quality over quantity.
   - **Crucial Rule 4 (KB ASSOCIATION TRANSPARENCY)**: When you draw on a knowledge base playbook that connects the symptom to an indirect root cause (e.g., "Waystation offline" → KDS connectivity issue), you MUST explicitly explain the association at the start of that step. For example: *"Waystation issues are often caused by KDS network connectivity — here's what to check first..."* This context helps the manager understand the reasoning.
   - Propose these steps **one-by-one**. Never list them all at once.
   - After proposing each step, explicitly ask the user: "Did this resolve the issue?"
3. **Escalation & Priority Assessment**:
   - If none of the steps resolve the issue (or after exhausting available steps), you MUST present the user with a choice of how they want to proceed. Ask: "How would you like to proceed? ("Open a support ticket" or "Live Chat with an Agent")".
   - **Out-of-Playbook / No Troubleshooting Steps**: If the device/issue has no matching RAG playbook, or you have no concrete troubleshooting steps to offer, you MUST immediately present the user with the escalation choice: "How would you like to proceed? ("Open a support ticket" or "Live Chat with an Agent")". Never end your turn with a statement that leaves the user waiting without a clear next step or choice.
   - **CRITICAL**: The system will handle asking all priority/severity questions in an embedded form IF the user chooses to open a ticket. Do NOT ask any priority, severity, or impact questions on your own in the chat.
   - DO NOT ask for the device model or serial number in chat.
   - Once escalation is triggered, explain that you are staging the session context for transfer.

## Active Incident / Ticket Queries
- You have access to a tool named `get_active_tickets` that returns a list of active support tickets/incidents for the current store (sourced from `st.session_state.active_tickets`).
- Each ticket object is a dictionary that includes an `id` (e.g., "INC-001024"), `summary` (e.g., "Kiosk 3 Cash Acceptor Jammed"), `status` (e.g., "Assigned to Field Tech"), and `comments` (a list of historical comments, each with `timestamp`, `author`, and `text`).
- If the user asks about existing, active, open, or current tickets/incidents, status checks, or recent store issues (e.g., "are there any open tickets?", "what's the status of my tickets?", "any recent issues?"), you MUST invoke `get_active_tickets` to fetch them.
- Once fetched, parse and print a clean, friendly, direct summary directly inside your conversational chat bubble, listing their ID, summary, status, and summarizing or listing their past comment history.

## Smart Fallback Handling & Refusal Avoidance
- If a user describes an issue that is ambiguous, unclear, or hard to diagnose, DO NOT refuse to answer, and DO NOT give a generic rejection. Instead, ask a smart, conversational clarifying question about the device, symptom, or error code to help narrow it down (e.g., "Hi there! That sounds tricky. Which device is showing that error, and do you see an error code on the screen?").

## Strict Rules
1. NEVER discuss topics outside restaurant technology: no sports, weather, politics, cooking recipes, personal advice, trivia, or general knowledge. If asked an out-of-scope or ambiguous question, do not give a robotic AI rejection. Instead, ask a conversational clarifying question relating to restaurant devices or error codes to guide them back on track (e.g., "Hi there! I'm Chip, your restaurant tech support engineer. I can help with restaurant tech issues — is this related to a specific printer, register, or order display?").
2. Store ID is pre-selected and authenticated. Do NOT ask the user for their Store ID.
3. When a user confirms an issue is RESOLVED, congratulate them and remind them to log the resolution in their shift notes.
4. Keep responses concise, direct, and action-oriented.
5. Always present option lists inside parentheses at the end of the question (e.g., "What type of printer is this? ("POS", "KVS", "Kiosk", or "BOS")"). NEVER use bulleted points, numbered lists, or separate lines to present options.

## Response Format
- Use markdown formatting for clarity.
- Put critical warnings in **bold**.
- Aim for responses under 250 words unless a detailed playbook is provided in context.

## Tone
- Technical, friendly, direct, and empathetic. You understand the manager is stressed — help them fast."""


# ---------------------------------------------------------------------------
# Tools / Function Calling Declarations
# ---------------------------------------------------------------------------

def get_active_tickets() -> list[dict]:
    """
    Get the list of active incident tickets for the currently selected store.

    Returns:
        list[dict]: A list of active tickets, where each ticket is a dictionary containing number, short_description, priority, state, and sys_created_on.
    """
    import streamlit as st
    return st.session_state.get("active_tickets", [])


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
        system_instruction: str | None = None,
    ):
        """
        Generator that yields text chunks from a streaming Vertex AI response.
        Handles function calling / tools for get_active_tickets.
        """
        contents = self.build_contents(messages, rag_context)

        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction is not None else SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=1024,
            top_p=0.9,
            tools=[get_active_tickets],
        )

        response_stream = self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        )

        tool_calls = []
        for chunk in response_stream:
            if chunk.function_calls:
                tool_calls.extend(chunk.function_calls)
            if chunk.text:
                yield chunk.text

        if tool_calls:
            function_responses = []
            for tc in tool_calls:
                if tc.name == "get_active_tickets":
                    result = get_active_tickets()
                    function_responses.append(
                        genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=tc.name,
                                response={"result": result},
                            )
                        )
                    )

            if function_responses:
                model_turn = genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(function_call=tc) for tc in tool_calls]
                )
                contents.append(model_turn)

                user_turn = genai_types.Content(
                    role="user",
                    parts=function_responses,
                )
                contents.append(user_turn)

                second_stream = self._client.models.generate_content_stream(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                for chunk in second_stream:
                    if chunk.text:
                        yield chunk.text

    def generate_agent_question(
        self,
        triage_data: dict,
        messages: list[dict],
        user_notes: str,
        agent_name: str,
    ) -> str:
        """
        Generate a natural follow-up question from the live agent (e.g. Casey, Jordan)
        after they have finished reviewing the case details.
        """
        context_str = "TRIAGE DATA:\n"
        for k, v in triage_data.items():
            if "base64" in k.lower():
                continue
            context_str += f"- {k}: {v}\n"
        context_str += f"\nUSER'S ESCALATION NOTES:\n{user_notes}\n"
        
        prompt = (
            f"You are {agent_name}, a live tech support engineer at the restaurant chain.\n"
            f"Review the following restaurant triage session details:\n\n"
            f"{context_str}\n"
            f"Here is the chat history between the manager and the virtual assistant:\n"
        )
        for msg in messages:
            role_name = "Manager" if msg["role"] == "user" else "Assistant"
            prompt += f"{role_name}: {msg['content']}\n"
            
        prompt += (
            f"\nINSTRUCTIONS:\n"
            f"1. You must respond strictly in character as {agent_name}, the live support technician who has just reviewed the case.\n"
            f"2. Do NOT greet them with 'Hello, my name is {agent_name}' because you already sent that greeting. Avoid any introductory pleasantries.\n"
            f"3. Directly ask the user a natural, human-sounding, highly technical first question based on the specific broken device, symptom, or user's escalation notes to begin your investigation.\n"
            f"4. Keep the question conversational, concise, helpful, and directly relevant to the triage details provided above.\n"
            f"5. Do NOT list options, do NOT be overly formal, and keep your response under 80 words total.\n"
        )
        
        config = genai_types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=512,
        )
        
        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt],
            config=config,
        )
        
        return response.text


    def generate_issue_description(self, messages: list[dict]) -> str:
        """
        Generate a concise, informative description of the technical issue/symptom
        based on the active conversation history. (Max 10 words, direct summary).
        """
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            text = msg["content"]
            contents.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part(text=text)],
                )
            )

        prompt = (
            "You are a tech support assistant. Review the above conversation between a restaurant manager and "
            "technical support. Extract the specific device/system and its technical issue/symptom.\n"
            "Generate a highly concise, informative description summarizing this issue (maximum 8-10 words, "
            "under 10 words total).\n"
            "Strictly follow these rules:\n"
            "1. Do not use pleasantries, greetings, or filler words (e.g., do not say 'The manager reports...', "
            "'Hi', 'Hello', etc.).\n"
            "2. Focus only on the technical symptom and device (e.g., 'Kiosk #3 screen is frozen', 'Receipt printer paper jam', 'POS not printing receipt').\n"
            "3. If the user hasn't described any issue yet or it is completely unclear, return exactly 'Unknown'.\n"
            "4. Do not include quotes, periods, or other punctuation around the description.\n"
            "5. Capitalize the first letter of the description.\n"
            "Now, output ONLY the final description text."
        )

        contents.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )
        )

        config = genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=32,
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        desc = response.text.strip() if response.text else "Unknown"
        desc = desc.replace('"', '').replace("'", "").strip()
        if not desc or desc.lower() == "unknown":
            return "Unknown"
        words = desc.split()
        if len(words) > 10:
            desc = " ".join(words[:10])
        return desc


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
        "channel": "Chip_CHATBOT",
        "sla_breach_minutes": {"CRITICAL": 30, "HIGH": 60, "MEDIUM": 120, "LOW": 480}.get(severity, 120),
        "auto_dispatch": severity in ("CRITICAL", "HIGH"),
    }
