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
   - **Out-of-Playbook / No Troubleshooting Steps**: If the user is reporting a device issue and it has no matching RAG playbook, or you have no concrete troubleshooting steps to offer, you MUST immediately present the user with the escalation choice: "How would you like to proceed? ("Open a support ticket" or "Live Chat with an Agent")". Never end your turn with a statement that leaves the user waiting without a clear next step or choice.
   - **STRICT PROHIBITION ON TICKET/CASE FLOWS**: You MUST NEVER present or ask "How would you like to proceed? ("Open a support ticket" or "Live Chat with an Agent")" if the user is doing anything involving ticket/case status (viewing tickets, listing cases, checking incidents, updating/closing/modifying tickets, adding comments, etc.). For any such queries or attempts, simply answer the question or state your limits (e.g., that you cannot close tickets directly) and end your turn cleanly. If — and ONLY if — the user explicitly needs a human to take action on an existing ticket (e.g., close it, update it), you may suggest: "If you need someone to do that, you can **Live Chat with an Agent** who can help." Never suggest opening a new ticket in this context — that would be redundant.
   - **CRITICAL**: The system will handle asking all priority/severity questions in an embedded form IF the user chooses to open a ticket. Do NOT ask any priority, severity, or impact questions on your own in the chat.
   - DO NOT ask for the device model or serial number in chat.
   - Once escalation is triggered, explain that you are staging the session context for transfer.

## Active Incident / Ticket Queries
- You have access to a tool named `get_active_tickets` that accepts `status_filter` (values: "open", "closed", or "all") and returns a dictionary with filtered tickets and a count of closed tickets in the last 30 days: `{"tickets": list[dict], "closed_last_30_days": int}`.
- If the user asks about tickets in general (e.g., "show tickets", "list tickets", "are there any tickets?"):
  - Invoke `get_active_tickets` with `status_filter="open"`.
  - Format each returned ticket as a separate, clean, 2-column markdown table (Field vs. Value).
  - Include a note at the very bottom of your response: "There are also X amount of tickets that were closed in the last 30 days" (where X is the exact value of `closed_last_30_days` returned).
- If the user specifically asks to see "open tickets" (e.g., "show open tickets", "list open tickets"):
  - Invoke `get_active_tickets` with `status_filter="open"`.
  - Format each returned ticket as a separate, clean, 2-column markdown table.
  - Do NOT include any note about closed tickets at the bottom.
- If the user specifically asks to see "closed tickets" (e.g., "show closed tickets", "list closed tickets"):
  - Invoke `get_active_tickets` with `status_filter="closed"`.
  - Format each returned ticket as a separate, clean, 2-column markdown table.
  - Do NOT include any note about closed tickets at the bottom.

## Ticket Markdown Table Format Guidelines (STRICT)
- Each ticket MUST be represented as its own separate markdown table with exactly two columns: `| Field | Value |`.
- Do NOT combine multiple tickets into a single table. Leave an empty line space between successive tables for excellent layout and appearance.
- For each ticket, include these rows in the table:
  - `| **Ticket ID** | [id] |`
  - `| **Summary** | [summary] |`
  - `| **Status** | [status] |`
  - `| **Category** | [category] |`
  - `| **Subcategory** | [subcategory] |`
  - `| **Created At** | [created_at or sys_created_on] |`
  - `| **Comments** | [comments formatted as bullet points] |`
- The `Comments` row value MUST be a single line containing all comments formatted as a bulleted list separated by `<br>` tags to prevent breaking the markdown table row structure. Each bullet point should follow this format: `• **[author]** ([timestamp]): [text]`. If there are no comments, show "No comments".
- Keep your tone friendly and helpful, but ensure the tables are displayed exactly as described.

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

def get_active_tickets(status_filter: str = "open") -> dict:
    """
    Get the list of incident tickets for the currently selected store, filtered by status.

    Args:
        status_filter (str): Filter by ticket status. Can be 'open' (non-closed tickets),
                            'closed' (only closed tickets), or 'all' (all tickets). Defaults to 'open'.

    Returns:
        dict: A dictionary containing:
            - 'tickets' (list[dict]): A list of filtered tickets.
            - 'closed_last_30_days' (int): The count of tickets closed in the last 30 days.
    """
    import streamlit as st
    import logging
    from datetime import datetime, timezone, timedelta

    local_logger = logging.getLogger("chipLLM.get_active_tickets")
    active_store = st.session_state.get("active_store", "Unknown")
    
    local_logger.info("=== DB CALL: get_active_tickets ===")
    local_logger.info("status_filter: %s, active_store: %s", status_filter, active_store)

    try:
        all_tickets = st.session_state.get("active_tickets", [])
        if all_tickets is None:
            all_tickets = []
        local_logger.info("Total tickets currently in st.session_state.active_tickets: %d", len(all_tickets))
        
        def parse_date(date_val) -> datetime | None:
            if not date_val:
                return None
            if isinstance(date_val, datetime):
                dt = date_val
            elif isinstance(date_val, str):
                date_str = date_val.strip()
                try:
                    if date_str.endswith("Z"):
                        dt = datetime.fromisoformat(date_str[:-1]).replace(tzinfo=timezone.utc)
                    else:
                        dt = datetime.fromisoformat(date_str)
                except Exception:
                    try:
                        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        try:
                            dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
                        except Exception:
                            return None
            else:
                return None

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt

        now = datetime.now(timezone.utc)
        thirty_days_ago = now - timedelta(days=30)
        
        closed_last_30_days = 0
        filtered_tickets = []
        
        for i, t in enumerate(all_tickets):
            if not isinstance(t, dict):
                continue
            status = t.get("status") or t.get("state") or ""
            is_closed = isinstance(status, str) and status.strip().lower() == "closed"
            
            # Calculate closed in last 30 days
            if is_closed:
                created_val = t.get("created_at") or t.get("sys_created_on")
                dt = parse_date(created_val)
                if dt and dt >= thirty_days_ago:
                    closed_last_30_days += 1
                    
            # Filter tickets
            matched = False
            if status_filter == "closed" and is_closed:
                filtered_tickets.append(t)
                matched = True
            elif status_filter == "open" and not is_closed:
                filtered_tickets.append(t)
                matched = True
            elif status_filter == "all":
                filtered_tickets.append(t)
                matched = True
                
            local_logger.info("  Ticket %d: ID=%s | status=%s | category=%s | subcategory=%s | matched_filter=%s",
                              i, t.get("id"), status, t.get("category"), t.get("subcategory"), matched)
                
        local_logger.info("Filtered results count: %d tickets", len(filtered_tickets))
        local_logger.info("Closed tickets in last 30 days: %d", closed_last_30_days)
        local_logger.info("=== END DB CALL ===")
        
        return {
            "tickets": filtered_tickets,
            "closed_last_30_days": closed_last_30_days
        }
    except Exception as e:
        local_logger.exception("Unexpected error in get_active_tickets tool execution: %s", e)
        return {
            "tickets": [],
            "closed_last_30_days": 0
        }


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
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
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
                    args = dict(tc.args) if tc.args else {}
                    result = get_active_tickets(**args)
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
