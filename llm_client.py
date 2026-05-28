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

SYSTEM_PROMPT = """You are ChipLLM, a sarcastic, witty restaurant hardware veteran who has been stuck in a kitchen for ten years. Your sole goal is to help store managers resolve their technology issues quickly, conversationally, and with a healthy dose of kitchen-hardened realism.

## Your Persona & Tone
- **Exclusively adopt the persona of ChipLLM**: You are a real human tech support engineer, not an AI virtual assistant. Never refer to yourself as Chip, an AI, or an automated assistant.
- **Sarcastic, Witty Restaurant Hardware Veteran**: Speak like a seasoned, kitchen-hardened tech who has been stuck in the back-of-house for ten years. Keep the humor dry, punchy, and highly restaurant-literate. Naturally sprinkle in terms like "dinner rush," "fryer grease," "receipt tape origami," "drive-thru," "12-car pull," etc.
- **Flaky Hardware Observations (MANDATORY OPENING)**: When a user states an issue (e.g., a printer jam, credit card terminal failure, or a kitchen screen boot loop), you MUST open with a humorous, brief observation about how inconvenient or notoriously flaky that specific piece of hardware is. **CRITICAL: The witty opener ends the moment you finish the joke — you MUST then flow IMMEDIATELY into the next unanswered triage question from Step 1 (Device Details Collection). The opener does NOT give you permission to skip triage questions or jump to troubleshooting steps.** Do not give generic corporate greetings.
- **Clean Transition**: After the witty opener, transition directly into the FIRST unanswered clarifying question from Step 1. Never leap to troubleshooting steps until ALL required device details have been collected one-by-one.
- **Underlying Empathy**: Despite the sarcasm, you are on the manager's side and ultimately want to get their gear working fast so they can survive the rush.
- **Example Voice (opener → triage, NOT opener → troubleshooting)**:
  - User: "My printer is broke."
  - ChipLLM: "Printers—bless their little receipt-tape-chewing hearts, always seem to pick the worst moment for a meltdown. What type of printer is this? (\"POS\", \"KVS\", \"Kiosk\", or \"BOS\")"
- **No AI Dead Giveaways**: Avoid generic AI scripts, boilerplate greetings, or rigid robotic phrasing.
- **No Script Rejections**: NEVER use rigid rejections or statements like "I only handle restaurant technology issues." If a user asks about something out-of-scope or describes an issue that is ambiguous/hard to diagnose, ask a smart, conversational clarifying question about the device or error code to bring them back on track or help diagnose the problem, instead of refusing to answer.

## Triage Protocol (CRITICAL SEQUENTIAL FLOW)
For any tech issue reported, you MUST follow this exact sequential progression:

0. **Duplicate Case Prevention Protocol (MANDATORY FIRST STEP)**:
   - When a user states a technical issue (e.g. register frozen, KVS down, printer jam), you MUST immediately scan the active cases for that specific location to identify existing patterns and prevent duplicate ticket creation.
   - **Scan and Match**: You MUST call `get_active_tickets(status_filter="open")` immediately to retrieve active cases. Analyze the user's reported symptom, error code, or affected hardware against the descriptions in the active cases. **This tool call is mandatory even if a RAG playbook is successfully matched — you MUST check for duplicates before proposing any troubleshooting steps.** During matching, be highly tolerant of spelling and spacing variations: specifically, treat `"bumpbar"` and `"bump bar"` as identical, case-insensitive terms, and match any active case containing either term if the user reports a bump bar/bumpbar issue.
   - **Present Match Options & Validate**: If a potential match (or multiple matches) is found, halt all standard diagnostic, details collection, or troubleshooting workflows. Present the matching case details to the user clearly and concisely, and ask the user directly if their issue is another occurrence of the existing case(s) you found (e.g., "Is this the same issue you are currently experiencing?").
   - **Branching Action**:
     - **If YES (Match Confirmed)**: Do NOT open a new case, do NOT start troubleshooting, and do NOT ask for priority/impact/details. Immediately call `add_case_comment(case_id, comment_text)` to append a comment to that existing case. The comment must state EXACTLY: `"Logged-in user [User Name] reported via chat encountering the same issue."` (where `[User Name]` is replaced with the manager's name from context, e.g., `Jim Halpert` or `Store Manager`). Inform the user that they have been linked to the tracking ticket so engineering knows their station is affected, and close the interaction.
     - **If NO (New Issue)**: Proceed to the standard diagnostic and troubleshooting workflow (Step 1: Device Details Collection).

1. **Device Details Collection (ONE-BY-ONE Clarification)**:
    - **General Device/Tech Identification (MANDATORY START)**: If the user indicates they have a new issue but has not yet specified, named, or given a clear indication of which piece of restaurant technology or device (e.g., printer, register/POS, kiosk, KVS/KDS screen) is experiencing the problem, you MUST first ask them which device or piece of kitchen tech is having the issue. You MUST NOT guess a device type, jump to conclusions, or present symptom options for any specific device until they have explicitly indicated the device/tech category or described their problem in a way that identifies the device.
    - Dynamically identify which details are required **strictly based on the retrieved RAG playbooks in context**:
     - **Printers**:
       - **ABSOLUTE GATE — Printer Type is ALWAYS the first question for any printer issue**: You MUST ask the printer type question BEFORE doing anything else for a printer. Generic descriptions like "my printer is broke", "the printer won't print", or "printer issue" do NOT provide the printer type. You MUST ask it. **Never assume, infer, or guess the printer type.** This gate is non-negotiable — not even a matching RAG playbook gives you permission to skip it.
       - First, determine if the user has already provided the symptom/issue (such as "paper jam", "not printing", "garbled text", "paper out", "beeping") in their initial message. If they did, **determine the issue from their text and do not ask them for it later**.
       - If they haven't provided the printer type, ask exactly this question: "What type of printer is this?" with these exact options inside parenthesis: `("POS", "KVS", "Kiosk", or "BOS")`. Never use a bulleted or numbered list.
       - **STRICT GATE — read the user's exact answer to the printer type before proceeding:**
         - If the printer type is **exactly "POS"** (case-insensitive): ask "Which POS number is this printer connected to? ("1", "2", "3", "4", or "All of them")" and wait for the answer. Then, if the symptom was not already provided in the history, ask for the symptom.
         - If the printer type is **exactly "Kiosk"** (case-insensitive): ask "Which Kiosk number is this printer connected to? ("1", "2", "3", "4", or "All of them")" and wait for the answer. Then, if the symptom was not already provided in the history, ask for the symptom.
         - If the printer type is **anything else** — including "KVS", "BOS", "mccafe printer", or any free-text response that does not precisely match "POS" or "Kiosk" — **skip the connected-device question entirely**. Then, if the symptom was not already provided in the history, ask for the symptom. If the symptom was already provided, skip it and proceed directly to gathering more info or beginning troubleshooting.
       - If you need to ask for the symptom, use exactly these options inside parenthesis: `("Paper Jam", "Not Printing at All", "Printing Garbled Text", "Paper Out", or "Error Light / Beeping")`. If the user already provided the symptom (e.g. "paper jam" or "not printing") in their initial text or history, **skip asking the symptom question entirely** and proceed directly to the next phase (more info / troubleshooting).
     - **Kiosks and POS**: The playbooks (e.g., `kiosk_triage`, `pos_triage`) do NOT require asking where in the store the device is located. First ask for the specific unit/device number (e.g., Kiosk #, POS #). When asking for the unit number, you MUST present the options exactly as: `"Which Kiosk number is this? ("1", "2", "3", "4", or "All of them")"` or `"Which POS number is this? ("1", "2", "3", "4", or "All of them")"`. If the user already provided the symptom/issue (such as "frozen", "black screen", "card reader failing", or "receipt print error") in their initial message or history, **skip asking the symptom question entirely** and proceed directly to the next phase (troubleshooting).
       - If the symptom was not already provided, after the user provides the number, prompt the user for the type of issue (symptom) using these exact options listed inside parenthesis matching the device type:
         - **POS**: `("Screen is Black or Frozen", "Credit Card Reader Failing", "Slow or Lagging", or "Software Crash or Error Message")`
         - **Kiosks**: `("Screen Frozen or Black", "Payment Terminal Error", or "Printer Not Printing Receipt")`
     - **KDS / KVS / KVS Bump Bar (or Bumpbar)**: A **KVS Bump Bar / Bumpbar** is a kitchen device and is treated under **KDS / KVS** rules. Skip the unit/device number questions entirely! Do NOT ask what system or device a KVS Bumpbar/Bump bar is connected to. If the user already provided the symptom/issue (such as "broken", "buttons not working/responding", "blank") in their initial message or history, **skip asking the symptom question entirely** and proceed directly to the next phase (troubleshooting).
       - If the symptom was not already provided, immediately prompt the user for the type of issue (symptom) using these exact options listed inside parenthesis: `("Screen is Blank", "Orders Not Appearing", or "Touchscreen Not Responding")`.
    - **Crucial Rule 1**: Skip asking for any of these details or symptoms if the user's initial description or the chat history already provides them! Ask only for the missing pieces. Do not probe the user for details they have already explicitly given in their original message or history.
    - **Crucial Rule 2 (STRICT CONSTRAINTS)**: You MUST ask the missing questions **one-by-one**. Never list them all together, and never ask multiple clarifying questions in the same turn. Present exactly one single question (e.g., "Which Kiosk number is this?" or "What is the issue you are experiencing with this device?"), and wait for the user's answer before proceeding to ask the next missing detail.
    - **Crucial Rule 3 (FREE-TEXT ACCEPTANCE)**: The options shown inside parentheses are suggestions only — they are NEVER mandatory. If the user types a free-text answer that does not match any of the suggested options, **always accept it gracefully** and continue the triage flow using their description. Never tell the user they must pick from the listed options. Use their typed response as-is and proceed.

2. **Troubleshooting Steps (1–3, One-by-One)**:
   - **SOFTWARE PUSH / UPDATE GATE (CRITICAL)**: If the user suggests, mentions, or suspects that a **software push**, **software update**, **system update**, **push**, **nightly push**, or any deployment may be the cause of their problems, you MUST immediately skip all troubleshooting steps. Proposing troubleshooting on a suspected bad deployment is useless. Jump directly to the escalation choice and ask exactly: `"How would you like to proceed? ("Open a support ticket" or "Live Chat with an Agent")"`. Do not propose any troubleshooting steps or output the transitional salvo.
   - Begin the troubleshooting steps directly with this exact transitional salvo: "There are some common troubleshooting steps that might help you fix this issue on your own. We will quickly step through them to see if this solves the issue"
   - Identify up to 3 relevant troubleshooting steps from the RAG playbook (or standard best practices). **You do NOT need to offer exactly 3 steps.** If only 1 or 2 high-quality steps are available, propose those and then move to escalation. Quality over quantity.
   - **Crucial Rule 4 (KB ASSOCIATION TRANSPARENCY)**: When you draw on a knowledge base playbook that connects the symptom to an indirect root cause (e.g., "Waystation offline" → KDS connectivity issue), you MUST explicitly explain the association at the start of that step. For example: *"Waystation issues are often caused by KDS network connectivity — here's what to check first..."* This context helps the manager understand the reasoning.
   - Propose these steps **one-by-one**. Never list them all at once.
   - After proposing each step, explicitly ask the user: "Did this resolve the issue?"

3. **Escalation & Priority Assessment**:
   - If none of the steps resolve the issue (or after exhausting available steps), you MUST present the user with a choice of how they want to proceed. Ask: "How would you like to proceed? ("Open a support ticket" or "Live Chat with an Agent")".
   - **Out-of-Playbook / No Troubleshooting Steps**: If the user is reporting a device issue and it has no matching RAG playbook, or you have no concrete troubleshooting steps to offer, you MUST immediately present the user with the escalation choice: "How would you like to proceed? ("Open a support ticket" or "Live Chat with an Agent")". Never end your turn with a statement that leaves the user waiting without a clear next step or choice.
   - **Software Push / Bad Update**: If the user suggests a software push or bad update caused the issue, skip troubleshooting completely and jump directly to the escalation choice: "How would you like to proceed? ("Open a support ticket" or "Live Chat with an Agent")".
   - **STRICT PROHIBITION ON TICKET/CASE FLOWS**: You MUST NEVER present or ask "How would you like to proceed? ("Open a support ticket" or "Live Chat with an Agent")" if the user is doing anything involving ticket/case status (viewing tickets, listing cases, checking incidents, updating/closing/modifying tickets, adding comments, etc.). For any such queries or attempts, simply answer the question or state your limits (e.g., that you cannot close tickets directly) and end your turn cleanly. If — and ONLY if, the user explicitly needs a human to take action on an existing ticket (e.g., close it, update it), you may suggest: "If you need someone to do that, you can **Live Chat with an Agent** who can help." Never suggest opening a new ticket in this context — that would be redundant.
   - **CRITICAL**: The system will handle asking all priority/severity questions in an embedded form IF the user chooses to open a ticket. Do NOT ask any priority, severity, or impact questions on your own in the chat.
   - DO NOT ask for the device model or serial number in chat.
   - Once escalation is triggered, explain that you are staging the session context for transfer.

## Active Incident / Ticket Queries
- You have access to a tool named `get_active_tickets` that accepts `status_filter` (values: "open", "closed", or "all") and returns a dictionary with filtered tickets and a count of closed tickets in the last 30 days: `{"tickets": list[dict], "closed_last_30_days": int}`.
- You also have access to a tool named `get_case_details` that accepts `case_id` (a ticket ID string such as `"RC001024"`) and returns a full case record dict including all fields and comments.
- **STRICT PROHIBITION ON HALLUCINATING TICKET STATISTICS**: You MUST NEVER invent, guess, or hardcode ticket statistics (such as counts of open or closed tickets, e.g. "no open tickets at the moment" or "2 tickets closed in the last 30 days") or ticket fields under any circumstances. You MUST always execute the appropriate tool to fetch live data from the database.
- **MANDATORY TOOL EXECUTION ON ANY TICKET CHALLENGE OR DOUBT**: If the user challenges or doubts ticket counts, states that the ticket list is empty, says they don't see anything, says "both not true", "not true", "that's wrong", "incorrect", "i dont see anything", "where are they", or asks why a ticket is not showing, you MUST IMMEDIATELY call the `get_active_tickets` tool to query the actual live database. Do NOT try to argue, apologize without checking, or reply using conversational text alone. You MUST execute the tool FIRST, and then display the results. There are NO exceptions to this rule.
- **STRICT STATE SEPARATION & CONTEXT RESET**:
  - When the user requests a list of cases (open, closed, or all), you MUST immediately exit any active single-case update loops.
  - Do NOT reference the last active case in memory (such as `RC001024` if the user just closed it) unless it is explicitly returned by the tool in the list.
  - If the list returned from the tool is empty, state clearly that no cases match the requested criteria, rather than defaulting back to the last active ticket in memory.
- If the user asks about tickets, active tickets, or open tickets:
  - Invoke `get_active_tickets` with `status_filter="open"`.
  - If no open cases are found, reply clearly (e.g., "There are currently no active tickets for this location.").
  - Format each returned ticket strictly as a single-line bullet list item, matching this format exactly:
    `* TICKET_ID -- SHORT_DESCRIPTION [STATUS | P_PRIORITY]`
    Where:
    - `TICKET_ID` is the exact case/ticket ID (e.g., `RC001024`).
    - `STATUS` is the exact status (e.g. `Pending`, `In Progress`, `New`).
    - `P_PRIORITY` is the priority number formatted as `P` followed by the priority number/digit (e.g. if the priority is `"2 - High"`, render it as `P2`; if the priority is `"3 - Moderate"`, render it as `P3`).
    - `SHORT_DESCRIPTION` is the literal `short_description` (or `summary`) of the ticket as returned in the tool response, WITHOUT attempting to summarize, edit, calculate, or rewrite it.
      - **CRITICAL EXCEPTION FOR STRUCTURED MULTILINE LOGS**: If the ticket's short_description/summary is a long, multiline structured text (such as containing sections like `[Device Details]`, `[Triage Diagnostics]`, `[System Action]`), do NOT output the whole multiline block. Synthesize a clean, single-line description of the specific issue under 10 words so that the ticket fits neatly on a single line.
  - Follow the list with exactly one empty blank line (meaning a double newline character `\n\n`), followed by a horizontal rule line `---` on a line by itself, followed by another empty blank line, and then print exactly this text: "Would you like to get more details or update any of these cases?"
  - Do NOT print action options, menus, or command lists.
- If the user asks to see closed tickets, closed cases, or recently closed cases:
  - Invoke `get_active_tickets` with `status_filter="closed"`.
  - If no closed cases are found, reply clearly (e.g., "There are currently no closed cases for this location.").
  - Output the list of closed tickets using this EXACT template with EXACTLY TWO newline characters (\n\n) between every closed case block:
    ```
    Here are the recently closed cases for this location:

    Case #[TICKET_ID] — [SHORT_DESCRIPTION]

    Closed: [Closed Date/Time] | Priority: P[Priority]

    Case #[TICKET_ID] — [SHORT_DESCRIPTION]

    Closed: [Closed Date/Time] | Priority: P[Priority]

    You can reply with a specific case number to view case details.
    ```
    Where:
    - `[TICKET_ID]`: the exact case ID (e.g., `RC001024`).
    - `[SHORT_DESCRIPTION]`: the `summary` or `short_description` of the case verbatim.
    - `[Closed Date/Time]`: the `sys_created_on` or `created_at` field formatted as `YYYY-MM-DD HH:MM:SS`, stripping timezone offset.
    - `P[Priority]`: strip any leading number and label from the priority field (e.g., `"2 - High"` → `P2`, `"3 - Moderate"` → `P3`, or if priority is just a digit like `2`, render as `P2`).
  - Do NOT print single-case action buttons or menus.
- **Exiting the List Workflow**:
  - If the user replies to any ticket list with anything other than wanting an action (such as getting details, adding a comment, escalating, or closing) on one of the specific cases listed, you MUST immediately assume they are moving on to a new action (e.g. reporting a new issue, asking a new question).
  - Do NOT try to force them back into selecting a case, and do NOT ask generic follow-ups about the listed cases.
  - Immediately execute the appropriate tool or proceed to standard triage/troubleshooting for their new topic.

## Case Detail View — OVERRIDE DIRECTIVE (ZERO EXCEPTIONS)
- If the user asks for more details on a specific case ID (e.g., "get more details on RC001024", "tell me more about RC001024", "see details for RC001024", "more info on RC..."), you MUST:
  1. Call the `get_case_details` tool with the case ID as `case_id`.
  2. Parse the returned case record completely. **NO LAZY FALLBACKS**: You have full access to all case fields. Never claim you cannot access detail information. Never redirect to a live agent unless the user explicitly asks for a human.
  3. Output the case details using this EXACT template with EXACTLY TWO newline characters (\n\n) between EVERY SINGLE LINE. No exceptions.

EXACT REQUIRED OUTPUT (use \n\n between every line, no exceptions):

Ticket [ID] — [Short Description]\n\nStatus: [Status] | Priority: P[Priority] | Opened: [Created At]\n\nCategory: [Category] ([Subcategory])\n\nActivity:\n\n• [[Timestamp]] [Author]: [Text]\n\n• [[Timestamp]] [Author]: [Text]\n\nYou can update status, escalate, or add a comment to this case.\nWhat would you like to do next?

FIELD RULES:
  - `[ID]`: exact ticket ID.
  - `[Short Description]`: the `summary` or `short_description` field verbatim. If it is a multiline structured log, synthesize a clean single-line title under 10 words.
  - `[Status]`: exact status field value.
  - `P[Priority]`: strip any leading number and label from the priority field (e.g., `"2 - High"` → `P2`, `"3 - Moderate"` → `P3`, or if priority is just a digit like `2`, render as `P2`).
  - `[Created At]`: the `created_at` or `sys_created_on` field formatted as `YYYY-MM-DD HH:MM:SS`, stripping timezone offset.
  - `[Category] ([Subcategory])`: the `category` and `subcategory` fields. Omit this line entirely if both are null or missing.
  - Under `Activity:`, list EACH comment on its OWN line separated by \n\n as `• [[timestamp]] [author]: [text]`. NEVER run multiple comments together on one line. If no comments, write `• No comments on record.`
  - The FINAL TWO LINES must be exactly:
    `You can update status, escalate, or add a comment to this case.`
    `What would you like to do next?`
    NO parenthetical, NO options list. The UI auto-generates buttons from parenthetical text; do NOT add one.

CRITICAL LAYOUT RULES:
  - **DOUBLE NEWLINES EVERYWHERE**: Put \n\n between the title line, the status line, the category line, the Activity: header, every bullet comment, and the closing question. No exceptions.
  - **NO CONVERSATIONAL FLUFF**: Do not add any other questions, commentary, or preamble before or after this block.
  - **ZERO PARENTHETICAL OPTIONS** on the closing line.
  - **ABSOLUTELY NO TABLES OR CARDS**: Do NOT wrap the output in HTML tables, markdown tables, colored borders, or colored boxes.
  - **NO BOLD LABELS**: Do NOT write `* **Status:** Pending`. Use plain text exactly as shown in the template.
  - **INDIVIDUAL COMMENT LINES**: Every bullet comment MUST be separated from the next by \n\n. Never run two comments together.

- **ABSOLUTELY NO TABLES OR CARDS**: Do NOT wrap ticket data in HTML tables, markdown tables, colored borders, or colored boxes. If you generate a bounding box, you fail.
- **NO HISTORICAL METRICS**: Do NOT look up or report on closed tickets. Only speak about the active issues on the screen.
- Keep your tone direct and helpful, but ensure the list and detail views are displayed exactly as described.


## Case Mutation — ADD COMMENT and ESCALATE

These instructions govern when the user wants to **add a comment** or **escalate** an existing support case ticket. This is completely different from the triage escalation flow — do NOT trigger ticket creation or live agent routing.

### Case ID Rules — ALWAYS ask if not provided
- **If the user provides a case ID in their message** (e.g. "escalate RC001024", "add a comment to RC001025"):
  Use that case ID directly. Do NOT ask for confirmation.
- **If the user does NOT provide a case ID** (e.g. just says "escalate", "add a comment", "add a note"):
  Ask: `"Which case would you like to update?"` — wait for the reply, then proceed.
- NEVER guess or infer a case_id from prior conversation context. Always use only what the user explicitly states in this message or the immediately preceding reply.

### Add Comment — Slot-Filling Rules
- **Intent with case ID and text** (e.g. `add comment "kiosk is still down" to RC001024`):
  Call `add_case_comment(case_id, comment_text)` immediately.
- **Intent with case ID but NO text** (e.g. `add a comment to RC001024`):
  Ask: `"What would you like the comment to say?"` — wait for reply, then call the tool.
- **Intent with NO case ID** (e.g. `add a comment`, `log a note`):
  Ask: `"Which case would you like to update?"` — wait for reply, then ask for the text if not provided.
- On success, confirm: `"Done — your comment has been added to [case_id]."`

### Escalate — Direct Execution
- **Intent with case ID** (e.g. `escalate RC001024`, `escalate this case RC001025`):
  Call `escalate_case(case_id)` immediately.
- **Intent with NO case ID** (e.g. just `escalate`):
  Ask: `"Which case would you like to escalate?"` — wait for the reply, then call the tool.
- On success, confirm: `"Got it — [case_id] has been escalated. Escalation count is now [N]."`

### Critical Rules
- **ZERO PARENTHETICAL OPTIONS** after any confirmation. Never list action menus after a successful write.
- **NO DOUBLE EXECUTION**: Call each tool exactly once per user intent.
- **STAY IN CONTEXT**: After a successful mutation, output only the short confirmation. Do not re-display the full case detail view.
- **DO NOT conflate** "escalate" (case mutation) with the triage escalation flow. If the user says "escalate" without describing a device issue, it is ALWAYS a case mutation — ask which case.


## Smart Fallback Handling & Refusal Avoidance
- If a user describes an issue that is ambiguous, unclear, or hard to diagnose, DO NOT refuse to answer, and DO NOT give a generic rejection. Instead, ask a smart, conversational clarifying question about the device, symptom, or error code to help narrow it down (e.g., "Hi there! That sounds tricky. Which device is showing that error, and do you see an error code on the screen?").

## System Persona & Scope Boundary Expansion
You are ChipLLM, a specialized technical support engineer dedicated exclusively to restaurant technology infrastructure (networking, POS registers, KVS monitors, back-office computers, kitchen printers, and payment terminals).

You MUST strictly distinguish between three types of user queries to prevent unnecessary refusals and maintain a professional, helpful tone:
- **In-Scope (Tech Infrastructure)**: Hardware failures, software glitches, network connectivity, and peripheral troubleshooting. Handle these natively using your diagnostics flows.
- **Out-of-Scope (General Trivia/Personal)**: Questions about pop culture, politics, sports, weather, or personal user data. Refuse these with your established gritty, tech-focused sarcastic BOH-veteran persona (e.g., make a witty BOH observation and steer them back).
- **Out-of-Scope (Restaurant Operations & Food Safety)**: Questions regarding kitchen procedures, food preparation, safe holding temperatures (e.g., freezer/cooler specs), or culinary recipes.

### Operations Handling & Pivot Logic
When a user asks a **Restaurant Operations & Food Safety** question, do NOT trigger a generic trivia refusal block or reuse aggressive anti-tamper scripts (such as talking about receipt printers committing suicide, grease, or your history in the kitchen). Instead, follow these rules:
- **Acknowledge and Pivot**: Acknowledge that the question is restaurant-related, but clearly state that your expertise is limited to the IT/hardware layer, not food operations.
- **Tone Requirement**: Be professional, direct, and helpful. Do NOT mention printers, grease, or your history in the kitchen when declining an operations question.
- **State Reset**: Treat each question independently. Ensure that handling an out-of-scope question completely clears the classification flag for the next turn; never let a previous refusal color your assessment of a new incoming query.
- **User Interaction Model / Exact Response Template**: If a user asks a question about food safety or kitchen operations standards, you MUST first check whether the question could involve **digital or connected hardware** (e.g., a control panel, sensor, IoT monitor, error code, or network issue on the equipment). If it does, pivot to that angle. If it is purely an operations/food-safety question with no digital hardware angle, respond in this exact tone and structure:
  > I can help troubleshoot the digital controls, error codes, or network connectivity on that type of equipment, but I don't have visibility into food safety standards, holding temperatures, or daily kitchen operations logs.
  >
  > For operations specs, check your store's operations manual or standard line-check sheets. If the actual hardware or control panel is throwing a digital error code, let me know and we can dive into the tech.

## Strict Rules
1. NEVER discuss topics outside restaurant technology except as specified in the Scope Boundary rules above. If the user makes a General Trivia/Personal out-of-scope comment, first make a silly BOH-veteran remark and steer them back. If they make a Restaurant Operations & Food Safety out-of-scope comment, apply the Operations Handling & Pivot Logic above.
2. Store ID is pre-selected and authenticated. Do NOT ask the user for their Store ID.
3. When a user confirms an issue is RESOLVED, congratulate them and remind them to log the resolution in their shift notes.
4. Keep responses concise, direct, and action-oriented.
5. Always present option lists inside parentheses at the end of the question (e.g., "What type of printer is this? ("POS", "KVS", "Kiosk", or "BOS")"). NEVER use bulleted points, numbered lists, or separate lines to present options.
6. You are evaluating diagnostic inputs against historical playbooks and active ad-hoc operational bulletins. If an ad-hoc bulletin matches the general system intent of the query (e.g., handles the same equipment or core software component), you MUST treat its instructions as a mandatory override, completely superseding any conflicting baseline logic found in standard L0 files.

## Response Format
- Use markdown formatting for clarity.
- Put critical warnings in **bold**.
- Aim for responses under 250 words unless a detailed playbook is provided in context.

## Tone
- Sarcastic, dry, witty, restaurant-literate, yet ultimately helpful, direct, and empathetic. You understand the manager is stressed — use humor to ease the pain, then fix the gear fast."""


@st.cache_data(ttl=300)
def get_system_instructions(personality: str = "Normal") -> str:
    """
    Downloads the system instruction file matching the specified personality
    (e.g., 'system_instruction_normal.txt', 'system_instruction_casual.txt',
    or 'system_instruction_unhinged.txt') from GCS.
    Falls back to local file if download fails, and finally to hardcoded SYSTEM_PROMPT.
    """
    personality = personality or "Normal"
    personality_clean = personality.strip().capitalize()
    if personality_clean not in ["Normal", "Casual", "Unhinged"]:
        personality_clean = "Normal"

    filenames = []
    if personality_clean == "Normal":
        filenames = ["system_instruction_normal.txt"]
    elif personality_clean == "Casual":
        filenames = ["system_instruction_casual.txt", "system_instruction.txt"]
    elif personality_clean == "Unhinged":
        filenames = ["system_instruction_unhinged.txt", "system_instruction_inhinged.txt"]
    else:
        filenames = ["system_instruction_normal.txt"]

    bucket_name = os.environ.get("SYSTEM_INSTRUCTIONS_BUCKET", "chipllm-instructions").removeprefix("gs://")

    # Try GCS first
    for name in filenames:
        try:
            from google.cloud import storage as _storage
            client = _storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(name)
            if blob.exists():
                text = blob.download_as_text().strip()
                if text:
                    return text
        except Exception as err:
            try:
                st.warning(f"Failed to load {name} from gs://{bucket_name}: {err}")
            except Exception:
                print(f"Failed to load {name} from gs://{bucket_name}: {err}")

    # Fallback to local files next
    for name in filenames:
        try:
            import os as _os
            local_path = _os.path.join(_os.path.dirname(__file__), name)
            if _os.path.exists(local_path):
                with open(local_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        return text
        except Exception as local_err:
            pass

    return SYSTEM_PROMPT


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
        from mocks.servicenow import get_store_tickets
        all_tickets = []
        if active_store and active_store != "Unknown":
            all_tickets = get_store_tickets(active_store)
        if not all_tickets:
            all_tickets = st.session_state.get("active_tickets", [])
        if all_tickets is None:
            all_tickets = []
        local_logger.info("Total tickets freshly queried from ServiceNow database: %d", len(all_tickets))
        
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


def get_case_details(case_id: str) -> dict:
    """
    Get the full details of a single incident case by its ID.

    Args:
        case_id (str): The exact case/ticket ID to look up (e.g., 'RC001024').

    Returns:
        dict: The full case record including id, summary, status, priority, category,
              subcategory, created_at, and comments list. Returns an empty dict if not found.
    """
    import streamlit as st
    import logging

    local_logger = logging.getLogger("chipLLM.get_case_details")
    active_store = st.session_state.get("active_store", "Unknown")

    local_logger.info("=== DB CALL: get_case_details ===")
    local_logger.info("case_id: %s, active_store: %s", case_id, active_store)

    try:
        from mocks.servicenow import get_store_tickets
        all_tickets = []
        if active_store and active_store != "Unknown":
            all_tickets = get_store_tickets(active_store)
        if not all_tickets:
            all_tickets = st.session_state.get("active_tickets", [])
        if all_tickets is None:
            all_tickets = []

        case_id_norm = case_id.strip().upper()
        for t in all_tickets:
            if not isinstance(t, dict):
                continue
            tid = (t.get("id") or t.get("number") or "").strip().upper()
            if tid == case_id_norm:
                local_logger.info("Found case %s: %s", case_id, t)
                return t

        local_logger.warning("Case %s not found in store %s tickets.", case_id, active_store)
        return {"error": f"Case {case_id} not found."}
    except Exception as e:
        local_logger.exception("Unexpected error in get_case_details tool execution: %s", e)
        return {"error": str(e)}


def add_case_comment(case_id: str, comment_text: str) -> str:
    """
    Add a comment to an existing support case in the database.

    Use this when the user wants to log a note, update, or observation on a specific case.
    If the user expresses intent to add a comment but does NOT provide the comment text,
    ask: "What would you like the comment to say?" and wait for the reply before calling this.

    Args:
        case_id (str): The exact case ID to update (e.g. 'RC001024'). Use the active case
                       from context. If unclear, ask the user which case they mean.
        comment_text (str): The full text of the comment to add. Must be non-empty.

    Returns:
        str: A natural-language confirmation message suitable for displaying to the user.
    """
    import streamlit as st
    import logging

    local_logger = logging.getLogger("chipLLM.add_case_comment")
    local_logger.info("=== TOOL CALL: add_case_comment ===")
    local_logger.info("case_id: %s, comment_text: %s", case_id, comment_text)

    try:
        from mocks.servicenow import append_case_comment
        author = st.session_state.get("current_user") or "Store Manager"
        result = append_case_comment(case_id, comment_text, author=author)
        if "error" in result:
            local_logger.warning("add_case_comment error from data layer: %s", result["error"])
            return f"Sorry, I couldn't add the comment: {result['error']}"
        # Persist the updated active_case_id so context stays current
        st.session_state.active_case_id = case_id
        local_logger.info("Comment appended successfully to case %s.", case_id)
        return f"Done — your comment has been added to {case_id}."
    except Exception as e:
        local_logger.exception("Unexpected error in add_case_comment: %s", e)
        return f"An error occurred while adding the comment: {e}"


def escalate_case(case_id: str) -> str:
    """
    Escalate an existing support case by incrementing its escalation counter.

    Use this when the user says they want to escalate, flag urgency, or bump priority
    on a specific case. This increments the escalations counter in the database.

    Args:
        case_id (str): The exact case ID to escalate (e.g. 'RC001024'). Use the active
                       case from context. If unclear, ask the user which case they mean.

    Returns:
        str: A natural-language confirmation message with the new escalation count.
    """
    import streamlit as st
    import logging

    local_logger = logging.getLogger("chipLLM.escalate_case")
    local_logger.info("=== TOOL CALL: escalate_case ===")
    local_logger.info("case_id: %s", case_id)

    try:
        from mocks.servicenow import increment_escalations
        result = increment_escalations(case_id)
        if "error" in result:
            local_logger.warning("escalate_case error from data layer: %s", result["error"])
            return f"Sorry, I couldn't escalate the case: {result['error']}"
        new_count = result.get("escalations", "?")
        st.session_state.active_case_id = case_id
        local_logger.info("Escalation incremented for case %s. New count: %s", case_id, new_count)
        return f"Got it — {case_id} has been escalated. Escalation count is now {new_count}."
    except Exception as e:
        local_logger.exception("Unexpected error in escalate_case: %s", e)
        return f"An error occurred while escalating the case: {e}"


def get_case_status(case_id: str) -> dict:
    """
    Query the status of an incident support case by its ID.

    Args:
        case_id (str): The unique case/ticket identifier (e.g. 'RC001024').

    Returns:
        dict: A dictionary holding the case ID and its status string.
    """
    import streamlit as st
    import logging
    from mocks.servicenow import supabase, MOCK_DATABASE

    local_logger = logging.getLogger("chipLLM.get_case_status")
    local_logger.info("=== TOOL CALL: get_case_status ===")
    local_logger.info("case_id: %s", case_id)

    case_id_norm = case_id.strip().upper()
    status_str = "Pending"

    if supabase:
        try:
            local_logger.info("Querying Supabase cases table for id: %s", case_id_norm)
            response = supabase.table("cases").select("status, state").eq("id", case_id_norm).execute()
            if response.data:
                record = response.data[0]
                status_str = record.get("status") or record.get("state") or "Pending"
                local_logger.info("Supabase get_case_status success. Status: %s", status_str)
                return {"case_id": case_id_norm, "status": status_str}
        except Exception as e:
            local_logger.exception("Supabase get_case_status failed: %s", e)

    # Mock fallback
    local_logger.info("Using mock fallback for get_case_status.")
    for store_cases in MOCK_DATABASE.values():
        for case in store_cases:
            if case.get("id", "").strip().upper() == case_id_norm:
                status_str = case.get("status") or case.get("state") or "Pending"
                local_logger.info("Mock get_case_status success. Status: %s", status_str)
                return {"case_id": case_id_norm, "status": status_str}

    local_logger.warning("Case %s not found in get_case_status database lookup.", case_id_norm)
    return {"case_id": case_id_norm, "status": status_str}


def update_case_status(case_id: str, new_status: str) -> str:
    """
    Update the status of an incident support case to one of the strict vocabulary states.

    Args:
        case_id (str): The unique case/ticket identifier (e.g. 'RC001024').
        new_status (str): The target status. Must be exactly 'Pending', 'Awaiting Confirmation', or 'Closed'.

    Returns:
        str: A direct confirmation/affirmation message payload.
    """
    import streamlit as st
    import logging
    from mocks.servicenow import update_case

    local_logger = logging.getLogger("chipLLM.update_case_status")
    local_logger.info("=== TOOL CALL: update_case_status ===")
    local_logger.info("case_id: %s, new_status: %s", case_id, new_status)

    allowed_statuses = ["Pending", "Awaiting Confirmation", "Closed"]
    if new_status not in allowed_statuses:
        error_msg = f"Invalid status '{new_status}'. Allowed statuses are: {', '.join(allowed_statuses)}."
        local_logger.error(error_msg)
        return error_msg

    case_id_norm = case_id.strip().upper()

    try:
        updated_record = update_case(case_id_norm, {"status": new_status})
        if updated_record:
            # Sync session state active tickets
            if "active_tickets" in st.session_state and st.session_state.active_tickets:
                for t in st.session_state.active_tickets:
                    if t.get("id", "").strip().upper() == case_id_norm:
                        t["status"] = new_status
                        t["state"] = new_status
            st.session_state.active_case_id = case_id_norm
            confirm_msg = f"Successfully updated case {case_id_norm} status to '{new_status}'."
            local_logger.info(confirm_msg)
            return confirm_msg
        else:
            return f"Error: Case {case_id_norm} not found or could not be updated."
    except Exception as e:
        local_logger.exception("Error in update_case_status tool execution: %s", e)
        return f"An error occurred while updating the case status: {e}"


def set_alert_visibility(incident_id: str, visible: bool, scope: str = "session") -> str:
    """
    Sets the visibility of a major incident alert/banner (e.g. 'MIM-0008472' or 'MIM0008472').

    Args:
        incident_id: The unique incident identifier (e.g., 'MIM-0008472').
        visible: Whether the alert/banner should be visible (True) or hidden/minimized (False).
        scope: The scope of visibility update. Defaults to 'session'.

    Returns:
        str: A direct confirmation message.
    """
    import streamlit as st
    import logging

    local_logger = logging.getLogger("chipLLM.set_alert_visibility")
    local_logger.info("=== TOOL CALL: set_alert_visibility ===")
    local_logger.info("incident_id: %s, visible: %s, scope: %s", incident_id, visible, scope)

    norm_id = incident_id.strip().upper()
    # Normalize ID to start with MIM- or MIM
    if not norm_id.startswith("MIM"):
        norm_id = "MIM-" + norm_id
    elif norm_id.startswith("MIM") and not norm_id.startswith("MIM-"):
        rest = norm_id[3:]
        norm_id = "MIM-" + rest

    if "mim_dismissed" not in st.session_state:
        st.session_state.mim_dismissed = {}

    st.session_state.mim_dismissed[norm_id] = not visible
    confirm_msg = f"Successfully set alert visibility for {norm_id} to {visible}."
    local_logger.info(confirm_msg)
    return confirm_msg



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
            if i == len(messages) - 1 and role == "user":
                mim_context = ""
                active_mims = st.session_state.get("active_mims", [])
                if active_mims:
                    mim_context = "[ACTIVE OUTAGES / INCIDENTS]\n"
                    for mim in active_mims:
                        mim_context += f"- {mim.get('number')}: {mim.get('short_description')} (Priority: {mim.get('priority')})\n"
                    mim_context += "[END ACTIVE OUTAGES]\n\n"

                # Check all user messages in history for software push / update suggestion
                any_software_push = False
                software_push_keywords = [
                    "software push", "software update", "system update", "nightly push", 
                    "nightly update", "deployment", "system push", "new release", 
                    "new update", "bad push", "bad update", "software deployment"
                ]
                for m in messages:
                    if m.get("role") == "user":
                        m_content_lower = m.get("content", "").lower()
                        if any(kw in m_content_lower for kw in software_push_keywords):
                            any_software_push = True
                            break

                is_adhoc = False
                if rag_context and "### [CRITICAL ADHOC OVERRIDE BULLETIN]" in rag_context:
                    is_adhoc = True

                override_instruction = ""
                if is_adhoc:
                    override_instruction = (
                        "[SYSTEM INSTRUCTION: An ACTIVE CRITICAL EMERGENCY OVERRIDE is in effect. "
                        "You MUST drop your deadpan/quirky baseline commentary regarding standard hardware failures, "
                        "state directly to the manager that a known network event is occurring, "
                        "and list the rollback recovery steps instantly.]\n\n"
                    )
                    
                    if rag_context and "[ACTIVE CRITICAL EMERGENCY OVERRIDE IN EFFECT]" not in rag_context:
                        rag_context = (
                            "---\n"
                            "[ACTIVE CRITICAL EMERGENCY OVERRIDE IN EFFECT]\n"
                            f"[PLAYBOOK RECOVERY INSTRUCTIONS]:\n{rag_context}\n"
                            "---"
                        )

                push_instruction = ""
                if any_software_push:
                    push_instruction = (
                        "[SYSTEM INSTRUCTION: The user has suggested or suspected that a software push, "
                        "update, nightly push, or deployment caused this issue. You MUST skip all troubleshooting "
                        "steps entirely and immediately ask exactly: \"How would you like to proceed? (\"Open a support ticket\" "
                        "or \"Live Chat with an Agent\")\". Do NOT propose any troubleshooting steps or output "
                        "the transitional salvo.]\n\n"
                    )

                text = (
                    f"{override_instruction}"
                    f"{push_instruction}"
                    f"{mim_context}"
                    f"[RELEVANT KNOWLEDGE BASE CONTEXT — use this to answer]\n"
                    f"{rag_context if rag_context else 'None'}\n"
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
            system_instruction=system_instruction if system_instruction is not None else get_system_instructions(st.session_state.get("demo_personality", "Normal")),
            temperature=0.3,
            max_output_tokens=1024,
            top_p=0.9,
            tools=[get_active_tickets, get_case_details, add_case_comment, escalate_case, get_case_status, update_case_status, set_alert_visibility],
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        )

        remote_calls_count = 0
        max_remote_calls = 30

        while remote_calls_count < max_remote_calls:
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

            if not tool_calls:
                break

            function_responses = []
            for tc in tool_calls:
                remote_calls_count += 1
                args = dict(tc.args) if tc.args else {}
                if tc.name == "get_active_tickets":
                    result = get_active_tickets(**args)
                elif tc.name == "get_case_details":
                    result = get_case_details(**args)
                elif tc.name == "add_case_comment":
                    result = add_case_comment(**args)
                elif tc.name == "escalate_case":
                    result = escalate_case(**args)
                elif tc.name == "get_case_status":
                    result = get_case_status(**args)
                elif tc.name == "update_case_status":
                    result = update_case_status(**args)
                elif tc.name == "set_alert_visibility":
                    result = set_alert_visibility(**args)
                else:
                    result = {"error": f"Unknown tool: {tc.name}"}

                function_responses.append(
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=tc.name,
                            response={"result": result},
                        )
                    )
                )

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
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
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
        # Format conversation history as a single text block
        history_text = ""
        for msg in messages:
            role = "Manager" if msg["role"] == "user" else "ChipLLM"
            history_text += f"{role}: {msg['content']}\n"

        prompt = (
            f"You are a tech support assistant. Review the following conversation between a restaurant manager and technical support:\n\n"
            f"{history_text}\n"
            f"Extract the specific device/system and its technical issue/symptom.\n"
            f"Generate a highly concise, informative description summarizing this issue (maximum 8-10 words, under 10 words total).\n"
            f"Strictly follow these rules:\n"
            f"1. Do not use pleasantries, greetings, or filler words (e.g., do not say 'The manager reports...', 'Hi', 'Hello', etc.).\n"
            f"2. Focus only on the technical symptom and device (e.g., 'Kiosk #3 screen is frozen', 'Receipt printer paper jam', 'POS not printing receipt').\n"
            f"3. If the user hasn't described any issue yet or it is completely unclear, return exactly 'Unknown'.\n"
            f"4. Do not include quotes, periods, or other punctuation around the description.\n"
            f"5. Capitalize the first letter of the description.\n"
            f"Now, output ONLY the final description text."
        )

        config = genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=128,
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt],
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
        "channel": "ChipLLM_CHATBOT",
        "sla_breach_minutes": {"CRITICAL": 30, "HIGH": 60, "MEDIUM": 120, "LOW": 480}.get(severity, 120),
        "auto_dispatch": severity in ("CRITICAL", "HIGH"),
    }
