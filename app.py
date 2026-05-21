"""
app.py
------
chipLLM – Mobile-focused Streamlit chat interface.
Vertex AI backend · In-memory RAG · Domain guardrails · Ticket metadata sidebar.
"""

from __future__ import annotations

import json
import time

import streamlit as st

from guardrails import check_guardrails
from knowledge_base import retrieve_context
from llm_client import ChipLLMClient, extract_ticket_metadata

# ---------------------------------------------------------------------------
# Store authentication config
# ---------------------------------------------------------------------------

DEFAULT_STORE  = "67067"               # pre-selected default
STORE_OPTIONS  = ["67067", "67068", "67069"]  # all authorized stores

# ---------------------------------------------------------------------------
# L0_KA Playbook Directory (Dynamic RAG)
# ---------------------------------------------------------------------------
# Handled in knowledge_base.py

# ---------------------------------------------------------------------------
# Page config – must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="chipLLM · Restaurant Tech Support",
    page_icon="🍔",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – mobile viewport, dark kitchen theme, micro-animations
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* ── Google Font ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root tokens ─────────────────────────────────────────────────────────── */
:root {
  --bg-app:       #0d0f14;
  --bg-panel:     #14171f;
  --bg-card:      #1a1d28;
  --bg-input:     #1f2333;
  --accent:       #da291c;       /* McDonald's Red */
  --accent-dim:   #a81a10;       /* Dark Red */
  --accent-glow:  rgba(218, 41, 28, .25);
  --user-bubble:  #1e3a5f;
  --bot-bubble:   #1a1d28;
  --border:       rgba(255,255,255,.07);
  --text-primary: #f0f2f8;
  --text-muted:   #8892a4;
  --text-accent:  #ffc72c;       /* McDonald's Yellow/Gold accent */
  --success:      #22d3a0;
  --danger:       #f43f5e;
  --radius:       18px;
  --radius-sm:    10px;
  --shadow:       0 8px 32px rgba(0,0,0,.55);
}

/* ── Base reset ──────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background-color: var(--bg-app) !important;
  color: var(--text-primary) !important;
}

/* ── Mobile viewport constraint ─────────────────────────────────────────── */
.main .block-container {
  max-width: 420px !important;
  padding: 0 !important;
  margin: 0 auto !important;
  padding-top: 68px !important;
  padding-bottom: 80px !important;
}

/* ── Sticky Header bar via sentinel ─────────────────────────────────────── */
[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] {
  position: fixed !important;
  top: 0 !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  width: 100% !important;
  max-width: 420px !important;     /* constraint to match mobile viewport width */
  background: linear-gradient(135deg, #0d0f14 0%, #1a1d28 100%) !important;
  border-bottom: 1px solid rgba(255,255,255,.07) !important;
  padding: 10px 14px !important;
  z-index: 9999 !important;        /* keep on top */
  box-shadow: 0 4px 20px rgba(0,0,0,.4) !important;
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 8px !important;
}

/* Ensure columns stack horizontally nicely inside header */
[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
  width: auto !important;
  flex: unset !important;
  min-width: 0 !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) {
  flex: 1.3 !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) {
  flex: 1.9 !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) {
  flex: 1.1 !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(4) {
  flex: 0.8 !important;
}

/* Custom styles for selectbox in header bar */
[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] > div > div {
  background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  color: #f0f2f8 !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  padding: 0 4px !important;
  min-height: 28px !important;
  height: 28px !important;
  border-radius: 6px !important;
  line-height: 1.2 !important;
  letter-spacing: .1px !important;
  margin-top: 6px !important;
  display: flex !important;
  align-items: center !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] > div > div:hover {
  background: rgba(218,41,28,.15) !important;
  border-color: rgba(255,199,44,.5) !important;
  color: #ffc72c !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] svg {
  fill: #8892a4 !important;
  width: 14px !important;
  height: 14px !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] [data-baseweb="select"] {
  height: 28px !important;
}

/* Override default large padding inside the header selectbox to fit store text */
[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] [class*="ValueContainer"] {
  padding-left: 2px !important;
  padding-right: 2px !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
  padding-left: 4px !important;
  padding-right: 18px !important;
}

/* Custom styles for header buttons */
[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] .stButton > button {
  background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  color: #f0f2f8 !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  padding: 4px 8px !important;
  min-height: 28px !important;
  height: 28px !important;
  border-radius: 6px !important;
  line-height: 1.2 !important;
  letter-spacing: .1px !important;
  transition: background .15s, border-color .15s, color .15s !important;
  white-space: nowrap !important;
  margin-top: 6px !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] .stButton > button:hover {
  background: rgba(218,41,28,.15) !important;
  border-color: rgba(255,199,44,.5) !important;
  color: #ffc72c !important;
  transform: none !important;
  opacity: 1 !important;
}

/* Pulse dot active store status indicator */
.pulse-dot-active::before {
  content: '';
  width: 5px;
  height: 5px;
  background: var(--success);
  border-radius: 50%;
  display: inline-block;
  animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── Native st.chat_message() bubble overrides ──────────────────────────── */
@keyframes slide-in {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 2px 12px !important;
  animation: slide-in .2s ease-out both;
}

[data-testid="stChatMessageContent"] {
  border-radius: var(--radius) !important;
  font-size: 14px !important;
  line-height: 1.55 !important;
  padding: 12px 16px !important;
  word-wrap: break-word !important;
}

/* Bot bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-bottom-left-radius: 4px !important;
  color: var(--text-primary) !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
  background: linear-gradient(135deg, #1e3a5f 0%, #1a4a7a 100%) !important;
  border-bottom-right-radius: 4px !important;
  color: #ddeeff !important;
}

/* Bot avatar */
[data-testid="chatAvatarIcon-assistant"] {
  background: linear-gradient(135deg, var(--accent) 0%, var(--text-accent) 100%) !important;
  box-shadow: 0 2px 10px var(--accent-glow) !important;
  border-radius: 50% !important;
}

/* User avatar */
[data-testid="chatAvatarIcon-user"] {
  background: linear-gradient(135deg, #1e3a5f 0%, #1a4a7a 100%) !important;
  border-radius: 50% !important;
}

.msg-time {
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 2px;
}

/* ── RAG badge ───────────────────────────────────────────────────────────── */
.rag-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: rgba(218,41,28,.12);
  border: 1px solid rgba(255,199,44,.3);
  color: var(--text-accent);
  font-size: 10px;
  font-weight: 600;
  padding: 3px 8px;
  border-radius: 20px;
  margin-bottom: 6px;
  letter-spacing: 0.3px;
}

/* ── Input area ──────────────────────────────────────────────────────────── */
.input-zone {
  background: var(--bg-panel);
  border-top: 1px solid var(--border);
  padding: 12px 14px 20px;
  position: sticky;
  bottom: 0;
}

/* Override Streamlit's chat input ──────────────────────────────────────── */
[data-testid="stChatInput"] {
  background: var(--bg-input) !important;
  border: 1.5px solid rgba(218,41,28,.35) !important;
  border-radius: 24px !important;
  padding: 14px 18px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  color: var(--text-primary) !important;
  min-height: 52px !important;
  transition: border-color .2s, box-shadow .2s;
}

[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

[data-testid="stChatInputSubmitButton"] {
  background: var(--accent) !important;
  border-radius: 50% !important;
  width: 42px !important;
  height: 42px !important;
}

/* ── Streamlit sidebar ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg-panel) !important;
  border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
  color: var(--text-primary) !important;
}

/* ── Ticket card in sidebar ──────────────────────────────────────────────── */
.ticket-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 14px;
  margin-top: 8px;
}

.ticket-field {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 7px 0;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
}

.ticket-field:last-child { border-bottom: none; }

.ticket-label {
  color: var(--text-muted);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-size: 10px;
}

.ticket-value {
  color: var(--text-primary);
  font-weight: 600;
  font-size: 12px;
  text-align: right;
}

.sev-critical { color: var(--accent) !important; } /* McDonald's Red */
.sev-high { color: var(--text-accent) !important; } /* McDonald's Yellow */
.sev-medium { color: #ffd97d !important; } /* Lighter Gold */
.sev-low { color: var(--success) !important; }

/* ── Streamlit default element overrides ────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 24px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  padding: 10px 22px !important;
  font-size: 13px !important;
  letter-spacing: 0.3px !important;
  min-height: 44px !important;
  transition: opacity .2s, transform .15s !important;
}

.stButton > button:hover {
  opacity: 0.9 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 0 0 2px #ffc72c !important;
}

/* ── Streamlit default markdown in chat ──────────────────────────────────── */
.stMarkdown p { margin: 0 0 6px; }
.stMarkdown ol, .stMarkdown ul { padding-left: 18px; margin: 4px 0; }
.stMarkdown code {
  background: rgba(218,41,28,.12);
  color: var(--text-accent);
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 12px;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(218,41,28,.3); border-radius: 4px; }

/* Streamlit selectbox in auth card */
[data-testid="stSelectbox"] > div > div {
  background: #1a1d28 !important;
  border: 1px solid rgba(218,41,28,.35) !important;
  border-radius: 12px !important;
  color: #f0f2f8 !important;
  font-size: 20px !important;
  font-weight: 700 !important;
  min-height: 52px !important;
}

/* ── Premium Suggestion Chips ───────────────────────────────────────────── */
div.chips-sentinel + div[data-testid="stHorizontalBlock"] button {
  background: rgba(255, 255, 255, 0.05) !important;
  color: #f0f2f8 !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 20px !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  min-height: 36px !important;
  height: 36px !important;
  padding: 0 16px !important;
  transition: all 0.2s ease !important;
  box-shadow: none !important;
  margin-bottom: 12px !important;
}

div.chips-sentinel + div[data-testid="stHorizontalBlock"] button:hover {
  background: rgba(218, 41, 28, 0.15) !important;
  border-color: rgba(255, 199, 44, 0.5) !important;
  color: #ffc72c !important;
  transform: translateY(-1px) !important;
}

/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu, header[data-testid="stHeader"], footer { display: none !important; }
.viewerBadge_container__1QSob { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ticket_metadata" not in st.session_state:
    st.session_state.ticket_metadata = None

if "escalation_triggered" not in st.session_state:
    st.session_state.escalation_triggered = False

if "rag_hits" not in st.session_state:
    st.session_state.rag_hits = {}  # msg_index -> playbook title

# Auth & Ticket state
if "store_confirmed" not in st.session_state:
    st.session_state.store_confirmed = True
if "active_store" not in st.session_state:
    st.session_state.active_store = DEFAULT_STORE
if "ticket_collection_active" not in st.session_state:
    st.session_state.ticket_collection_active = False
if "ticket_collection_is_agent" not in st.session_state:
    st.session_state.ticket_collection_is_agent = False

if "live_agent_pending_connection" not in st.session_state:
    st.session_state.live_agent_pending_connection = False

if "escalation_triage_active" not in st.session_state:
    st.session_state.escalation_triage_active = False
if "escalation_triage_step" not in st.session_state:
    st.session_state.escalation_triage_step = None
if "triage_model" not in st.session_state:
    st.session_state.triage_model = None
if "triage_serial" not in st.session_state:
    st.session_state.triage_serial = None
if "triage_q1" not in st.session_state:
    st.session_state.triage_q1 = None
if "triage_q2" not in st.session_state:
    st.session_state.triage_q2 = None
if "triage_q3" not in st.session_state:
    st.session_state.triage_q3 = None
if "triage_priority" not in st.session_state:
    st.session_state.triage_priority = None
if "show_restart_chat_btn" not in st.session_state:
    st.session_state.show_restart_chat_btn = False

def _is_printer_issue() -> bool:
    """Return True if the current session is about a printer."""
    all_text = " ".join(
        m["content"] for m in st.session_state.get("messages", [])
    ).lower()
    return "printer" in all_text or "print" in all_text


def render_device_detail_form() -> None:
    """
    Renders a minimal embedded form to optionally collect device model/serial
    before the priority assessment questions. Shown only for printer issues.
    """
    ts_now = time.strftime("%H:%M")

    st.markdown(
        '<div style="background: rgba(218,41,28,0.10); border: 1px solid rgba(218,41,28,0.45); '
        'border-left: 4px solid var(--accent); padding: 12px 16px; border-radius: 8px; '
        'margin: 8px 0 12px; font-size: 13.5px; line-height: 1.5; color: var(--text-primary);">'
        '🎫 <b>Ticket Creation In Progress</b><br/>'
        'Device details help route your ticket faster. Enter what you know — all fields are optional.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("device_detail_form", border=True):
        st.markdown(
            "<div style='font-size:13px; color:#8892a4; margin-bottom:8px;'>"
            "To help with troubleshooting, can you enter (optional):"
            "</div>",
            unsafe_allow_html=True
        )
        model_val = st.text_input(
            "Device Model Number",
            placeholder="e.g. Epson TM-T88VI",
            key="device_form_model_input"
        )
        serial_val = st.text_input(
            "Device Serial Number",
            placeholder="e.g. X1A2B3C4D5",
            key="device_form_serial_input"
        )
        submitted = st.form_submit_button("Submit", use_container_width=True)

    if submitted:
        # Store whatever was entered (empty = unknown, we skip it)
        if model_val.strip():
            st.session_state.triage_model = model_val.strip()
        if serial_val.strip():
            st.session_state.triage_serial = serial_val.strip()
        # Advance to q1
        st.session_state.escalation_triage_step = "q1"
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "To help determine priority for this ticket, let's dig in on impact.\n\n"
                "Is this issue completely stopping your store from taking orders or payments right now?"
            ),
            "timestamp": ts_now,
            "blocked": False
        })
        st.rerun()


def start_escalation_triage(append_welcome: bool = True):
    st.session_state.escalation_triage_active = True
    st.session_state.triage_model = None
    st.session_state.triage_serial = None
    st.session_state.triage_q1 = None
    st.session_state.triage_q2 = None
    st.session_state.triage_q3 = None
    st.session_state.triage_priority = None
    st.session_state.manual_ticket_flow = False

    is_printer = _is_printer_issue()
    # Printers use embedded device form; non-printers jump straight to q1
    first_step = "device_form" if is_printer else "q1"
    st.session_state.escalation_triage_step = first_step

    ts_now = time.strftime("%H:%M")
    if append_welcome:
        if is_printer:
            notice = (
                "Since these troubleshooting steps didn't resolve the issue, we need to escalate this and create a support ticket.\n\n"
                "For printer issues, device details help route your ticket faster. "
                "Please fill in the optional form below."
            )
        else:
            notice = (
                "Since these troubleshooting steps didn't resolve the issue, we need to escalate this and create a support ticket.\n\n"
                "To help determine priority for this ticket, let's dig in on impact.\n\n"
                "Is this issue completely stopping your store from taking orders or payments right now?"
            )
        st.session_state.messages.append({
            "role": "assistant",
            "content": notice,
            "timestamp": ts_now,
            "blocked": False
        })
    else:
        # LLM already explained escalation — for printers, device form will appear automatically.
        # For non-printers, append the q1 question if not already present.
        if not is_printer:
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                last_msg = st.session_state.messages[-1]
                if "stopping your store" not in last_msg["content"].lower():
                    last_msg["content"] = (
                        last_msg["content"].strip()
                        + "\n\nTo help determine priority for this ticket, let's dig in on impact.\n\n"
                        "Is this issue completely stopping your store from taking orders or payments right now?"
                    )

def reset_triage_state():
    st.session_state.escalation_triage_active = False
    st.session_state.escalation_triage_step = None
    st.session_state.triage_model = None
    st.session_state.triage_serial = None
    st.session_state.triage_q1 = None
    st.session_state.triage_q2 = None
    st.session_state.triage_q3 = None
    st.session_state.triage_priority = None

def calculate_priority_and_finish_triage():
    q1 = st.session_state.get("triage_q1", "No")
    q2 = st.session_state.get("triage_q2", "Only one")
    q3 = st.session_state.get("triage_q3", "Yes")
    
    is_q1_yes = "yes" in q1.lower()
    is_q2_multiples = "multiple" in q2.lower()
    is_q3_yes = "yes" in q3.lower() if is_q2_multiples else False
    
    if is_q1_yes:
        if not is_q2_multiples:
            priority = "P1"
        else: # multiples
            if not is_q3_yes:
                priority = "P1"
            else:
                priority = "P2"
    else: # q1 is no
        if not is_q2_multiples:
            priority = "P3"
        else: # multiples
            if not is_q3_yes:
                priority = "P3"
            else:
                priority = "P4"
                
    st.session_state.triage_priority = priority
    st.session_state.escalation_triage_active = False
    st.session_state.escalation_triage_step = "complete"
    st.session_state.ticket_collection_active = True



# ---------------------------------------------------------------------------
# Ticket Collection Interface
# ---------------------------------------------------------------------------

# Keys that belong to "Troubleshooting Steps" section, not core diagnostics
_TROUBLESHOOTING_KEYS = {
    "Is there visible paper",
    "Internal Jam Roller C",
    "Are you OTP or OTP cer",
}


def get_diagnostic_summary() -> list[tuple[str, str]]:
    """
    Extract key-value pairs (clarifying questions and responses) from session messages.
    Returns (core_rows, troubleshooting_rows, triage_rows) packed as a flat list with
    sentinels so the render function can split them into separate sections.
    Converts verbatim chat prompts into clean high-fidelity keys like Type, Number, etc.
    """
    import re
    messages = st.session_state.get("messages", [])
    summary_dict = {}
    
    # Process message history pairs
    for i in range(len(messages) - 1):
        msg = messages[i]
        next_msg = messages[i+1]
        if msg["role"] == "assistant" and next_msg["role"] == "user":
            q_content = msg["content"].lower()
            ans_clean = next_msg["content"].replace("**", "").replace("`", "").strip()
            
            # 1. Type detection
            if any(kw in q_content for kw in ["where is the printer", "printer located", "location", "what type", "which type"]):
                t_val = "POS"
                if "kvs" in ans_clean.lower() or "kitchen" in ans_clean.lower():
                    t_val = "KVS"
                elif "kiosk" in ans_clean.lower():
                    t_val = "Kiosk"
                elif "bos" in ans_clean.lower() or "back office" in ans_clean.lower():
                    t_val = "BOS"
                summary_dict["Type"] = t_val
                
            # 2. Number detection
            elif any(kw in q_content for kw in ["which pos number", "which device number", "which kiosk number", "which unit", "device number", "unit number"]):
                current_type = summary_dict.get("Type", "POS")
                num_only = re.sub(r"\D", "", ans_clean)
                if num_only:
                    summary_dict["Number"] = f"{current_type}{num_only}"
                else:
                    summary_dict["Number"] = ans_clean
                    
            # 3. What is the issue
            elif any(kw in q_content for kw in ["describe the issue", "what is the issue", "what symptom", "experiencing"]):
                summary_dict["What is the issue"] = ans_clean.capitalize()
                
            # 4. Is there visible paper  (troubleshooting step)
            elif "visible paper" in q_content:
                summary_dict["Is there visible paper"] = ans_clean.capitalize()
                
            # 5. Internal Jam Roller C  (troubleshooting step)
            elif any(kw in q_content for kw in ["internal jam roller", "roller c", "roller"]):
                summary_dict["Internal Jam Roller C"] = f"{ans_clean.capitalize()} — not resolved" if ans_clean.lower() == "no" else ans_clean
                
            # 6. Device Information
            elif any(kw in q_content for kw in ["serial number", "model", "device information"]):
                summary_dict["Device Information"] = ans_clean
                
            # 7. Are you OTP  (troubleshooting step)
            elif "otp" in q_content:
                summary_dict["Are you OTP or OTP cer"] = ans_clean.capitalize()

    # If first user message exists, use it to seed "What is the issue"
    if "What is the issue" not in summary_dict:
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        if user_msgs:
            first_msg = user_msgs[0].replace("**", "").replace("`", "").strip()
            if len(first_msg) > 30:
                first_msg = first_msg[:27] + "..."
            summary_dict["What is the issue"] = first_msg.capitalize()

    # Core device/issue rows (always shown in Triage Diagnostics)
    core_defaults = [
        ("Type", "POS"),
        ("Number", "POS1"),
        ("What is the issue", "Unknown"),
        ("Device Information", "Skipped device info"),
    ]
    # Troubleshooting Q&A rows (shown under Troubleshooting Steps)
    ts_defaults = [
        ("Is there visible paper", "No"),
        ("Internal Jam Roller C", "No — not resolved"),
        ("Are you OTP or OTP cer", "No"),
    ]
    
    core_rows = []
    for key, def_val in core_defaults:
        val = summary_dict.get(key)
        if not val:
            if key == "Type":
                all_text = " ".join([m["content"] for m in messages]).lower()
                if "kiosk" in all_text:
                    val = "Kiosk"
                elif "kvs" in all_text or "kitchen" in all_text:
                    val = "KVS"
                else:
                    val = def_val
            elif key == "Number":
                all_text = " ".join([m["content"] for m in messages]).lower()
                num_match = re.search(r"\b(pos|kiosk|kds|kvs|unit)?\s*#?\s*(\d)\b", all_text)
                t = summary_dict.get("Type", "POS")
                if num_match:
                    val = f"{t}{num_match.group(2)}"
                else:
                    val = f"{t}1"
            else:
                val = def_val
        if key == "Device Information":
            t_model = st.session_state.get("triage_model")
            t_serial = st.session_state.get("triage_serial")
            if t_model or t_serial:
                m_str = t_model if t_model else "Unknown"
                s_str = t_serial if t_serial else "Unknown"
                val = f"Model: {m_str}, S/N: {s_str}"
        core_rows.append((key, val))

    # Append impact triage answers to core rows
    if st.session_state.get("triage_priority"):
        core_rows.append(("Device Model", st.session_state.get("triage_model", "Unknown")))
        core_rows.append(("Serial Number", st.session_state.get("triage_serial", "Unknown")))
        core_rows.append(("Stopping Orders/Payments", st.session_state.get("triage_q1", "No")))
        core_rows.append(("Only Device of its Kind", st.session_state.get("triage_q2", "Only one")))
        if "multiple" in st.session_state.get("triage_q2", "").lower():
            core_rows.append(("Other Devices Functional", st.session_state.get("triage_q3", "Yes")))
        core_rows.append(("Calculated Priority", st.session_state.get("triage_priority", "P3")))

    # Build troubleshooting rows — only include keys that were actually answered
    ts_rows = []
    for key, _ in ts_defaults:
        if key in summary_dict:
            ts_rows.append((key, summary_dict[key]))

    # Flat format with section sentinels for backward compat (ticket dump)
    # Use a special "__section__" prefix to mark section headers
    final_summary = [("__section__", "Triage Diagnostics")] + core_rows
    if ts_rows:
        final_summary += [("__section__", "Troubleshooting Steps Attempted")] + ts_rows
    return final_summary


def get_choices_from_message(content: str) -> list[str]:
    """
    Extract discrete choice options from typical assistant questions.
    Supports Yes/No questions, Location questions, and Unit numbers.
    Ensures that options do not contain quotes and filters out 'e.g.' prefixes.
    """
    import re
    content_lower = content.lower()
    
    # 0. Sequential Triage questions — device model/serial now use embedded form, not chips
    if "completely stopping your store from taking orders or payments" in content_lower:
        return ["Yes", "No"]
    if "only device of its kind" in content_lower:
        return ["Only one", "There are multiples"]
    if "other devices of this kind online and functional" in content_lower:
        return ["Yes", "No"]

    # "More info" prompt is free-text — no chips
    if "more information about the problem" in content_lower or "when it began" in content_lower:
        return []

    # 0. Case routing choices
    if "cases with more details" in content_lower or "quicker resolution" in content_lower:
        return ["Manual Ticket Flow", "Continue with Automated Flow"]
    
    # 1. Yes/No questions
    if any(phrase in content_lower for phrase in [
        "did this resolve", "resolve the issue", "visible paper", "otp or otp cert", 
        "readable now", "connectivity restored", "showing the welcome screen", 
        "drawer opening", "performance acceptable", "lights on now", "orders appearing",
        "touchscreen responding", "reader working", "fixed the issue", "is paper out"
    ]):
        return ["Yes", "No"]
        
    # 2. Location/Type questions
    if ("where" in content_lower and "printer" in content_lower and "located" in content_lower) or "type of printer" in content_lower or all(kw in content_lower for kw in ["pos", "kvs", "kiosk", "bos"]):
        return ["POS", "KVS", "Kiosk", "BOS"]
        
    # 3. Unit number questions
    if any(phrase in content_lower for phrase in [
        "which pos number", "which kiosk number", "which device number", "which unit number", 
        "what # device", "what number", "device number is exper", "number is exper"
    ]):
        return ["1", "2", "3", "4"]

    # 4. Explicit paren-bounded list of options
    match_paren = re.search(r"\(([^)]+)\)\s*\??\s*$", content.strip())
    if match_paren:
        s = match_paren.group(1).strip()
        
        # Clean leading e.g. prefixes completely from the start of the paren text
        s_clean = re.sub(r"^e\.g\.?,?\s*", "", s, flags=re.IGNORECASE).strip()
        
        # Try to find all quoted substrings (standard or curly quotes)
        quoted = re.findall(r'["\'“’‘”]([^"\'“’‘”]+)["\'“’‘”]', s_clean)
        if quoted:
            raw_choices = quoted
        else:
            # Fallback to comma/or/slash splitting
            parts = re.split(r",\s*|\s+or\s+|\s*/\s*", s_clean)
            raw_choices = parts
            
        cleaned = []
        for choice in raw_choices:
            # Remove all forms of quotes
            c = choice.replace('"', '').replace("'", "").replace('“', '').replace('”', '').replace('‘', '').replace('’', '').strip()
            
            # Ignore e.g. prefixes again if they somehow remain
            if c.lower().startswith("e.g."):
                c = c[4:].strip()
            elif c.lower().startswith("e.g"):
                c = c[3:].strip()
                
            # Strip trailing commas or punctuation
            c = c.rstrip(",.? ")
            
            # Final filter to prevent "e.g" or empty options
            if c and c.lower() not in ["e.g.", "e.g", "example", "or", "and"]:
                cleaned.append(c)
                
        if 1 < len(cleaned) <= 6:
            return cleaned

    return []


def clean_assistant_message(content: str) -> str:
    """
    Remove parenthesized options list from assistant messages if suggestion chips will be displayed,
    append context-appropriate instruction suffixes, and wrap the troubleshooting salvo in a colored box.
    """
    import re
    
    # 1. Intercept troubleshooting salvo and put inside a colored box
    salvo_text = "There are some common troubleshooting steps that might help you fix this issue on your own. We will quickly step through them to see if this solves the issue"
    if salvo_text.lower() in content.lower():
        pattern = re.compile(re.escape(salvo_text) + r"\.?", re.IGNORECASE)
        replacement = (
            '<div style="background: rgba(218, 41, 28, 0.08); border: 1px solid rgba(255, 199, 44, 0.3); '
            'border-left: 4px solid var(--accent); padding: 12px 14px; border-radius: 8px; margin: 10px 0; '
            'font-size: 13.5px; line-height: 1.5; color: var(--text-primary);">'
            '⚡ <b>Recommended Troubleshooting</b><br/>'
            'There are some common troubleshooting steps that might help you fix this issue on your own. '
            'We will quickly step through them to see if this solves the issue.'
            '</div>'
        )
        content = pattern.sub(replacement, content)

    # 1.5 Intercept escalation notice and put inside a colored box
    escalation_phrases = [
        "since these troubleshooting steps didn't resolve the issue, we need to escalate",
        "since the troubleshooting steps did not resolve the issue",
        "these troubleshooting steps didn't resolve the issue",
    ]
    for esc_phrase in escalation_phrases:
        if esc_phrase in content.lower():
            # Wrap the sentence containing the escalation notice
            import re as _re
            esc_pattern = _re.compile(
                r"((?:I understand\.?\s*)?(?:Since|Because)[^.!?]*(?:escalate|support ticket)[^.!?]*[.!?])",
                _re.IGNORECASE
            )
            def _esc_replacer(m):
                return (
                    '<div style="background: rgba(218, 41, 28, 0.10); border: 1px solid rgba(218, 41, 28, 0.4); '
                    'border-left: 4px solid var(--accent); padding: 12px 14px; border-radius: 8px; margin: 10px 0; '
                    'font-size: 13.5px; line-height: 1.5; color: var(--text-primary);">'
                    '🚨 <b>Escalation Required</b><br/>'
                    + m.group(0) +
                    '</div>'
                )
            new_content = esc_pattern.sub(_esc_replacer, content, count=1)
            if new_content != content:
                content = new_content
                break

    # 1.6 Intercept priority/impact notice and put inside a colored box
    priority_text = "To help determine priority for this ticket, let's dig in on impact"
    if priority_text.lower() in content.lower():
        pattern = re.compile(re.escape(priority_text) + r"\.?", re.IGNORECASE)
        replacement = (
            '<div style="background: rgba(218, 41, 28, 0.08); border: 1px solid rgba(255, 199, 44, 0.3); '
            'border-left: 4px solid var(--accent); padding: 12px 14px; border-radius: 8px; margin: 10px 0; '
            'font-size: 13.5px; line-height: 1.5; color: var(--text-primary);">'
            '📊 <b>Ticket Impact Assessment</b><br/>'
            "To help determine priority for this ticket, let's dig in on impact."
            '</div>'
        )
        content = pattern.sub(replacement, content)

    # Prevent duplicate cleanups if suffix is already present
    if any(suffix in content for suffix in [
        "(type model # in the text box below)",
        "(type serial # in the text box below)",
        "(If complicated, describe in the steps below)",
        "(or type your answer below)",
        "(Impact, when it began, possible causes)",
    ]):
        return content

    choices = get_choices_from_message(content)
    if not choices:
        return content
        
    # Strip any bulleted/numbered lines presenting the choices
    lines = content.split("\n")
    filtered_lines = []
    for line in lines:
        stripped_line = line.strip()
        # check if it looks like a list item: starting with *, -, +, \d+. or •
        is_list_item = re.match(r"^(?:[\*\-\+\•]|\d+\.?)\s*", stripped_line)
        if is_list_item and stripped_line:
            line_lower = stripped_line.lower()
            if any(choice.lower() in line_lower for choice in choices if not choice.isdigit()):
                continue
        filtered_lines.append(line)
        
    # If the last remaining non-empty line ends with a colon or is a short list introduction, drop it
    while filtered_lines:
        last_line = filtered_lines[-1].strip()
        if not last_line:
            filtered_lines.pop()
            continue
        last_line_lower = last_line.lower()
        if len(last_line) < 45 and (last_line.endswith(":") or any(phrase in last_line_lower for phrase in ["is it a", "is it"])):
            filtered_lines.pop()
        else:
            break
            
    content = "\n".join(filtered_lines).strip()
        
    # Remove final parenthesized block that contains options or 'e.g.'
    cleaned = re.sub(
        r"\s*\([^)]*(?:e\.g\.|pos|kvs|kiosk|bos|yes|no|or|1|2|3|4|flow|only|multiples|screen|reader|payment|print|jam|garbled)[^)]*\)\s*\??\s*$", 
        "", 
        content.strip(), 
        flags=re.IGNORECASE
    )
    
    if cleaned == content.strip():
        cleaned = re.sub(
            r"\s*\([^)]*(?:e\.g\.|pos|kvs|kiosk|bos|yes|no|or|1|2|3|4|flow|only|multiples|screen|reader|payment|print|jam|garbled)[^)]*\)\s*\??\s*$", 
            "", 
            content.strip(), 
            flags=re.IGNORECASE
        )
        
    cleaned = cleaned.rstrip("?:. ")
    
    # 4. Dynamic Suffixes Selection
    content_lower = content.lower()
    if "device model" in content_lower or "model number" in content_lower or "model #" in content_lower:
        suffix = "(type model # in the text box below)"
    elif "serial number" in content_lower or "serial #" in content_lower:
        suffix = "(type serial # in the text box below)"
    elif any(phrase in content_lower for phrase in [
        "did this resolve", "resolve the issue", "visible paper", "otp or otp cert", 
        "readable now", "connectivity restored", "showing the welcome screen", 
        "drawer opening", "performance acceptable", "lights on now", "orders appearing",
        "touchscreen responding", "reader working", "fixed the issue", "is paper out"
    ]):
        suffix = "(If complicated, describe in the steps below)"
    else:
        suffix = "(or type your answer below)"
        
    return f"{cleaned}? {suffix}"


def trigger_live_agent_flow(user_message_text: str) -> None:
    """
    Simulates escalating to a live chat agent by dumping collected diagnostics,
    showing a connecting message, and scheduling Franklin to join in the next loop.
    Uses the same diagnostic data as the ticket creation form.
    """
    import time
    ts_now = time.strftime("%H:%M")
    st.session_state.ticket_collection_active = False
    
    # 1. Append user's escalation request if not already present
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_message_text:
        st.session_state.messages.append({
            "role": "user",
            "content": user_message_text,
            "timestamp": ts_now,
            "blocked": False
        })
        
    # 2. Get diagnostic summary — same data as ticket form, filtered of __section__ sentinels
    diag_summary = get_diagnostic_summary()
    active_store = st.session_state.get("active_store", "67067")
    
    case_dump_lines = [
        "🤖 **Live Agent Handoff – Case Details Collected So Far:**\n\n",
        "| Parameter | Value |\n",
        "| :--- | :--- |\n",
        f"| **Active Store** | `Store #{active_store}` |\n"
    ]
    for q, a in diag_summary:
        if q == "__section__":
            # Render section headers as separator rows in the table
            case_dump_lines.append(f"| **— {a} —** |  |\n")
            continue
        case_dump_lines.append(f"| **{q}** | `{a}` |\n")
        
    case_dump_text = "".join(case_dump_lines)
    st.session_state.messages.append({
        "role": "assistant",
        "content": case_dump_text,
        "timestamp": ts_now,
        "blocked": False
    })
    
    # 3. Append "Connecting to live agent, please hold.."
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Connecting to live agent, please hold..",
        "timestamp": ts_now,
        "blocked": False
    })
    
    # Schedule Franklin's message on next run loop
    st.session_state.live_agent_pending_connection = True
    st.rerun()


def ask_case_flow_options(user_message_text: str | None = None) -> None:
    """
    Asks the user if they want to use Manual Ticket Flow or Continue with Automated Flow,
    showing suggestion buttons for both.
    """
    ts_now = time.strftime("%H:%M")
    st.session_state.ticket_collection_active = False
    
    if user_message_text:
        # Append user's action/message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_message_text,
            "timestamp": ts_now,
            "blocked": False
        })
        
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Cases with more details will lead to quicker resolution. "
            "Do you want to describe the problem further and attempt basic troubleshooting, "
            "or jump directly to the ticket creation flow?"
        ),
        "timestamp": ts_now,
        "blocked": False
    })
    st.rerun()



def render_ticket_collection_form(is_live_agent: bool = False) -> None:
    """
    Renders an inline form above the chat input to gather final ticket details,
    styled exactly like the premium dark theme screenshot.
    """
    import re
    active_store = st.session_state.get("active_store", 67067)
    
    # Store addresses map
    STORE_ADDRESSES = {
        67067: {
            "address": "1725 Slough Avenue, Scranton, PA 18503",
            "location": "Scranton Restaurant"
        },
        67068: {
            "address": "120 Paper Place, Scranton, PA 18508",
            "location": "Scranton North Restaurant"
        },
        67069: {
            "address": "420 Paper Mill Road, Scranton, PA 18512",
            "location": "Scranton East Restaurant"
        }
    }
    
    # Check if we should override store ID to matches the screenshot "12345" for perfect fidelity
    display_store = str(active_store)
    
    store_info = STORE_ADDRESSES.get(int(active_store) if str(active_store).isdigit() else 67067, {
        "address": "1725 Slough Avenue, Scranton, PA 18503",
        "location": "Scranton Restaurant"
    })
    store_address = store_info["address"]
    store_location = store_info["location"]
    
    st.write("---")
    
    # Style override for the ticket form inputs and layout to look exactly like the screenshot
    st.markdown(
        """
        <style>
        .ticket-form-title {
            font-size: 20px;
            font-weight: 700;
            color: #f0f2f8;
            margin-bottom: 16px;
            text-align: left;
        }
        .confirm-header {
            font-size: 11px;
            font-weight: 700;
            color: #ffc72c;
            letter-spacing: 1px;
            margin-bottom: 12px;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .lock-lbl {
            font-size: 11px;
            font-weight: 700;
            color: #8892a4;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
            margin-top: 10px;
        }
        
        /* Container border wrapper override styling to match confirmation box */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            background-color: #161720 !important;
            border-radius: 14px !important;
            padding: 20px !important;
            margin-bottom: 16px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        }
        
        /* Make disabled input widgets match the screenshot dark lock style */
        div[data-testid="stTextInput"] div[data-baseweb="input"] input:disabled {
            background-color: #1e2030 !important;
            color: #8892a4 !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            font-size: 13px !important;
            height: 38px !important;
            border-radius: 8px !important;
        }
        
        /* Make modifiable input widgets match premium style */
        div[data-testid="stTextInput"] div[data-baseweb="input"] input:not(:disabled) {
            background-color: #1e2030 !important;
            color: #f0f2f8 !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            font-size: 13px !important;
            height: 38px !important;
            border-radius: 8px !important;
        }
        
        /* Focus state selection ring matching screenshot orange border */
        div[data-testid="stTextInput"] div[data-baseweb="input"] input:not(:disabled):focus {
            border-color: #ffc72c !important;
            box-shadow: 0 0 0 1px #ffc72c !important;
        }
        
        .diagnostic-card {
            background-color: #161720 !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 14px;
            padding: 20px;
            margin-top: 14px;
            margin-bottom: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        }
        .diagnostic-header {
            font-size: 10px;
            font-weight: 600;
            color: #8892a4;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 6px;
        }
        .diagnostic-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-size: 11px;
        }
        .diagnostic-row:last-child {
            border-bottom: none;
        }
        .diagnostic-key {
            color: #8892a4;
            max-width: 60%;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .diagnostic-val {
            color: #f0f2f8;
            font-weight: 700;
            text-align: right;
        }
        
        /* Submit button custom styling override via sibling marker */
        div.submit-btn-marker + div.stButton > button {
            background-color: #da291c !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            height: 44px !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 12px rgba(218,41,28,0.2) !important;
            margin-top: 12px !important;
            margin-bottom: 4px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        div.submit-btn-marker + div.stButton > button:hover {
            background-color: #a81a10 !important;
            box-shadow: 0 4px 16px rgba(218,41,28,0.35) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Start Over button custom styling override via sibling marker */
        div.cancel-btn-marker + div.stButton > button {
            background-color: #1f2333 !important;
            color: #8892a4 !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            height: 40px !important;
            border-radius: 10px !important;
            margin-top: 6px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        div.cancel-btn-marker + div.stButton > button:hover {
            color: #f0f2f8 !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
            background-color: #262a3f !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Form Title
    st.markdown("<div class='ticket-form-title'>Submitting Support Ticket</div>", unsafe_allow_html=True)
    
    # Highlight Warning/Notification Box
    st.markdown(
        '<div style="background: rgba(218, 41, 28, 0.08); border: 1px solid rgba(255, 199, 44, 0.3); '
        'border-left: 4px solid var(--accent); padding: 14px 16px; border-radius: 10px; margin-bottom: 18px; '
        'box-shadow: 0 4px 15px rgba(218, 41, 28, 0.05);">'
        '<div style="font-size: 13.5px; font-weight: 700; color: var(--text-accent); display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">'
        '<span>⚠️</span> Escalation Phase: Ticket Submission'
        '</div>'
        '<div style="font-size: 12px; line-height: 1.4; color: #cbd5e1;">'
        'We have initialized a ServiceNow ticket escalation draft. Please verify the contact '
        'information and diagnostic details collected so far before finalizing the submission.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Card 1: Information Confirmation Card (in container with border=True to match wrapper styles)
    with st.container(border=True):
        st.markdown(
            f"<div class='confirm-header'>👤 Confirm Your Information</div>",
            unsafe_allow_html=True
        )
        
        # Name (non-modifiable)
        st.markdown("<div class='lock-lbl'>Name 🔒</div>", unsafe_allow_html=True)
        name_val = st.text_input("Name", value="Jim Halpert", disabled=True, label_visibility="collapsed", key="ticket_name")
        
        # Store # (non-modifiable)
        st.markdown("<div class='lock-lbl'>Store # 🔒</div>", unsafe_allow_html=True)
        store_val = st.text_input("Store #", value=display_store, disabled=True, label_visibility="collapsed", key="ticket_store_num")
        
        # Store Address (non-modifiable)
        st.markdown("<div class='lock-lbl'>Store Address 🔒</div>", unsafe_allow_html=True)
        address_val = st.text_input("Store Address", value=store_address, disabled=True, label_visibility="collapsed", key="ticket_store_address")
        
        # Phone Number (modifiable, present)
        st.markdown("<div class='lock-lbl' style='color:#f0f2f8; margin-top:10px;'>Phone Number</div>", unsafe_allow_html=True)
        phone_val = st.text_input("Phone Number", value="(570) 555-0142", label_visibility="collapsed", key="ticket_phone")
        
        st.markdown("<hr style='margin: 14px 0; border: none; border-top: 1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
        
        # Backup contact (optional)
        st.markdown("<div style='font-size:11px; font-weight:700; color:var(--text-accent); letter-spacing:0.8px; text-transform:uppercase; margin-bottom:8px;'>Backup Contact <span style='font-size:9px; color:#8892a4; text-transform:lowercase; font-weight:normal;'>optional</span></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='lock-lbl' style='color:#8892a4;'>Backup Name</div>", unsafe_allow_html=True)
        backup_name_val = st.text_input("Backup Name", placeholder="Backup contact name", label_visibility="collapsed", key="ticket_backup_name")
        
        st.markdown("<div class='lock-lbl' style='color:#8892a4;'>Backup Phone</div>", unsafe_allow_html=True)
        backup_phone_val = st.text_input("Backup Phone", placeholder="(555) 555-0100", label_visibility="collapsed", key="ticket_backup_phone")
        
    # Card 2: Manual Ticket Details (if manual flow is active)
    manual_flow = st.session_state.get("manual_ticket_flow", False)
    if manual_flow:
        # Pre-populate short description from first user message
        first_user_msg = ""
        for m in st.session_state.get("messages", []):
            if m["role"] == "user":
                first_user_msg = m["content"].replace("**", "").replace("`", "").strip()
                break

        with st.container(border=True):
            st.markdown(
                "<div class='confirm-header'>📝 Enter Ticket Details "
                "<span style='color: var(--accent); font-size: 13px; font-weight: 700; margin-left: 6px; "
                "letter-spacing: 0.5px;'>* REQUIRED</span></div>",
                unsafe_allow_html=True
            )
            st.markdown("<div class='lock-lbl' style='color:#f0f2f8; text-transform:none;'>Short Description</div>", unsafe_allow_html=True)
            st.text_input("Short Description", value=first_user_msg, label_visibility="collapsed", key="ticket_manual_short_desc")
            
            st.markdown(
                "<div class='lock-lbl' style='color:#f0f2f8; text-transform:none; margin-top:10px;'>"
                "Description <span style='font-size:10px; color:#8892a4; font-weight:400; text-transform:none;'>"
                "(Include details of issue, specific device info, etc)</span></div>",
                unsafe_allow_html=True
            )
            st.text_area("Description", placeholder="Describe the problem with as much detail as possible...", label_visibility="collapsed", key="ticket_manual_desc")

    # Card 3: Diagnostic Summary Card
    diag_summary = get_diagnostic_summary()
    
    st.markdown("<div class='diagnostic-card'>", unsafe_allow_html=True)
    current_section = "Triage Diagnostics"
    st.markdown(f"<div class='diagnostic-header'>{current_section}</div>", unsafe_allow_html=True)
    for q, a in diag_summary:
        if q == "__section__":
            # Close previous section and open a new one
            current_section = a
            st.markdown(
                f"<div class='diagnostic-header' style='margin-top:12px; padding-top:12px; "
                f"border-top: 1px solid rgba(255,255,255,0.06);'>{a}</div>",
                unsafe_allow_html=True
            )
            continue
        st.markdown(
            f"<div class='diagnostic-row'>"
            f"  <div class='diagnostic-key'>{q}</div>"
            f"  <div class='diagnostic-val'>{a}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Buttons with dynamic style markers
    st.markdown("<div class='submit-btn-marker'></div>", unsafe_allow_html=True)
    if st.button("🎫 Submit Support Ticket", use_container_width=True, key="submit_ticket_collection"):
        if manual_flow:
            m_s_desc = st.session_state.get("ticket_manual_short_desc", "").strip()
            m_desc = st.session_state.get("ticket_manual_desc", "").strip()
            if not m_s_desc or not m_desc:
                st.error("⚠️ Please enter both a Short Description and Description.")
                st.stop()
                
        import random
        # ServiceNow Ticket Number format: INC followed by 7 digits
        inc_num = f"INC{random.randint(1000000, 9999999)}"
        
        # Determine category and sub-category dynamically
        all_chat_text = " ".join([m["content"] for m in st.session_state.messages]).lower()
        summary_text = " ".join([f"{q} {a}" for q, a in diag_summary]).lower()
        combined_text = all_chat_text + " " + summary_text
        
        sub_category = "kiosk"
        if "printer" in combined_text or "print" in combined_text:
            sub_category = "printer"
        elif "pos" in combined_text or "terminal" in combined_text or "register" in combined_text:
            sub_category = "pos"
            
        category = "hardware"
        if any(kw in combined_text for kw in ["slow", "lagging", "crash", "network", "offline", "login", "password"]):
            category = "software"
            
        # Parse serial number if present in history or triage
        serial_val = "—"
        if st.session_state.get("triage_serial") and st.session_state.get("triage_serial") != "Unknown":
            serial_val = st.session_state.triage_serial
        else:
            for msg in st.session_state.messages:
                content = msg["content"]
                match = re.search(r"\b([A-Z0-9]{2,6}[-]?[\d]{5,12})\b", content)
                if match:
                    serial_val = match.group(1)
                    break
            
        # Compile structured dump
        diagnostic_dump_lines = []
        diagnostic_dump_lines.append("[Device Details]")
        diagnostic_dump_lines.append(f"- Store ID: {active_store}")
        diagnostic_dump_lines.append(f"- Location: {store_location}")
        diagnostic_dump_lines.append(f"- Reporter: Jim Halpert (Phone: {phone_val})")
        if st.session_state.get("triage_model"):
            diagnostic_dump_lines.append(f"- Device Model: {st.session_state.triage_model}")
        if st.session_state.get("triage_serial"):
            diagnostic_dump_lines.append(f"- Serial Number: {st.session_state.triage_serial}")
        if st.session_state.get("triage_priority"):
            diagnostic_dump_lines.append(f"- Calculated Priority: {st.session_state.triage_priority}")
            
        if backup_name_val.strip() or backup_phone_val.strip():
            b_name = backup_name_val.strip() if backup_name_val.strip() else "None specified"
            b_phone = backup_phone_val.strip() if backup_phone_val.strip() else "None specified"
            diagnostic_dump_lines.append(f"- Backup Contact: {b_name} (Phone: {b_phone})")

        # Split diag_summary into core triage rows vs troubleshooting Q&A rows
        diagnostic_dump_lines.append("\n[Triage Diagnostics]")
        in_ts_section = False
        ts_dump_lines = []
        for q, a in diag_summary:
            if q == "__section__":
                if a == "Troubleshooting Steps Attempted":
                    in_ts_section = True
                else:
                    in_ts_section = False
                continue
            if in_ts_section:
                ts_dump_lines.append(f"- {q}: {a}")
            else:
                diagnostic_dump_lines.append(f"- {q}: {a}")

        # Troubleshooting steps section
        diagnostic_dump_lines.append("\n[Troubleshooting Steps Attempted]")
        # First add the Q&A pairs captured from chat (visible paper, roller, OTP, etc.)
        diagnostic_dump_lines.extend(ts_dump_lines)
        # Then add any assistant-driven step summaries from chat messages
        step_index = len(ts_dump_lines) + 1
        for msg in st.session_state.messages:
            if msg["role"] == "assistant":
                content = msg["content"]
                if any(step_kw in content for step_kw in ["Reboot", "Power cycle", "Clear the Paper Jam", "Clean the Card Reader", "Restart"]):
                    step_title = "Step"
                    for line in content.split('\n'):
                        if any(kw in line for kw in ["Reboot", "Jam", "Cable", "Power", "Reader"]):
                            step_title = line.replace("**", "").replace("`", "").replace("##", "").strip()
                            break
                    diagnostic_dump_lines.append(f"{step_index}. {step_title} (Attempted) -> Outcome: Did not resolve the issue.")
                    step_index += 1

        if step_index == 1 and not ts_dump_lines:
            diagnostic_dump_lines.append("- No troubleshooting steps could be attempted or they were skipped.")
            
        diagnostic_dump_lines.append("\n[System Action]")
        diagnostic_dump_lines.append("- Escalated to ServiceNow via virtual assistant chat session.")
        
        if manual_flow:
            short_desc = st.session_state.get("ticket_manual_short_desc", "").strip()
            full_description_dump = st.session_state.get("ticket_manual_desc", "").strip()
        else:
            full_description_dump = "\n".join(diagnostic_dump_lines)
            
            # Build Short Description
            short_desc = f"{sub_category.capitalize()} triage escalation - {diag_summary[0][1] if diag_summary else 'Hardware issue'}"
            if len(short_desc) > 80:
                short_desc = short_desc[:77] + "..."
            
        # Store metadata
        meta = {
            "ticket_id": inc_num,
            "store_id": f"STORE-{active_store}",
            "asset_type": sub_category.upper(),
            "asset_serial_number": serial_val,
            "severity": "CRITICAL" if category == "software" else "HIGH",
            "sla_breach_minutes": 30 if category == "software" else 60,
            "routing_target": "Unisys RTS L1 - SD - US",
            "auto_dispatch": True,
            "channel": "chat",
            "description": full_description_dump,
            "short_description": short_desc,
            "category": category,
            "sub_category": sub_category,
            "caller": f"Jim Halpert, Store #{active_store}, {store_location}",
            "contact_type": "chat",
            "assignment_group": "Unisys RTS L1 - SD - US",
            "priority": st.session_state.get("triage_priority", "P3")
        }
        
        st.session_state.ticket_metadata = meta
        st.session_state.escalation_triggered = True
        
        # Compile response message for the chat history
        _ts = time.strftime("%H:%M")
        
        success_content = (
            f"🎫 **ServiceNow Ticket Generated Successfully!**\n\n"
            f"Below are the registered integration parameters sent to ServiceNow:\n\n"
            f"| ServiceNow Parameter | Registered Value |\n"
            f"| :--- | :--- |\n"
            f"| **Ticket Number** | `{meta['ticket_id']}` |\n"
            f"| **Priority** | `{meta.get('priority', 'P3')}` |\n"
            f"| **Caller** | `{meta['caller']}` |\n"
            f"| **Category** | `{meta['category']}` |\n"
            f"| **Sub-category** | `{meta['sub_category']}` |\n"
            f"| **Short Description** | `{meta['short_description']}` |\n"
            f"| **Contact Type** | `{meta['contact_type']}` |\n"
            f"| **Assignment Group** | `{meta['assignment_group']}` |\n\n"
            f"#### 📝 Full Description & Diagnostic Dump:\n"
            f"```text\n"
            f"{meta['description']}\n"
            f"```\n\n"
            f"An engineer from the **{meta['assignment_group']}** group has been dispatched and is reviewing this ticket."
        )
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": success_content,
            "timestamp": _ts,
            "blocked": False
        })
        
        st.session_state.ticket_collection_active = False
        st.session_state.show_restart_chat_btn = True
        st.rerun()
        
    st.markdown("<div class='cancel-btn-marker'></div>", unsafe_allow_html=True)
    if st.button("← Start Over", use_container_width=True, key="cancel_ticket_collection"):
        st.session_state.messages = []
        st.session_state.ticket_metadata = None
        st.session_state.escalation_triggered = False
        st.session_state.ticket_collection_active = False
        st.session_state.rag_hits = {}
        st.session_state.manual_ticket_flow = False
        st.session_state.show_restart_chat_btn = False
        reset_triage_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar – ticket metadata panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 16px 0 12px;'>
          <div style='font-size:28px; margin-bottom:6px;'>🎫</div>
          <div style='font-size:15px; font-weight:700; color:#f0f2f8;'>Dispatch Console</div>
          <div style='font-size:11px; color:#8892a4;'>ServiceNow · Genesys Routing</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if st.session_state.ticket_metadata:
        meta = st.session_state.ticket_metadata
        sev = meta.get("severity", "MEDIUM")
        sev_class = f"sev-{sev.lower()}"

        st.markdown(
            f"""
            <div style='font-size:11px; font-weight:600; color:var(--text-accent); text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:8px;'>
              ● Live Ticket Generated
            </div>
            """,
            unsafe_allow_html=True,
        )

        prio = meta.get("priority", "—")
        prio_class = "sev-critical" if prio == "P1" else "sev-high" if prio == "P2" else "sev-medium" if prio == "P3" else "sev-low" if prio == "P4" else ""

        fields = [
            ("Ticket ID",      meta.get("ticket_id", "—"),             ""),
            ("Store ID",       meta.get("store_id", "—"),               ""),
            ("Asset Type",     meta.get("asset_type", "—"),             ""),
            ("Serial Number",  meta.get("asset_serial_number", "—"),    ""),
            ("Priority",       prio,                                    prio_class),
            ("Severity",       sev,                                      sev_class),
            ("SLA Breach",     f"{meta.get('sla_breach_minutes')} min", ""),
            ("Routing",        meta.get("routing_target", "—"),         ""),
            ("Auto Dispatch",  "✅ YES" if meta.get("auto_dispatch") else "⏸ NO", ""),
            ("Channel",        meta.get("channel", "—"),                ""),
        ]

        rows_html = ""
        for label, value, extra_class in fields:
            rows_html += f"""
            <div class="ticket-field">
              <span class="ticket-label">{label}</span>
              <span class="ticket-value {extra_class}">{value}</span>
            </div>"""

        st.markdown(
            f'<div class="ticket-card">{rows_html}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Description**")
        st.caption(meta.get("description", "—")[:200])

        st.divider()

        st.markdown("**Raw Payload (JSON)**")
        st.code(json.dumps(meta, indent=2), language="json")

        if st.button("🔄 Clear Ticket"):
            st.session_state.ticket_metadata = None
            st.session_state.escalation_triggered = False
            reset_triage_state()
            st.rerun()

    else:
        st.markdown(
            """
            <div style='text-align:center; padding:32px 12px; color:#8892a4;'>
              <div style='font-size:36px; margin-bottom:10px;'>📡</div>
              <div style='font-size:13px; font-weight:500;'>No active ticket</div>
              <div style='font-size:11px; margin-top:6px; line-height:1.5;'>
                Say "agent", "ticket", or "escalate"<br>to trigger zero-touch routing.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Quick action buttons
    st.markdown(
        "<div style='font-size:11px; font-weight:600; color:#8892a4; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Quick Actions</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.ticket_metadata = None
            st.session_state.escalation_triggered = False
            st.session_state.rag_hits = {}
            st.session_state.ticket_collection_active = False
            st.session_state.manual_ticket_flow = False
            reset_triage_state()
            st.rerun()
    with col2:
        if st.button("🎫 Force Ticket", use_container_width=True):
            ask_case_flow_options("Force Ticket")

    # Connection status
    st.divider()
    st.markdown(
        """
        <div style='font-size:10px; color:#8892a4; text-align:center; line-height:1.6;'>
          <span style='color:#22d3a0;'>●</span> Vertex AI Connected<br>
          <span style='color:#22d3a0;'>●</span> Guardrails Active<br>
          <span style='color:#22d3a0;'>●</span> RAG Grounding On
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div id="header-sentinel"></div>', unsafe_allow_html=True)
h_col1, h_col2, h_col3, h_col4 = st.columns([1.3, 1.9, 1.1, 0.8], gap="small")

with h_col1:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:8px; height: 100%; margin-top: 4px;">
            <div style="font-size:20px; background:linear-gradient(135deg, var(--accent) 0%, var(--text-accent) 100%); width:32px; height:32px; border-radius:6px; display:flex; align-items:center; justify-content:center; box-shadow: 0 2px 8px var(--accent-glow); flex-shrink:0;">🍔</div>
            <div style="min-width:0;">
                <div style="font-size:22px; font-weight:800; color:#f0f2f8; line-height:1.0; letter-spacing:-0.5px; padding-bottom: 2px;">chipLLM</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with h_col2:
    selected_store = st.selectbox(
        "Header Store select",
        options=STORE_OPTIONS,
        index=STORE_OPTIONS.index(st.session_state.active_store) if st.session_state.active_store in STORE_OPTIONS else 0,
        format_func=lambda x: f"Store: {x}",
        label_visibility="collapsed",
        key="header_store_selector"
    )
    if selected_store != st.session_state.active_store:
        st.session_state.active_store = selected_store
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"🔄 **Switched active context to Store #{selected_store}.** How can I assist you with this store?",
            "timestamp": time.strftime("%H:%M"),
            "blocked": False
        })
        st.rerun()

with h_col3:
    if st.button("🎫 Open Case", use_container_width=True, key="btn_header_case"):
        ask_case_flow_options("Open Case")

with h_col4:
    if st.button("Live Agent", use_container_width=True, key="btn_header_agent"):
        trigger_live_agent_flow("Live Agent")

# ---------------------------------------------------------------------------
# Welcome message (injected once)
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hello! I am chipLLM, your restaurant tech support virtual engineer. "
            "I'll help you troubleshoot on-site issues and if we can't resolve it here, "
            "I'll create a support ticket or get you over to a live chat agent.\n\n"
            "Please describe your issue."
        ),
        "timestamp": time.strftime("%H:%M"),
        "blocked": False
    })

# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    avatar = "👤" if role == "user" else "🤖"
    
    with st.chat_message(role, avatar=avatar):
        # RAG grounding badge above bot responses
        if role == "assistant" and i in st.session_state.rag_hits:
            st.markdown(
                f'<div class="rag-badge">📚 KB: {st.session_state.rag_hits[i]}</div>',
                unsafe_allow_html=True,
            )

        if msg.get("blocked", False):
            st.markdown(
                f'<div style="color:#fda4af;border-left:3px solid #f43f5e;'
                f'padding-left:10px;">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            display_content = msg["content"]
            if role == "assistant":
                display_content = clean_assistant_message(display_content)
            st.markdown(display_content, unsafe_allow_html=True)

        st.markdown(f'<div class="msg-time">{msg.get("timestamp", "")}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Live Agent Franklin connection gate
# ---------------------------------------------------------------------------
if st.session_state.get("live_agent_pending_connection", False):
    st.session_state.live_agent_pending_connection = False
    time.sleep(2)
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello, my name is Franklin. I am reviewing your case now.",
        "timestamp": time.strftime("%H:%M"),
        "blocked": False
    })
    st.rerun()


# ---------------------------------------------------------------------------
# Restart Chat button (shown after ticket submission)
# ---------------------------------------------------------------------------
if st.session_state.get("show_restart_chat_btn", False) and not st.session_state.ticket_collection_active:
    st.markdown(
        """
        <style>
        div.restart-btn-marker + div.stButton > button {
            background: linear-gradient(135deg, #1a1d28 0%, #1f2333 100%) !important;
            color: var(--text-accent) !important;
            border: 1.5px solid rgba(255, 199, 44, 0.4) !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            height: 46px !important;
            border-radius: 12px !important;
            margin-top: 8px !important;
            margin-bottom: 16px !important;
            letter-spacing: 0.3px !important;
            transition: all 0.2s ease !important;
        }
        div.restart-btn-marker + div.stButton > button:hover {
            background: rgba(255, 199, 44, 0.08) !important;
            border-color: rgba(255, 199, 44, 0.8) !important;
            transform: translateY(-1px) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='restart-btn-marker'></div>", unsafe_allow_html=True)
    if st.button("🔄 Click Here to Restart Chat", use_container_width=True, key="restart_chat_after_ticket"):
        st.session_state.messages = []
        st.session_state.ticket_metadata = None
        st.session_state.escalation_triggered = False
        st.session_state.ticket_collection_active = False
        st.session_state.rag_hits = {}
        st.session_state.manual_ticket_flow = False
        st.session_state.show_restart_chat_btn = False
        reset_triage_state()
        st.rerun()

# ---------------------------------------------------------------------------
# Ticket Form Render Gate
# ---------------------------------------------------------------------------

if st.session_state.ticket_collection_active:
    is_live_agent = st.session_state.get("ticket_collection_is_agent", False)
    render_ticket_collection_form(is_live_agent=is_live_agent)

# ---------------------------------------------------------------------------
# Device Detail Mini-Form (printer escalation: optional model/serial before priority)
# ---------------------------------------------------------------------------

_triage_active = st.session_state.get("escalation_triage_active", False)
_triage_step = st.session_state.get("escalation_triage_step")

if _triage_active and _triage_step == "device_form":
    render_device_detail_form()

# ---------------------------------------------------------------------------
# Suggestion Chips Rendering Block
# ---------------------------------------------------------------------------
if "suggestion_click" not in st.session_state:
    st.session_state.suggestion_click = None

# Suppress chips during device_form step (the mini-form replaces them)
_show_chips = (
    not st.session_state.ticket_collection_active
    and _triage_step != "device_form"
)
    last_msg = st.session_state.messages[-1] if st.session_state.messages else None
    if last_msg and last_msg["role"] == "assistant":
        choices = get_choices_from_message(last_msg["content"])
        if choices:
            st.markdown("<div class='chips-sentinel'></div>", unsafe_allow_html=True)
            # Make sure we use proper flex columns so they sit on a single line
            cols = st.columns(len(choices))
            for idx, (col, choice) in enumerate(zip(cols, choices)):
                with col:
                    if st.button(choice, key=f"chip_{choice}_{idx}", use_container_width=True):
                        st.session_state.suggestion_click = choice
                        st.rerun()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

chat_placeholder = "Describe the issue… (e.g. 'Printer jammed')" if len(st.session_state.messages) <= 1 else "Type here"

chat_val = st.chat_input(
    placeholder=chat_placeholder,
    key="chat_input",
)

user_input = None
if st.session_state.suggestion_click:
    user_input = st.session_state.suggestion_click
    st.session_state.suggestion_click = None
elif chat_val:
    user_input = chat_val

if user_input:
    ts_now = time.strftime("%H:%M")
    lower_input = user_input.lower()
    
    # Intercept commands like "start over"
    if "start over" in lower_input or "restart" in lower_input:
        st.session_state.messages = []
        st.session_state.ticket_metadata = None
        st.session_state.escalation_triggered = False
        st.session_state.ticket_collection_active = False
        st.session_state.rag_hits = {}
        st.session_state.manual_ticket_flow = False
        reset_triage_state()
        st.rerun()
        
    # Intercept sequential triage state machine
    elif st.session_state.get("escalation_triage_active", False):
        step = st.session_state.get("escalation_triage_step")
        
        # Append user's choice to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": ts_now,
            "blocked": False
        })
        
        if step == "device_form":
            # User typed in chat while device form was shown — treat as skip
            st.session_state.escalation_triage_step = "q1"
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": ts_now,
                "blocked": False
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "To help determine priority for this ticket, let's dig in on impact.\n\n"
                    "Is this issue completely stopping your store from taking orders or payments right now?"
                ),
                "timestamp": ts_now,
                "blocked": False
            })
            st.rerun()

        elif step == "model":
            st.session_state.triage_model = user_input
            st.session_state.escalation_triage_step = "serial"
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Got it. What is the device serial number?",
                "timestamp": ts_now,
                "blocked": False
            })
            st.rerun()
            
        elif step == "serial":
            st.session_state.triage_serial = user_input
            st.session_state.escalation_triage_step = "q1"
            st.session_state.messages.append({
                "role": "assistant",
                "content": "To help determine priority for this ticket, let's dig in on impact.\n\nIs this issue completely stopping your store from taking orders or payments right now?",
                "timestamp": ts_now,
                "blocked": False
            })
            st.rerun()
            
        elif step == "q1":
            st.session_state.triage_q1 = user_input
            st.session_state.escalation_triage_step = "q2"
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Is this the only device of its kind in your store?",
                "timestamp": ts_now,
                "blocked": False
            })
            st.rerun()
            
        elif step == "q2":
            st.session_state.triage_q2 = user_input
            if "multiple" in user_input.lower():
                st.session_state.escalation_triage_step = "q3"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Are the other devices of this kind online and functional?",
                    "timestamp": ts_now,
                    "blocked": False
                })
            else:
                calculate_priority_and_finish_triage()
            st.rerun()
            
        elif step == "q3":
            st.session_state.triage_q3 = user_input
            calculate_priority_and_finish_triage()
            st.rerun()
        
    # Intercept live agent keywords
    elif any(kw in lower_input for kw in ["live agent", "human", "agent", "talk to an agent", "talk to a human"]):
        trigger_live_agent_flow(user_input)
        
    # Intercept case routing choices
    elif "manual ticket flow" in lower_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": ts_now,
            "blocked": False
        })
        st.session_state.manual_ticket_flow = True
        st.session_state.ticket_collection_active = True
        st.session_state.ticket_collection_is_agent = False
        st.rerun()
        
    elif "continue with automated flow" in lower_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": ts_now,
            "blocked": False
        })
        st.session_state.manual_ticket_flow = False
        st.session_state.ticket_collection_active = False
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Understood! Let's continue with the automated troubleshooting. Please describe your issue.",
            "timestamp": ts_now,
            "blocked": False
        })
        st.rerun()
        
    # Intercept manual ticket escalation keywords
    elif "escalate" in lower_input or "ticket" in lower_input or "open case" in lower_input:
        ask_case_flow_options(user_input)
        
    else:
        # Standard AI chat logic
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
                "timestamp": ts_now,
                "blocked": False,
            }
        )

        # --- Layer 1: Guardrails -----------------------------------------------
        guard = check_guardrails(user_input)

        if guard.blocked:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": guard.refusal_text,
                    "timestamp": ts_now,
                    "blocked": True,
                }
            )
            st.rerun()

        # --- Layer 2: Escalation keyword check ---------------------------------
        if guard.escalation_triggered:
            start_escalation_triage()
            st.rerun()

        # --- Layer 3: Dynamic HTML RAG retrieval -------------------------------
        relevant_playbooks = retrieve_context(user_input, top_k=1)
        rag_context: str | None = None
        rag_title: str | None = None

        if relevant_playbooks:
            playbook = relevant_playbooks[0]
            rag_context = playbook["content"]
            rag_title = playbook["title"]
        else:
            # No RAG match — only route to manual ticket on the FIRST user message
            # (mid-flow replies like "No" or "yes" should not trigger this)
            num_user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
            if num_user_msgs <= 1:
                _ts_no_rag = time.strftime("%H:%M")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        "I wasn't able to find a matching troubleshooting playbook for that issue in our knowledge base. "
                        "Let me open a support ticket so our team can assist you directly."
                    ),
                    "timestamp": _ts_no_rag,
                    "blocked": False,
                })
                st.session_state.manual_ticket_flow = True
                st.session_state.ticket_collection_active = True
                st.session_state.ticket_collection_is_agent = False
                st.rerun()
            # else: fall through to LLM call with no RAG context

        # --- Layer 4: LLM call (streaming) -------------------------------------
        client = ChipLLMClient()

        with st.chat_message("assistant", avatar="🤖"):
            full_response = ""
            response_placeholder = st.empty()

            try:
                for chunk in client.stream_response(
                    messages=st.session_state.messages,
                    rag_context=rag_context,
                ):
                    full_response += chunk
                    response_placeholder.markdown(f"{full_response}▌")

                response_placeholder.markdown(full_response)

            except Exception as exc:
                full_response = (
                    f"⚠️ **LLM connectivity issue:** `{exc}`\n\n"
                    "Verify that ADC is active (`gcloud auth application-default login`) "
                    "and that `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` are set correctly."
                )

        # --- Store assistant response -----------------------------------------
        msg_idx = len(st.session_state.messages)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "timestamp": ts_now,
                "blocked": False,
            }
        )

        # Tag with RAG hit
        if rag_title:
            st.session_state.rag_hits[msg_idx] = rag_title

        # --- Post-response: check if response triggers escalation -------------
        if not st.session_state.ticket_collection_active and not st.session_state.get("escalation_triage_active", False):
            post_guard = check_guardrails(full_response)
            if post_guard.escalation_triggered:
                start_escalation_triage(append_welcome=False)

        st.rerun()
