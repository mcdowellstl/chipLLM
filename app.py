"""
app.py
------
chipLLM – Mobile-focused Streamlit chat interface.
Vertex AI backend · In-memory RAG · Domain guardrails · Ticket metadata sidebar.
"""

from __future__ import annotations

import json
import time
import logging
import sys

# Force configure logging to write to sys.stdout with a clear custom format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger("chipLLM")

import streamlit as st

from guardrails import check_guardrails, is_greeting_or_small_talk, is_cafe_issue, is_ticket_status_lookup_intent
from knowledge_base import (
    retrieve_context,
    get_file_list,
    fetch_file_content,
    build_playbook_pool,
    L0_BUCKET,
    ADHOC_BUCKET,
)
import knowledge_base as _kb
from llm_client import ChipLLMClient, extract_ticket_metadata, set_alert_visibility


@st.cache_data(ttl=300)
def _cached_fetch_file_content(bucket_name: str, blob_name: str):
    """Streamlit-cached wrapper around the st-free knowledge_base fetcher."""
    return fetch_file_content(bucket_name, blob_name)


_KB_REFRESH_INTERVAL = 300  # seconds (5 minutes)


def refresh_playbook_pool(*, force: bool = False) -> list[dict]:
    """
    Returns the current playbook pool, refreshing from GCS only when needed:
      - First call in this session (no pool in session_state yet)
      - More than _KB_REFRESH_INTERVAL seconds since the last GCS scan
      - force=True (triggered by /nuke-knowledge)

    Between refreshes the session_state pool is returned immediately with
    zero network calls, keeping the chat input fully responsive.
    """
    now = time.time()
    last_loaded = st.session_state.get("kb_loaded_at", 0)
    pool_exists = bool(st.session_state.get("knowledge_base"))

    if force or not pool_exists or (now - last_loaded) >= _KB_REFRESH_INTERVAL:
        logger.info("Refreshing playbook pool from GCS (force=%s).", force)
        l0_names   = get_file_list(L0_BUCKET)
        adhoc_names = get_file_list(ADHOC_BUCKET)
        pool = build_playbook_pool(l0_names, adhoc_names, _cached_fetch_file_content)
        if pool:
            st.session_state.knowledge_base = pool
            st.session_state.kb_loaded_at   = now
            logger.info("Playbook pool refreshed: %d entries.", len(pool))
        else:
            # GCS unreachable — keep whatever we have (or fall back to module-init)
            if not pool_exists:
                st.session_state.knowledge_base = _kb.PLAYBOOKS
                st.session_state.kb_loaded_at   = now

    return st.session_state.get("knowledge_base", _kb.PLAYBOOKS)


import re as _re_guardrail

_TICKET_CREATION_PATTERNS = [
    # Direct create/open/file/raise/submit + ticket/case/incident
    r"\b(open|create|file|raise|submit|start|log|make)\s+(a\s+|an\s+|new\s+)*(support\s+)?(ticket|case|incident|report)\b",
    # "need to open / need a ticket / need a case"
    r"\bneed\s+(to\s+)?(open|create|file|raise|a|an)\s+(support\s+)?(ticket|case|incident)\b",
    # "get a ticket / get a case opened"
    r"\bget\s+(a\s+|an\s+)?(new\s+)?(support\s+)?(ticket|case|incident)\b",
    # Bare phrases
    r"\b(new\s+)?(support\s+)ticket\b",
    r"\bopen\s+(a\s+)?case\b",
    r"\bnew\s+case\b",
]

def _is_ticket_creation_intent(text: str) -> bool:
    """
    Returns True if the user's message is clearly requesting ticket/case creation.
    Catches natural phrasing that would otherwise reach the LLM and trigger the
    hallucination that ChipLLM cannot create support tickets.
    """
    for pattern in _TICKET_CREATION_PATTERNS:
        if _re_guardrail.search(pattern, text):
            return True
    return False

def check_and_update_mim_state():
    """
    Checks if there are active MIMs, initializes the dismissal & last seen states,
    and resets the user's dismissal state if the backend metadata of an active MIM has changed.
    """
    if "mim_dismissed" not in st.session_state:
        st.session_state.mim_dismissed = {}
    if "mim_last_seen_metadata" not in st.session_state:
        st.session_state.mim_last_seen_metadata = {}
        
    active_mims = st.session_state.get("active_mims", [])
    for mim in active_mims:
        norm_id = mim["number"].upper().strip()
        current_metadata = {
            "short_description": mim.get("short_description", ""),
            "description": mim.get("description", ""),
            "priority": mim.get("priority", "")
        }
        
        last_seen = st.session_state.mim_last_seen_metadata.get(norm_id)
        if last_seen:
            # If any of the metadata has changed, override the dismissal state
            if (last_seen.get("short_description") != current_metadata["short_description"] or
                last_seen.get("description") != current_metadata["description"] or
                last_seen.get("priority") != current_metadata["priority"]):
                
                st.session_state.mim_dismissed[norm_id] = False
                logger.info("Incident %s metadata updated on the backend. Overriding user dismissal state.", norm_id)
                
        # Save or update the last seen metadata
        st.session_state.mim_last_seen_metadata[norm_id] = current_metadata

# ---------------------------------------------------------------------------
# Store authentication config
# ---------------------------------------------------------------------------

DEFAULT_STORE  = "67067"               # pre-selected default
STORE_OPTIONS  = ["67067", "67068", "67069"]  # all authorized stores

# Authenticated user identity — single source of truth for display name and case authorship
CURRENT_USER_NAME = "Jim Halpert"

# ---------------------------------------------------------------------------
# L0_KA Playbook Directory (Dynamic RAG)
# ---------------------------------------------------------------------------
# Handled in knowledge_base.py

# ---------------------------------------------------------------------------
# Page config – must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ChipLLM · Restaurant Tech Support",
    page_icon="🍔",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS – mobile viewport, dark kitchen theme, micro-animations
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* ── Google Font ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Completely hide the sidebar and collapsed control arrow */
[data-testid="stSidebar"] {
  display: none !important;
}
[data-testid="collapsedControl"] {
  display: none !important;
}

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
  padding-top: 68px !important;   /* single header row: ~52px + 16px buffer */
  padding-bottom: 80px !important;
}

/* ── Sticky Header bar via sentinel ─────────────────────────────────────── */
/* One unified row: 🍔 ChipLLM | store selector | active-cases button */
[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] {
  position: fixed !important;
  top: 0 !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  width: 100% !important;
  max-width: 420px !important;
  background: #0f111a !important;
  border-bottom: 1px solid rgba(255,255,255,.08) !important;
  padding: 8px 14px !important;
  z-index: 9999 !important;
  box-shadow: 0 2px 20px rgba(0,0,0,.55) !important;
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
}

/* Column sizing inside header: logo col is auto/small, rest split equally */
[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) {
  flex: 0 0 auto !important;
  min-width: 0 !important;
}
[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) {
  flex: 1 1 0 !important;
  min-width: 0 !important;
}
[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) {
  flex: 1 1 0 !important;
  min-width: 0 !important;
}

/* Brand logo + wordmark in the first col */
.header-brand {
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
  padding: 0 4px;
}

.header-logo {
  width: 38px;
  height: 38px;
  background: linear-gradient(135deg, #da291c 0%, #ffc72c 100%);
  border-radius: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  box-shadow: 0 2px 10px rgba(218,41,28,.45);
  flex-shrink: 0;
}

.header-title {
  font-size: 18px;
  font-weight: 800;
  color: #f0f2f8;
  letter-spacing: -0.4px;
  line-height: 1;
}

/* Store selectbox in header */
[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] > div > div {
  background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.13) !important;
  color: #f0f2f8 !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  min-height: 36px !important;
  height: 36px !important;
  border-radius: 8px !important;
  line-height: 1 !important;
  display: flex !important;
  align-items: center !important;
  padding: 0 8px !important;
  white-space: nowrap !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] > div > div:hover {
  border-color: rgba(255,199,44,.5) !important;
  background: rgba(255,255,255,.09) !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] svg {
  fill: #8892a4 !important;
  width: 13px !important;
  height: 13px !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] [data-baseweb="select"] {
  height: 36px !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] [class*="ValueContainer"] {
  padding: 0 4px !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
  padding-left: 8px !important;
  padding-right: 24px !important;
}

/* Active Cases button in header */
[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] .stButton > button {
  background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.13) !important;
  color: #f0f2f8 !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  padding: 0 10px !important;
  min-height: 36px !important;
  height: 36px !important;
  border-radius: 8px !important;
  line-height: 1 !important;
  letter-spacing: .1px !important;
  transition: background .15s, border-color .15s, color .15s !important;
  white-space: nowrap !important;
  margin-top: 0 !important;
  width: 100% !important;
}

[data-testid="stMarkdownContainer"]:has(#header-sentinel)
  + [data-testid="stHorizontalBlock"] .stButton > button:hover {
  background: rgba(255,255,255,.11) !important;
  border-color: rgba(255,199,44,.4) !important;
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

/* Metric button style override — kept for legacy fallback, header CSS above wins */
div[data-testid="element-container"]:has(div.metric-btn-marker) + div[data-testid="element-container"] button {
  background: rgba(255, 255, 255, 0.06) !important;
  border: 1px solid rgba(255, 255, 255, 0.13) !important;
  color: #f0f2f8 !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  min-height: 36px !important;
  height: 36px !important;
  border-radius: 8px !important;
  white-space: nowrap !important;
  transition: all 0.2s ease !important;
}

/* ── Active Outage banner ──────────────────────────────────────────────────── */
/* The whole card uses st.container(border=True); we just style the inner content */
.outage-banner-title {
  font-size: 12.5px;
  font-weight: 700;
  color: #f5c842;
  line-height: 1.3;
  margin-bottom: 4px;
}

.outage-banner-desc {
  font-size: 11.5px;
  color: #8892a4;
  line-height: 1.4;
}

/* Outage card container — use border=True in Python */
div[data-testid="stVerticalBlockBorderWrapper"]:has(div.active-outage-box) {
  border: 1px solid rgba(255, 180, 0, 0.22) !important;
  background: rgba(255, 160, 0, 0.04) !important;
  border-radius: 10px !important;
  padding: 10px 12px !important;
  margin-bottom: 8px !important;
  box-shadow: none !important;
}

/* Both Affected + Dismiss buttons inside the outage card: small gray pills */
div[data-testid="stVerticalBlockBorderWrapper"]:has(div.active-outage-box) button {
  background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.14) !important;
  color: #9aa0b2 !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  min-height: 26px !important;
  height: 26px !important;
  border-radius: 6px !important;
  padding: 0 9px !important;
  box-shadow: none !important;
  letter-spacing: 0.2px !important;
  white-space: nowrap !important;
  width: 100% !important;
  transition: all 0.15s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(div.active-outage-box) button:hover {
  background: rgba(255,255,255,.10) !important;
  color: #d0d4e0 !important;
  border-color: rgba(255,255,255,.22) !important;
  transform: none !important;
  box-shadow: none !important;
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

/* ── Recommended Troubleshooting Banner Button Styling ───────────────────── */
/* Style the button container */
div[data-testid="stMarkdownContainer"]:has(.recommended-troubleshooting-banner) + div[data-testid="stButton"] {
  position: relative !important;
  margin-top: -38px !important; /* Move the button up into the banner space */
  margin-bottom: 22px !important; /* Offset the negative margin to prevent overlap with elements below */
  padding-left: 14px !important;
  z-index: 10 !important;
  display: block !important;
}

/* Style the button itself to make it small, premium, and neat */
div[data-testid="stMarkdownContainer"]:has(.recommended-troubleshooting-banner) + div[data-testid="stButton"] button {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%) !important;
  color: white !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  padding: 4px 10px !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  border-radius: 4px !important;
  height: 24px !important;
  min-height: 24px !important;
  line-height: 1 !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 5px rgba(0,0,0,0.25) !important;
  width: auto !important;
  text-transform: none !important;
}

div[data-testid="stMarkdownContainer"]:has(.recommended-troubleshooting-banner) + div[data-testid="stButton"] button:hover {
  background: rgba(255, 199, 44, 0.2) !important;
  border-color: #ffc72c !important;
  color: #ffc72c !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 8px rgba(0,0,0,0.35) !important;
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

if st.session_state.get("needs_reset", False):
    st.session_state.needs_reset = False
    st.session_state.messages = []
    st.session_state.ticket_metadata = None
    st.session_state.escalation_triggered = False
    st.session_state.rag_hits = {}
    st.session_state.ticket_collection_active = False
    st.session_state.manual_ticket_flow = False
    st.session_state.show_restart_chat_btn = False
    st.session_state.escalation_triage_active = False
    st.session_state.escalation_triage_step = None
    st.session_state.triage_model = None
    st.session_state.triage_serial = None
    st.session_state.triage_q1 = None
    st.session_state.triage_q2 = None
    st.session_state.triage_q3 = None
    st.session_state.triage_priority = None
    st.session_state.current_triage = {}
    st.session_state.escalated = False
    st.session_state.escalation_stage = None
    st.session_state.live_agent_chat_turns = 0
    st.session_state.live_agent_pending_connection = False
    st.session_state.live_agent_pending_question = False
    
    # Also delete widget key states directly so they are completely fresh on reload
    widget_keys = [
        "ticket_extra_info", "ticket_manual_short_desc", "ticket_manual_desc", 
        "ticket_name", "ticket_phone", "ticket_backup_name", "ticket_backup_phone",
        "priority_form_q1", "priority_form_q2", "priority_form_q3", 
        "device_form_model_input", "device_form_serial_input",
        "escalation_user_notes_input", "escalation_image_upload", "ticket_image_upload"
    ]
    for key in widget_keys:
        if key in st.session_state:
            del st.session_state[key]

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

# ---------------------------------------------------------------------------
# One-time knowledge base load per session
# ---------------------------------------------------------------------------
if "knowledge_base" not in st.session_state:
    logger.info("Session KB cold-start: loading playbook pool from GCS.")
    refresh_playbook_pool()  # populates session_state.knowledge_base + kb_loaded_at
if "active_store" not in st.session_state:
    st.session_state.active_store = DEFAULT_STORE
    logger.info("Session initialized. Default store set to %s.", st.session_state.active_store)
# Track the last case the user viewed/discussed so mutation tools have a target
if "active_case_id" not in st.session_state:
    st.session_state.active_case_id = None
# Authenticated user identity — available to tool functions via session state
if "current_user" not in st.session_state:
    st.session_state.current_user = CURRENT_USER_NAME
if "active_tickets" not in st.session_state:
    from mocks.servicenow import get_store_tickets
    st.session_state.active_tickets = get_store_tickets(st.session_state.active_store)
    logger.info("Loaded default store tickets: %d tickets.", len(st.session_state.active_tickets))
if "active_mims" not in st.session_state:
    from mocks.servicenow import get_store_mims
    st.session_state.active_mims = get_store_mims(st.session_state.active_store)
    logger.info("Loaded default store MIMs: %d MIMs.", len(st.session_state.active_mims))
check_and_update_mim_state()
if "ticket_collection_active" not in st.session_state:
    st.session_state.ticket_collection_active = False
if "ticket_collection_is_agent" not in st.session_state:
    st.session_state.ticket_collection_is_agent = False

if "live_agent_pending_connection" not in st.session_state:
    st.session_state.live_agent_pending_connection = False

if "live_agent_name" not in st.session_state:
    import random
    st.session_state.live_agent_name = random.choice(["Alex", "Taylor", "Jordan", "Morgan", "Casey", "Robin", "Pat", "Jamie", "Sam", "Chris"])

if "live_agent_chat_turns" not in st.session_state:
    st.session_state.live_agent_chat_turns = 0

if "escalated" not in st.session_state:
    st.session_state.escalated = False

if "escalation_stage" not in st.session_state:
    st.session_state.escalation_stage = None

if "current_triage" not in st.session_state:
    st.session_state.current_triage = {}

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

def file_to_base64_data_url(uploaded_file) -> str | None:
    """Converts a Streamlit uploaded file into a Base64 data URL."""
    if uploaded_file is None:
        return None
    import base64
    try:
        file_bytes = uploaded_file.read()
        # Reset pointer in case it needs to be read again
        uploaded_file.seek(0)
        encoded = base64.b64encode(file_bytes).decode("utf-8")
        mime_type = uploaded_file.type if uploaded_file.type else "image/png"
        return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        logger.exception("Failed to convert file to base64 data URL")
        return None

# ---------------------------------------------------------------------------
# Automatic Triage Synchronization
# ---------------------------------------------------------------------------

def get_current_flow_messages() -> list:
    """
    Returns messages in st.session_state.messages starting AFTER the last
    ServiceNow Ticket Generated Successfully marker message, to isolate the current triage flow.
    """
    messages = st.session_state.get("messages", [])
    if not messages:
        return []
    last_idx = -1
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant" and "ServiceNow Ticket Generated Successfully!" in msg.get("content", ""):
            last_idx = idx
    if last_idx != -1:
        return messages[last_idx + 1:]
    return messages

def sync_triage_from_messages() -> None:
    """
    Parses conversation history on every rerun to sync conversational triage fields
    into st.session_state.current_triage without overwriting explicit form fields.
    """
    import re
    if "current_triage" not in st.session_state:
        st.session_state.current_triage = {}
        
    messages = get_current_flow_messages()
    if not messages:
        return

    # Check if we need to call AI to generate/update the issue description
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    if user_msgs:
        last_count = st.session_state.get("last_message_count_for_ai_desc", -1)
        if len(messages) != last_count or "ai_issue_description" not in st.session_state:
            try:
                client = ChipLLMClient()
                desc = client.generate_issue_description(messages)
                st.session_state.ai_issue_description = desc
                st.session_state.last_message_count_for_ai_desc = len(messages)
            except Exception as e:
                logger.exception("Failed to generate AI issue description")
                if "ai_issue_description" not in st.session_state:
                    st.session_state.ai_issue_description = "Unknown"
    else:
        st.session_state.ai_issue_description = "Unknown"
        
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
                elif any(dt_kw in ans_clean.lower() for dt_kw in ["drive through", "drive-thru", "dt"]):
                    t_val = "Drive-Thru"
                summary_dict["Device Type"] = t_val
                
            # 2. Number detection
            elif any(kw in q_content for kw in ["which pos number", "which device number", "which kiosk number", "which unit", "device number", "unit number"]):
                current_type = summary_dict.get("Device Type", "Unknown")
                num_only = re.sub(r"\D", "", ans_clean)
                if num_only:
                    if current_type != "Unknown":
                        summary_dict["Asset Number"] = f"{current_type}{num_only}"
                    else:
                        summary_dict["Asset Number"] = num_only
                else:
                    summary_dict["Asset Number"] = ans_clean
                    
            # 3. What is the issue
            elif any(kw in q_content for kw in ["describe the issue", "what is the issue", "what symptom", "experiencing"]):
                summary_dict["Issue Description"] = ans_clean.capitalize()
                
            # 4. Is there visible paper
            elif "visible paper" in q_content:
                summary_dict["Is there visible paper"] = ans_clean.capitalize()
                
            # 5. Internal Jam Roller C
            elif (any(re.search(rf"\b{re.escape(kw)}\b", q_content) for kw in ["internal jam roller", "roller c"]) or
                  (re.search(r"\broller\b", q_content) and ("printer" in q_content or "paper" in q_content or "jam" in q_content or "printer" in st.session_state.get("ai_issue_description", "").lower() or any("printer" in m["content"].lower() for m in messages if m["role"] == "user")))):
                summary_dict["Internal Jam Roller C"] = f"{ans_clean.capitalize()} — not resolved" if ans_clean.lower() == "no" else ans_clean
                
            # 7. Are you OTP
            elif re.search(r"\botp\b", q_content):
                summary_dict["Are you OTP or OTP cer"] = ans_clean.capitalize()

    # Seed the "Issue Description" with the AI-generated summary if it is valid (not "Unknown"), otherwise fallback to first non-lookup user message
    if "Issue Description" not in summary_dict or summary_dict["Issue Description"] in ["Unknown", ""]:
        ai_desc = st.session_state.get("ai_issue_description", "Unknown")
        if ai_desc and ai_desc.lower() != "unknown":
            summary_dict["Issue Description"] = ai_desc
        else:
            first_user_raw = ""
            for m in messages:
                if m["role"] == "user":
                    m_content = m["content"].strip()
                    if not is_ticket_status_lookup_intent(m_content) and len(m_content) > 3:
                        first_user_raw = m["content"].replace("**", "").replace("`", "").strip()
                        break
            if first_user_raw:
                summary_dict["Issue Description"] = first_user_raw[:200]  # cap at 200 chars
            else:
                summary_dict["Issue Description"] = "Unknown"

    # Pre-populate Device Type and Asset Number based strictly on USER messages to avoid assistant prompts polluting the parsing
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    user_text = " ".join(user_msgs).lower()
    
    if "Device Type" not in summary_dict:
        if "kiosk" in user_text:
            summary_dict["Device Type"] = "Kiosk"
        elif "kvs" in user_text or "kitchen" in user_text:
            summary_dict["Device Type"] = "KVS"
        elif any(dt_kw in user_text for dt_kw in ["drive through", "drive-thru", "dt"]):
            summary_dict["Device Type"] = "Drive-Thru"
        elif "pos" in user_text or "register" in user_text:
            summary_dict["Device Type"] = "POS"
        elif "bos" in user_text or "back office" in user_text:
            summary_dict["Device Type"] = "BOS"
        else:
            summary_dict["Device Type"] = "Unknown"
            
    if "Asset Number" not in summary_dict:
        num_match = re.search(r"\b(pos|kiosk|kds|kvs|unit)?\s*#?\s*(\d)\b", user_text)
        t = summary_dict.get("Device Type", "Unknown")
        if num_match and t != "Unknown":
            summary_dict["Asset Number"] = f"{t}{num_match.group(2)}"
        else:
            summary_dict["Asset Number"] = "Unknown"
    else:
        # Re-derive the Asset Number prefix from the now-resolved Device Type
        # to handle cases where the number was captured before the type was known
        t = summary_dict.get("Device Type", "Unknown")
        if t != "Unknown":
            asset_val = summary_dict["Asset Number"]
            num_only = re.sub(r"\D", "", asset_val)
            if num_only:
                summary_dict["Asset Number"] = f"{t}{num_only}"
        else:
            summary_dict["Asset Number"] = "Unknown"

    # Merge into current_triage without overwriting existing explicit form data
    for k, v in summary_dict.items():
        current_val = st.session_state.current_triage.get(k)
        # Only write if key is missing or currently set to a default fallback
        is_fallback = current_val is None or current_val in ["Unknown", "Skipped device info"]
        if k == "Issue Description" and current_val is not None:
            # If the current value is a bare generic word, and the new value is a richer symptom, allow overwrite!
            is_current_bare = current_val.lower().strip() in {"printer", "pos", "kiosk", "kvs", "kds", "bos", "hardware", "software", "device", "unknown", ""}
            is_new_rich = v.lower().strip() not in {"printer", "pos", "kiosk", "kvs", "kds", "bos", "hardware", "software", "device", "unknown", ""}
            if is_current_bare and is_new_rich:
                is_fallback = True
        if is_fallback:
            st.session_state.current_triage[k] = v

sync_triage_from_messages()

def _is_printer_issue() -> bool:
    """Return True if the current session is about a printer."""
    all_text = " ".join(
        m["content"] for m in get_current_flow_messages()
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
        'Device details help route your ticket faster. Enter what you know — all fields are optional.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("device_detail_form", border=True):
        model_val = st.text_input(
            "device model number:",
            placeholder="e.g. Epson TM-T88VI",
            key="device_form_model_input"
        )
        serial_val = st.text_input(
            "device serial number:",
            placeholder="e.g. X1A2B3C4D5",
            key="device_form_serial_input"
        )
        submitted = st.form_submit_button("submit", use_container_width=True)

    if submitted:
        # Store whatever was entered (empty = unknown, we skip it)
        m_val = model_val.strip() if model_val.strip() else "Unknown"
        s_val = serial_val.strip() if serial_val.strip() else "Unknown"
        logger.info("Device detail form submitted. Model: %s, Serial: %s", m_val, s_val)
        st.session_state.triage_model = m_val
        st.session_state.triage_serial = s_val
        st.session_state.current_triage["Device Model"] = m_val
        st.session_state.current_triage["Serial Number"] = s_val
        # Finish triage
        st.session_state.escalation_triage_active = False
        st.session_state.escalation_triage_step = "complete"
        st.session_state.ticket_collection_active = True
        st.rerun()


def render_priority_form() -> None:
    """
    Renders a cute, styled form with 3 mandatory priority/impact questions.
    """
    st.markdown(
        '<div style="background: rgba(255, 199, 44, 0.12); border: 1.5px solid #ffc72c; '
        'border-left: 5px solid #da291c; padding: 14px 18px; border-radius: 8px; margin: 12px 0; '
        'font-size: 13.5px; line-height: 1.5; color: var(--text-primary);">'
        '📊 <b>Ticket Impact Assessment</b><br/>'
        'Please answer the following mandatory questions to help determine the priority of your ticket.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("priority_assessment_form", border=True):
        st.markdown(
            f"<div class='confirm-header'>📊 Priority Questions</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='lock-lbl' style='color:#f0f2f8; text-transform:none;'>1. Is this issue completely stopping your store from taking orders or payments right now? <span style='color:var(--accent); font-weight:bold;'>*</span></div>", unsafe_allow_html=True)
        q1_val = st.radio(
            "Q1",
            options=["Yes", "No"],
            index=None,
            label_visibility="collapsed",
            key="priority_form_q1"
        )

        st.markdown("<div class='lock-lbl' style='color:#f0f2f8; text-transform:none; margin-top:14px;'>2. Is this the only device of its kind in your store? <span style='color:var(--accent); font-weight:bold;'>*</span></div>", unsafe_allow_html=True)
        q2_val = st.radio(
            "Q2",
            options=["Only one", "There are multiples"],
            index=None,
            label_visibility="collapsed",
            key="priority_form_q2"
        )

        st.markdown("<div class='lock-lbl' style='color:#f0f2f8; text-transform:none; margin-top:14px;'>3. Are the other devices of this kind online and functional? <span style='color:var(--accent); font-weight:bold;'>*</span></div>", unsafe_allow_html=True)
        q3_val = st.radio(
            "Q3",
            options=["Yes", "No"],
            index=None,
            label_visibility="collapsed",
            key="priority_form_q3"
        )

        submitted = st.form_submit_button("submit", use_container_width=True)

    if submitted:
        if q1_val is None or q2_val is None or q3_val is None:
            st.error("⚠️ All priority questions are mandatory. Please select an option for each question.")
        else:
            st.session_state.triage_q1 = q1_val
            st.session_state.triage_q2 = q2_val
            st.session_state.triage_q3 = q3_val
            st.session_state.current_triage["Stopping Orders/Payments"] = q1_val
            st.session_state.current_triage["Only Device of its Kind"] = q2_val
            st.session_state.current_triage["Other Devices Functional"] = q3_val
            calculate_priority_and_finish_triage()
            st.rerun()


def start_escalation_triage(append_welcome: bool = True):
    logger.info("Starting escalation triage flow. active_store: %s", st.session_state.active_store)
    st.session_state.escalation_triage_active = True
    st.session_state.ticket_collection_active = False
    st.session_state.triage_model = None
    st.session_state.triage_serial = None
    st.session_state.triage_q1 = None
    st.session_state.triage_q2 = None
    st.session_state.triage_q3 = None
    st.session_state.triage_priority = None
    st.session_state.manual_ticket_flow = False
 
    # All devices start with priority form step
    st.session_state.escalation_triage_step = "priority_form"
 
    ts_now = time.strftime("%H:%M")
    if append_welcome:
        notice = (
            "Got it. Since those steps didn't do the trick, let's get you transferred over to a support engineer.\n\n"
            "To help package up the exact priority and severity for this transfer queue, an embedded form will launch below. Please fill out the mandatory questions."
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": notice,
            "timestamp": ts_now,
            "blocked": False
        })
    else:
        # LLM already explained escalation — append the priority check notice if not already present.
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            last_msg = st.session_state.messages[-1]
            
            # Clean up any LLM-initiated priority/impact questions from the message
            content = last_msg["content"]
            lines = content.split("\n")
            cleaned_lines = []
            for line in lines:
                line_lower = line.lower()
                if any(phrase in line_lower for phrase in [
                    "how many kiosks are currently affected",
                    "how many pos are currently affected",
                    "how many devices are currently affected",
                    "how many are affected",
                    "how many are impacted",
                    "assess the priority of this issue",
                    "questions to assess the priority",
                ]):
                    continue
                cleaned_lines.append(line)
            content = "\n".join(cleaned_lines).strip()
            
            if "priority assessment" not in content.lower():
                last_msg["content"] = (
                    content
                    + "\n\nTo help determine priority for this ticket, an embedded form will launch below. Please fill out the mandatory questions."
                )

def check_escalation_intent(user_input: str) -> bool:
    """
    Checks if the user input implies an escalation to a human agent.
    Uses regex word boundaries to prevent false positives from substring matching.
    """
    if not user_input:
        return False
    lower_input = user_input.lower()
    
    # Strict action patterns indicating human transfer intent
    patterns = [
        r"\b(?:talk\s+to|speak\s+with|transfer\s+to|connect\s+to|connect\s+me\s+to)(?:\s+(?:a|an|the))?\s+(?:agent|human|representative|person|operator|supervisor|manager)\b",
        r"\b(?:live\s+agent|live\s+chat|real\s+person|human\s+support)\b",
        r"\b(?:escalate\s+to|escalate\s+this\s+to|escalate\s+my|escalate\s+our)\b",
        r"\b(?:open|create|submit|raise)\s+(?:a\s+)?ticket\b",
        r"\b(?:servicenow|genesys)\b"
    ]
    
    import re
    return any(re.search(pat, lower_input) for pat in patterns)


# (is_ticket_status_lookup_intent is now imported from guardrails)


def parse_status_change_intent(text: str) -> tuple[bool, str | None]:
    """
    Parses conversational user intent to change/update a case/ticket status.
    Returns (is_status_change_intent, ticket_id)
    """
    import re
    if not text:
        return False, None
    lower_text = text.strip().lower()
    
    # 1. Keywords list for status update intent
    keywords = ["status", "close", "closed", "pending", "awaiting confirmation", "awaiting"]
    verbs = ["change", "update", "modify", "set", "transition", "switch", "patch", "close"]
    
    # Simple check: does user mention status change or closing?
    # Enforce word boundary matching
    has_verb = any(re.search(r"\b" + re.escape(v) + r"\b", lower_text) for v in verbs)
    has_keyword = any(re.search(r"\b" + re.escape(k) + r"\b", lower_text) for k in keywords)
    is_intent = has_verb and has_keyword
    
    # Or does it match phrases like "change status", "update status", "modify status", "set status", 
    # "close case", "close ticket", "close the case", "close the ticket", "update case status", "update ticket status"
    direct_phrases = [
        "change status", "update status", "modify status", "set status", 
        "close case", "close ticket", "close the case", "close the ticket",
        "update case status", "update ticket status", "change case status"
    ]
    if any(phrase in lower_text for phrase in direct_phrases):
        is_intent = True
        
    # Also if the user says exactly "modify status", "change status", "update status", "close", "close ticket", "close case"
    if lower_text in ["modify status", "change status", "update status", "close", "close ticket", "close case"]:
        is_intent = True

    # 2. Extract ticket/case ID if present
    ticket_id_match = re.search(r"\b((?:INC|RC)[-_]?\d+)\b", text.upper())
    ticket_id = ticket_id_match.group(1) if ticket_id_match else None
    
    return is_intent, ticket_id



def parse_comment_update_intent(text: str) -> tuple[str, str] | None:
    """
    Parses conversational user intent to add a note/comment to a ticket.
    Returns (ticket_id, comment_text) if matched, otherwise None.
    """
    import re
    if not text:
        return None
    text_clean = text.strip()
    
    # 1. Match the prefix including verb and ticket ID
    # Verbs: add a note/comment to/on, update, comment on
    prefix_pattern = r"(?i)^(?:add\s+(?:a\s+)?(?:note|comment)\s+(?:to|on)|update|comment\s+on)\s+((?:INC|RC)[-_]?\d+)"
    match = re.match(prefix_pattern, text_clean)
    if not match:
        return None
        
    ticket_id = match.group(1).upper()
    # The remaining text after the ticket ID prefix
    remaining = text_clean[match.end():].strip()
    
    # 2. Repeatedly strip transition elements from the start of remaining text
    # Elements: spaces, colons, commas, semicolons, quotes, saying, stating, that, with (a) note/comment, to say, etc.
    while True:
        prev_len = len(remaining)
        # Strip leading punctuation/whitespace
        remaining = re.sub(r'^(?:[\s:;,"\'\-\(\)]+)', '', remaining)
        # Strip leading transition words
        remaining = re.sub(r'(?i)^(?:saying|stating|that|with\s+(?:a\s+)?(?:note|comment)?(?:\s+saying)?|to\s+say|say)\b', '', remaining)
        # Strip any new leading punctuation/whitespace that resulted
        remaining = re.sub(r'^(?:[\s:;,"\'\-\(\)]+)', '', remaining)
        if len(remaining) == prev_len:
            break
            
    # Clean up trailing quotes/punctuation
    remaining = re.sub(r'["\'\s\.]+$', '', remaining).strip()
    
    if ticket_id and remaining:
        return ticket_id, remaining
        
    return None


def package_conversational_context() -> None:
    """
    Mock utility that packages and prints the current conversational context
    and active store ID to the background terminal console for human queue handoff.
    Exclusively consumes contact details, current_triage, and comments.
    """
    store_id = st.session_state.get("active_store", "Unknown")
    
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
    
    store_info = STORE_ADDRESSES.get(int(store_id) if str(store_id).isdigit() else 67067, {
        "address": "1725 Slough Avenue, Scranton, PA 18503",
        "location": "Scranton Restaurant"
    })
    store_address = store_info["address"]
    store_location = store_info["location"]
    
    name = st.session_state.get("ticket_name", "Jim Halpert")
    phone = st.session_state.get("ticket_phone", "(570) 555-0142")
    backup_name = st.session_state.get("ticket_backup_name", "").strip()
    backup_phone = st.session_state.get("ticket_backup_phone", "").strip()
    comments = st.session_state.get("ticket_extra_info", "").strip()
    current_triage = st.session_state.get("current_triage", {})
    
    logger.info("📞 [HUMAN AGENT ESCALATION INTERCEPTED]")
    logger.info("  Name: %s", name)
    logger.info("  Store #: %s (%s)", store_id, store_location)
    logger.info("  Address: %s", store_address)
    logger.info("  Phone: %s", phone)
    if backup_name or backup_phone:
        logger.info("  Backup Contact: %s (Phone: %s)", backup_name or 'None', backup_phone or 'None')
        
    logger.info("  [Triage Details]")
    for k, v in current_triage.items():
        logger.info("    %s: %s", k, v)
        
    if comments:
        logger.info("  [Additional Info / Comments]")
        logger.info("    %s", comments)


def reset_triage_state():
    st.session_state.escalation_triage_active = False
    st.session_state.escalation_triage_step = None
    st.session_state.triage_model = None
    st.session_state.triage_serial = None
    st.session_state.triage_q1 = None
    st.session_state.triage_q2 = None
    st.session_state.triage_q3 = None
    st.session_state.triage_priority = None
    st.session_state.current_triage = {}
    st.session_state.escalated = False
    st.session_state.escalation_stage = None
    st.session_state.pop("ticket_extra_info", None)

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
    st.session_state.current_triage["Priority"] = priority
    logger.info("Triage Priority Assessment completed. Q1: %s, Q2: %s, Q3: %s. Calculated priority: %s", q1, q2, q3, priority)

    st.session_state.escalation_triage_step = "device_form"
    ts_now = time.strftime("%H:%M")
    st.session_state.messages.append({
        "role": "assistant",
        "content": "An optional embedded form will launch to collect additional details like the device's model and serial number, which will help our technicians? (type serial # in the text box below)",
        "timestamp": ts_now,
        "blocked": False
    })




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
                elif any(dt_kw in ans_clean.lower() for dt_kw in ["drive through", "drive-thru", "dt"]):
                    t_val = "Drive-Thru"
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
            elif (any(re.search(rf"\b{re.escape(kw)}\b", q_content) for kw in ["internal jam roller", "roller c"]) or
                  (re.search(r"\broller\b", q_content) and ("printer" in q_content or "paper" in q_content or "jam" in q_content or "printer" in st.session_state.get("ai_issue_description", "").lower() or any("printer" in m["content"].lower() for m in messages if m["role"] == "user")))):
                summary_dict["Internal Jam Roller C"] = f"{ans_clean.capitalize()} — not resolved" if ans_clean.lower() == "no" else ans_clean
                
            # 6. Device Information
            elif any(kw in q_content for kw in ["serial number", "model", "device information"]):
                summary_dict["Device Information"] = ans_clean
                
            # 7. Are you OTP  (troubleshooting step)
            elif re.search(r"\botp\b", q_content):
                summary_dict["Are you OTP or OTP cer"] = ans_clean.capitalize()

    # Seed the "What is the issue" with the AI-generated description
    if "What is the issue" not in summary_dict or summary_dict["What is the issue"] == "Unknown":
        summary_dict["What is the issue"] = st.session_state.get("ai_issue_description", "Unknown")

    # Pre-populate Type and Number if missing in summary_dict based strictly on USER message search
    # to avoid default fallbacks overriding parsed user values across different fields
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    user_text = " ".join(user_msgs).lower()
    
    if "Type" not in summary_dict:
        if "kiosk" in user_text:
            summary_dict["Type"] = "Kiosk"
        elif "kvs" in user_text or "kitchen" in user_text:
            summary_dict["Type"] = "KVS"
        elif any(dt_kw in user_text for dt_kw in ["drive through", "drive-thru", "dt"]):
            summary_dict["Type"] = "Drive-Thru"
        # Do NOT guess POS — leave unknown if no type was explicitly stated

    if "Number" not in summary_dict:
        num_match = re.search(r"\b(pos|kiosk|kds|kvs|unit)?\s*#?\s*(\d)\b", user_text)
        t = summary_dict.get("Type", "")
        if num_match and t:
            summary_dict["Number"] = f"{t}{num_match.group(2)}"
        # Do NOT guess a number — leave unknown if no asset number was explicitly stated

    # Core device/issue rows (always shown in Triage Diagnostics)
    core_defaults = [
        ("Type", "Unknown"),
        ("Number", "Unknown"),
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
        if key == "Device Information" and st.session_state.get("triage_priority"):
            # skip Device Information if we are spelling out Device Model and Serial Number
            continue
            
        val = summary_dict.get(key)
        if not val:
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


def is_duplicate_matching_question(content: str) -> bool:
    """
    Detects if the assistant message is asking the user if their issue matches
    an existing active ticket/case.
    """
    import re
    content_lower = content.lower()
    
    # Looks for phrases like "same issue", "same printer issue", "fresh hell", "another occurrence", "is this the same"
    has_same_phrase = any(phrase in content_lower for phrase in [
        "same issue", 
        "same printer", 
        "same kiosk", 
        "same pos", 
        "same device", 
        "same bumpbar", 
        "same bump bar", 
        "same register", 
        "same station", 
        "same kvs",
        "fresh hell",
        "another occurrence",
        "is this the same"
    ])
    
    # Must also look like a question or contains "or is this" / "or a fresh hell"
    is_question = "?" in content_lower or "or is" in content_lower or "is this the" in content_lower
    
    # Must contain reference to a ticket/case
    has_ticket_ref = bool(re.search(r'\b(?:rc|inc|case|ticket|match)\s*#?\s*\d+\b', content_lower))
    
    return has_same_phrase and is_question and (has_ticket_ref or "open case" in content_lower or "existing case" in content_lower)


def get_choices_from_message(content: str) -> list[str]:
    """
    Extract discrete choice options from typical assistant questions.
    Supports Yes/No questions, Location questions, and Unit numbers.
    Ensures that options do not contain quotes and filters out 'e.g.' prefixes.
    """
    import re
    content_lower = content.lower()
    
    # 0. Case detail view — ALWAYS suppress chips/buttons entirely.
    # Matches all case detail view markers including mutation confirmation responses.
    if ("activity:" in content_lower
            or "recent activity" in content_lower
            or "how do you want to handle this" in content_lower
            or "what's the plan for this issue" in content_lower
            or "you can update status" in content_lower
            or "what would you like to do next" in content_lower
            or "your comment has been added" in content_lower
            or "has been escalated" in content_lower
            or "escalation count is now" in content_lower
            or "servicenow ticket generated successfully" in content_lower
            or "servicenow parameter" in content_lower):
        return []

    # 0. Case options prompt bubbles - BANNED under UI Reboot
    if "case options:" in content_lower:
        return []

    # Duplicate case matching flow check
    if is_duplicate_matching_question(content):
        return ["Same Issue", "Different Issue"]

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

    # 0. No-KB-match or troubleshooting exhausted stopper choices
    # Do NOT display buttons when the user is doing anything involving ticket/case status
    last_user_input = st.session_state.get("last_user_input", "")
    is_lookup = is_ticket_status_lookup_intent(last_user_input)

    # Full two-button escalation — only for genuine troubleshooting exhaustion
    if "how would you like to proceed" in content_lower and ("open a support ticket" in content_lower or "live chat" in content_lower):
        if not is_lookup:
            return ["Open a support ticket", "Live Chat with an Agent"]

    # Single "Live Chat with an Agent" chip — when LLM says it can't act on a ticket
    # and directs the user to an agent (but not to open a new ticket)
    live_chat_phrases = [
        "live chat with an agent",
        "live chat with an agent who can help",
        "chat with an agent who can",
        "chat with a live agent",
    ]
    if is_lookup and any(phrase in content_lower for phrase in live_chat_phrases):
        return ["Live Chat with an Agent"]

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
        return ["1", "2", "3", "4", "All of them"]

    # 4. Explicit paren-bounded list of options
    match_paren = re.search(r"\(([^)]+)\)\s*\??\s*$", content.strip())
    if match_paren:
        s = match_paren.group(1).strip()
        s_clean = re.sub(r"^e\.g\.?,?\s*", "", s, flags=re.IGNORECASE).strip()
        quoted = re.findall(r'["\'"\u2018\u2019\u201c\u201d]([^"\'"\u2018\u2019\u201c\u201d]+)["\'"\u2018\u2019\u201c\u201d]', s_clean)
        if quoted:
            raw_choices = quoted
        else:
            parts = re.split(r",\s*|\s+or\s+|\s*/\s*", s_clean)
            raw_choices = parts
        cleaned = []
        for choice in raw_choices:
            c = choice.replace('"', '').replace("'", '').replace('\u201c', '').replace('\u201d', '').replace('\u2018', '').replace('\u2019', '').strip()
            if c.lower().startswith("e.g."):
                c = c[4:].strip()
            elif c.lower().startswith("e.g"):
                c = c[3:].strip()
            c = c.rstrip(",.? ")
            if c and c.lower() not in ["e.g.", "e.g", "example", "or", "and"]:
                cleaned.append(c)
        if 1 < len(cleaned) <= 6:
            return cleaned

    return []


# Status → CSS background color mapping for ticket tables
_TICKET_STATUS_COLORS = {
    "new":                    "rgba(255, 250, 180, 0.25)",   # light yellow
    "pending":                "rgba(173, 216, 255, 0.25)",   # light blue
    "in progress":            "rgba(210, 180, 255, 0.25)",   # light purple
    "awaiting confirmation":  "rgba(180, 255, 200, 0.25)",   # light green
    "closed":                 "rgba(255, 182, 193, 0.25)",   # light pink
}
_TICKET_STATUS_BORDERS = {
    "new":                    "rgba(200, 180, 0, 0.45)",
    "pending":                "rgba(70, 130, 200, 0.45)",
    "in progress":            "rgba(130, 80, 200, 0.45)",
    "awaiting confirmation":  "rgba(50, 160, 80, 0.45)",
    "closed":                 "rgba(200, 80, 100, 0.45)",
}


def _colorize_ticket_tables(content: str) -> str:
    """
    Parses the response content to find ticket tables, including those with broken
    multi-line comments or missing pipes, converts them to HTML tables, and wraps
    them in status-colored divs. This prevents layout issues where comments
    might be broken into multiple lines or fall outside the table bounds.
    """
    import re
    lines = content.splitlines()
    processed_lines = []
    
    in_ticket = False
    ticket_lines = []
    
    def process_ticket_block(t_lines: list[str]) -> str:
        # 1. Normalize status mapping
        def normalize_status(status_str: str) -> str:
            status_clean = status_str.strip().lower()
            if "new" in status_clean:
                return "new"
            if "pending" in status_clean or "assign" in status_clean:
                return "pending"
            if "progress" in status_clean:
                return "in progress"
            if "confirm" in status_clean:
                return "awaiting confirmation"
            if "close" in status_clean:
                return "closed"
            return status_clean

        # 2. Find the status
        raw_status = None
        for line in t_lines:
            status_match = re.search(
                r'\|\s*\*{0,2}Status\*{0,2}\s*\|\s*([^|\n]+?)\s*\|',
                line, re.IGNORECASE
            )
            if status_match:
                raw_status = status_match.group(1).strip().lower()
                break
                
        # Look up status by Ticket ID if status is not found yet (helps colorize streaming tables instantly!)
        if not raw_status:
            ticket_id = None
            for line in t_lines:
                id_match = re.search(
                    r'\|\s*\*{0,2}Ticket ID\*{0,2}\s*\|\s*([^|\n]+?)\s*\|',
                    line, re.IGNORECASE
                )
                if id_match:
                    ticket_id = id_match.group(1).strip()
                    # Strip markdown bold, spaces, or brackets if any
                    ticket_id = re.sub(r'[\*\`\[\]\s]', '', ticket_id)
                    break
            
            if ticket_id:
                active_tickets = st.session_state.get("active_tickets", [])
                for t in active_tickets:
                    t_id_clean = re.sub(r'[^A-Z0-9]', '', t.get("id", "").upper())
                    target_id_clean = re.sub(r'[^A-Z0-9]', '', ticket_id.upper())
                    if t_id_clean == target_id_clean:
                        raw_status = (t.get("status") or t.get("state") or "").strip().lower()
                        break
                        
        norm_status = normalize_status(raw_status) if raw_status else None
        bg = _TICKET_STATUS_COLORS.get(norm_status) if norm_status else "var(--bg-card)"
        border = _TICKET_STATUS_BORDERS.get(norm_status) if norm_status else "var(--border)"
        
        # 3. Parse and normalize rows
        normalized_rows = []
        current_cells = []
        
        for line in t_lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            if re.fullmatch(r'[\|\s\-\:]+', line_stripped):
                continue
                
            if line_stripped.startswith('|'):
                raw_parts = line_stripped.split('|')
                if raw_parts[0] == '':
                    raw_parts = raw_parts[1:]
                if raw_parts and raw_parts[-1] == '':
                    raw_parts = raw_parts[:-1]
                    
                cells = [c.strip() for c in raw_parts]
                if len(cells) >= 2:
                    if current_cells:
                        normalized_rows.append(current_cells)
                    current_cells = [cells[0], cells[1]]
                elif len(cells) == 1:
                    if current_cells:
                        current_cells[1] += "<br>" + cells[0]
                    else:
                        current_cells = ["", cells[0]]
            else:
                clean_line = line_stripped.rstrip('|').strip()
                if current_cells:
                    current_cells[1] += "<br>" + clean_line
                else:
                    current_cells = ["", clean_line]
                    
        if current_cells:
            normalized_rows.append(current_cells)
            
        if not normalized_rows:
            return '\n'.join(t_lines)
            
        def clean_cell(text: str) -> str:
            return re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            
        html_rows = []
        for row_idx, row in enumerate(normalized_rows):
            key = row[0]
            val = row[1]
            
            key_clean = clean_cell(key)
            val_clean = clean_cell(val)
            
            # Robust removal of [Field | Value] headers with absolute reliability
            is_header = (
                key_clean.strip().lower() in ("field", "field name") or 
                val_clean.strip().lower() in ("value", "value name") or
                ("field" in key_clean.lower() and "value" in val_clean.lower())
            )
            if is_header:
                continue
            
            cells_html = (
                f'<td style="padding:8px 12px; border-bottom:1px solid rgba(255,255,255,0.06); '
                f'font-weight:600; font-size:12.5px; color:#8892a4; vertical-align:top; width:25%;">'
                f'{key_clean}'
                f'</td>'
                f'<td style="padding:8px 12px; border-bottom:1px solid rgba(255,255,255,0.06); '
                f'font-size:13.5px; color:var(--text-primary); vertical-align:top; word-break:break-word;">'
                f'{val_clean}'
                f'</td>'
            )
            html_rows.append(f'<tr>{cells_html}</tr>')
            
        if not html_rows:
            return ""
            
        html_table = (
            f'<table style="width:100%; border-collapse:collapse; '
            f'font-family:inherit;">'
            + ''.join(html_rows)
            + '</table>'
        )
        
        # Always wrap in a card to prevent flashing from un-styled black text tables while streaming
        return (
            f'<div style="background:{bg}; border:1px solid {border}; '
            f'border-left:4px solid {border}; border-radius:10px; '
            f'padding:14px 16px; margin:14px 0;">'
            f'{html_table}'
            f'</div>'
        )

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        line_stripped = line.strip()
        
        is_table_header = (
            line_stripped.startswith('|') and 
            ("field" in line_stripped.lower() or "ticket id" in line_stripped.lower())
        )
        
        if not in_ticket and is_table_header:
            in_ticket = True
            ticket_lines = [line]
            idx += 1
            continue
            
        if in_ticket:
            stop_ticket = False
            
            is_new_table_header = (
                line_stripped.startswith('|') and 
                "field" in line_stripped.lower()
            )
            
            if is_new_table_header:
                stop_ticket = True
            elif "there are also" in line_stripped.lower() and "closed" in line_stripped.lower():
                stop_ticket = True
            elif not line_stripped:
                peek_idx = idx + 1
                next_non_empty = None
                while peek_idx < len(lines):
                    if lines[peek_idx].strip():
                        next_non_empty = lines[peek_idx].strip()
                        break
                    peek_idx += 1
                    
                if next_non_empty:
                    # Always stop if a new ticket table is starting
                    if next_non_empty.startswith('|') and ("field" in next_non_empty.lower() or "ticket id" in next_non_empty.lower()):
                        stop_ticket = True
                    else:
                        # Find the last field name in the ticket lines we have collected so far
                        last_field = None
                        for t_line in reversed(ticket_lines):
                            t_line_stripped = t_line.strip()
                            if t_line_stripped.startswith('|'):
                                if re.fullmatch(r'[\|\s\-\:]+', t_line_stripped):
                                    continue
                                raw_parts = [p.strip() for p in t_line_stripped.split('|')]
                                if raw_parts and raw_parts[0] == '':
                                    raw_parts = raw_parts[1:]
                                if raw_parts and raw_parts[-1] == '':
                                    raw_parts = raw_parts[:-1]
                                if raw_parts:
                                    last_field = raw_parts[0]
                                    break
                        
                        # If we have parsed any rows, stop if the next content is plain conversational text (not a table row or a list bullet)
                        if last_field:
                            if not next_non_empty.startswith('|') and not next_non_empty.startswith('•') and not next_non_empty.startswith('*') and not next_non_empty.startswith('-'):
                                stop_ticket = True
                        # If we do not have any active row yet, default to safety
                        else:
                            stop_ticket = True
                else:
                    stop_ticket = True
                    
            if stop_ticket:
                in_ticket = False
                processed_lines.append(process_ticket_block(ticket_lines))
                ticket_lines = []
                continue
            else:
                ticket_lines.append(line)
                idx += 1
        else:
            processed_lines.append(line)
            idx += 1
            
    if in_ticket:
        processed_lines.append(process_ticket_block(ticket_lines))
        
    return '\n'.join(processed_lines)


def _reformat_case_detail_view(content: str) -> str:
    """
    Deterministically reformat a case detail view response so every section
    is separated by exactly two newlines (\n\n), regardless of what the LLM produced.

    Detected by the presence of 'Ticket ' at the start and 'Activity:' in the body.
    Preserves the exact field values; only normalises whitespace/separators.
    """
    import re

    # Detect: must start with "Ticket " line and contain "Activity:"
    stripped = content.strip()
    if not re.match(r'^Ticket\s+\S', stripped) or 'activity:' not in stripped.lower():
        return content

    # Collect all meaningful lines (non-empty after stripping)
    raw_lines = [ln.strip() for ln in re.split(r'\r?\n', stripped)]
    sections = [ln for ln in raw_lines if ln]  # drop blank lines — we'll add our own

    # Canonical closing block — always appended; any LLM-generated closing line is dropped
    result_parts = []
    _CLOSING_LINES = {
        "how do you want to handle this",
        "what's the plan for this issue",
        "what would you like to do with this case",
        "what would you like to do next",
        "you can update status, escalate, or add a comment to this case",
    }
    _CANONICAL_CLOSING = (
        "You can update status, escalate, or add a comment to this case.\n\n"
        "What would you like to do next?"
    )

    for line in sections:
        # Strip trailing punctuation and spaces before comparing
        line_bare = re.sub(r'[?.!,\s]+$', '', line.lower())
        if line_bare not in _CLOSING_LINES:
            result_parts.append(line)

    # Insert a markdown hr divider before the closing block so the UI can't collapse the gap
    result_parts.append("---")
    result_parts.append(_CANONICAL_CLOSING)

    # Join with exactly \n\n between every element
    return "\n\n".join(result_parts)


def clean_assistant_message(content: str, msg_idx: int | None = None) -> str:
    """
    Remove parenthesized options list from assistant messages if suggestion chips will be displayed,
    append context-appropriate instruction suffixes, and wrap the troubleshooting salvo in a colored box.
    """
    import re
    import streamlit as st

    # Skip processing for all live agent messages — render verbatim
    live_agent_names = ["Alex", "Taylor", "Jordan", "Morgan", "Casey", "Robin", "Pat", "Jamie", "Sam", "Chris"]
    is_live_agent_msg = False
    
    if msg_idx is None:
        # During streaming of the live agent response
        if st.session_state.get("escalation_stage") == "connected":
            is_live_agent_msg = True
    else:
        messages = st.session_state.get("messages", [])
        for idx, msg in enumerate(messages):
            msg_content = msg.get("content", "")
            if msg.get("role") == "assistant":
                # Check if this message or any message before it in the history was the live agent greeting
                if any(f"Hello, my name is {name}." in msg_content for name in live_agent_names) and "reviewing your case now" in msg_content:
                    if msg_idx >= idx:
                        is_live_agent_msg = True
                        break

    if is_live_agent_msg:
        return content

    # Skip processing for the case details summary block
    if "Case Details Collected So Far" in content:
        return content

    # Skip processing for the ticket success/generated block
    if "ServiceNow Ticket Generated Successfully!" in content or "ServiceNow Parameter" in content:
        return content

    # 0. Case detail view — reformat deterministically with \n\n spacing
    content = _reformat_case_detail_view(content)
    # If this is a case detail view, return immediately (skip all other processing)
    if content.strip().startswith("Ticket ") and "Activity:" in content:
        return content

    # Colorize ticket markdown tables by status (must run before any further manipulation)
    content = _colorize_ticket_tables(content)

    # 1. Intercept troubleshooting salvo and put inside a colored box
    salvo_text = "There are some common troubleshooting steps that might help you fix this issue on your own. We will quickly step through them to see if this solves the issue"

    # Verify if the content contains actual steps/instructions
    has_concrete_steps = False
    lower_content = content.lower()
    if "did this resolve" in lower_content or "resolve the issue" in lower_content or "did that work" in lower_content or "did it work" in lower_content:
        has_concrete_steps = True

    if salvo_text.lower() in lower_content:
        if not has_concrete_steps:
            # No actual steps — strip the salvo entirely and don't render the card
            pattern = re.compile(re.escape(salvo_text) + r"\.?", re.IGNORECASE)
            content = pattern.sub("", content).strip()
        else:
            pattern = re.compile(re.escape(salvo_text) + r"\.?", re.IGNORECASE)
            replacement = (
                '<div class="recommended-troubleshooting-banner" style="background: rgba(218, 41, 28, 0.08); border: 1px solid rgba(255, 199, 44, 0.3); '
                'border-left: 4px solid var(--accent); padding: 12px 14px 45px 14px; border-radius: 8px; margin: 10px 0; '
                'font-size: 13.5px; line-height: 1.5; color: var(--text-primary); position: relative;">'
                '⚡ <b>Recommended Troubleshooting</b><br/>'
                'There are some common troubleshooting steps that might help you fix this issue on your own. '
                'We will quickly step through them to see if this solves the issue.'
                '</div>'
            )
            content = pattern.sub(replacement, content)

    # Check if the user's intent is ticket status lookup. If so, completely exempt from Ticket Impact Assessment banner injection.
    is_lookup = False
    if "messages" in st.session_state and st.session_state.messages:
        for msg in reversed(st.session_state.messages):
            if msg.get("role") == "user":
                if is_ticket_status_lookup_intent(msg.get("content", "")):
                    is_lookup = True
                break

    if not is_lookup:
        # 1.5 Intercept escalation/exhausted steps and insert Ticket Impact Assessment banner
        # immediately after it, completely removing the "Entering Ticket Creation Flow" banner.
        exhausted_phrases = [
            "exhausted the initial troubleshooting steps",
            "exhausted the common troubleshooting steps",
            "troubleshooting steps didn't resolve the issue",
            "troubleshooting steps did not resolve the issue",
            "did not resolve the issue, we need to escalate",
            "didn't resolve the issue, we need to escalate",
            "need to ask a few questions to assess the priority",
            "assess the priority of this issue",
            "how many kiosks are currently affected",
            "how many pos are currently affected",
            "how many devices are currently affected",
        ]
        
        has_inserted_impact_banner = False
        
        for phrase in exhausted_phrases:
            if phrase in content.lower():
                # Find the sentence containing this phrase
                import re as _re
                match = _re.search(rf'([^.!?\n]*{_re.escape(phrase)}[^.!?\n]*[.!?]?)', content, _re.IGNORECASE)
                if match:
                    sentence = match.group(1)
                    banner = (
                        '<div style="background: rgba(255, 199, 44, 0.12); border: 1.5px solid #ffc72c; '
                        'border-left: 5px solid #da291c; padding: 14px 18px; border-radius: 8px; margin: 12px 0; '
                        'font-size: 13.5px; line-height: 1.5; color: var(--text-primary);">'
                        '📊 <b>Ticket Impact Assessment</b><br/>'
                        "To help determine priority for this ticket, let's dig in on impact."
                        '</div>'
                    )
                    # Replace the sentence with the banner followed by the sentence
                    content = content.replace(sentence, banner + "\n\n" + sentence, 1)
                    has_inserted_impact_banner = True
                    break

        # 1.6 Intercept priority/impact notice and put inside a colored box if not already shown
        priority_text = "To help determine priority for this ticket, let's dig in on impact"
        if priority_text.lower() in content.lower():
            # The device_form step is completely after the priority check, so they haven't seen the banner yet
            already_shown_in_prev_turn = False
            
            if has_inserted_impact_banner or already_shown_in_prev_turn:
                # Strip the redundant plain text/phrase of priority_text
                pattern = re.compile(r'To help determine priority for this ticket, let\'s dig in on impact\.?\s*', re.IGNORECASE)
                content = pattern.sub("", content)
            else:
                # Otherwise show the premium McDonald's branded banner
                pattern = re.compile(re.escape(priority_text) + r"\.?", re.IGNORECASE)
                replacement = (
                    '<div style="background: rgba(255, 199, 44, 0.12); border: 1.5px solid #ffc72c; '
                    'border-left: 5px solid #da291c; padding: 14px 18px; border-radius: 8px; margin: 12px 0; '
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
        
    # Strip any trailing sentence fragments commonly left behind after removing parenthesized option list
    # e.g., "for example, is it", "is it", "such as", "like", "for instance, is it a", etc.
    fragment_pattern = re.compile(
        r"\b(?:can|could)\s+you\s+(?:tell|let\s+me\s+know)(?:\s+me)?\s*$"
        r"|\b(?:tell|let\s+me\s+know)(?:\s+me)?\s*$"
        r"|\b(?:is\s+it|are\s+they)(?:\s+a)?\s*$"
        r"|\b(?:such\s+as|like|specifically|for\s+example|for\s+instance)\s*$"
        r"|\b(?:can|could)\s+you\s*$"
        r"|\b(?:to\s+assess|assess|determine)\s*$"
        r"|\b(?:which|what)\s+(?:one|symptom|issue|device)\s*$"
        r"|\b(?:if\s+it\s+is|if\s+they\s+are|whether\s+it\s+is|whether\s+they\s+are)\s*$",
        re.IGNORECASE
    )
    while True:
        prev = cleaned
        cleaned = fragment_pattern.sub("", cleaned).strip()
        cleaned = cleaned.rstrip("?:.,; \t-–—")
        if cleaned == prev:
            break
    
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
        suffix = "(If complicated, describe in the steps below)"  # type: ignore[assignment]
    else:
        suffix = "(or type your answer below)"
        
    return f"{cleaned}? {suffix}"


def trigger_live_agent_flow(user_message_text: str) -> None:
    """
    Triggers Stage 1 of the Live Agent escalation flow (acknowledged state).
    Builds the high-fidelity summary table and connecting notice in a single assistant message.
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
        
    active_store = st.session_state.get("active_store", "67067")
    logger.info("Live Agent escalation flow triggered (acknowledged stage). Store ID: %s. User message: %s", active_store, user_message_text)
    
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
    
    store_info = STORE_ADDRESSES.get(int(active_store) if str(active_store).isdigit() else 67067, {
        "address": "1725 Slough Avenue, Scranton, PA 18503",
        "location": "Scranton Restaurant"
    })
    store_address = store_info["address"]
    store_location = store_info["location"]
    
    name = st.session_state.get("ticket_name", "Jim Halpert")
    phone = st.session_state.get("ticket_phone", "(570) 555-0142")
    backup_name = st.session_state.get("ticket_backup_name", "").strip()
    backup_phone = st.session_state.get("ticket_backup_phone", "").strip()

    case_dump_lines = [
        "Got it, let me pull in one of our support engineers to look at this with you. Hang tight for a second while I package up what we've gone over so far.\n\n",
        "### 📋 Case Transfer Context Staged by ChipLLM\n\n",
        "| Parameter | Value |\n",
        "| :--- | :--- |\n",
        f"| **Active Store** | `Store #{active_store} ({store_location})` |\n",
        f"| **Store Address** | `{store_address}` |\n",
        f"| **Contact Name** | `{name}` |\n",
        f"| **Contact Phone** | `{phone}` |\n"
    ]
    
    if backup_name:
        case_dump_lines.append(f"| **Backup Contact** | `{backup_name}` |\n")
    if backup_phone:
        case_dump_lines.append(f"| **Backup Phone** | `{backup_phone}` |\n")
        
    for k, v in st.session_state.current_triage.items():
        if v is not None and str(v).strip():
            case_dump_lines.append(f"| **{k}** | `{v}` |\n")
            
    case_dump_text = "".join(case_dump_lines)
    st.session_state.messages.append({
        "role": "assistant",
        "content": case_dump_text,
        "timestamp": ts_now,
        "blocked": False
    })
    
    package_conversational_context()
    st.session_state.escalation_stage = "acknowledged"
    st.session_state.escalated = True
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
            font-size: 13.5px;
            font-weight: 700;
            color: #ffc72c;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 6px;
        }
        .diagnostic-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 14px;
        }
        .diagnostic-row:last-child {
            border-bottom: none;
        }
        .diagnostic-key {
            color: #8892a4;
            font-weight: 500;
            padding-right: 12px;
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
        name_val = st.text_input("Name", value=CURRENT_USER_NAME, disabled=True, label_visibility="collapsed", key="ticket_name")
        
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

    # Card 1.5: Additional Information Card
    # Only show when NOT in manual_ticket_flow — if manual flow is active the user
    # already has a free-form Description field below, so double-prompting is redundant.
    if not manual_flow:
        with st.container(border=True):
            st.markdown(
                f"<div class='confirm-header'>📝 Additional Information</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<div class='lock-lbl' style='color:#f0f2f8; text-transform:none;'>"
                "Is there any other information you can provide to the agent about this issue "
                "(when the problem began, other troubleshooting that has been tried)</div>",
                unsafe_allow_html=True
            )
            st.text_area(
                "Additional Info",
                placeholder="e.g. Started after power surge, already rebooted twice...",
                label_visibility="collapsed",
                key="ticket_extra_info"
            )
    if manual_flow:
        # Pre-populate short description from AI issue description or fallback to first user message
        first_user_msg = ""
        ai_desc = st.session_state.get("ai_issue_description", "")
        if ai_desc and ai_desc != "Unknown":
            first_user_msg = ai_desc
        else:
            for m in get_current_flow_messages():
                if m["role"] == "user":
                    first_user_msg = m["content"].replace("**", "").replace("`", "").strip()
                    break

        # Pre-populate description field with troubleshooting diagnostics and steps if not already filled
        if "ticket_manual_desc" not in st.session_state:
            p_val = st.session_state.get("ticket_phone", "(570) 555-0142")
            b_name_val = st.session_state.get("ticket_backup_name", "")
            b_phone_val = st.session_state.get("ticket_backup_phone", "")
            
            diagnostic_dump_lines = []

            # Always include the user's own words at the top of the description
            first_user_msg_raw = ""
            for m in get_current_flow_messages():
                if m["role"] == "user":
                    first_user_msg_raw = m["content"].replace("**", "").replace("`", "").strip()
                    break
            if first_user_msg_raw:
                is_bare_word = (
                    first_user_msg_raw.lower().strip() in {"printer", "pos", "kiosk", "kvs", "kds", "bos", "hardware", "software", "device", "unknown"}
                    or len(first_user_msg_raw.split()) <= 2
                )
                if not is_bare_word:
                    diagnostic_dump_lines.append("[User Report]")
                    diagnostic_dump_lines.append(f"- Initial message: {first_user_msg_raw}")
                    diagnostic_dump_lines.append("")

            diagnostic_dump_lines.append("[Device Details]")
            diagnostic_dump_lines.append(f"- Store ID: {active_store}")
            diagnostic_dump_lines.append(f"- Location: {store_location}")
            diagnostic_dump_lines.append(f"- Reporter: {CURRENT_USER_NAME} (Phone: {p_val})")
            
            t_model = st.session_state.current_triage.get("Device Model")
            if t_model and t_model != "Unknown":
                diagnostic_dump_lines.append(f"- Device Model: {t_model}")
            t_serial = st.session_state.current_triage.get("Serial Number")
            if t_serial and t_serial != "Unknown":
                diagnostic_dump_lines.append(f"- Serial Number: {t_serial}")
            t_priority = st.session_state.current_triage.get("Priority")
            if t_priority:
                diagnostic_dump_lines.append(f"- Calculated Priority: {t_priority}")
                
            if b_name_val.strip() or b_phone_val.strip():
                b_name_str = b_name_val.strip() if b_name_val.strip() else "None specified"
                b_phone_str = b_phone_val.strip() if b_phone_val.strip() else "None specified"
                diagnostic_dump_lines.append(f"- Backup Contact: {b_name_str} (Phone: {b_phone_str})")
                
            extra_info_val = st.session_state.get("ticket_extra_info", "").strip()
            if extra_info_val:
                diagnostic_dump_lines.append(f"- Additional Agent Info: {extra_info_val}")
    
            # Split current_triage into core triage rows vs troubleshooting Q&A rows
            diagnostic_dump_lines.append("\n[Triage Diagnostics]")
            triage_keys_local = [
                "Device Type", "Asset Number", "Issue Description", "Device Model",
                "Serial Number", "Stopping Orders/Payments", "Only Device of its Kind",
                "Other Devices Functional", "Priority"
            ]
            for k in triage_keys_local:
                v = st.session_state.current_triage.get(k)
                if v is not None and str(v).strip():
                    diagnostic_dump_lines.append(f"- {k}: {v}")
    
            # Troubleshooting steps section
            diagnostic_dump_lines.append("\n[Troubleshooting Steps Attempted]")
            troubleshooting_keys_local = [
                "Is there visible paper", "Internal Jam Roller C", "Are you OTP or OTP cer"
            ]
            ts_dump_count = 0
            for k in troubleshooting_keys_local:
                v = st.session_state.current_triage.get(k)
                if v is not None and str(v).strip():
                    diagnostic_dump_lines.append(f"- {k}: {v}")
                    ts_dump_count += 1
                    
            # Then add any assistant-driven step summaries from chat messages
            step_index = ts_dump_count + 1
            for msg in get_current_flow_messages():
                if msg["role"] == "assistant":
                    content = msg["content"]
                    content_lower = content.lower()
                    step_kws = ["reboot", "power cycle", "paper jam", "card reader", "restart", "cable", "connection", "software update"]
                    if any(kw in content_lower for kw in step_kws):
                        step_title = "Step"
                        if "reboot" in content_lower or "restart" in content_lower or "power cycle" in content_lower:
                            step_title = "Power Cycle / Restart Device"
                        elif "cable" in content_lower or "connection" in content_lower:
                            step_title = "Check Cables & Connections"
                        elif "software update" in content_lower or "outdated software" in content_lower or "pending software" in content_lower:
                            step_title = "Check & Install Software Updates"
                        elif "paper jam" in content_lower or "roller c" in content_lower:
                            step_title = "Clear Paper Jam"
                        elif "card reader" in content_lower:
                            step_title = "Clean Card Reader"
                        else:
                            for line in content.split('\n'):
                                line_lower = line.lower()
                                if any(kw in line_lower for kw in ["reboot", "jam", "cable", "connection", "power", "reader", "restart", "software", "update"]):
                                    step_title = line.replace("**", "").replace("`", "").replace("##", "").strip()
                                    break
                        diagnostic_dump_lines.append(f"{step_index}. {step_title} (Attempted) -> Outcome: Did not resolve the issue.")
                        step_index += 1
    
            if step_index == 1 and ts_dump_count == 0:
                diagnostic_dump_lines.append("- No troubleshooting steps could be attempted or they were skipped.")
                
            diagnostic_dump_lines.append("\n[System Action]")
            diagnostic_dump_lines.append("- Escalated to ServiceNow via chat session with support engineer ChipLLM.")
            
            st.session_state.ticket_manual_desc = "\n".join(diagnostic_dump_lines)

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

    # Card 2.5: Optional Image Attachment Card
    with st.container(border=True):
        st.markdown(
            f"<div class='confirm-header'>📷 Attach a Photo</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='lock-lbl' style='color:#f0f2f8; text-transform:none; margin-bottom: 8px;'>"
            "Upload an optional photo or screenshot of the issue (optional)</div>",
            unsafe_allow_html=True
        )
        st.file_uploader(
            "Attach Photo",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="ticket_image_upload"
        )

    # Card 3: Diagnostic Summary Card
    triage_keys = [
        "Device Type", "Asset Number", "Issue Description", "Device Model",
        "Serial Number", "Stopping Orders/Payments", "Only Device of its Kind",
        "Other Devices Functional", "Priority"
    ]
    troubleshooting_keys = [
        "Is there visible paper", "Internal Jam Roller C", "Are you OTP or OTP cer"
    ]
    
    if not manual_flow:
        st.markdown("<div class='diagnostic-card'>", unsafe_allow_html=True)
        
        # Section 1: Triage Diagnostics
        st.markdown(f"<div class='diagnostic-header'>Triage Diagnostics</div>", unsafe_allow_html=True)
        for k in triage_keys:
            v = st.session_state.current_triage.get(k)
            if v is not None and str(v).strip():
                st.markdown(
                    f"<div class='diagnostic-row'>"
                    f"  <div class='diagnostic-key'>{k}</div>"
                    f"  <div class='diagnostic-val'>{v}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Buttons with dynamic style markers
    st.markdown("<div class='submit-btn-marker'></div>", unsafe_allow_html=True)
    if st.button("🎫 Submit Support Ticket", use_container_width=True, key="submit_ticket_collection"):
        try:
            if manual_flow:
                m_s_desc = st.session_state.get("ticket_manual_short_desc", "").strip()
                m_desc = st.session_state.get("ticket_manual_desc", "").strip()
                if not m_s_desc or not m_desc:
                    st.error("⚠️ Please enter both a Short Description and Description.")
                    st.stop()
                    
            import random
            # ServiceNow Ticket Number format: RC00 followed by 4 digits
            inc_num = f"RC00{random.randint(1000, 9999)}"
            
            # Determine category and sub-category dynamically
            all_chat_text = " ".join([m["content"] for m in get_current_flow_messages()]).lower()
            summary_text = " ".join([f"{k} {v}" for k, v in st.session_state.current_triage.items()]).lower()
            combined_text = all_chat_text + " " + summary_text
            
            sub_category = "kiosk"
            device_type = st.session_state.current_triage.get("Device Type", "").lower()
            if "printer" in device_type or "print" in device_type or "printer" in combined_text or "print" in combined_text:
                sub_category = "printer"
            elif "pos" in device_type or "register" in device_type or "pos" in combined_text or "terminal" in combined_text or "register" in combined_text:
                sub_category = "pos"
            elif "kvs" in device_type or "kitchen" in device_type or "kvs" in combined_text:
                sub_category = "kvs"
            elif "kiosk" in device_type or "kiosk" in combined_text:
                sub_category = "kiosk"
            elif "bos" in device_type or "back office" in device_type or "bos" in combined_text:
                sub_category = "bos"
                
            category = "hardware"
            if any(kw in combined_text for kw in ["slow", "lagging", "crash", "network", "offline", "login", "password"]):
                category = "software"
                
            # Parse serial number if present in triage
            serial_val = st.session_state.current_triage.get("Serial Number", "—")
            if serial_val in ["—", "Unknown"]:
                for msg in get_current_flow_messages():
                    content = msg["content"]
                    match = re.search(r"\b([A-Z0-9]{2,6}[-]?[\d]{5,12})\b", content)
                    if match:
                        serial_val = match.group(1)
                        break
                
            # Compile structured dump
            diagnostic_dump_lines = []

            # Always include the user's own words at the top of the description
            first_user_msg_raw = ""
            for m in get_current_flow_messages():
                if m["role"] == "user":
                    first_user_msg_raw = m["content"].replace("**", "").replace("`", "").strip()
                    break
            if first_user_msg_raw:
                is_bare_word = (
                    first_user_msg_raw.lower().strip() in {"printer", "pos", "kiosk", "kvs", "kds", "bos", "hardware", "software", "device", "unknown"}
                    or len(first_user_msg_raw.split()) <= 2
                )
                if not is_bare_word:
                    diagnostic_dump_lines.append("[User Report]")
                    diagnostic_dump_lines.append(f"- Initial message: {first_user_msg_raw}")
                    diagnostic_dump_lines.append("")

            diagnostic_dump_lines.append("[Device Details]")
            diagnostic_dump_lines.append(f"- Store ID: {active_store}")
            diagnostic_dump_lines.append(f"- Location: {store_location}")
            diagnostic_dump_lines.append(f"- Reporter: {CURRENT_USER_NAME} (Phone: {phone_val})")
            
            t_model = st.session_state.current_triage.get("Device Model")
            if t_model and t_model != "Unknown":
                diagnostic_dump_lines.append(f"- Device Model: {t_model}")
            t_serial = st.session_state.current_triage.get("Serial Number")
            if t_serial and t_serial != "Unknown":
                diagnostic_dump_lines.append(f"- Serial Number: {t_serial}")
            t_priority = st.session_state.current_triage.get("Priority")
            if t_priority:
                diagnostic_dump_lines.append(f"- Calculated Priority: {t_priority}")
                
            if backup_name_val.strip() or backup_phone_val.strip():
                b_name = backup_name_val.strip() if backup_name_val.strip() else "None specified"
                b_phone = backup_phone_val.strip() if backup_phone_val.strip() else "None specified"
                diagnostic_dump_lines.append(f"- Backup Contact: {b_name} (Phone: {b_phone})")
                
            extra_info = st.session_state.get("ticket_extra_info", "").strip()
            if extra_info:
                diagnostic_dump_lines.append(f"- Additional Agent Info: {extra_info}")
    
            # Split current_triage into core triage rows vs troubleshooting Q&A rows
            diagnostic_dump_lines.append("\n[Triage Diagnostics]")
            for k in triage_keys:
                v = st.session_state.current_triage.get(k)
                if v is not None and str(v).strip():
                    diagnostic_dump_lines.append(f"- {k}: {v}")
    
            # Troubleshooting steps section
            diagnostic_dump_lines.append("\n[Troubleshooting Steps Attempted]")
            ts_dump_count = 0
            for k in troubleshooting_keys:
                v = st.session_state.current_triage.get(k)
                if v is not None and str(v).strip():
                    diagnostic_dump_lines.append(f"- {k}: {v}")
                    ts_dump_count += 1
                    
            # Then add any assistant-driven step summaries from chat messages
            step_index = ts_dump_count + 1
            for msg in get_current_flow_messages():
                if msg["role"] == "assistant":
                    content = msg["content"]
                    content_lower = content.lower()
                    step_kws = ["reboot", "power cycle", "paper jam", "card reader", "restart", "cable", "connection", "software update"]
                    if any(kw in content_lower for kw in step_kws):
                        step_title = "Step"
                        if "reboot" in content_lower or "restart" in content_lower or "power cycle" in content_lower:
                            step_title = "Power Cycle / Restart Device"
                        elif "cable" in content_lower or "connection" in content_lower:
                            step_title = "Check Cables & Connections"
                        elif "software update" in content_lower or "outdated software" in content_lower or "pending software" in content_lower:
                            step_title = "Check & Install Software Updates"
                        elif "paper jam" in content_lower or "roller c" in content_lower:
                            step_title = "Clear Paper Jam"
                        elif "card reader" in content_lower:
                            step_title = "Clean Card Reader"
                        else:
                            for line in content.split('\n'):
                                line_lower = line.lower()
                                if any(kw in line_lower for kw in ["reboot", "jam", "cable", "connection", "power", "reader", "restart", "software", "update"]):
                                    step_title = line.replace("**", "").replace("`", "").replace("##", "").strip()
                                    break
                        diagnostic_dump_lines.append(f"{step_index}. {step_title} (Attempted) -> Outcome: Did not resolve the issue.")
                        step_index += 1
    
            if step_index == 1 and ts_dump_count == 0:
                diagnostic_dump_lines.append("- No troubleshooting steps could be attempted or they were skipped.")
                
            diagnostic_dump_lines.append("\n[System Action]")
            diagnostic_dump_lines.append("- Escalated to ServiceNow via chat session with support engineer ChipLLM.")
            
            if manual_flow:
                short_desc = st.session_state.get("ticket_manual_short_desc", "").strip()
                full_description_dump = st.session_state.get("ticket_manual_desc", "").strip()
                extra_info = st.session_state.get("ticket_extra_info", "").strip()
                if extra_info:
                    full_description_dump += f"\n\n[Additional Information]\n- {extra_info}"
            else:
                full_description_dump = "\n".join(diagnostic_dump_lines)

                # Build Short Description deterministically from structured triage data
                _dev_type = st.session_state.current_triage.get("Device Type", "")
                _asset_num = st.session_state.current_triage.get("Asset Number", "")
                _issue_desc = st.session_state.current_triage.get("Issue Description", "")

                # Prefer AI description if triage captured only the device name, then fallback to first non-lookup user message
                _bare_device_words = {"pos", "kiosk", "kvs", "kds", "bos", "printer", "unknown", "hardware", "software", "device", ""}
                if not _issue_desc or _issue_desc.lower().strip() in _bare_device_words:
                    ai_desc = st.session_state.get("ai_issue_description", "Unknown")
                    if ai_desc and ai_desc.lower() != "unknown":
                        _issue_desc = ai_desc
                    else:
                        for _m in get_current_flow_messages():
                            if _m["role"] == "user":
                                _m_content = _m["content"].strip()
                                if not is_ticket_status_lookup_intent(_m_content) and len(_m_content) > 3:
                                    _issue_desc = _m["content"].replace("**", "").replace("`", "").strip()
                                    break

                # Resolve device base identifier
                if _asset_num and _asset_num not in ("Unknown", ""):
                    dev_id = _asset_num
                elif _dev_type and _dev_type not in ("Unknown", ""):
                    dev_id = _dev_type
                else:
                    dev_id = sub_category.capitalize()

                # Ensure the sub_category (e.g. Printer) is part of the identifier
                sub_cap = sub_category.capitalize()
                if sub_cap.lower() not in dev_id.lower():
                    dev_identifier = f"{dev_id} {sub_cap}"
                else:
                    dev_identifier = dev_id

                # Enrich symptom formatting
                if _issue_desc and _issue_desc.lower().strip() not in _bare_device_words:
                    short_desc = f"{dev_identifier} - {_issue_desc}"
                else:
                    short_desc = f"{dev_identifier} Issue"

                if len(short_desc) > 80:
                    short_desc = short_desc[:77] + "..."
            # Handle photo upload conversion for ticket submission
            attachment_b64 = None
            attachment_name = None
            ticket_image = st.session_state.get("ticket_image_upload")
            if ticket_image is not None:
                base64_url = file_to_base64_data_url(ticket_image)
                if base64_url:
                    attachment_b64 = base64_url
                    attachment_name = ticket_image.name

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
                "caller": f"{CURRENT_USER_NAME}, Store #{active_store}, {store_location}",
                "contact_type": "chat",
                "assignment_group": "Unisys RTS L1 - SD - US",
                "priority": st.session_state.current_triage.get("Priority", "P3"),
                "attachment_base64": attachment_b64,
                "attachment_name": attachment_name
            }
            
            logger.info("Submitting support ticket to ServiceNow. Store ID: %s, Category: %s, Sub-category: %s", active_store, category, sub_category)
            logger.info("ServiceNow Ticket Payload: %s", json.dumps(meta, indent=2))
            
            # Create the case in Supabase/storage layer
            from mocks.servicenow import create_case, get_active_cases
            username = st.session_state.get("ticket_name", CURRENT_USER_NAME)
            # Map 'P1'/'P2' priority string to integer for SMALLINT column
            _priority_str = st.session_state.current_triage.get("Priority", "P3")
            try:
                _priority_int = int(str(_priority_str).strip().lstrip("Pp"))
            except (ValueError, TypeError):
                _priority_int = 3
            create_case({
                "id": inc_num,
                "store_id": active_store,
                "description": full_description_dump,      # full diagnostic dump → description
                "short_description": short_desc,           # AI summary → short_description
                "status": "New",
                "priority": _priority_int,
                "escalations": 0,
                "category": category,
                "subcategory": sub_category,
                "comments": [
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "author": "System",
                        "text": f"Case opened via ChipLLM by {username}"
                    }
                ]
            })
            # Refresh active cases count and list immediately
            st.session_state.active_tickets = get_active_cases(active_store)
            
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
                f"An engineer from the **{meta['assignment_group']}** group has been dispatched and is reviewing this ticket.\n\n"
                f"Is there anything else I can do for you?"
            )
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": success_content,
                "timestamp": _ts,
                "blocked": False,
                "attachment_b64": attachment_b64,
                "attachment_name": attachment_name
            })
            
            st.session_state.ticket_collection_active = False
            st.session_state.show_restart_chat_btn = True
            st.rerun()
        except Exception as e:
            logger.exception("Failed to submit ServiceNow ticket")
            st.error("⚠️ An unexpected error occurred while generating your support ticket. Please try again or contact support directly.")
        
    st.markdown("<div class='cancel-btn-marker'></div>", unsafe_allow_html=True)
    if st.button("← Start Over", use_container_width=True, key="cancel_ticket_collection"):
        st.session_state.needs_reset = True
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

        if meta.get("attachment_base64"):
            st.markdown("**Attached Image**")
            st.markdown(
                f'<div style="border-radius: 8px; overflow: hidden; border: 1px solid var(--border); '
                f'margin-top: 4px; margin-bottom: 12px; background: #0f1016; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);">'
                f'  <img src="{meta["attachment_base64"]}" style="width: 100%; height: auto; display: block; max-height: 160px; object-fit: contain;" />'
                f'  <div style="padding: 6px 10px; font-size: 10px; color: var(--text-muted); border-top: 1px solid var(--border); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">'
                f'    📎 {meta.get("attachment_name", "Attachment")}'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("**Description**")
        st.caption(meta.get("description", "—")[:200])

        st.divider()

        st.markdown("**Raw Payload (JSON)**")
        st.code(json.dumps(meta, indent=2), language="json")

        if st.button("🔄 Clear Ticket"):
            st.session_state.needs_reset = True
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
            st.session_state.needs_reset = True
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
def load_store_context():
    store_id = st.session_state.get("header_store_selector", DEFAULT_STORE)
    st.session_state.active_store = store_id
    from mocks.servicenow import get_store_tickets, get_store_mims
    st.session_state.active_tickets = get_store_tickets(store_id)
    st.session_state.active_mims = get_store_mims(store_id)
    check_and_update_mim_state()
    logger.info("Active store changed to %s. Loaded %d tickets and %d MIMs.", store_id, len(st.session_state.active_tickets), len(st.session_state.active_mims))
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"🔄 **Switched active context to Store #{store_id}.** How can I assist you with this store?",
        "timestamp": time.strftime("%H:%M"),
        "blocked": False
    })

# ---------------------------------------------------------------------------
# Header — single sticky row: logo+name | store selector | active cases
# ---------------------------------------------------------------------------
st.markdown('<div id="header-sentinel"></div>', unsafe_allow_html=True)

active_mims = st.session_state.get("active_mims", [])
has_dismissed_mim = False
dismissed_mim_id = None

if active_mims:
    if "mim_dismissed" not in st.session_state:
        st.session_state.mim_dismissed = {}
    for mim in active_mims:
        norm_id = mim["number"].upper().strip()
        if st.session_state.mim_dismissed.get(norm_id, False):
            has_dismissed_mim = True
            dismissed_mim_id = norm_id
            break

# Three-column header: brand | store | active-cases  (or brand | store | cases | ⚠️)
if has_dismissed_mim:
    col_logo, col_store, col_metric, col_mim = st.columns([1.2, 1.5, 1.5, 0.3], gap="small")
else:
    col_logo, col_store, col_metric = st.columns([1.2, 1.5, 1.5], gap="small")

with col_logo:
    st.markdown(
        """
        <div class="header-brand">
          <div class="header-logo">🍔</div>
          <span class="header-title">ChipLLM</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_store:
    selected_store = st.selectbox(
        "Header Store select",
        options=STORE_OPTIONS,
        index=STORE_OPTIONS.index(st.session_state.active_store) if st.session_state.active_store in STORE_OPTIONS else 0,
        format_func=lambda x: f"🏪 {x}",
        label_visibility="collapsed",
        key="header_store_selector",
        on_change=load_store_context
    )

with col_metric:
    active_tickets = st.session_state.get("active_tickets", [])
    active_count = sum(
        1 for t in active_tickets
        if (t.get("status") or t.get("state") or "").strip().lower() != "closed"
    )
    metric_btn_clicked = st.button(
        f"📋 Active Cases: {active_count}",
        key="header_active_cases_btn",
        use_container_width=True
    )
    if metric_btn_clicked:
        st.session_state.suggestion_click = "show active cases"
        st.rerun()

if has_dismissed_mim:
    with col_mim:
        restore_clicked = st.button("⚠️", key="restore_mim_button", help="Restore outage banner")
        if restore_clicked:
            st.session_state.mim_dismissed[dismissed_mim_id] = False
            st.rerun()

# ---------------------------------------------------------------------------
# Major Outages / MIMs Alert
# ---------------------------------------------------------------------------
if st.session_state.get("active_mims"):
    for mim in st.session_state.active_mims:
        norm_id = mim['number'].upper().strip()
        if "mim_dismissed" not in st.session_state:
            st.session_state.mim_dismissed = {}
        if st.session_state.mim_dismissed.get(norm_id, False):
            continue

        mim_id = mim['number'].replace("MIM", "")
        mim_title = mim['short_description']
        mim_desc  = mim.get('description', '')

        with st.container(border=True):
            st.markdown('<div class="active-outage-box"></div>', unsafe_allow_html=True)
            col_left, col_right = st.columns([3, 1], gap="small", vertical_alignment="center")
            with col_left:
                st.markdown(
                    f"""
                    <div class="outage-banner-title">⚡ Active Outage: {mim_title}</div>
                    <div class="outage-banner-desc">{mim_desc}</div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_right:
                btn_clicked = st.button(
                    "Affected",
                    key=f"report_mim_button_{mim_id}",
                    use_container_width=True,
                )
                dismiss_clicked = st.button(
                    "Dismiss",
                    key=f"dismiss_mim_button_{mim_id}",
                    use_container_width=True,
                )
            if btn_clicked:
                import logging
                logging.info(f"[MOCK TRACKING] Store {st.session_state.active_store} reported MIM {mim['number']}")
                st.success("Linked to master incident.")
            if dismiss_clicked:
                st.session_state.mim_dismissed[norm_id] = True
                st.rerun()

# ---------------------------------------------------------------------------
# Status Update Success Alert
# ---------------------------------------------------------------------------
if st.session_state.get("status_update_success"):
    st.success(st.session_state.status_update_success, icon="✅")
    del st.session_state.status_update_success

# ---------------------------------------------------------------------------
# Welcome message (injected once)
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hi there! I'm ChipLLM, your restaurant tech support engineer. "
            "Let me know what's acting up or what case status you need to check, "
            "and we'll get it sorted out."
        ),
        "timestamp": time.strftime("%H:%M"),
        "blocked": False
    })

# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    avatar = "👤" if role == "user" else "👨‍💻"
    
    with st.chat_message(role, avatar=avatar):
        if msg.get("blocked", False):
            st.markdown(
                f'<div style="color:#fda4af;border-left:3px solid #f43f5e;'
                f'padding-left:10px;">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            display_content = msg["content"]
            if role == "assistant":
                display_content = clean_assistant_message(display_content, msg_idx=i)
            st.markdown(display_content, unsafe_allow_html=True)
            
            if role == "assistant" and "⚡ <b>Recommended Troubleshooting</b>" in display_content:
                if st.button("Skip Directly to Tickeet Creation", key=f"skip_trouble_{i}"):
                    ts_now = time.strftime("%H:%M")
                    st.session_state.messages.append({
                        "role": "user",
                        "content": "Skip Directly to Tickeet Creation",
                        "timestamp": ts_now,
                        "blocked": False
                    })
                    st.session_state.manual_ticket_flow = True
                    start_escalation_triage(append_welcome=False)
                    st.rerun()
            
            # Render optional image attachment if present in the message
            if msg.get("attachment_b64"):
                st.markdown(
                    f'<div style="margin-top: 12px; border-radius: 12px; overflow: hidden; max-width: 320px; '
                    f'border: 1px solid rgba(255, 255, 255, 0.08); background-color: #161720; '
                    f'box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35); transition: all 0.2s ease;">'
                    f'  <img src="{msg["attachment_b64"]}" style="width: 100%; height: auto; display: block; max-height: 240px; object-fit: contain; background: #0f1016;" />'
                    f'  <div style="padding: 8px 12px; font-size: 11px; color: #8892a4; font-weight: 600; '
                    f'border-top: 1px solid rgba(255, 255, 255, 0.06); text-overflow: ellipsis; overflow: hidden; white-space: nowrap; display: flex; align-items: center; gap: 6px;">'
                    f'    <span>📎</span> {msg.get("attachment_name", "Attachment")}'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown(f'<div class="msg-time">{msg.get("timestamp", "")}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Extra Context Form (Stage 2 of Live Agent Escalation)
# ---------------------------------------------------------------------------
def render_extra_context_form() -> None:
    """
    Renders an inline form to gather additional notes/context from the user
    before escalating to live technician ChipLLM.
    """
    st.markdown(
        '<div style="background: rgba(218, 41, 28, 0.10); border: 1px solid rgba(218, 41, 28, 0.45); '
        'border-left: 4px solid var(--accent); padding: 12px 16px; border-radius: 8px; '
        'margin: 8px 0 12px; font-size: 13.5px; line-height: 1.5; color: var(--text-primary);">'
        '📋 <b>Support Queue Transfer Staging</b>'
        '</div>',
        unsafe_allow_html=True
    )
    
    with st.form("extra_context_form", border=True):
        user_notes = st.text_area(
            "Is there any other information you can provide to the support engineer? (When the problem started, other troubleshooting that has been attempted)",
            placeholder="e.g. Device stopped working after a new version rollout. The screen shows error 067.",
            key="escalation_user_notes_input"
        )
        
        uploaded_image = st.file_uploader(
            "Attach a photo of the physical damage / issue (optional)",
            type=["png", "jpg", "jpeg"],
            key="escalation_image_upload"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Submit & Transfer to Live Queue", use_container_width=True)
        with col2:
            cancelled = st.form_submit_button("Cancel & Return to Chat", use_container_width=True)
            
    if cancelled:
        logger.info("Stage 2 extra context form cancelled. Returning to automated chat.")
        st.session_state.escalated = False
        st.session_state.escalation_stage = None
        st.session_state.escalation_triage_active = False
        st.session_state.escalation_triage_step = None
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ **Escalation cancelled.** We are back in the automated support chat. How can I help you troubleshoot today?",
            "timestamp": time.strftime("%H:%M"),
            "blocked": False
        })
        st.rerun()
        
    if submitted:
        notes_str = user_notes.strip() if user_notes.strip() else "None provided."
        logger.info("Stage 2 extra context form submitted. Notes: %s", notes_str)
        st.session_state.current_triage["User Escalation Notes"] = notes_str
        
        attachment_b64 = None
        attachment_name = None
        if uploaded_image is not None:
            base64_url = file_to_base64_data_url(uploaded_image)
            if base64_url:
                st.session_state.current_triage["Attachment Base64"] = base64_url
                st.session_state.current_triage["Attachment Name"] = uploaded_image.name
                attachment_b64 = base64_url
                attachment_name = uploaded_image.name
        
        # Append User's Escalation Handoff response as a chat message
        user_msg_content = f"**Additional Handoff Notes:** {notes_str}"
        if attachment_name:
            user_msg_content += f"\n\n📎 **Attachment:** `{attachment_name}`"
            
        st.session_state.messages.append({
            "role": "user",
            "content": user_msg_content,
            "timestamp": time.strftime("%H:%M"),
            "blocked": False,
            "attachment_b64": attachment_b64,
            "attachment_name": attachment_name
        })
        
        st.session_state.escalation_stage = "connected"
        st.session_state.live_agent_pending_connection = True
        st.rerun()

if st.session_state.get("escalation_stage") == "acknowledged":
    with st.chat_message("assistant", avatar="👨‍💻"):
        render_extra_context_form()

# ---------------------------------------------------------------------------
# Live Agent connection gate
# ---------------------------------------------------------------------------
if st.session_state.get("live_agent_pending_connection", False):
    st.session_state.live_agent_pending_connection = False
    agent_name = st.session_state.live_agent_name
    
    # Send the first greeting immediately
    greeting_msg = f"Hello, my name is {agent_name}. I am reviewing your case now."
    st.session_state.messages.append({
        "role": "assistant",
        "content": greeting_msg,
        "timestamp": time.strftime("%H:%M"),
        "blocked": False
    })
    
    st.session_state.live_agent_pending_question = True
    st.rerun()

if st.session_state.get("live_agent_pending_question", False):
    st.session_state.live_agent_pending_question = False
    agent_name = st.session_state.live_agent_name
    
    # Simulate a realistic 10 second wait where they read the case history
    with st.spinner(f"{agent_name} is reviewing the case details (takes about 10 seconds)..."):
        time.sleep(10)
        
        try:
            logger.info("Invoking Gemini for live agent %s's question.", agent_name)
            client = ChipLLMClient()
            user_notes = st.session_state.current_triage.get("User Escalation Notes", "None provided.")
            
            question_msg = client.generate_agent_question(
                triage_data=st.session_state.current_triage,
                messages=st.session_state.messages,
                user_notes=user_notes,
                agent_name=agent_name
            )
            question_msg = question_msg.strip() if question_msg else ""
        except Exception as exc:
            logger.exception("Failed to generate agent question. Using fallback.")
            question_msg = f"Thanks for hanging tight. Can you confirm exactly what symptoms you're seeing on the screen?"
            
        st.session_state.messages.append({
            "role": "assistant",
            "content": question_msg,
            "timestamp": time.strftime("%H:%M"),
            "blocked": False
        })
        st.rerun()




# ---------------------------------------------------------------------------
# Device Detail & Priority Forms (escalation triage flow)
# ---------------------------------------------------------------------------

_triage_active = st.session_state.get("escalation_triage_active", False)
_triage_step = st.session_state.get("escalation_triage_step")

if _triage_active:
    with st.chat_message("assistant", avatar="👨‍💻"):
        if _triage_step == "priority_form":
            render_priority_form()
        elif _triage_step == "device_form":
            render_device_detail_form()

# ---------------------------------------------------------------------------
# Ticket Form Render Gate
# ---------------------------------------------------------------------------

elif st.session_state.ticket_collection_active:
    is_live_agent = st.session_state.get("ticket_collection_is_agent", False)
    with st.chat_message("assistant", avatar="👨‍💻"):
        render_ticket_collection_form(is_live_agent=is_live_agent)

# ---------------------------------------------------------------------------
# Suggestion Chips Rendering Block
# ---------------------------------------------------------------------------
if "suggestion_click" not in st.session_state:
    st.session_state.suggestion_click = None

# Suppress chips during form steps (the forms replace them)
_show_chips = (
    not st.session_state.ticket_collection_active
    and _triage_step not in ["device_form", "priority_form"]
    and not st.session_state.get("escalated", False)
)
if _show_chips:
    last_msg = st.session_state.messages[-1] if st.session_state.messages else None
    if last_msg and last_msg["role"] == "assistant":
        status_flow = last_msg.get("status_change_flow")
        if status_flow:
            case_id = status_flow["case_id"]
            current_status = status_flow["current_status"]
            
            # Strict vocab states
            all_statuses = ["Pending", "Awaiting Confirmation", "Closed"]
            remaining_statuses = [s for s in all_statuses if s.lower() != current_status.lower()]
            
            st.markdown("<div class='chips-sentinel'></div>", unsafe_allow_html=True)
            cols = st.columns(len(remaining_statuses))
            for idx, status_opt in enumerate(remaining_statuses):
                with cols[idx]:
                    if st.button(status_opt, key=f"status_btn_{case_id}_{status_opt}_{idx}", use_container_width=True):
                        # Zero-Friction Click Processing
                        from llm_client import update_case_status
                        update_msg = update_case_status(case_id, status_opt)
                        
                        ts_now = time.strftime("%H:%M")
                        st.session_state.messages.append({
                            "role": "user",
                            "content": f"Update status to {status_opt}",
                            "timestamp": ts_now,
                            "blocked": False
                        })
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Done — case **{case_id}** status has been updated to **{status_opt}**.",
                            "timestamp": ts_now,
                            "blocked": False
                        })
                        
                        st.session_state.status_update_success = f"Case {case_id} status updated to {status_opt}!"
                        st.rerun()
        else:
            choices = get_choices_from_message(last_msg["content"])
            if choices:
                st.markdown("<div class='chips-sentinel'></div>", unsafe_allow_html=True)
                cols = st.columns(len(choices))
                for idx, (col, choice) in enumerate(zip(cols, choices)):
                    with col:
                        if st.button(choice, key=f"chip_{choice}_{idx}", use_container_width=True):
                            if choice == "Modify Status":
                                st.session_state.suggestion_click = choice
                                st.rerun()
                            elif choice in ["See Details", "Add Comment", "Escalate", "None"]:
                                # Don't do anything for now
                                pass
                            else:
                                st.session_state.suggestion_click = choice
                                st.rerun()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

chat_placeholder = "Describe the issue… (e.g. 'Printer jammed')" if len(st.session_state.messages) <= 1 else "Type here"

is_connected_stage = st.session_state.get("escalation_stage") == "connected"
chat_val = st.chat_input(
    placeholder=chat_placeholder,
    key="chat_input",
    disabled=st.session_state.get("escalated", False) and not is_connected_stage,
)

user_input = None
if st.session_state.suggestion_click:
    user_input = st.session_state.suggestion_click
    st.session_state.suggestion_click = None
elif chat_val:
    user_input = chat_val

if user_input:
    if user_input.strip() == "/nuke-knowledge":
        st.cache_data.clear()
        # Also evict the session_state pool so the next turn triggers a full GCS re-scan
        st.session_state.pop("knowledge_base", None)
        st.session_state.pop("kb_loaded_at", None)
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Knowledge caches purged. Re-fetching operational bulletins...",
            "timestamp": time.strftime("%H:%M"),
            "blocked": False
        })
        st.rerun()

    logger.info("Received user chat input: %s", user_input)
    st.session_state.last_user_input = user_input
    ts_now = time.strftime("%H:%M")
    lower_input = user_input.lower()
    
    # Force bypass of triage, ticket collection, and escalation if user is checking ticket/case status
    if is_ticket_status_lookup_intent(user_input):
        logger.info("Ticket status lookup intent detected. Resetting all triage, escalation, and ticket collection states to bypass Priority/Escalation forms.")
        reset_triage_state()
        st.session_state.ticket_collection_active = False
        st.session_state.manual_ticket_flow = False
        
        # Clear active_case_id context if this is a general/bulk list query (does not contain a specific case ID)
        import re as _re
        if not _re.search(r"\b(?:inc|rc00|rc)[-_]?\d+\b", lower_input):
            st.session_state.active_case_id = None
            logger.info("Cleared active_case_id context for general/bulk query.")

    # Detect cafe issues and route immediately to live human agent
    if is_cafe_issue(user_input) and not st.session_state.get("escalated", False):
        logger.info("Cafe issue detected: %s. Routing to live human agent.", user_input)
        trigger_live_agent_flow(user_input)

    # Global human agent escalation interceptor
    if check_escalation_intent(user_input) and not is_ticket_status_lookup_intent(user_input):
        trigger_live_agent_flow(user_input)
    
    # Check if we were awaiting a case ID for a status change
    last_assistant_msg = None
    if st.session_state.messages:
        for msg in reversed(st.session_state.messages):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg
                break
                
    if last_assistant_msg and last_assistant_msg.get("awaiting_case_id_for_status_change"):
        import re
        case_id_match = re.search(r"\b((?:INC|RC)[-_]?\d+)\b", user_input.upper())
        if case_id_match:
            resolved_case_id = case_id_match.group(1)
            last_assistant_msg["awaiting_case_id_for_status_change"] = False
            
            # Append user's message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": ts_now,
                "blocked": False
            })
            
            # Resolve current status
            from llm_client import get_case_status
            status_res = get_case_status(resolved_case_id)
            current_status = status_res.get("status", "Pending")
            
            summary_content = f"Case **{resolved_case_id}** is currently set to **{current_status}**. Please select the updated state below:"
            st.session_state.messages.append({
                "role": "assistant",
                "content": summary_content,
                "timestamp": ts_now,
                "blocked": False,
                "status_change_flow": {
                    "case_id": resolved_case_id,
                    "current_status": current_status
                }
            })
            st.session_state.active_case_id = resolved_case_id
            st.rerun()
        else:
            # User replied with something other than a case ID, assume they are moving on
            last_assistant_msg["awaiting_case_id_for_status_change"] = False
            logger.info("Cleared awaiting_case_id_for_status_change state as user moved on.")

    # Intercept status change updates
    is_status_intent, status_ticket_id = parse_status_change_intent(user_input)
    if is_status_intent:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": ts_now,
            "blocked": False
        })
        
        resolved_case_id = status_ticket_id
        if not resolved_case_id:
            resolved_case_id = st.session_state.get("active_case_id")
            
        if resolved_case_id:
            from llm_client import get_case_status
            status_res = get_case_status(resolved_case_id)
            current_status = status_res.get("status", "Pending")
            
            summary_content = f"Case **{resolved_case_id}** is currently set to **{current_status}**. Please select the updated state below:"
            st.session_state.messages.append({
                "role": "assistant",
                "content": summary_content,
                "timestamp": ts_now,
                "blocked": False,
                "status_change_flow": {
                    "case_id": resolved_case_id,
                    "current_status": current_status
                }
            })
            st.session_state.active_case_id = resolved_case_id
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Which case would you like to update?",
                "timestamp": ts_now,
                "blocked": False,
                "awaiting_case_id_for_status_change": True
            })
        st.rerun()

    # Intercept comment updates
    comment_match = parse_comment_update_intent(user_input)
    if comment_match:
        ticket_id, comment_text = comment_match
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": ts_now,
            "blocked": False
        })
        
        # Search for target ticket
        found_ticket = None
        normalized_target = ticket_id.replace("-", "").replace("_", "").upper()
        active_tickets = st.session_state.get("active_tickets", [])
        for ticket in active_tickets:
            tid = ticket.get("id", "")
            normalized_tid = tid.replace("-", "").replace("_", "").upper()
            if normalized_tid == normalized_target:
                found_ticket = ticket
                break
                
        if found_ticket:
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            if "comments" not in found_ticket:
                found_ticket["comments"] = []
            new_comment = {
                "timestamp": timestamp_str,
                "author": "Manager Jim",
                "text": comment_text
            }
            found_ticket["comments"].append(new_comment)
            
            # Persist comment update to the database layer
            from mocks.servicenow import update_case
            update_case(found_ticket["id"], {"comments": found_ticket["comments"]})
            
            st.session_state.active_tickets = active_tickets
            
            confirm_msg = f"Got it, I've added that note to case {found_ticket.get('id', ticket_id)} for you."
            st.session_state.messages.append({
                "role": "assistant",
                "content": confirm_msg,
                "timestamp": ts_now,
                "blocked": False
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I parsed your request to add a note to **{ticket_id}**, but I couldn't find a matching active ticket with that ID in store #{st.session_state.get('active_store', 'Unknown')}. Please check the ticket number and try again.",
                "timestamp": ts_now,
                "blocked": False
            })
        st.rerun()

    # Intercept commands like "start over"
    elif "start over" in lower_input or "restart" in lower_input:
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
        
        if step == "priority_form":
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Please complete the mandatory Priority Assessment form above to proceed.",
                "timestamp": ts_now,
                "blocked": False
            })
            st.rerun()
            
        elif step == "device_form":
            # User typed in chat while device form was shown — treat as skip
            logger.info("Device detail form skipped by user input: %s. Setting model/serial to Unknown.", user_input)
            st.session_state.triage_model = "Unknown"
            st.session_state.triage_serial = "Unknown"
            st.session_state.current_triage["Device Model"] = "Unknown"
            st.session_state.current_triage["Serial Number"] = "Unknown"
            st.session_state.escalation_triage_active = False
            st.session_state.escalation_triage_step = "complete"
            st.session_state.ticket_collection_active = True
            st.rerun()
        
    # Intercept live agent keywords
    elif check_escalation_intent(user_input) and not is_ticket_status_lookup_intent(user_input):
        trigger_live_agent_flow(user_input)
        
    # Intercept ticket creation intent — broad fuzzy match so natural phrasing
    # ("need to open a case", "open a case", "file a ticket", etc.) never reaches
    # the LLM and causes it to hallucinate that it can't create tickets.
    elif _is_ticket_creation_intent(lower_input) and not is_ticket_status_lookup_intent(user_input):
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": ts_now,
            "blocked": False
        })
        st.session_state.manual_ticket_flow = True
        start_escalation_triage(append_welcome=False)
        st.rerun()

    elif lower_input in ["live chat with an agent", "live chat", "chat with an agent"]:
        trigger_live_agent_flow(user_input)
        
    # Intercept case routing choices
    elif "manual ticket flow" in lower_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": ts_now,
            "blocked": False
        })
        # Route through the priority form first so we always capture P1-P4 before the ticket form
        st.session_state.manual_ticket_flow = True
        start_escalation_triage(append_welcome=False)
        st.rerun()

    elif "continue with automated flow" in lower_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": ts_now,
            "blocked": False
        })
        # Resume the LLM conversation — do NOT jump to ticket collection
        st.session_state.manual_ticket_flow = False
        st.session_state.ticket_collection_active = False
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Got it — let's keep troubleshooting. What happened after the last step you tried?",
            "timestamp": ts_now,
            "blocked": False
        })
        st.rerun()

    # Intercept only explicit escalate/live-chat keyword requests from outside an active troubleshoot session
    # NOTE: Do NOT intercept ticket-related words here — that would block ticket status/action requests.
    # The escalation buttons are triggered ONLY by the LLM via the post-response check when troubleshooting is exhausted.
    elif ("escalate" in lower_input or "live chat" in lower_input or "live agent" in lower_input) and not is_ticket_status_lookup_intent(user_input):
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

        is_connected_stage = st.session_state.get("escalation_stage") == "connected"
        if is_connected_stage:
            st.session_state.live_agent_chat_turns = st.session_state.get("live_agent_chat_turns", 0) + 1
            guard_blocked = False
            rag_context = None
            rag_title = None
        else:
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
            is_ticket_query = is_ticket_status_lookup_intent(user_input)
            
            relevant_playbooks = []
            if not is_ticket_query:
                logger.info("Attempting RAG retrieval for input: '%s'", user_input)
                # Timestamp-gated refresh: hits GCS only if pool is stale or missing
                _kb.PLAYBOOKS = refresh_playbook_pool()
                relevant_playbooks = retrieve_context(user_input, top_k=1)
            else:
                logger.info("Ticket query detected. Skipping RAG retrieval to prevent model distraction.")
                
            rag_context: str | None = None
            rag_title: str | None = None

            if relevant_playbooks:
                rag_context_parts = []
                rag_titles_list = []
                for p in relevant_playbooks:
                    content = p["content"]
                    if p.get("source") == "adhoc":
                        rag_context_parts.append(f"### [CRITICAL ADHOC OVERRIDE BULLETIN]\n{content}")
                    else:
                        rag_context_parts.append(content)
                    rag_titles_list.append(p["title"])
                rag_context = "\n\n".join(rag_context_parts)
                rag_title = ", ".join(rag_titles_list)
                logger.info("RAG playbook hits: '%s'", rag_title)
            else:
                logger.info("RAG playbook miss (no match found). Proceeding to conversational LLM fallback.")

        # --- Layer 4: LLM call (streaming) -------------------------------------
        with st.chat_message("assistant", avatar="👨‍💻"):
            full_response = ""
            response_placeholder = st.empty()
            if is_ticket_status_lookup_intent(user_input):
                response_placeholder.markdown("Retrieving data...")

            try:
                logger.info("Starting Gemini API streaming response.")
                client = ChipLLMClient()
                is_connected_stage = st.session_state.get("escalation_stage") == "connected"
                sys_inst = None
                if is_connected_stage:
                    agent_name = st.session_state.live_agent_name
                    turns = st.session_state.get("live_agent_chat_turns", 0)
                    if turns >= 2:
                        sys_inst = (
                            f"You are {agent_name}, a highly experienced, casual, and friendly live tech support engineer at the restaurant chain.\n"
                            f"The user is a restaurant manager who has just been connected to you.\n"
                            f"This conversation has gone on for a few turns. You must now politely direct the user to the formal ticket creation flow "
                            f"to schedule a field dispatch or a deeper investigation by hardware specialists.\n"
                            f"Explain to them clearly and warmly that we need to open a support ticket to track this issue, and that you are opening the form for them right now.\n"
                            f"Keep the message concise and under 60 words."
                        )
                    else:
                        sys_inst = (
                            f"You are {agent_name}, a highly experienced, casual, and friendly live tech support engineer at the restaurant chain.\n"
                            f"The user is a restaurant manager who has just been connected to you.\n"
                            f"Continue the conversation in character as {agent_name}. Keep your natural, casual, empathetic, and highly technical tone.\n"
                            f"Use phrases like 'Hi there!' or 'Thanks for that detail,' when appropriate, and be extremely helpful.\n"
                            f"Keep responses relatively concise and do not repeat your initial greeting."
                        )
                for chunk in client.stream_response(
                    messages=st.session_state.messages,
                    rag_context=rag_context,
                    system_instruction=sys_inst,
                ):
                    full_response += chunk
                    cleaned_chunk = clean_assistant_message(full_response)
                    response_placeholder.markdown(f"{cleaned_chunk}▌", unsafe_allow_html=True)

                cleaned_final = clean_assistant_message(full_response)
                response_placeholder.markdown(cleaned_final, unsafe_allow_html=True)
                logger.info("Completed Gemini API streaming response. Response length: %d chars", len(full_response))

                # Auto-track active_case_id: if the response is a detail view, capture the case ID
                import re as _re
                _case_match = _re.search(r'\bTicket\s+(RC\w+)\b', full_response)
                if _case_match:
                    _detected_id = _case_match.group(1)
                    if st.session_state.get("active_case_id") != _detected_id:
                        st.session_state.active_case_id = _detected_id
                        logger.info("active_case_id set to %s from detail view response.", _detected_id)

            except Exception as exc:
                logger.exception("Gemini API streaming failed")
                full_response = (
                    "⚠️ **An unexpected backend error occurred.** Please try again, "
                    "or type 'escalate' to route this issue to our support team."
                )
                response_placeholder.markdown(full_response)

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



        # If they are connected and chat turns >= 2, trigger ticket creation flow
        if is_connected_stage and st.session_state.get("live_agent_chat_turns", 0) >= 2:
            st.session_state.manual_ticket_flow = True
            start_escalation_triage(append_welcome=False)

        st.rerun()
