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
  --accent:       #f97316;       /* vivid orange — kitchen urgency */
  --accent-dim:   #c2580e;
  --accent-glow:  rgba(249,115,22,.25);
  --user-bubble:  #1e3a5f;
  --bot-bubble:   #1a1d28;
  --border:       rgba(255,255,255,.07);
  --text-primary: #f0f2f8;
  --text-muted:   #8892a4;
  --text-accent:  #f97316;
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
}

/* ── Header bar ─────────────────────────────────────────────────────────── */
.chip-header {
  background: linear-gradient(135deg, #0d0f14 0%, #1a1d28 100%);
  border-bottom: 1px solid var(--border);
  padding: 14px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 2px 20px rgba(0,0,0,.4);
}

.chip-header .chip-logo {
  width: 44px;
  height: 44px;
  background: linear-gradient(135deg, var(--accent) 0%, #ea580c 100%);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  box-shadow: 0 4px 14px var(--accent-glow);
  flex-shrink: 0;
}

.chip-header .chip-info {
  flex: 1;
}

.chip-header .chip-name {
  font-size: 16px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.3px;
}

.chip-header .chip-status {
  font-size: 11px;
  color: var(--success);
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 5px;
}

.chip-header .chip-status::before {
  content: '';
  width: 7px;
  height: 7px;
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
  background: linear-gradient(135deg, var(--accent) 0%, #ea580c 100%) !important;
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
  background: rgba(249,115,22,.12);
  border: 1px solid rgba(249,115,22,.3);
  color: var(--accent);
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
  border: 1.5px solid rgba(249,115,22,.35) !important;
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

.sev-critical { color: var(--danger) !important; }
.sev-high { color: #fb923c !important; }
.sev-medium { color: #facc15 !important; }
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
}

/* ── Streamlit default markdown in chat ──────────────────────────────────── */
.stMarkdown p { margin: 0 0 6px; }
.stMarkdown ol, .stMarkdown ul { padding-left: 18px; margin: 4px 0; }
.stMarkdown code {
  background: rgba(249,115,22,.12);
  color: var(--accent);
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 12px;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(249,115,22,.3); border-radius: 4px; }

/* ── Bottom action bar ────────────────────────────────────────────────────────────── */
.action-bar {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr;
  gap: 6px;
  padding: 8px 14px 4px;
}

.action-bar .stButton > button {
  background: rgba(255,255,255,.05) !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  color: #8892a4 !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  padding: 7px 4px !important;
  min-height: 40px !important;
  border-radius: 10px !important;
  line-height: 1.3 !important;
  letter-spacing: .2px !important;
  transition: background .15s, border-color .15s, color .15s !important;
}

.action-bar .stButton > button:hover {
  background: rgba(249,115,22,.12) !important;
  border-color: rgba(249,115,22,.4) !important;
  color: #f97316 !important;
  transform: none !important;
  opacity: 1 !important;
}

/* ticket button — subtle warm tint */
.action-bar-ticket .stButton > button {
  border-color: rgba(249,115,22,.3) !important;
  color: #fb923c !important;
}

/* agent button — subtle teal tint */
.action-bar-agent .stButton > button {
  border-color: rgba(34,211,160,.25) !important;
  color: #22d3a0 !important;
}

/* quit button — subtle red tint */
.action-bar-quit .stButton > button {
  border-color: rgba(244,63,94,.25) !important;
  color: #f43f5e !important;
}

/* Streamlit selectbox in auth card */
[data-testid="stSelectbox"] > div > div {
  background: #1a1d28 !important;
  border: 1px solid rgba(249,115,22,.35) !important;
  border-radius: 12px !important;
  color: #f0f2f8 !important;
  font-size: 20px !important;
  font-weight: 700 !important;
  min-height: 52px !important;
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

# Auth state
if "store_confirmed" not in st.session_state:
    st.session_state.store_confirmed = False
if "active_store" not in st.session_state:
    st.session_state.active_store = DEFAULT_STORE


# ---------------------------------------------------------------------------
# Store authentication gate  (blocks all chat UI until resolved)
# ---------------------------------------------------------------------------

def _confirm_store(store: str) -> None:
    st.session_state.active_store = store
    st.session_state.store_confirmed = True


if not st.session_state.store_confirmed:

    st.markdown(
        """
        <div style='max-width:420px;margin:60px auto 0;padding:32px 28px;
                    background:#14171f;border:1px solid rgba(255,255,255,.08);
                    border-radius:20px;box-shadow:0 12px 40px rgba(0,0,0,.55);'>

          <div style='text-align:center;margin-bottom:24px;'>
            <div style='font-size:36px;margin-bottom:8px;'>🏪</div>
            <div style='font-size:18px;font-weight:700;color:#f0f2f8;letter-spacing:-.3px;'>Store Verification</div>
            <div style='font-size:12px;color:#8892a4;margin-top:4px;'>chipLLM · Secure Access</div>
          </div>

          <div style='font-size:11px;color:#8892a4;text-transform:uppercase;
                      letter-spacing:.8px;margin-bottom:10px;'>Select your store</div>
        """,
        unsafe_allow_html=True,
    )

    selected_store = st.selectbox(
        "Select your store",
        options=STORE_OPTIONS,
        index=STORE_OPTIONS.index(DEFAULT_STORE),
        label_visibility="collapsed",
        key="store_select",
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if st.button(
        f"✅  Confirm Store #{selected_store}",
        use_container_width=True,
        key="auth_confirm",
    ):
        _confirm_store(selected_store)
        st.rerun()

    # Close the card div
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


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
            <div style='font-size:11px; font-weight:600; color:#f97316; text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:8px;'>
              ● Live Ticket Generated
            </div>
            """,
            unsafe_allow_html=True,
        )

        fields = [
            ("Ticket ID",      meta.get("ticket_id", "—"),             ""),
            ("Store ID",       meta.get("store_id", "—"),               ""),
            ("Asset Type",     meta.get("asset_type", "—"),             ""),
            ("Serial Number",  meta.get("asset_serial_number", "—"),    ""),
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
            # Reset auth so user is re-prompted on next load (optional safeguard)
            st.rerun()
    with col2:
        if st.button("🎫 Force Ticket", use_container_width=True):
            if st.session_state.messages:
                meta = extract_ticket_metadata(st.session_state.messages)
                meta["store_id"] = f"STORE-{st.session_state.active_store}"
                st.session_state.ticket_metadata = meta
                st.rerun()

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

st.markdown(
    f"""
    <div class="chip-header">
      <div class="chip-logo">🍔</div>
      <div class="chip-info">
        <div class="chip-name">chipLLM</div>
        <div class="chip-status">Tech Support · Online</div>
      </div>
      <div style='display:flex;align-items:center;gap:6px;'>
        <div style='background:rgba(249,115,22,.15);border:1px solid rgba(249,115,22,.35);
                    border-radius:8px;padding:3px 9px;font-size:11px;font-weight:600;
                    color:#f97316;letter-spacing:.3px;'>
          Store #{st.session_state.active_store}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Welcome message (injected once)
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                f"👋 **Hey there, Manager!** I'm **chipLLM**, your restaurant tech support assistant "
                f"for **Store #{st.session_state.active_store}**.\n\n"
                "I can help you troubleshoot:\n"
                "- \U0001f5a8\ufe0f **POS Printers** (jams, offline, receipt issues)\n"
                "- \U0001f4bb **POS Terminals** (crashes, payment hardware, login)\n"
                "- \U0001f4fa **Kitchen Displays / Waystations** (boot loops, display faults)\n"
                "- \U0001f916 **Self-Order Kiosks** (freezes, payment modules, scanner)\n\n"
                "What's going wrong? Describe the issue and I'll get you sorted fast."
            ),
            "timestamp": time.strftime("%H:%M"),
            "blocked": False,
        }
    )

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
            st.markdown(msg["content"])

        st.markdown(f'<div class="msg-time">{msg.get("timestamp", "")}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Bottom action bar
# ---------------------------------------------------------------------------

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

_ts = time.strftime("%H:%M")
_ab1, _ab2, _ab3, _ab4 = st.columns(4, gap="small")

with _ab1:
    st.markdown('<div class="action-bar-ticket">', unsafe_allow_html=True)
    if st.button("🎫\nOpen a\nTicket Now", use_container_width=True, key="btn_ticket"):
        meta = extract_ticket_metadata(st.session_state.messages)
        meta["store_id"] = f"STORE-{st.session_state.active_store}"
        st.session_state.ticket_metadata = meta
        st.session_state.escalation_triggered = True
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"🎫 **Ticket opened** for Store #{st.session_state.active_store}.\n\n"
                f"Ticket ID: `{meta['ticket_id']}`  \nSeverity: **{meta['severity']}**\n\n"
                "Your ticket has been submitted. A technician will follow up within the SLA window."
            ),
            "timestamp": _ts, "blocked": False,
        })
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with _ab2:
    st.markdown('<div class="action-bar-agent">', unsafe_allow_html=True)
    if st.button("🧑‍💻\nLive\nAgent", use_container_width=True, key="btn_agent"):
        meta = extract_ticket_metadata(st.session_state.messages)
        meta["store_id"] = f"STORE-{st.session_state.active_store}"
        meta["routing_target"] = "GENESYS_TIER1_PRIORITY"
        meta["auto_dispatch"] = True
        st.session_state.ticket_metadata = meta
        st.session_state.escalation_triggered = True
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🧑\u200d💻 **Connecting you to a live agent...**\n\n"
                       f"Your session for Store #{st.session_state.active_store} has been escalated to **Tier 1 Support**. "
                       "An agent will join this chat shortly. Please stay on the line.",
            "timestamp": _ts, "blocked": False,
        })
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Col 3 — Start Over
with st.container():
    if st.button("🔄\nStart\nOver", use_container_width=True, key="btn_restart"):
        st.session_state.messages = []
        st.session_state.ticket_metadata = None
        st.session_state.escalation_triggered = False
        st.session_state.rag_hits = {}
        st.rerun()

# Col 4 — Quit (returns to store selection)
with _ab4:
    st.markdown('<div class="action-bar-quit">', unsafe_allow_html=True)
    if st.button("❌\nQuit", use_container_width=True, key="btn_quit"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if user_input := st.chat_input(
    placeholder="Describe the issue… (e.g. 'Printer jammed')",
    key="chat_input",
):
    ts_now = time.strftime("%H:%M")

    # --- Store user message ------------------------------------------------
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

    # --- Layer 2: Escalation / resolution detection ------------------------
    if guard.escalation_triggered and not st.session_state.escalation_triggered:
        st.session_state.escalation_triggered = True
        meta = extract_ticket_metadata(st.session_state.messages)
        meta["store_id"] = f"STORE-{st.session_state.active_store}"
        st.session_state.ticket_metadata = meta

    # --- Layer 3: RAG retrieval --------------------------------------------
    relevant_playbooks = retrieve_context(user_input, top_k=1)
    rag_context: str | None = None
    rag_title: str | None = None

    if relevant_playbooks:
        playbook = relevant_playbooks[0]
        rag_context = playbook["content"]
        rag_title = playbook["title"]

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
    if not st.session_state.escalation_triggered:
        post_guard = check_guardrails(full_response)
        if post_guard.escalation_triggered:
            st.session_state.escalation_triggered = True
            meta = extract_ticket_metadata(st.session_state.messages)
            meta["store_id"] = f"STORE-{st.session_state.active_store}"
            st.session_state.ticket_metadata = meta

    st.rerun()
