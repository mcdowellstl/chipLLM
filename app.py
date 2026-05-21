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

/* ── Chat scroll container ───────────────────────────────────────────────── */
.chat-scroll {
  padding: 16px 14px 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 60vh;
}

/* ── Message bubbles ─────────────────────────────────────────────────────── */
.msg-row {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  animation: slide-in .2s ease-out both;
}

@keyframes slide-in {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

.msg-row.user { flex-direction: row-reverse; }

.msg-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  flex-shrink: 0;
}

.msg-avatar.bot-av {
  background: linear-gradient(135deg, var(--accent) 0%, #ea580c 100%);
  box-shadow: 0 2px 10px var(--accent-glow);
}

.msg-avatar.user-av {
  background: linear-gradient(135deg, #1e3a5f 0%, #1a4a7a 100%);
}

.msg-bubble {
  max-width: 82%;
  padding: 12px 16px;
  border-radius: var(--radius);
  font-size: 14px;
  line-height: 1.55;
  word-wrap: break-word;
  position: relative;
}

.msg-bubble.bot {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-bottom-left-radius: 4px;
  color: var(--text-primary);
}

.msg-bubble.user {
  background: linear-gradient(135deg, #1e3a5f 0%, #1a4a7a 100%);
  border-bottom-right-radius: 4px;
  color: #ddeeff;
}

.msg-bubble.blocked {
  background: linear-gradient(135deg, rgba(244,63,94,.15) 0%, rgba(244,63,94,.08) 100%);
  border: 1px solid rgba(244,63,94,.3);
  color: #fda4af;
}

.msg-time {
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 4px;
  padding: 0 4px;
}

.msg-row.user .msg-time { text-align: right; }

/* ── Typing indicator ────────────────────────────────────────────────────── */
.typing-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
}

.typing-dots {
  display: flex;
  gap: 4px;
}

.typing-dots span {
  width: 7px;
  height: 7px;
  background: var(--accent);
  border-radius: 50%;
  animation: bounce 1.2s ease-in-out infinite;
}

.typing-dots span:nth-child(2) { animation-delay: .2s; }
.typing-dots span:nth-child(3) { animation-delay: .4s; }

@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
  40%           { transform: translateY(-8px); opacity: 1; }
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
            st.rerun()
    with col2:
        if st.button("🎫 Force Ticket", use_container_width=True):
            if st.session_state.messages:
                st.session_state.ticket_metadata = extract_ticket_metadata(
                    st.session_state.messages
                )
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
    """
    <div class="chip-header">
      <div class="chip-logo">🍔</div>
      <div class="chip-info">
        <div class="chip-name">chipLLM</div>
        <div class="chip-status">Tech Support · Online</div>
      </div>
      <div style='font-size:20px; cursor:pointer;'>⚡</div>
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
                "👋 **Hey there, Manager!** I'm **chipLLM**, your restaurant tech support assistant.\n\n"
                "I can help you troubleshoot:\n"
                "- 🖨️ **POS Printers** (jams, offline, receipt issues)\n"
                "- 💻 **POS Terminals** (crashes, payment hardware, login)\n"
                "- 📺 **Kitchen Displays / Waystations** (boot loops, display faults)\n"
                "- 🤖 **Self-Order Kiosks** (freezes, payment modules, scanner)\n\n"
                "What's going wrong? Give me your **Store ID** and a quick description."
            ),
            "timestamp": time.strftime("%H:%M"),
            "blocked": False,
        }
    )

# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

for i, msg in enumerate(st.session_state.messages):
    is_user = msg["role"] == "user"
    is_blocked = msg.get("blocked", False)
    ts = msg.get("timestamp", "")

    avatar_html = (
        '<div class="msg-avatar user-av">👤</div>'
        if is_user
        else '<div class="msg-avatar bot-av">🤖</div>'
    )

    bubble_class = "user" if is_user else ("blocked" if is_blocked else "bot")
    row_class = "user" if is_user else "bot"

    # RAG badge
    rag_badge = ""
    if not is_user and i in st.session_state.rag_hits:
        rag_badge = (
            f'<div class="rag-badge">📚 KB: {st.session_state.rag_hits[i]}</div>'
        )

    # Render text via st.markdown inside a styled container
    with st.container():
        st.markdown(
            f"""
            <div class="msg-row {row_class}">
              {avatar_html}
              <div>
                {rag_badge}
                <div class="msg-bubble {bubble_class}">
            """,
            unsafe_allow_html=True,
        )
        st.markdown(msg["content"])
        st.markdown(
            f"""
                </div>
                <div class="msg-time">{ts}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if user_input := st.chat_input(
    placeholder="Describe the issue… (e.g. 'Printer jammed, Store 4421')",
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
        st.session_state.ticket_metadata = extract_ticket_metadata(
            st.session_state.messages
        )

    # --- Layer 3: RAG retrieval --------------------------------------------
    relevant_playbooks = retrieve_context(user_input, top_k=1)
    rag_context: str | None = None
    rag_title: str | None = None

    if relevant_playbooks:
        playbook = relevant_playbooks[0]
        rag_context = playbook["content"]
        rag_title = playbook["title"]

    # --- Layer 4: LLM call (streaming) -------------------------------------
    # ChipLLMClient is lightweight; the underlying genai.Client is process-cached
    # via @st.cache_resource in llm_client.py — no new connection is created here.
    client = ChipLLMClient()

    # Placeholder for the streaming response
    with st.spinner(""):
        full_response = ""
        response_placeholder = st.empty()

        try:
            for chunk in client.stream_response(
                messages=st.session_state.messages,
                rag_context=rag_context,
            ):
                full_response += chunk
                response_placeholder.markdown(
                    f'<div class="msg-bubble bot" style="max-width:100%;padding:12px 16px;">'
                    f"{full_response}▌"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            response_placeholder.empty()

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
            st.session_state.ticket_metadata = extract_ticket_metadata(
                st.session_state.messages
            )

    st.rerun()
