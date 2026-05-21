# CLAUDE.md — chipLLM Agent Constitution
> **For any agent (human or AI) working in this repo.**
> Read this file before writing a single line of code.
> These rules are non-negotiable architectural decisions made by the Lead Architect.
> Do not override them without an explicit instruction from the CTO.

---

## 1. Project Identity

| Property | Value |
|---|---|
| **Project** | chipLLM |
| **Purpose** | Mobile-focused restaurant-tech support chatbot PoC |
| **Runtime** | Google Cloud Run (serverless, stateless containers) |
| **Backend** | Vertex AI — `gemini-2.0-flash-001` via `google-genai` SDK |
| **Frontend** | Streamlit (single-page, mobile-viewport-constrained) |
| **Domain** | Restaurant technology only: POS, Printers, Kiosks, KDS/KVS |

---

## 2. Module Boundaries — Touch Nothing You Don't Own

Each file has a single responsibility. Agents must not collapse modules or add cross-cutting logic outside its designated owner.

| File | Owner Responsibility | What Does NOT Belong Here |
|---|---|---|
| `app.py` | Streamlit UI, session state, orchestration loop | No LLM calls, no raw SDK imports, no business logic |
| `guardrails.py` | Layer 1 keyword filtering + intent signals | No LLM calls, no Streamlit imports, no RAG |
| `knowledge_base.py` | Playbook data + `retrieve_context()` retrieval | No LLM calls, no session state, no HTTP |
| `llm_client.py` | SDK wrapper, context assembly, ticket extraction, **ADC client factory** | No guardrail logic — `import streamlit` is permitted here **exclusively** for the `@st.cache_resource` decorator on `get_genai_client()` |

**Rule:** If a new feature touches more than two modules, stop and create a new dedicated module.

---

## 3. Token-Saving Rules — Enforced, Not Suggested

These rules exist to minimize Vertex AI token spend at $100M-account scale.

### 3.1 Guardrails Run First, Always
```
User message → guardrails.check_guardrails() → [BLOCKED?] → return static string
                                               → [PASS] → RAG → LLM
```
- **Blocked messages never reach the LLM.** The refusal is a hardcoded string.
- Do NOT restructure this pipeline to call the LLM first and filter after.
- Do NOT add LLM-based intent classification — local keyword sets are intentional.

### 3.2 RAG Context Is Injected Once, Into the Final User Turn Only
- `llm_client.ChipLLMClient.build_contents()` injects RAG text into `messages[-1]` only.
- Never inject context into every message in the history — this multiplies token cost linearly.
- RAG context is raw playbook markdown, prepended with a `[RELEVANT KNOWLEDGE BASE CONTEXT]` sentinel the model is instructed to use.

### 3.3 Model Parameters Are Fixed for Consistency
```python
temperature   = 0.3   # Low creativity — this is a support bot, not a poet
max_output_tokens = 1024  # Hard ceiling — no runaway responses
top_p         = 0.9
```
Do not raise `max_output_tokens` without CTO approval. Verbose responses increase cost and slow mobile UX.

### 3.4 Streaming Is Mandatory
- Always use `generate_content_stream()`, never `generate_content()`.
- Streaming lets the UI render progressively — critical for mobile perceived performance.
- Do not buffer the full response before rendering.

### 3.5 History Window Management (Future Agents: Implement This Before Adding Memory)
- `st.session_state.messages` is the **only** conversation store. No DB, no vector store, no external cache.
- If history grows beyond ~20 turns, implement a **sliding window** (keep last 12 turns + system context).
  Trim from the front, never the back. Never truncate the most recent user message.
- Do NOT implement persistent cross-session memory without explicit CTO instruction.

---

## 4. Guardrail Architecture — Rules for Extension

### 4.1 Adding a New Restricted Topic
1. Add a new `frozenset` or `set` named `_<TOPIC>_KEYWORDS` in `guardrails.py`.
2. Union it into `_ALL_RESTRICTED`.
3. That is the complete change. Do not touch `app.py`.

### 4.2 Adding an Escalation Keyword
- Add to `_ESCALATION_KEYWORDS` in `guardrails.py`.
- `check_guardrails()` already returns `escalation_triggered=True` — `app.py` picks it up automatically.

### 4.3 The Refusal String Is Static and Hardcoded
- `_REFUSAL_RESPONSE` in `guardrails.py` must remain a static string.
- Do NOT make it dynamic, templated, or LLM-generated. Latency and cost must be zero.

---

## 5. RAG Knowledge Base — Rules for Extension

### 5.1 Adding a New Playbook
Add a new dict to the `PLAYBOOKS` list in `knowledge_base.py` with this exact schema:
```python
{
    "id":       "UNIQUE_SNAKE_CASE_ID",       # str, required
    "keywords": ["keyword1", "multi word"],   # list[str], all lowercase
    "title":    "Human-Readable Title",       # str
    "content":  "## Markdown playbook...",    # str, full troubleshooting content
}
```
- Keywords must be **lowercase**. The retriever lowercases input before matching.
- Prefer multi-word keywords for precision (e.g., `"kitchen display"` over `"display"`).
- Each playbook should end with a `### Metadata Tags` block: `severity | asset_type | sla`.

### 5.2 Retrieval Is Keyword Hit-Count, Not Embedding
- `retrieve_context()` scores by keyword overlap count. No vector DB. No embeddings. Intentional.
- `top_k=1` by default — only the best-matching playbook is injected per turn.
- Do NOT add embedding-based retrieval without creating a new module (`rag_semantic.py`) that
  leaves `knowledge_base.py` untouched.

---

## 6. Ticket Metadata — ServiceNow / Genesys Payload Contract

The JSON payload emitted by `extract_ticket_metadata()` is a **contract**. Do not rename or remove fields.
Downstream integrations (ServiceNow, Genesys) depend on this exact schema:

```jsonc
{
  "ticket_id":            "CHK-XXXXXXXX",   // chipLLM-generated, ephemeral
  "created_at":           "ISO8601Z",
  "store_id":             "STORE-NNNN",     // extracted via regex or STORE-UNCONFIRMED
  "asset_type":           "POS_TERMINAL | PRINTER | KIOSK | KDS_KVS | UNKNOWN",
  "asset_serial_number":  "SN-PENDING",     // regex-extracted or pending
  "severity":             "CRITICAL | HIGH | MEDIUM | LOW",
  "description":          "string",         // last 200 chars of last assistant turn
  "routing_target":       "GENESYS_TIER2_IMMEDIATE | GENESYS_TIER1_PRIORITY | SERVICENOW_AUTO_TICKET | SERVICENOW_STANDARD_QUEUE",
  "channel":              "chipLLM_CHATBOT",
  "sla_breach_minutes":   30 | 60 | 120 | 480,
  "auto_dispatch":        true | false
}
```

**Adding fields:** Append only. Never rename or delete existing keys.
**Severity → SLA mapping is fixed:**
- CRITICAL → 30 min → GENESYS_TIER2_IMMEDIATE
- HIGH     → 60 min → GENESYS_TIER1_PRIORITY
- MEDIUM   → 120 min → SERVICENOW_AUTO_TICKET
- LOW      → 480 min → SERVICENOW_STANDARD_QUEUE

---

## 7. UI / CSS Architecture Rules

### 7.1 Mobile-First, Always
- `max-width: 420px` on `.main .block-container` is the mobile viewport lock. Do not remove it.
- All touch targets (buttons, inputs) must have `min-height: 44px` (Apple HIG standard).
- Font size floor: `13px`. Nothing smaller on mobile.

### 7.2 Design Tokens Are in CSS Variables
All colors, radii, shadows live in `:root` in `app.py`'s CSS block:
```css
:root {
  --accent: #f97316;       /* kitchen orange — brand color */
  --bg-app: #0d0f14;
  --success: #22d3a0;
  --danger:  #f43f5e;
  /* ... */
}
```
- Never hardcode hex values in component CSS. Always use a `var(--token)`.
- Adding a new color = add a token to `:root` first, then reference it.

### 7.3 Streamlit Chrome Is Hidden
```css
#MainMenu, header[data-testid="stHeader"], footer { display: none !important; }
```
This is intentional for the full-screen mobile app feel. Do not re-enable it.

### 7.4 Animations Are Micro Only
- Existing animations: `slide-in` (messages), `pulse-dot` (status), `bounce` (typing indicator).
- No full-page transitions, no heavy keyframe sequences. This runs on kitchen tablets and phones.

---

## 8. Session State Contract

`st.session_state` keys and their invariants:

| Key | Type | Invariant |
|---|---|---|
| `messages` | `list[dict]` | Each dict has `role`, `content`, `timestamp`, `blocked`. Never mutate in place — append only. |
| `ticket_metadata` | `dict \| None` | `None` until escalation triggered. Set by `extract_ticket_metadata()` only. |
| `escalation_triggered` | `bool` | Monotonically true — never reset to False mid-session except via "Clear Chat". |
| `rag_hits` | `dict[int, str]` | Maps message index → playbook title for RAG badge rendering. |

> **Note:** `llm_client` is no longer stored in session_state. The underlying `genai.Client` is a process-level singleton managed by `@st.cache_resource` in `llm_client.py`. `ChipLLMClient()` is instantiated inline per message — it is trivially cheap because it only calls `get_genai_client()` which returns the cached object.

---

## 9. Environment & Deployment Rules

### 9.1 Auth — ADC Only (No API Keys, Ever)
Authentication is exclusively via **Application Default Credentials (ADC)**.

```
Local dev:   gcloud auth application-default login
Cloud Run:   Workload Identity — service account attached to the Cloud Run revision
```

- `genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)` — this is the **only** permitted initialization pattern.
- `GOOGLE_API_KEY` is **prohibited** in all code and config files. It is not a valid fallback.
- Service account JSON key files are **prohibited**. Use `roles/aiplatform.user` on the service account, not downloaded keys.
- The `get_genai_client()` function in `llm_client.py` is decorated with `@st.cache_resource` — it creates the client **once per server process** and reuses it. Do not create additional client instances.

### 9.2 Model Selection
- Default model: `gemini-2.0-flash-001`
- Override via env var: `CHIPLLM_MODEL=gemini-2.0-flash-lite-001` (for cost reduction experiments)
- Do NOT hardcode a model string in any file other than `llm_client.py`.

### 9.3 Cloud Run Constraints
- Container must be stateless. No file writes at runtime. No sqlite. No local cache that persists across instances.
- Port is always `8080`. Configured in both `Dockerfile` and `.streamlit/config.toml`.
- Non-root user (`chipllm`) is required. Do not change the Dockerfile USER directive.

---

## 10. What Agents Must Never Do

| ❌ Prohibited Action | ✅ Correct Alternative |
|---|---|
| Call the LLM before running guardrails | Always run `check_guardrails()` first |
| Add `import streamlit` to `guardrails.py` or `knowledge_base.py` | Only `app.py` and `llm_client.py` (for `@st.cache_resource`) may import streamlit |
| Use `generate_content()` (non-streaming) | Always use `generate_content_stream()` |
| Inject RAG context into every message in history | Inject into `messages[-1]` only |
| Store conversation history in a database | Use `st.session_state.messages` exclusively |
| Raise `max_output_tokens` above 1024 without approval | Keep the ceiling at 1024 |
| Rename or remove fields from the ticket JSON payload | Append new fields only |
| Hardcode hex color values in CSS | Use CSS custom properties (`var(--token)`) |
| Remove the mobile viewport constraint | `max-width: 420px` is a product requirement |
| Add a new playbook without the full schema | Follow the schema in Section 5.1 exactly |
| Use `GOOGLE_API_KEY` or a service account JSON key path | Use ADC — `gcloud auth application-default login` |
| Create a second `genai.Client()` anywhere in the codebase | Call `get_genai_client()` from `llm_client.py` — it is the single source of truth |
| Store `llm_client` in `st.session_state` | The client is managed by `@st.cache_resource`; instantiate `ChipLLMClient()` inline |

---

## 11. Recommended Extension Points (Pre-Approved Patterns)

These are the **right** way to grow the system without breaking it:

- **New playbook** → add dict to `PLAYBOOKS` list in `knowledge_base.py`
- **New restricted topic** → add keyword set + union into `_ALL_RESTRICTED` in `guardrails.py`
- **New escalation keyword** → add to `_ESCALATION_KEYWORDS` in `guardrails.py`
- **New severity level or SLA** → update both `extract_ticket_metadata()` and this doc
- **Semantic RAG** → create `rag_semantic.py`; swap `retrieve_context` import in `app.py` only
- **Persistent history** → create `session_store.py`; do not modify `app.py` message append logic
- **New LLM model** → set `CHIPLLM_MODEL` env var; zero code changes required
- **A/B prompt testing** → parameterize `SYSTEM_PROMPT` in `llm_client.py`; keep the single constant name

---

*Last updated: 2026-05-20 by chipLLM Lead Architect*
*Any agent modifying rules in this file must leave a dated comment explaining the change.*
