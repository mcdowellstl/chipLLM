"""
knowledge_base.py
-----------------
RAG knowledge base engine for chipLLM.

Content lives entirely in GCS buckets (configured via env vars).
No client-specific content is hardcoded here — this file is
client-agnostic infrastructure only.

Each playbook entry contains:
  - keywords: list of trigger terms for lightweight fuzzy matching
  - title:    human-readable playbook title
  - content:  full troubleshooting playbook injected into context window

KB_AVAILABLE is set to False at module init if GCS is unreachable.
Callers (app.py) should check this flag and degrade gracefully rather
than letting the LLM hallucinate troubleshooting steps.
"""

import os
import re
import logging
from bs4 import BeautifulSoup
from google.cloud import storage

_log = logging.getLogger("chipLLM.knowledge_base")

# Will be set to False at module init if GCS is unreachable with no local fallback
KB_AVAILABLE: bool = True



def _strip_gs_prefix(value: str) -> str:
    """Normalizes a GCS bucket identifier: strips leading 'gs://' if present.
    Accepts both 'gs://my-bucket' (Cloud Run style) and 'my-bucket' (bare name).
    """
    return value.removeprefix("gs://")


L0_BUCKET    = _strip_gs_prefix(os.environ.get("KNOWLEDGE_STORAGE_BUCKET", "chipllm-l0-playbooks"))
ADHOC_BUCKET = _strip_gs_prefix(os.environ.get("KNOWLEDGE_ADHOC_BUCKET",   "chipllm-adhoc-playbooks"))


def get_file_list(bucket_name: str) -> list[str]:
    """
    Returns the live list of HTML blob names in the given GCS bucket.
    Always performs a fresh list_blobs() call — never cached.
    Callers use this to build a live manifest for diffing against cached content.
    """
    try:
        client = storage.Client()
        return [
            blob.name
            for blob in client.list_blobs(bucket_name)
            if blob.name.endswith(".html")
        ]
    except Exception as exc:
        _log.warning("get_file_list failed for bucket '%s': %s", bucket_name, exc)
        return []


def fetch_file_content(bucket_name: str, blob_name: str) -> dict | None:
    """
    Downloads and parses a single HTML playbook blob.
    Returns a structured playbook dict, or None on any error.
    NOTE: This function is intentionally cache-free. Callers in app.py
    wrap it with @st.cache_data so caching stays in the Streamlit layer.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        html_text = blob.download_as_text()
        soup = BeautifulSoup(html_text, "html.parser")

        meta_id = soup.find("meta", attrs={"name": "id"})
        meta_keywords = soup.find("meta", attrs={"name": "keywords"})
        body_text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""

        record_id = meta_id["content"] if meta_id else blob_name
        keywords_list = (
            [k.strip().lower() for k in meta_keywords["content"].split(",")]
            if meta_keywords else []
        )
        title_text = (
            soup.title.string.strip()
            if (soup.title and soup.title.string)
            else record_id.replace("_", " ").title()
        )

        return {
            "id": record_id,
            "keywords": keywords_list,
            "title": title_text,
            "content": body_text,
        }
    except Exception as exc:
        _log.warning("fetch_file_content failed for gs://%s/%s: %s", bucket_name, blob_name, exc)
        return None


def build_playbook_pool(
    l0_names: list[str],
    adhoc_names: list[str],
    fetch_fn,
) -> list[dict]:
    """
    Builds the merged playbook list from pre-filtered live manifests.
    `fetch_fn(bucket_name, blob_name)` is a callable — in production this
    will be the @st.cache_data-wrapped version supplied by app.py.

    Merge rules:
    - L0 baseline entries are loaded first.
    - Ad-hoc entries overwrite any matching record_id.
    - Blobs not present in the live manifest are simply never fetched,
      which effectively evicts deleted files from the active pool.
    """
    parsed_ledger: dict[str, dict] = {}

    for blob_name in l0_names:
        entry = fetch_fn(L0_BUCKET, blob_name)
        if entry:
            entry = dict(entry, source="l0_baseline")
            parsed_ledger.setdefault(entry["id"], entry)

    for blob_name in adhoc_names:
        entry = fetch_fn(ADHOC_BUCKET, blob_name)
        if entry:
            entry = dict(entry, source="adhoc")
            parsed_ledger[entry["id"]] = entry  # always overwrite

    return list(parsed_ledger.values())


def load_and_merge_cloud_knowledge_base() -> list[dict]:
    """
    Loads playbooks exclusively from GCS. No local or hardcoded fallbacks.
    Sets the module-level KB_AVAILABLE flag to reflect whether content was loaded.
    Returns an empty list if GCS is unreachable — callers should check KB_AVAILABLE.
    """
    global KB_AVAILABLE
    pool = build_playbook_pool(
        l0_names=get_file_list(L0_BUCKET),
        adhoc_names=get_file_list(ADHOC_BUCKET),
        fetch_fn=fetch_file_content,
    )

    if pool:
        KB_AVAILABLE = True
        _log.info("Knowledge base loaded: %d playbooks from GCS.", len(pool))
    else:
        KB_AVAILABLE = False
        _log.error(
            "GCS returned no playbooks (buckets: L0=%s, adhoc=%s). "
            "KB_AVAILABLE=False — app will degrade gracefully.",
            L0_BUCKET, ADHOC_BUCKET
        )

    return pool


# Grounding runtime context vector array initialization
PLAYBOOKS = load_and_merge_cloud_knowledge_base()
playbooks_pool = PLAYBOOKS


# ---------------------------------------------------------------------------
# RAG retrieval – lightweight keyword matching
# ---------------------------------------------------------------------------

ABBREVIATIONS = {
    "wst": "waystation",
    "waistation": "waystation",
    "kds": "kitchen display system",
    "kvs": "kitchen video system",
    "pos": "point of sale",
    "bos": "back office system",
}

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def fuzzy_match_ratio(s1: str, s2: str) -> float:
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - (dist / max_len)

def preprocess_text(text: str) -> list[str]:
    # Lowercase and clean
    cleaned = text.lower().strip()
    words = re.findall(r'\b\w+\b', cleaned)
    resolved_words = []
    for w in words:
        resolved_words.append(ABBREVIATIONS.get(w, w))
    return resolved_words

def retrieve_context(user_message: str, top_k: int = 1) -> list[dict]:
    """
    Perform fuzzy matching retrieval against the playbook knowledge base.
    Handles typos, abbreviations, and implements a two-pass context builder.
    Ad-hoc playbooks bypass truncation limits and sit at the top.

    Scoring strategy:
    - Exact keyword phrase hits score up to 2.0 (boosted) to ensure genuinely
      relevant playbooks beat ones that only share incidental words.
    - Single-word fuzzy hits only count when ratio >= 0.85 to reduce false
      positives (e.g. "not" in "Cash Drawer Not Opening" matching "not printing").
    - Full-phrase substring hits score 0.6–1.0 against title/id candidates.
    - Final threshold to be included in results: 0.6.
    """
    query_words = preprocess_text(user_message)
    if not query_words:
        return []

    STOPWORDS = {"my", "is", "the", "a", "an", "on", "of", "to", "in", "at",
                 "for", "with", "and", "or", "having", "it", "are", "you"}
    query_clean = " ".join(query_words)
    scored: list[tuple[float, dict]] = []

    for playbook in PLAYBOOKS:
        # Separate keyword candidates (high-signal) from title/id (lower signal)
        kw_candidates  = [kw.lower() for kw in playbook.get("keywords", [])]
        id_title_cands = []
        if playbook.get("id"):
            id_title_cands.append(playbook["id"].lower().replace("_", " "))
        if playbook.get("title"):
            id_title_cands.append(playbook["title"].lower())
        all_candidates = id_title_cands + kw_candidates

        best_score = 0.0

        # --- Pass A: exact keyword phrase match (boosted up to 2.0) ---
        for kw in kw_candidates:
            if query_clean in kw or kw in query_clean:
                length_ratio = (
                    min(len(query_clean), len(kw)) / max(len(query_clean), len(kw))
                    if max(len(query_clean), len(kw)) > 0 else 1.0
                )
                # Boost: keyword phrase hits score in range [1.0, 2.0]
                kw_score = 1.0 + length_ratio
                if kw_score > best_score:
                    best_score = kw_score
            # Full-phrase fuzzy match against keyword (boosted by 1.0)
            r = fuzzy_match_ratio(query_clean, kw)
            boosted = r + 1.0 if r >= 0.75 else r
            if boosted > best_score:
                best_score = boosted

        # --- Pass B: title/id substring match (unboosted, 0.6–1.0) ---
        for cand in id_title_cands:
            if query_clean in cand or cand in query_clean:
                score = (
                    min(len(query_clean), len(cand)) / max(len(query_clean), len(cand))
                    if max(len(query_clean), len(cand)) > 0 else 1.0
                )
                score = max(0.6, score)
                if score > best_score:
                    best_score = score
            r = fuzzy_match_ratio(query_clean, cand)
            if r > best_score:
                best_score = r

        # --- Pass C: single-word fuzzy match (high threshold to avoid noise) ---
        for q_word in query_words:
            if len(q_word) < 3 or q_word in STOPWORDS:
                continue
            for cand in all_candidates:
                for c_word in re.findall(r'\b\w+\b', cand):
                    r = fuzzy_match_ratio(q_word, c_word)
                    # Only count strong single-word matches to avoid false positives
                    if r >= 0.85 and r > best_score:
                        best_score = r

        # Include if above the minimum relevance threshold
        if best_score >= 0.6:
            scored.append((best_score, playbook))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    matched_playbooks = [p for _, p in scored]

    # Pass 2: Separate adhoc and baseline matches
    adhoc_matches  = [p for p in matched_playbooks if p.get("source") == "adhoc"]
    baseline_matches = [p for p in matched_playbooks if p.get("source") != "adhoc"]

    # Ad-hoc matches bypass length truncation and sit at the top.
    # Baseline matches are limited to top_k.
    return adhoc_matches + baseline_matches[:top_k]

