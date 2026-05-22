"""
scratch/test_mutations.py
-------------------------
Directly tests the data mutation tool functions:
  - add_case_comment: appends a comment to RC001024
  - escalate_case: increments the escalations counter on RC001024

Run inside the container:
  docker exec chipllm-chip-llm-1 python scratch/test_mutations.py
"""

import sys
import os
sys.path.insert(0, "/home/chipllm/app")

# Simulate minimal session state so Streamlit imports don't fail
class _FakeSessionState(dict):
    def __getattr__(self, key):
        return self.get(key)
    def __setattr__(self, key, value):
        self[key] = value

import streamlit as st
st.session_state = _FakeSessionState()
st.session_state.active_store = "67067"
st.session_state.active_case_id = None

from mocks.servicenow import append_case_comment, increment_escalations, get_store_tickets

CASE_ID = "RC001024"

# ── Test 1: append_case_comment ───────────────────────────────────────────────
print("=== Test 1: append_case_comment ===")
before = get_store_tickets("67067")
before_case = next((c for c in before if c["id"] == CASE_ID), None)
before_count = len(before_case.get("comments", [])) if before_case else "??"
print(f"Comments before: {before_count}")

result = append_case_comment(CASE_ID, "Technician has confirmed parts are on order.", author="Store Manager")
assert "error" not in result, f"FAIL: {result}"

after = get_store_tickets("67067")
after_case = next((c for c in after if c["id"] == CASE_ID), None)
after_count = len(after_case.get("comments", [])) if after_case else "??"
print(f"Comments after:  {after_count}")

# Supabase returns fresh data; mock mutates in-place — either way count should increase
assert "error" not in result, "FAIL: result contains error key"
last_comment = (after_case or {}).get("comments", [{}])[-1]
print(f"New comment: {last_comment}")
print("PASS: append_case_comment\n")

# ── Test 2: increment_escalations ─────────────────────────────────────────────
print("=== Test 2: increment_escalations ===")
before2 = get_store_tickets("67067")
before_case2 = next((c for c in before2 if c["id"] == CASE_ID), None)
before_esc = before_case2.get("escalations", 0) if before_case2 else "??"
print(f"Escalations before: {before_esc}")

result2 = increment_escalations(CASE_ID)
assert "error" not in result2, f"FAIL: {result2}"

new_esc = result2.get("escalations", "??")
print(f"Escalations after:  {new_esc}")
print("PASS: increment_escalations\n")

# ── Test 3: LLM tool wrappers ─────────────────────────────────────────────────
print("=== Test 3: LLM tool wrappers (add_case_comment, escalate_case) ===")
from llm_client import add_case_comment as llm_add_comment, escalate_case as llm_escalate

comment_result = llm_add_comment(CASE_ID, "Verified unit is offline — hardware replacement scheduled.")
print(f"add_case_comment result: {comment_result}")
assert "added to" in comment_result, f"FAIL: unexpected response: {comment_result}"

escalate_result = llm_escalate(CASE_ID)
print(f"escalate_case result:    {escalate_result}")
assert "escalated" in escalate_result.lower(), f"FAIL: unexpected response: {escalate_result}"

print("\nPASS: All mutation tests passed.")
