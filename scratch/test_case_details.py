import sys
import os

# Parse .env file manually and load into os.environ
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from mocks.servicenow import get_store_tickets

if "active_tickets" not in st.session_state:
    st.session_state.active_tickets = get_store_tickets("67067")
if "active_store" not in st.session_state:
    st.session_state.active_store = "67067"

from llm_client import ChipLLMClient, get_case_details

def test_case_details_tool():
    """Directly test the get_case_details tool function."""
    print("\n=== Direct Tool Test: get_case_details ===")
    result = get_case_details("RC001024")
    print(f"Tool result for RC001024: {result}")
    assert result.get("id") == "RC001024", f"Expected RC001024, got {result.get('id')}"
    assert "comments" in result, "Missing comments field"
    print("PASS: get_case_details returned correct case data.\n")

def test_case_detail_streaming():
    """Test the full LLM streaming response for a case detail request."""
    print("=== Streaming Test: Case Detail View ===")
    client = ChipLLMClient()

    messages = [
        {"role": "user", "content": "Can you get more details on RC001024?"}
    ]

    print(f"User: {messages[-1]['content']}\nChip: ", end="", flush=True)
    full = []
    for chunk in client.stream_response(messages):
        print(chunk, end="", flush=True)
        full.append(chunk)
    output = "".join(full)
    print("\n--------------------------------------")

    assert "RC001024" in output, "Missing ticket ID in detail view"
    assert "Recent Activity" in output or "Comments" in output, "Missing comments section"
    assert "What would you like to do" in output, "Missing follow-up prompt"
    assert "| **" not in output, "Found bold category prefix formatting — forbidden!"
    print("\nPASS: Case detail view formatted correctly.")

if __name__ == "__main__":
    test_case_details_tool()
    test_case_detail_streaming()
