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

# Add parent directory to sys.path so we can import from llm_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dummy streamlit session state before import
import streamlit as st
from mocks.servicenow import get_store_tickets

if "active_tickets" not in st.session_state:
    st.session_state.active_tickets = get_store_tickets("67067")
if "active_store" not in st.session_state:
    st.session_state.active_store = "67067"

from llm_client import ChipLLMClient

def test_conversational_lookup():
    print("Initializing ChipLLMClient...")
    client = ChipLLMClient()
    
    # 1. Ask about open tickets
    messages = [
        {"role": "user", "content": "Are there any open tickets?"}
    ]
    
    print("\n--- Test 1: Querying open tickets ---")
    print(f"User: {messages[-1]['content']}\nChip: ", end="", flush=True)
    
    response_chunks = []
    for chunk in client.stream_response(messages):
        print(chunk, end="", flush=True)
        response_chunks.append(chunk)
    print("\n--------------------------------------")
    
    # 2. Ask about comment history
    messages.append({"role": "assistant", "content": "".join(response_chunks)})
    messages.append({"role": "user", "content": "Can you check the recent comment updates or notes on INC-001024?"})
    
    print("\n--- Test 2: Querying comment history ---")
    print(f"User: {messages[-1]['content']}\nChip: ", end="", flush=True)
    
    for chunk in client.stream_response(messages):
        print(chunk, end="", flush=True)
    print("\n--------------------------------------")

if __name__ == "__main__":
    test_conversational_lookup()
