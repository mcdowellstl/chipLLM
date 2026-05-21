import os
import sys

# Set up environment variables first to avoid importing defaults
os.environ["GOOGLE_CLOUD_PROJECT"] = "chip-496911"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-east1"

sys.path.append("/Users/jeffmickd/src/chipllm")

import time
from guardrails import check_guardrails
from knowledge_base import retrieve_context
from llm_client import ChipLLMClient
from app import clean_assistant_message, get_choices_from_message

print("Initializing test...")
print(f"Project ID: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
print(f"Location: {os.environ.get('GOOGLE_CLOUD_LOCATION')}")

user_input = "printer"
print(f"User Input: '{user_input}'")

# Guardrails
guard = check_guardrails(user_input)
print(f"Guardrail blocked: {guard.blocked}, escalation: {guard.escalation_triggered}")

# Context
relevant_playbooks = retrieve_context(user_input, top_k=1)
rag_context = None
rag_title = None
if relevant_playbooks:
    playbook = relevant_playbooks[0]
    rag_context = playbook["content"]
    rag_title = playbook["title"]
    print(f"RAG context title: '{rag_title}'")

# Messages setup
messages = [
    {
        "role": "assistant",
        "content": (
            "Hello! I am chipLLM, your restaurant tech support virtual engineer. "
            "I'll help you troubleshoot on-site issues and if we can't resolve it here, "
            "I'll create a support ticket or get you over to a live chat agent.\n\n"
            "Please describe your issue."
        ),
        "timestamp": time.strftime("%H:%M"),
        "blocked": False
    },
    {
        "role": "user",
        "content": user_input,
        "timestamp": time.strftime("%H:%M"),
        "blocked": False,
    }
]

print("Calling stream_response...")
client = ChipLLMClient()
full_response = ""
try:
    for chunk in client.stream_response(messages=messages, rag_context=rag_context):
        full_response += chunk
    print(f"LLM Response:\n{full_response}\n---")
except Exception as e:
    print("LLM Call Failed:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Calling clean_assistant_message...")
try:
    cleaned = clean_assistant_message(full_response)
    print(f"Cleaned response:\n{cleaned}")
    
    choices = get_choices_from_message(full_response)
    print(f"Choices extracted: {choices}")
except Exception as e:
    print("Cleaning Failed:")
    import traceback
    traceback.print_exc()
