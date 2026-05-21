import sys
import os

# Set up paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from knowledge_base import PLAYBOOKS, retrieve_context

print("=== VERIFYING DYNAMIC RAG LOADING ===")
print(f"Total playbooks loaded: {len(PLAYBOOKS)}")

if len(PLAYBOOKS) >= 30:
    print("✅ SUCCESS: 30+ playbooks loaded.")
else:
    print("❌ ERROR: Less than 30 playbooks loaded.")

print("\n=== TESTING RETRIEVAL KERNEL ===")
queries = [
    "printer is jammed",
    "KDS monitor is blank",
    "card reader check",
    "kiosk boot loop"
]

for q in queries:
    hits = retrieve_context(q, top_k=1)
    if hits:
        print(f"Query: '{q}' -> Matched KB: '{hits[0]['title']}' (ID: {hits[0]['id']})")
    else:
        print(f"Query: '{q}' -> ❌ No match found.")

print("\nVerification completed successfully.")
