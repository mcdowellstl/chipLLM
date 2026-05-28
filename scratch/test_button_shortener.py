import os
import sys

os.environ["GOOGLE_CLOUD_PROJECT"] = "chip-496911"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-east1"

sys.path.append("/Users/jeffmickd/src/chipllm")

from app import shorten_choice_via_llm, get_choices_from_message

print("Testing static mappings...")
tests = {
    "Not Printing at All": "Not Printing",
    "not printing at all": "Not Printing",
    "Printing Garbled Text": "Garbled Text",
    "printing garbled text": "Garbled Text",
    "Error Light / Beeping": "Error Light",
    "Error Light/Beeping": "Error Light",
    "error light or beeping": "Error Light",
    
    # Bypassed action buttons
    "Open a support ticket": "Open a support ticket",
    "Live Chat with an Agent": "Live Chat with an Agent",
    "⚡ Mark as Affected": "⚡ Mark as Affected",
}

for inp, expected in tests.items():
    res = shorten_choice_via_llm(inp)
    print(f"Input: '{inp}' -> Output: '{res}' (Expected: '{expected}')")
    assert res == expected, f"Failed for {inp}: got {res}"

print("\nTesting dynamic LLM shortening...")
long_inputs = [
    "Screen is Black or Frozen",
    "Credit Card Reader Failing",
    "Printer Not Printing Receipt"
]

for inp in long_inputs:
    res = shorten_choice_via_llm(inp)
    print(f"Input: '{inp}' -> Output: '{res}' (Length: {len(res)})")
    assert len(res) <= 15, f"Failed for {inp}: length {len(res)} > 15"

print("\nTesting extraction & shortening integration...")
msg = "What is the issue with this printer? (e.g. \"Paper Jam\", \"Not Printing at All\", \"Printing Garbled Text\", \"Paper Out\", or \"Error Light / Beeping\")"
choices = get_choices_from_message(msg)
print(f"Message: '{msg}'")
print(f"Choices: {choices}")
for c in choices:
    assert len(c) <= 15, f"Choice '{c}' exceeds 15 chars"
assert "Not Printing" in choices
assert "Garbled Text" in choices
assert "Error Light" in choices

print("\nAll tests passed successfully!")
