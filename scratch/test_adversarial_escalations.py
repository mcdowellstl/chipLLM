import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardrails import check_guardrails
from app import check_escalation_intent

def run_tests():
    # Words/phrases that previously caused false positives
    false_positives = [
        "can you help me troubleshoot",
        "is there a person assigned to this case",
        "i have a management question",
        "the personnel needs help",
        "just standard small talk help",
    ]
    
    # Words/phrases that SHOULD trigger escalation
    true_positives = [
        "talk to a human representative",
        "escalate to a live agent",
        "open a ticket",
        "servicenow incident support",
        "please connect me to a supervisor",
    ]
    
    print("=== Testing check_escalation_intent (app.py) ===")
    for text in false_positives:
        val = check_escalation_intent(text)
        print(f"FP check: '{text}' -> {val} (Expected: False)")
        
    for text in true_positives:
        val = check_escalation_intent(text)
        print(f"TP check: '{text}' -> {val} (Expected: True)")
        
    print("\n=== Testing check_guardrails (guardrails.py) ===")
    for text in false_positives:
        guard = check_guardrails(text)
        print(f"FP guardrail: '{text}' -> {guard.escalation_triggered} (Expected: False)")
        
    for text in true_positives:
        guard = check_guardrails(text)
        print(f"TP guardrail: '{text}' -> {guard.escalation_triggered} (Expected: True)")

if __name__ == "__main__":
    run_tests()
