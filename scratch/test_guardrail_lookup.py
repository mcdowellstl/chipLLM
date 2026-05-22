import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardrails import check_guardrails

def run_tests():
    # 1. Ticket status inquiries (should NOT trigger escalation)
    lookup_cases = [
        "are there any open tickets for my store?",
        "are there any open tickets?",
        "what's the status of RC001024?",
        "check status of my cases",
    ]
    
    # 2. Genuine escalation requests (SHOULD trigger escalation)
    escalation_cases = [
        "open a ticket",
        "i need to speak with a human",
        "please escalate to a live agent",
        "this is still broken, open ticket",
    ]
    
    all_passed = True
    
    print("=== Testing Ticket Status Lookups (should NOT trigger escalation) ===")
    for text in lookup_cases:
        guard = check_guardrails(text)
        if not guard.escalation_triggered:
            print(f"PASS: '{text}' -> escalation_triggered = False")
        else:
            print(f"FAIL: '{text}' -> escalation_triggered = True (Expected False)")
            all_passed = False
            
    print("\n=== Testing Escalation Requests (SHOULD trigger escalation) ===")
    for text in escalation_cases:
        guard = check_guardrails(text)
        if guard.escalation_triggered:
            print(f"PASS: '{text}' -> escalation_triggered = True")
        else:
            print(f"FAIL: '{text}' -> escalation_triggered = False (Expected True)")
            all_passed = False
            
    if all_passed:
        print("\nAll guardrail unit checks passed successfully!")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
