import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def is_ticket_status_lookup_intent(text: str) -> bool:
    """
    Checks if the user's message is asking about existing ticket statuses, open cases,
    or recent store issues, rather than asking to create/open a new ticket.
    """
    import re
    if not text:
        return False
    lower_text = text.strip().lower()
    
    # If a specific ticket ID is mentioned, it's definitely a lookup/status check
    if re.search(r"\b(?:inc|rc00|rc)[-_]?\d+\b", lower_text):
        return True
        
    # Flexible check for recent ticket inquiries
    if "recent" in lower_text and any(w in lower_text for w in ["issue", "ticket", "case"]):
        return True
        
    # Phrases indicating lookup/inquiry about existing tickets/cases
    lookup_keywords = [
        "active tickets", "open tickets", "my tickets", "store tickets", "existing tickets",
        "active cases", "open cases", "my cases", "store cases", "existing cases",
        "ticket status", "case status", "status of my", "status of the", "status of inc", "status of rc", "status of rc00",
        "check status", "check ticket", "check case", "any tickets", "any cases",
        "list tickets", "list cases", "show tickets", "show cases", "recent issues",
        "recent tickets", "recent cases", "view tickets", "view cases", "what tickets",
        "what cases", "current tickets", "current cases", "outstanding tickets",
        "outstanding cases", "status check"
    ]
    
    return any(kw in lower_text for kw in lookup_keywords)

def test_lookup_intent():
    positive_cases = [
        "are there any active tickets open for my store?",
        "what is the status of my tickets?",
        "are there any open tickets?",
        "check ticket RC001024",
        "show my tickets",
        "do we have any open tickets?",
        "any tickets open?",
        "status check",
        "status check on tickets",
        "ticket RC001024 status",
        "recent store issues",
        "are there any open cases?",
    ]
    
    negative_cases = [
        "i need to open a support ticket",
        "how do i escalate this issue?",
        "please open a ticket for me",
        "can we escalate to a human?",
        "my pos is broken, please create a ticket",
    ]
    
    passed = 0
    total = len(positive_cases) + len(negative_cases)
    
    print("--- Testing Positive Cases (Should be True) ---")
    for text in positive_cases:
        res = is_ticket_status_lookup_intent(text)
        if res is True:
            print(f"PASS: '{text}' -> {res}")
            passed += 1
        else:
            print(f"FAIL: '{text}' -> Expected True, got {res}")
            
    print("\n--- Testing Negative Cases (Should be False) ---")
    for text in negative_cases:
        res = is_ticket_status_lookup_intent(text)
        if res is False:
            print(f"PASS: '{text}' -> {res}")
            passed += 1
        else:
            print(f"FAIL: '{text}' -> Expected False, got {res}")
            
    print(f"\nPassed {passed}/{total} cases.")
    if passed == total:
        print("All lookup intent unit checks passed successfully!")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    test_lookup_intent()
