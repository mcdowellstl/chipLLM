import sys
import os
import re

def parse_comment_update_intent(text: str) -> tuple[str, str] | None:
    """
    Parses conversational user intent to add a note/comment to a ticket.
    Returns (ticket_id, comment_text) if matched, otherwise None.
    """
    if not text:
        return None
    text_clean = text.strip()
    
    # 1. Match the prefix including verb and ticket ID
    # Verbs: add a note/comment to/on, update, comment on
    prefix_pattern = r"(?i)^(?:add\s+(?:a\s+)?(?:note|comment)\s+(?:to|on)|update|comment\s+on)\s+(INC[-_]?\d+)"
    match = re.match(prefix_pattern, text_clean)
    if not match:
        return None
        
    ticket_id = match.group(1).upper()
    # The remaining text after the ticket ID prefix
    remaining = text_clean[match.end():].strip()
    
    # 2. Repeatedly strip transition elements from the start of remaining text
    # Elements: spaces, colons, commas, semicolons, quotes, saying, stating, that, with (a) note/comment, to say, etc.
    while True:
        prev_len = len(remaining)
        # Strip leading punctuation/whitespace
        remaining = re.sub(r'^(?:[\s:;,"\'\-\(\)]+)', '', remaining)
        # Strip leading transition words
        remaining = re.sub(r'(?i)^(?:saying|stating|that|with\s+(?:a\s+)?(?:note|comment)?(?:\s+saying)?|to\s+say|say)\b', '', remaining)
        # Strip any new leading punctuation/whitespace that resulted
        remaining = re.sub(r'^(?:[\s:;,"\'\-\(\)]+)', '', remaining)
        if len(remaining) == prev_len:
            break
            
    # Clean up trailing quotes/punctuation
    remaining = re.sub(r'["\'\s\.]+$', '', remaining).strip()
    
    if ticket_id and remaining:
        return ticket_id, remaining
        
    return None

def test_intent_parsing():
    test_cases = [
        ("add a note to INC-001024 saying the parts arrived", ("INC-001024", "the parts arrived")),
        ("add comment to INC-001025: the screen is black", ("INC-001025", "the screen is black")),
        ("update INC-001026 saying sensor is still red", ("INC-001026", "sensor is still red")),
        ("add a comment to INC001024 that the kiosk is working now", ("INC001024", "the kiosk is working now")),
        ("comment on INC-001025: Tech support has arrived", ("INC-001025", "Tech support has arrived")),
        ("update INC-001026: \"The parts are delayed by 2 days.\"", ("INC-001026", "The parts are delayed by 2 days")),
        ("add note to INC-001024, the issue is resolved", ("INC-001024", "the issue is resolved")),
        ("add a comment to INC-001025 saying: no updates yet.", ("INC-001025", "no updates yet")),
    ]

    passed = 0
    for text, expected in test_cases:
        result = parse_comment_update_intent(text)
        if result == expected:
            print(f"PASS: '{text}' -> {result}")
            passed += 1
        else:
            print(f"FAIL: '{text}' -> Expected {expected}, got {result}")
            
    print(f"\nPassed {passed}/{len(test_cases)} cases.")
    if passed == len(test_cases):
        print("All intent parsing unit checks passed successfully!")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    test_intent_parsing()

