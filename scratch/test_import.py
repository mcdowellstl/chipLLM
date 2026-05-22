import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app import parse_comment_update_intent
    print("SUCCESS: Imported parse_comment_update_intent from app successfully!")
    
    # Quick sanity check
    res = parse_comment_update_intent("add note to INC-001024 saying test message")
    print(f"Sanity check result: {res}")
    assert res == ("INC-001024", "test message")
    print("Sanity check passed!")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
