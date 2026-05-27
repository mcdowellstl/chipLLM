import re

variants = [
    # 1. Trailing "can you tell"
    ('Ah, a KVS printer, the unsung hero of the back-of-house, always getting splashed with something. You said it\'s "broken" – can you tell',
     'Ah, a KVS printer, the unsung hero of the back-of-house, always getting splashed with something. You said it\'s "broken"'),
    
    # 2. Trailing "For example, is it" -> Keeps the prior complete question sentence!
    ('Oh no, a broken bump bar can really slow things down in the kitchen! Can you tell me a bit more about what\'s happening with it? For example, is it',
     'Oh no, a broken bump bar can really slow things down in the kitchen! Can you tell me a bit more about what\'s happening with it'),
    
    # 3. Trailing "Can you tell me"
    ('Oh no, a broken bump bar can really slow things down in the kitchen! Can you tell me',
     'Oh no, a broken bump bar can really slow things down in the kitchen!'),
    
    # 4. Full question: "can you tell me what type of printer this is" -> Should NOT be truncated!
    ('Ah, a KVS printer, the unsung hero of the back-of-house, always getting splashed with something. You said it\'s "broken" – can you tell me what type of printer this is',
     'Ah, a KVS printer, the unsung hero of the back-of-house, always getting splashed with something. You said it\'s "broken" – can you tell me what type of printer this is'),
    
    # 5. Trailing "can you tell me specifically"
    ('Ah, a KVS printer, the unsung hero of the back-of-house, always getting splashed with something. You said it\'s "broken" – can you tell me specifically',
     'Ah, a KVS printer, the unsung hero of the back-of-house, always getting splashed with something. You said it\'s "broken"'),
    
    # 6. Complete sentence containing "tell me" -> Should NOT be truncated!
    ('I can tell you that the printer is online. What type of printer is this',
     'I can tell you that the printer is online. What type of printer is this'),

    # 7. Trailing "let me know if it is"
    ('The printer is showing an error. Let me know if it is',
     'The printer is showing an error'),

    # 8. Trailing "specifically"
    ('Could you describe the issue specifically',
     'Could you describe the issue'),
]

# Refined combined pattern using end-anchored sub-patterns
combined_pattern = re.compile(
    r"\b(?:can|could)\s+you\s+(?:tell|let\s+me\s+know)(?:\s+me)?\s*$"
    r"|\b(?:tell|let\s+me\s+know)(?:\s+me)?\s*$"
    r"|\b(?:is\s+it|are\s+they)(?:\s+a)?\s*$"
    r"|\b(?:such\s+as|like|specifically|for\s+example|for\s+instance)\s*$"
    r"|\b(?:can|could)\s+you\s*$"
    r"|\b(?:to\s+assess|assess|determine)\s*$"
    r"|\b(?:which|what)\s+(?:one|symptom|issue|device)\s*$"
    r"|\b(?:if\s+it\s+is|if\s+they\s+are|whether\s+it\s+is|whether\s+they\s+are)\s*$",
    re.IGNORECASE
)

print("Testing refined regex pattern:")
failed = False
for idx, (var, expected) in enumerate(variants, 1):
    cleaned = var.strip()
    
    while True:
        prev = cleaned
        cleaned = combined_pattern.sub("", cleaned).strip()
        cleaned = cleaned.rstrip("?:.,; \t-–—")
        if cleaned == prev:
            break
            
    matched = (cleaned == expected)
    print(f"Variant {idx}: {'PASS' if matched else 'FAIL'}")
    print(f"Original: {var}")
    print(f"Expected: {expected}")
    print(f"Actual:   {cleaned}")
    print("="*60)
    if not matched:
        failed = True

if failed:
    print("Some tests failed!")
else:
    print("All tests passed successfully!")
