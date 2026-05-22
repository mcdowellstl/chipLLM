import sys
sys.path.append('.')
from app import _colorize_ticket_tables, get_choices_from_message

# Test case 1: Rebooted single-line text format
sample_reboot_text = """
You have 3 active cases right now:
* RC001024 -- Kiosk 3 Cash Acceptor Jammed [Pending | P2]
* RC001025 -- KVS Bumpbar buttons unresponsive [In Progress | P3]
* RC001026 -- POS 2 receipt printer paper jam sensor failure [New | P2]

Would you like to get more details or update any of these cases?
"""

print("=== Running Colorizer Test (UI Reboot) ===")
result = _colorize_ticket_tables(sample_reboot_text)
print("Colorized Output (should be identical to original):")
print(result)
assert result.strip() == sample_reboot_text.strip(), "Failed! The rebooted text list format was modified by the colorizer."
print("Success! Colorizer correctly ignored the rebooted format.")

print("\n=== Running Suggestion Chips Test (UI Reboot) ===")
# Even if a legacy/older text contains "Case Options:", get_choices_from_message MUST return empty list
chips_legacy = get_choices_from_message("Case Options: See Details, Add Comment, Modify Status, Escalate, None")
print("Chips Extracted (Legacy):", chips_legacy)
assert chips_legacy == [], f"Failed! Got {chips_legacy} but expected [] under UI Reboot rule."

chips_reboot = get_choices_from_message(result)
print("Chips Extracted (Reboot):", chips_reboot)
assert chips_reboot == [], f"Failed! Got {chips_reboot} but expected [] under UI Reboot rule."

print("Success! Chips extraction is safely disabled for Case Options.")

