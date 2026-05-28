import sys
import os

# Set up environment variables
os.environ["GOOGLE_CLOUD_PROJECT"] = "chip-496911"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-east1"

sys.path.append("/Users/jeffmickd/src/chipllm")

from app import clean_assistant_message

salvo = "There are some common troubleshooting steps that might help you fix this issue on your own. We will quickly step through them to see if this solves the issue"
content = f"{salvo}. First, check if the printer is plugged in. Did this resolve the issue?"

print("Testing clean_assistant_message with troubleshooting salvo...")
result = clean_assistant_message(content)

print(f"Result:\n{result}\n")

# Verify that the class, padding and split marker are in the output
expected_class = 'class="recommended-troubleshooting-banner"'
expected_style = 'padding: 12px 14px 14px 14px;'
expected_split_marker = '<!-- SPLIT_TROUBLESHOOTING_BANNER -->'

assert expected_class in result, f"Expected '{expected_class}' to be in the result"
assert expected_style in result, f"Expected '{expected_style}' to be in the result"
assert expected_split_marker in result, f"Expected '{expected_split_marker}' to be in the result"

print("Verification SUCCESS: Custom banner class, padding and split comment marker injected correctly!")
