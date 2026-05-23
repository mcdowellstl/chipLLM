"""
guardrails.py
-------------
Layer 1: Fast local middleware guardrails for Chip.
Performs intent classification BEFORE hitting the LLM to save tokens
and enforce strict domain boundaries.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Restricted-topic keyword lists
# ---------------------------------------------------------------------------

_SPORTS_KEYWORDS = {
    "baseball", "football", "basketball", "soccer", "nba", "nfl", "mlb",
    "nhl", "mls", "hockey", "tennis", "golf", "espn", "nascar", "formula 1",
    "f1", "olympics", "sport", "sports", "game score", "standings", "playoffs",
    "superbowl", "super bowl", "world cup", "champions league", "ufc", "mma",
    "wrestling", "wwe", "boxing", "athlete", "team", "stadium", "touchdown",
    "homerun", "home run", "batting", "pitcher", "quarterback", "dribble",
}

_WEATHER_KEYWORDS = {
    "weather", "forecast", "temperature", "rain", "sunny", "cloudy", "snow",
    "storm", "hurricane", "tornado", "celsius", "fahrenheit", "humidity",
    "wind", "thunder", "lightning", "blizzard", "heatwave", "heat wave",
    "cold front", "warm front", "barometric", "meteorology", "climate change",
    "el nino", "la nina", "drought", "flood", "rainfall",
}

_POLITICS_KEYWORDS = {
    "politics", "political", "democrat", "republican", "congress", "senate",
    "president", "election", "vote", "ballot", "campaign", "liberal",
    "conservative", "trump", "biden", "obama", "policy", "law", "bill",
    "legislation", "government", "irs", "taxes", "tariff", "immigration",
    "abortion", "gun control", "healthcare reform", "medicare", "social security",
    "supreme court", "white house", "capitol", "lobbyist",
}

_TRIVIA_KEYWORDS = {
    "trivia", "quiz", "riddle", "joke", "fact", "did you know", "fun fact",
    "history of", "biography", "who invented", "where was", "what year",
    "capital of", "population of", "currency of", "language of",
    "recipe", "cooking", "baking", "movie", "film", "actor", "actress",
    "music", "song", "album", "artist", "book", "novel", "author",
    "travel", "vacation", "tourism", "hotel", "flight", "airline",
    "stock market", "crypto", "bitcoin", "investment",
}

_ALL_RESTRICTED: set[str] = (
    _SPORTS_KEYWORDS
    | _WEATHER_KEYWORDS
    | _POLITICS_KEYWORDS
    | _TRIVIA_KEYWORDS
)

_REFUSAL_RESPONSE = (
    "Look, I've spent the last ten years trying to convince grease-covered receipt printers not to commit suicide "
    "during the Friday dinner rush. I don't know anything about sports, weather, politics, or trivia—my entire "
    "world exists in a 100-degree kitchen. Let's steer this back to restaurant tech: if you've got a POS register, "
    "a kitchen screen, or a printer that is currently acting up, let me know, and we'll get it sorted out!"
)

# ---------------------------------------------------------------------------
# Escalation / ticket intent signals
# ---------------------------------------------------------------------------

_ESCALATION_KEYWORDS = {
    "agent", "human", "escalate", "servicenow", "genesys",
    "call center", "open a ticket", "create ticket", "submit ticket", "raise ticket",
    "open ticket", "new ticket", "escalate ticket", "i need help from", "connect me",
    "live agent", "live support", "real person", "not working still",
    "still broken", "urgent", "down completely", "totally down",
    "talk to a manager", "talk to the manager", "speak with a manager",
    "speak with the manager", "escalate to manager", "contact manager",
    "talk to a supervisor", "talk to the supervisor", "speak with a supervisor",
    "speak with the supervisor", "escalate to supervisor", "contact supervisor",
}

_RESOLUTION_KEYWORDS = {
    "fixed", "resolved", "working now", "it works", "all good", "sorted",
    "thank you", "thanks", "that worked", "solved", "no longer", "issue gone",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GuardrailResult:
    """Holds the output of the guardrail middleware check."""

    def __init__(
        self,
        blocked: bool,
        refusal_text: str | None = None,
        escalation_triggered: bool = False,
        resolution_detected: bool = False,
    ) -> None:
        self.blocked = blocked
        self.refusal_text = refusal_text
        self.escalation_triggered = escalation_triggered
        self.resolution_detected = resolution_detected


def check_guardrails(user_message: str) -> GuardrailResult:
    """
    Run the full guardrail stack against a user message.

    Returns a GuardrailResult indicating:
      - blocked: True if the message should be refused immediately
      - refusal_text: the static refusal string to return
      - escalation_triggered: True if intent suggests live-agent routing
      - resolution_detected: True if user has indicated the issue is resolved
    """
    msg_lower = user_message.lower()
    tokens = set(msg_lower.split())

    # --- Check 1: Restricted topic detection (exact keyword match) ----------
    matched_restricted = _ALL_RESTRICTED.intersection(tokens)
    # Also check multi-word phrases
    for phrase in _ALL_RESTRICTED:
        if " " in phrase and phrase in msg_lower:
            matched_restricted.add(phrase)

    if matched_restricted:
        return GuardrailResult(
            blocked=True,
            refusal_text=_REFUSAL_RESPONSE,
        )

    # --- Check 2: Escalation intent detection --------------------------------
    import re
    escalation_hit = any(re.search(r"\b" + re.escape(kw) + r"\b", msg_lower) for kw in _ESCALATION_KEYWORDS)
    if escalation_hit and is_ticket_status_lookup_intent(user_message):
        escalation_hit = False

    # --- Check 3: Resolution detection ---------------------------------------
    resolution_hit = any(kw in msg_lower for kw in _RESOLUTION_KEYWORDS)

    return GuardrailResult(
        blocked=False,
        escalation_triggered=escalation_hit,
        resolution_detected=resolution_hit,
    )


def is_greeting_or_small_talk(user_message: str) -> bool:
    """
    Detect if the user's message is a pure greeting, pleasantry, or simple small-talk,
    rather than a description of a technical issue.
    """
    import re
    # Clean the message: lowercase, remove punctuation except spaces
    cleaned = re.sub(r"[^\w\s]", "", user_message.lower()).strip()
    if not cleaned:
        return True

    # 1. Check exact phrase matches
    greeting_phrases = {
        "hows it going", "how are you", "how are you doing", "how do you do", "nice to meet you",
        "good morning", "good afternoon", "good evening", "good day", "anyone there",
        "are you there", "is anyone there", "anybody there", "hello there", "hi there",
        "hey there", "howdy partner", "howdy chip", "hi chip", "hello chip", "hey chip"
    }
    if cleaned in greeting_phrases:
        return True

    # 2. Check if the message consists entirely of greeting/pleasantry/filler words
    greeting_words = {
        "hi", "hello", "hey", "howdy", "hola", "greetings", "morning", "afternoon", "evening",
        "yo", "whats up", "sup", "test", "testing", "chip", "there", "partner", "buddy",
        "friend", "man", "dude", "sir", "maam", "everyone", "all", "here", "good", "whats", "up",
        "what", "is"
    }
    words = cleaned.split()
    if all(word in greeting_words for word in words):
        return True

    # 3. If it's very short (1-2 words) and contains at least one greeting word,
    # and doesn't contain device-specific keywords, treat it as a greeting.
    device_keywords = {
        "pos", "kiosk", "printer", "kvs", "kds", "bumpbar", "bump bar", "screen",
        "register", "terminal", "receipt", "paper", "jam", "drawer", "cash",
        "card", "payment", "reader", "scanner", "network", "offline", "wifi",
        "internet", "power", "cable", "cord", "button", "buttons", "monitor",
        "controller", "waystation"
    }
    if len(words) <= 2:
        has_greeting = any(word in greeting_words for word in words)
        has_device = any(word in device_keywords for word in words)
        if has_greeting and not has_device:
            return True

    return False


def is_cafe_issue(user_message: str) -> bool:
    """
    Detect if the user's message is related to a cafe, McCafe, coffee, or espresso issue
    which is different from the normal tech we cover, so we can route them immediately
    to chat with a human.
    """
    import re
    cleaned = re.sub(r"[^\w\s\-]", "", user_message.lower()).strip()
    if not cleaned:
        return False
        
    cafe_keywords = {
        "mccafe", "cafe", "coffee", "espresso", "latte", "cappuccino", "frappe",
        "macchiato", "americano", "beverage", "brewer", "bunnomatic", "blender",
        "drink", "frappuccino", "tea", "caffeine"
    }
    
    # Check word boundaries using split
    words = re.split(r"[\s\-]+", cleaned)
    if any(word in cafe_keywords for word in words):
        return True
        
    # Check multi-word phrases explicitly
    multi_word_phrases = ["iced coffee", "hot chocolate", "iced tea"]
    if any(phrase in cleaned for phrase in multi_word_phrases):
        return True
        
    return False


def is_ticket_status_lookup_intent(text: str) -> bool:
    """
    Checks if the user's message is asking about existing ticket statuses, open cases,
    or recent store issues — OR is a case mutation action (add comment, escalate a case).
    Either way these must bypass the triage escalation flow.
    """
    import re
    if not text:
        return False
    lower_text = text.strip().lower()

    # 0a. Explicit ticket creation phrases must NOT be treated as lookups/mutations
    creation_phrases = [
        "open a support ticket", "open support ticket", "open a ticket", "open ticket",
        "create a support ticket", "create support ticket", "create a ticket", "create ticket",
        "submit a support ticket", "submit support ticket", "submit a ticket", "submit ticket",
        "raise a support ticket", "raise support ticket", "raise a ticket", "raise ticket",
    ]
    if any(phrase in lower_text for phrase in creation_phrases):
        return False

    # 0. Case mutation intents — treat exactly like a lookup to bypass triage routing
    mutation_phrases = [
        "add comment", "add a comment", "add note", "add a note",
        "log a comment", "log comment", "log note", "log a note",
        "escalate case", "escalate the case", "escalate rc",
        "escalate ticket", "escalate this",
    ]
    if any(phrase in lower_text for phrase in mutation_phrases):
        return True
    # Bare "escalate" with no device/issue context is a case mutation intent
    if re.search(r'^escalate$', lower_text.strip()):
        return True

    # 1. Broad standalone keywords check: status, tickets, cases, incidents, incident, ticket, case, history
    keywords = ["status", "tickets", "cases", "incidents", "incident", "ticket", "case", "history"]
    has_kw = any(re.search(r"\b" + re.escape(kw) + r"\b", lower_text) for kw in keywords)

    # 2. Check standard ticket ID patterns like RC001024 or INC12345
    has_id = bool(re.search(r"\b(?:inc|rc00|rc)[-_]?\d+\b", lower_text))

    # 3. Check for ticket challenge or empty list follow-up phrases
    challenge_phrases = [
        "both not true", "not true", "incorrect", "that's incorrect", "that is incorrect",
        "i dont see", "i don't see", "not showing", "where are they", "refresh", "reload",
        "empty", "missing", "wrong count", "wrong statistic", "wrong stats", "lies", "hallucination",
        "hallucinating", "that is wrong", "thats wrong"
    ]
    has_challenge = any(phrase in lower_text for phrase in challenge_phrases)

    if not has_kw and not has_id and not has_challenge:
        return False

    # We found a ticket keyword or ID.
    # Now check if this is explicitly a ticket/case/incident creation intent:
    creation_pattern = r"\b(open|create|submit|raise|file|make)\s+(?:a\s+|an\s+|new\s+)*(ticket|case|incident)\b"
    if re.search(creation_pattern, lower_text):
        if any(w in lower_text for w in ["status", "check", "list", "show", "track", "history", "view"]):
            return True
        return False

    # Any other mention of ticket/case/status/incidents is a lookup/view attempt!
    return True

