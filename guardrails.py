"""
guardrails.py
-------------
Layer 1: Fast local middleware guardrails for chipLLM.
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
    "⛔ **Topic restricted.** Please ask a store-related support tech question.\n\n"
    "I'm chipLLM — I only handle restaurant technology issues: "
    "**POS terminals, receipt printers, self-order kiosks, and kitchen display systems (KDS/KVS)**. "
    "How can I help you troubleshoot today?"
)

# ---------------------------------------------------------------------------
# Escalation / ticket intent signals
# ---------------------------------------------------------------------------

_ESCALATION_KEYWORDS = {
    "agent", "human", "escalate", "ticket", "servicenow", "genesys",
    "call center", "supervisor", "manager", "open a ticket", "create ticket",
    "submit ticket", "raise ticket", "i need help from", "connect me",
    "live agent", "live support", "real person", "not working still",
    "still broken", "urgent", "down completely", "totally down",
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
    escalation_hit = any(kw in msg_lower for kw in _ESCALATION_KEYWORDS)

    # --- Check 3: Resolution detection ---------------------------------------
    resolution_hit = any(kw in msg_lower for kw in _RESOLUTION_KEYWORDS)

    return GuardrailResult(
        blocked=False,
        escalation_triggered=escalation_hit,
        resolution_detected=resolution_hit,
    )
