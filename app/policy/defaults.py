"""Compiled-in default Use-Case Profiles.

These two profiles are always available even when no external policy file is
present, ensuring the gateway can start without configuration.
"""

from app.models import UseCaseProfile

CUSTOMER_CHATBOT = UseCaseProfile(
    name="customer_chatbot",
    latency_budget_ms=10_000,
    complexity_threshold=0.7,
    token_compression_threshold=512,
    groundedness_pass_threshold=0.85,
    inspection_timeout_ms=3_000,
    pii_masking_enabled=True,
    human_escalation_enabled=True,
)

INTERNAL_COPILOT = UseCaseProfile(
    name="internal_copilot",
    latency_budget_ms=30_000,
    complexity_threshold=0.6,
    token_compression_threshold=2_048,
    groundedness_pass_threshold=0.75,
    inspection_timeout_ms=8_000,
    pii_masking_enabled=False,
    human_escalation_enabled=True,
)

# Map of name → profile for quick lookup
BUILT_IN_PROFILES: dict[str, UseCaseProfile] = {
    CUSTOMER_CHATBOT.name: CUSTOMER_CHATBOT,
    INTERNAL_COPILOT.name: INTERNAL_COPILOT,
}
