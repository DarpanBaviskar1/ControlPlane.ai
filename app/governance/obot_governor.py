"""Obot agent governance layer — MCP tool call interception (Req. 11).

Design:
- OBotGovernor evaluates each MCP_Tool_Call against the Use_Case_Profile
  authorisation policy (allowed_tools, blocked_tools, max_tool_calls_per_request).
- Authorisation runs synchronously and must complete within OBOT_LATENCY_BUDGET_MS (Req. 11.7).
- If Obot SDK is not installed, the governor falls back to a rule-based policy
  check using the profile fields directly, so basic tool governance still works.
- A real Obot integration would forward the request to the Obot server.
  Here we implement the policy semantics directly (drop-in replacement once
  the Obot server is wired up).

MCP Tool Call structure:
    {
        "tool_name": "query_database",
        "parameters": {"query": "SELECT * FROM users", "limit": 10}
    }
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.models import UseCaseProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP Tool Call model
# ---------------------------------------------------------------------------

@dataclass
class MCPToolCall:
    """Represents a single Model Context Protocol tool call from an agent."""
    tool_name: str
    parameters: dict[str, Any]
    # Populated by the governance layer
    agent_identity: str = ""
    timestamp: float = 0.0


@dataclass
class ToolCallDecision:
    """Result of Obot policy evaluation for a single MCP tool call."""
    allowed: bool
    tool_name: str
    # Populated when allowed=False
    blocked_reason: str | None = None
    # Which policy clause was violated
    policy_clause: str | None = None


# ---------------------------------------------------------------------------
# OBotGovernor
# ---------------------------------------------------------------------------

class OBotGovernor:
    """Evaluates MCP tool calls against the profile authorisation policy (Req. 11).

    Usage::

        governor = OBotGovernor()
        decision = governor.evaluate(tool_call, profile, request_id, call_count)
    """

    def evaluate(
        self,
        tool_call: MCPToolCall,
        profile: UseCaseProfile,
        request_id: str,
        current_call_count: int,
    ) -> ToolCallDecision:
        """Evaluate a single MCP tool call against the profile policy.

        Must complete within OBOT_LATENCY_BUDGET_MS (20 ms default, Req. 11.7).
        """
        start = time.monotonic()
        decision = self._evaluate_policy(tool_call, profile, request_id, current_call_count)
        elapsed_ms = (time.monotonic() - start) * 1000

        if elapsed_ms > settings.OBOT_LATENCY_BUDGET_MS:
            logger.warning(
                "OBot evaluation exceeded budget: %.1f ms > %d ms for tool '%s'",
                elapsed_ms,
                settings.OBOT_LATENCY_BUDGET_MS,
                tool_call.tool_name,
            )

        return decision

    def _evaluate_policy(
        self,
        tool_call: MCPToolCall,
        profile: UseCaseProfile,
        request_id: str,
        current_call_count: int,
    ) -> ToolCallDecision:
        tool_name = tool_call.tool_name

        # --- Rule 1: Explicit block list (Req. 11.2) ---
        if profile.blocked_tools and tool_name in profile.blocked_tools:
            logger.info(
                "TOOL_CALL_BLOCKED request_id=%s tool=%s reason=blocked_list",
                request_id,
                tool_name,
            )
            return ToolCallDecision(
                allowed=False,
                tool_name=tool_name,
                blocked_reason=f"Tool '{tool_name}' is explicitly blocked for profile '{profile.name}'",
                policy_clause="blocked_tools",
            )

        # --- Rule 2: Allow list (if non-empty, only listed tools are permitted, Req. 11.2) ---
        if profile.allowed_tools and tool_name not in profile.allowed_tools:
            logger.info(
                "TOOL_CALL_BLOCKED request_id=%s tool=%s reason=not_in_allow_list",
                request_id,
                tool_name,
            )
            return ToolCallDecision(
                allowed=False,
                tool_name=tool_name,
                blocked_reason=(
                    f"Tool '{tool_name}' is not in the allowed_tools list for profile '{profile.name}'"
                ),
                policy_clause="allowed_tools",
            )

        # --- Rule 3: Per-request call limit (Req. 11.5, 11.6) ---
        if current_call_count >= profile.max_tool_calls_per_request:
            logger.info(
                "TOOL_CALL_LIMIT_EXCEEDED request_id=%s tool=%s count=%d limit=%d",
                request_id,
                tool_name,
                current_call_count,
                profile.max_tool_calls_per_request,
            )
            return ToolCallDecision(
                allowed=False,
                tool_name=tool_name,
                blocked_reason=(
                    f"max_tool_calls_per_request ({profile.max_tool_calls_per_request}) exceeded"
                ),
                policy_clause="max_tool_calls_per_request",
            )

        logger.info(
            "TOOL_CALL_ALLOWED request_id=%s tool=%s call_count=%d",
            request_id,
            tool_name,
            current_call_count + 1,
        )
        return ToolCallDecision(allowed=True, tool_name=tool_name)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_governor: OBotGovernor | None = None


def get_governor() -> OBotGovernor:
    """Return the module-level OBotGovernor singleton."""
    global _governor  # noqa: PLW0603
    if _governor is None:
        _governor = OBotGovernor()
    return _governor
