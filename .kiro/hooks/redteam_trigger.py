"""Hook handler: Redteam Adversarial Suite on Policy Save.

Called by the PostFileSave hook whenever a .yaml/.yml/.json file is saved.
Reads the Kiro session context from stdin (JSON), decides whether the saved
file is actually a policy file, then hits POST /v1/redteam/run on the running
gateway and prints a summary.

Exit codes:
  0  – redteam run completed (breakthroughs printed as warnings)
  1  – gateway unreachable or run failed (non-blocking — hook does not block
       the save operation; we never exit 2)

The script is intentionally dependency-light (stdlib + httpx which is already
in the project venv).
"""

from __future__ import annotations

import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Read hook context from stdin
# ---------------------------------------------------------------------------
try:
    context = json.loads(sys.stdin.read() or "{}")
except json.JSONDecodeError:
    context = {}

saved_file: str = context.get("filePath", context.get("file", ""))

# Only run for files that look like policy files.
# Pattern: any yaml/json that lives under a "policy" directory, OR is named
# "policy.yaml" / "policy.json" / ends with "_policy.yaml" etc.
POLICY_INDICATORS = ("policy", "profiles", "use_case")

file_lower = saved_file.replace("\\", "/").lower()
is_policy_file = any(ind in file_lower for ind in POLICY_INDICATORS)

if not is_policy_file:
    # Not a policy file — skip silently
    sys.exit(0)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GATEWAY_URL = os.getenv("CONTROLPLANE_GATEWAY_URL", "http://localhost:8000")
REDTEAM_ENDPOINT = f"{GATEWAY_URL}/v1/redteam/run"
REPORT_ENDPOINT = f"{GATEWAY_URL}/v1/redteam/report"
REQUEST_TIMEOUT = int(os.getenv("REDTEAM_HOOK_TIMEOUT_S", "90"))

print(f"[redteam-hook] Policy file saved: {saved_file}")
print(f"[redteam-hook] Triggering adversarial test suite against {REDTEAM_ENDPOINT}")

# ---------------------------------------------------------------------------
# MCP health check (Req. 5.11)
# ---------------------------------------------------------------------------
MCP_HEALTH_URL = os.getenv("REDTEAM_MCP_URL", "http://localhost:9200") + "/health"

try:
    import httpx  # available in the project venv
except ImportError:
    print(
        "[redteam-hook] httpx not available — install it with 'pip install httpx'",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    with httpx.Client(timeout=2.0) as _mcp_client:
        _mcp_resp = _mcp_client.get(MCP_HEALTH_URL)
        if _mcp_resp.status_code == 200:
            print(f"[redteam-hook] MCP server healthy at {MCP_HEALTH_URL}")
        else:
            print(
                f"[redteam-hook] REDTEAM_MCP_UNAVAILABLE: MCP health check returned "
                f"{_mcp_resp.status_code}",
                file=sys.stderr,
            )
except Exception as _mcp_exc:
    print(
        f"[redteam-hook] REDTEAM_MCP_UNAVAILABLE: MCP health check failed — {_mcp_exc}",
        file=sys.stderr,
    )
# Always proceed to call the Gateway regardless of MCP health result (Req. 5.11)

# ---------------------------------------------------------------------------
# Call the redteam endpoint
# ---------------------------------------------------------------------------
start = time.monotonic()
try:
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(REDTEAM_ENDPOINT)
except httpx.ConnectError:
    print(
        f"[redteam-hook] Gateway not reachable at {GATEWAY_URL} — "
        "start the gateway with `uvicorn app.main:app` before saving policy files.",
        file=sys.stderr,
    )
    sys.exit(1)
except httpx.TimeoutException:
    print(
        f"[redteam-hook] Redteam run timed out after {REQUEST_TIMEOUT}s.",
        file=sys.stderr,
    )
    sys.exit(1)
except Exception as exc:  # noqa: BLE001
    print(f"[redteam-hook] Unexpected error: {exc}", file=sys.stderr)
    sys.exit(1)

elapsed = time.monotonic() - start

if resp.status_code == 409:
    print("[redteam-hook] A redteam run is already in progress — skipping.")
    sys.exit(0)

if not resp.is_success:
    print(
        f"[redteam-hook] Redteam endpoint returned {resp.status_code}: {resp.text[:300]}",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Parse and summarise the report
# ---------------------------------------------------------------------------
try:
    report = resp.json()
except Exception:  # noqa: BLE001
    print("[redteam-hook] Could not parse redteam report JSON.", file=sys.stderr)
    sys.exit(1)

total = report.get("total_prompts_sent", 0)
breakthroughs = report.get("total_breakthroughs", 0)
blocks = report.get("total_blocks", 0)
block_rate = report.get("block_rate", 0.0)
status = report.get("status", "unknown")
session_id = report.get("session_id", "?")

print(f"[redteam-hook] Run complete in {elapsed:.1f}s  status={status}  session={session_id}")
print(
    f"[redteam-hook] Prompts: {total}  |  Blocked: {blocks}  |  "
    f"Block-rate: {block_rate:.0%}  |  Breakthroughs: {breakthroughs}"
)

# Emit a prominent warning for any breakthroughs so the developer sees them
if breakthroughs > 0:
    attack_results = report.get("attack_results", [])
    breakthrough_categories = [
        r.get("category", "unknown")
        for r in attack_results
        if r.get("breakthrough")
    ]
    print(
        f"\n[redteam-hook] ⚠️  WARNING: {breakthroughs} adversarial prompt(s) "
        f"bypassed the updated policy!\n"
        f"  Categories: {', '.join(breakthrough_categories)}\n"
        f"  Review the full report at GET {REPORT_ENDPOINT}\n",
        file=sys.stderr,
    )
    # Still exit 0 — this is a warning, not a hard block of the save
    sys.exit(0)

print("[redteam-hook] ✓ No breakthroughs detected. Policy change looks safe.")
sys.exit(0)
