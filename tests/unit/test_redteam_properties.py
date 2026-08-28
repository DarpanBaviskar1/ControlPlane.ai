"""Property-based tests for the Redteam MCP server and RedTeamRunner.

Tasks: 12.3, 13.2, 13.3
Properties:
  RT-2 — No app.* imports: static AST import scan of mcp_servers/redteam/server.py
          must find zero references to modules starting with 'app.'
  RT-1 — MCP fallback: when MCP server returns non-2xx or raises, run() must
          complete without raising and return a valid RedTeamReport
  RT-3 — Breakthrough logging: for any AttackResult with breakthrough=True, the
          Langfuse tracer mock must be called with name="RED_TEAM_BREAKTHROUGH"
          and level="ERROR", whether run originates from MCP or in-process

Requirements: 5.1, 5.7, 5.8, 5.10
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from app.redteam.runner import RedTeamRunner, RedTeamReport, AttackResult


# ---------------------------------------------------------------------------
# RT-2: No app.* imports — static AST scan
# Task 12.3
# Requirements: 5.1
# ---------------------------------------------------------------------------

class TestRT2NoAppImports:
    """Property RT-2: mcp_servers/redteam/server.py must have zero app.* imports."""

    _SERVER_PATH = (
        pathlib.Path(__file__).resolve().parents[2]
        / "mcp_servers" / "redteam" / "server.py"
    )

    def test_server_file_exists(self) -> None:
        assert self._SERVER_PATH.exists(), f"server.py not found at {self._SERVER_PATH}"

    def test_no_app_star_imports(self) -> None:
        """
        Property RT-2: static AST import scan of mcp_servers/redteam/server.py
        must find zero references to modules whose fully-qualified name starts with 'app.'
        """
        source = self._SERVER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(self._SERVER_PATH))

        app_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("app.") or alias.name == "app":
                        app_imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("app.") or module == "app":
                    app_imports.append(
                        f"from {module} import ..."
                    )

        assert app_imports == [], (
            f"Found {len(app_imports)} app.* import(s) in server.py:\n"
            + "\n".join(f"  {imp}" for imp in app_imports)
        )

    def test_server_file_is_valid_python(self) -> None:
        """The server.py must parse as valid Python (no syntax errors)."""
        source = self._SERVER_PATH.read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"server.py has a syntax error: {exc}")

    def test_server_has_required_endpoints(self) -> None:
        """server.py must define /run, /health, and /report endpoint functions."""
        source = self._SERVER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Collect all function names
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        # Check for the three required endpoint handlers
        assert "health" in func_names, "Missing /health endpoint function"
        assert "report" in func_names, "Missing /report endpoint function"
        assert "run" in func_names, "Missing /run endpoint function"


# ---------------------------------------------------------------------------
# RT-1: MCP fallback
# Task 13.2
# Requirements: 5.7, 5.8
# ---------------------------------------------------------------------------

class TestRT1MCPFallback:
    """Property RT-1: when MCP server is unavailable, run() must still return a valid report."""

    def _make_runner(self) -> RedTeamRunner:
        runner = RedTeamRunner(gateway_base_url="http://localhost:8000")
        return runner

    @pytest.mark.asyncio
    async def test_connection_error_falls_back_to_in_process(self) -> None:
        """ConnectError from MCP must trigger fallback and return a valid RedTeamReport."""
        runner = self._make_runner()

        with patch("app.redteam.runner.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value = mock_client

            report = await runner.run(tracer=None)

        assert isinstance(report, RedTeamReport)
        assert report.status in ("completed", "failed")
        assert RedTeamRunner._mcp_healthy is False

    @pytest.mark.asyncio
    @given(status_code=st.integers(min_value=400, max_value=599))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    async def test_non_2xx_status_falls_back(self, status_code: int) -> None:
        """Any 4xx/5xx response from MCP must cause fallback."""
        runner = self._make_runner()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                f"HTTP {status_code}",
                request=MagicMock(),
                response=mock_response,
            )
        )

        with patch("app.redteam.runner.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            report = await runner.run(tracer=None)

        assert isinstance(report, RedTeamReport)
        assert report.status in ("completed", "failed")

    @pytest.mark.asyncio
    async def test_timeout_falls_back(self) -> None:
        """Timeout from MCP must trigger fallback without raising."""
        runner = self._make_runner()

        with patch("app.redteam.runner.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client_cls.return_value = mock_client

            report = await runner.run(tracer=None)

        assert isinstance(report, RedTeamReport)

    @pytest.mark.asyncio
    async def test_mcp_success_returns_report_without_in_process(self) -> None:
        """When MCP returns 200, run() must use that report and not run in-process."""
        runner = self._make_runner()

        mcp_payload = {
            "session_id": "mcp-session-001",
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "completed_at": datetime.now(tz=timezone.utc).isoformat(),
            "total_prompts_sent": 5,
            "total_blocks": 5,
            "total_breakthroughs": 0,
            "block_rate": 1.0,
            "breakthrough_rate": 0.0,
            "attack_results": [],
            "status": "completed",
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mcp_payload)

        with patch("app.redteam.runner.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            report = await runner.run(tracer=None)

        assert report.session_id == "mcp-session-001"
        assert RedTeamRunner._mcp_healthy is True


# ---------------------------------------------------------------------------
# RT-3: Breakthrough logging
# Task 13.3
# Requirements: 5.10
# ---------------------------------------------------------------------------

class TestRT3BreakthroughLogging:
    """Property RT-3: breakthrough AttackResults must be logged to tracer with ERROR level."""

    def _make_mock_tracer(self) -> MagicMock:
        tracer = MagicMock()
        tracer.add_span = MagicMock()
        return tracer

    @pytest.mark.asyncio
    async def test_in_process_breakthrough_logged(self) -> None:
        """In-process breakthrough must call tracer.add_span with RED_TEAM_BREAKTHROUGH + ERROR."""
        runner = RedTeamRunner(gateway_base_url="http://localhost:8000")
        tracer = self._make_mock_tracer()

        # Force MCP to be unavailable so we go in-process
        with patch.object(runner, "_try_mcp_run", AsyncMock(return_value=None)):
            # Make _execute_attack return a breakthrough
            breakthrough_result = AttackResult(
                category="test",
                session_id="s1",
                prompts=["attack"],
                response="leaked",
                breakthrough=True,
            )
            with patch.object(runner, "_collect_attack_sequences", AsyncMock(return_value=[("test", ["attack"])])):
                with patch.object(runner, "_execute_attack", AsyncMock(return_value=breakthrough_result)):
                    with patch.object(runner, "_run_garak_probes", AsyncMock(return_value=[])):
                        # Allow event loop to process create_task calls
                        report = await runner.run(tracer=tracer)

        # Give the event loop a tick to process any fire-and-forget tasks
        await asyncio.sleep(0)

        # Verify tracer was called with correct parameters
        calls = tracer.add_span.call_args_list
        breakthrough_calls = [
            c for c in calls
            if c.kwargs.get("name") == "RED_TEAM_BREAKTHROUGH"
            or (c.args and c.args[0] == "RED_TEAM_BREAKTHROUGH")  # type: ignore[index]
        ]
        # At least one call with name=RED_TEAM_BREAKTHROUGH and level=ERROR
        assert len(breakthrough_calls) >= 1, (
            f"Expected at least one RED_TEAM_BREAKTHROUGH call, got: {calls}"
        )
        call = breakthrough_calls[0]
        assert call.kwargs.get("level") == "ERROR" or "ERROR" in str(call), (
            f"Expected level=ERROR, got: {call}"
        )

    @pytest.mark.asyncio
    async def test_mcp_breakthrough_logged(self) -> None:
        """Breakthroughs from MCP response must also trigger tracer.add_span with ERROR level."""
        runner = RedTeamRunner(gateway_base_url="http://localhost:8000")
        tracer = self._make_mock_tracer()

        mcp_payload = {
            "session_id": "mcp-bt-001",
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "completed_at": datetime.now(tz=timezone.utc).isoformat(),
            "total_prompts_sent": 1,
            "total_blocks": 0,
            "total_breakthroughs": 1,
            "block_rate": 0.0,
            "breakthrough_rate": 1.0,
            "attack_results": [
                {
                    "category": "jailbreak",
                    "prompts": ["attack prompt"],
                    "response": "leaked content",
                    "breakthrough": True,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
            ],
            "status": "completed",
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mcp_payload)

        with patch("app.redteam.runner.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            report = await runner.run(tracer=tracer)

        # Allow fire-and-forget tasks to run
        await asyncio.sleep(0)

        calls = tracer.add_span.call_args_list
        breakthrough_calls = [
            c for c in calls
            if c.kwargs.get("name") == "RED_TEAM_BREAKTHROUGH"
        ]
        assert len(breakthrough_calls) >= 1, (
            f"Expected RED_TEAM_BREAKTHROUGH call for MCP breakthrough, got: {calls}"
        )
        assert breakthrough_calls[0].kwargs.get("level") == "ERROR"

    @pytest.mark.asyncio
    async def test_no_breakthrough_no_tracer_call(self) -> None:
        """When no breakthroughs occur, tracer must not be called with RED_TEAM_BREAKTHROUGH."""
        runner = RedTeamRunner()
        tracer = self._make_mock_tracer()

        clean_result = AttackResult(
            category="test",
            session_id="s2",
            prompts=["safe"],
            response="blocked",
            breakthrough=False,
        )
        with patch.object(runner, "_try_mcp_run", AsyncMock(return_value=None)):
            with patch.object(runner, "_collect_attack_sequences", AsyncMock(return_value=[("test", ["safe"])])):
                with patch.object(runner, "_execute_attack", AsyncMock(return_value=clean_result)):
                    with patch.object(runner, "_run_garak_probes", AsyncMock(return_value=[])):
                        await runner.run(tracer=tracer)

        await asyncio.sleep(0)

        calls = tracer.add_span.call_args_list
        breakthrough_calls = [
            c for c in calls
            if c.kwargs.get("name") == "RED_TEAM_BREAKTHROUGH"
        ]
        assert len(breakthrough_calls) == 0, (
            f"Expected no RED_TEAM_BREAKTHROUGH calls, but got: {breakthrough_calls}"
        )

    @pytest.mark.asyncio
    @given(num_breakthroughs=st.integers(min_value=1, max_value=5))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    async def test_multiple_breakthroughs_all_logged(self, num_breakthroughs: int) -> None:
        """Each breakthrough in a run must produce a separate tracer call."""
        runner = RedTeamRunner()
        tracer = self._make_mock_tracer()

        results = [
            AttackResult(
                category=f"cat{i}",
                session_id="s3",
                prompts=[f"attack {i}"],
                response="leaked",
                breakthrough=True,
            )
            for i in range(num_breakthroughs)
        ]

        call_count = 0

        async def _multi_execute(client, session_id, category, turns):
            nonlocal call_count
            r = results[call_count % len(results)]
            call_count += 1
            return r

        sequences = [(f"cat{i}", [f"attack {i}"]) for i in range(num_breakthroughs)]

        with patch.object(runner, "_try_mcp_run", AsyncMock(return_value=None)):
            with patch.object(runner, "_collect_attack_sequences", AsyncMock(return_value=sequences)):
                with patch.object(runner, "_execute_attack", _multi_execute):
                    with patch.object(runner, "_run_garak_probes", AsyncMock(return_value=[])):
                        await runner.run(tracer=tracer)

        await asyncio.sleep(0)

        calls = tracer.add_span.call_args_list
        bt_calls = [c for c in calls if c.kwargs.get("name") == "RED_TEAM_BREAKTHROUGH"]
        assert len(bt_calls) == num_breakthroughs, (
            f"Expected {num_breakthroughs} breakthrough log calls, got {len(bt_calls)}"
        )
