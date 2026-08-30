# Vendor Independence & Packaging Remediation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every commercial-SaaS dependency from ControlPlane.ai's required path — replacing Portkey with LiteLLM, the Guardrails Hub with local validators, and OpenAI embeddings with a local model — while fixing the packaging and dead-code defects that currently prevent the app from starting.

**Architecture:** A new `app/router/providers.py` becomes the single LLM egress point, wrapping `litellm.acompletion` (BSD-3, no account, no proxy) for both buffered and streaming calls. A new `app/router/complexity.py` supplies a real local SLM/FRONTIER routing score, replacing the hardcoded `score = 0.5` stub. `pyproject.toml` collapses to a six-package required floor with everything else behind extras, matching the `try/except ImportError` guards the code already has.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2 / pydantic-settings, httpx, watchdog, LiteLLM (BSD-3), sentence-transformers (Apache-2.0, optional), pytest + hypothesis.

**Spec:** This plan is self-contained; it is derived from `current_state.md` plus the codebase audit recorded in the "Findings" section below. `current_state.md` is itself a target of Task 9 because several of its claims are wrong.

---

## Findings — the concerns this plan closes

Recorded here because the plan argues from them, and because three of them contradict `current_state.md`.

### Commercial-SaaS dependencies in the required path

| # | Finding | Evidence |
|---|---|---|
| C1 | **Portkey (commercial SaaS) is preferred over the user's own key.** `route_and_call()` checks `PORTKEY_API_KEY` *before* `LLM_API_KEY`, so a Portkey key hijacks dispatch. | `app/router/model_router.py:88` vs `:115` |
| C2 | **SSE streaming has no non-Portkey path at all.** With only a Gemini/OpenAI key set, `/v1/chat/stream` cannot reach a real model — it silently serves canned text. | `app/ingress/streaming_router.py:261-279` |
| C3 | **Langfuse defaults to the vendor cloud.** | `app/config.py:78` → `https://cloud.langfuse.com` |
| C4 | **Guardrails runs a network install at boot.** `load_validators()` shells out to `python -m guardrails hub install <id>` against a remote registry, with a token env var for private validators. | `app/judges/output_validator.py:47-80` |
| C5 | **Default embedding model is a paid OpenAI endpoint.** | `app/config.py:94`, `app/router/semantic_cache.py:82` → `text-embedding-3-small` |

**Correcting the stated hypothesis:** Portkey hosts **no models of its own**. Its "virtual keys" are stored references to *your* OpenAI/Anthropic/Google credentials, so there is no "Portkey native SLM/frontier model" and no model-access lock-in to escape. What Portkey actually supplies is provider abstraction, retries, fallback and cost dashboards — which is precisely what LiteLLM replaces. The dependency is real and worth removing; the mechanism is routing, not model hosting.

### The dual-tier routing it is supposed to preserve does not exist

| # | Finding | Evidence |
|---|---|---|
| D1 | `FRONTIER_CONFIG` and `SLM_CONFIG` are dead dicts — nothing in the repo reads them. | `app/router/model_router.py:35-43` |
| D2 | The Portkey call always sends `PORTKEY_SLM_VIRTUAL_KEY`. `PORTKEY_FRONTIER_VIRTUAL_KEY` is never used in a request. | `app/router/model_router.py:97`, `app/ingress/streaming_router.py:278` |
| D3 | **Installing RouteLLM makes the gateway worse.** When `_controller` is non-None the function hardcodes `score = 0.5` and returns the literal string `"This is a mock LLM response."`, discarding the real-LLM paths entirely. | `app/router/model_router.py:181`, `:190` |
| D4 | Streaming hardcodes `"model": "gpt-3.5-turbo"`, ignoring `LLM_FALLBACK_MODEL`. | `app/ingress/streaming_router.py:286` |

Consequence: tiering must be **built**, not migrated. Task 2 is therefore net-new capability, not a port.

### Packaging and startup defects (carried over from the previous round)

| # | Finding | Evidence |
|---|---|---|
| P1 | **The app cannot start in the current environment.** `httpx` and `watchdog` are unguarded imports and are both absent; there is no venv in the project root. | `app/policy/loader.py:26-27`, `app/oversight/worldsense_oversight.py:23` |
| P2 | **`faiss-cpu` is a hard dependency that is never imported.** No `import faiss` exists anywhere in `app/` or `mcp_servers/`. | `pyproject.toml:17` |
| P3 | **`portkey-ai` is a hard dependency that is never imported.** Portkey is called as a plain `httpx` POST with three headers. | `pyproject.toml:15` |
| P4 | Nine packages the code guards with `try/except ImportError` are listed as **hard** requirements: `llm-guard`, `routellm`, `spacy`, `langfuse`, `guardrails-ai`, `pyrit`, `garak` (+ P2, P3). Baseline install is multi-gigabyte for a service that runs without any of them. | `pyproject.toml:10-30` |
| P5 | `pyrit` / `garak` belong to the red-team sidecar, which already has its own `requirements.txt`. | `mcp_servers/redteam/requirements.txt` |
| P6 | **Stage 3 groundedness is a stub presented as real.** `vector_store.py` is 22 lines returning `Document("mock document content")`; `auditor.py` uses `embedding = [0.1] * 128`. `current_state.md` describes a working FAISS pipeline. | `app/groundedness/vector_store.py`, `app/groundedness/auditor.py:59` |
| P7 | Both MCP sidecars default to enabled against `localhost`, costing ~4s of dead boot time (two 2s probes) on a fresh clone. | `app/config.py:74,84` |
| P8 | `PORTKEY_API_KEY` defaults to the literal string `"dummy-portkey-key"`. | `app/config.py:52` |
| P9 | The "245 tests pass, 4 skipped" claim is **unverified** — `pytest` and `hypothesis` are not installed. | `current_state.md` |

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Python floor:** `>=3.11` (do not raise it; `pyproject.toml` already declares this).
- **Required-dependency allowlist.** After Task 1 the `[project].dependencies` array contains **exactly** these six: `fastapi>=0.111.0`, `uvicorn[standard]>=0.29.0`, `pydantic>=2.7.1`, `pydantic-settings>=2.2.1`, `httpx>=0.27.0`, `watchdog>=4.0.0`. Adding anything else to that array requires an unguarded top-level import to justify it.
- **No commercial SaaS in any default.** No default config value may point at a vendor-hosted endpoint or name a paid model. This includes `LANGFUSE_HOST`, `EMBEDDING_MODEL`, and anything Portkey.
- **No network I/O at import or startup** beyond the explicitly opt-in MCP probes. Specifically: no `subprocess` package installation, ever.
- **Licences:** every newly added dependency must be OSI-approved and self-hostable. LiteLLM is BSD-3; sentence-transformers is Apache-2.0.
- **Graceful degradation is preserved.** Every optional import stays inside `try/except ImportError`, and every absent optional dependency must downgrade behaviour without raising.
- **Test command:** `.venv/bin/python -m pytest tests/unit -q` from the project root.
- **Commit style:** `<type>: <short description>` with type in `feat|fix|refactor|test|docs|chore`.
- **Field-name stability:** `RoutingDecision.routellm_score` keeps its name even though RouteLLM stops being the scorer. Renaming it would ripple through `app/models.py`, telemetry records, and the existing test suite for zero functional gain; a code comment records why.

---

## File Structure

**New files**

| Path | Responsibility |
|---|---|
| `app/router/providers.py` | The single LLM egress point. Wraps `litellm.acompletion` for buffered and streaming calls, owns retries/fallback and the SLM→FRONTIER model map. Nothing else in the app may call an LLM directly. |
| `app/router/complexity.py` | Pure-function local complexity scorer. Zero dependencies, zero I/O — the real replacement for the `score = 0.5` stub. |
| `app/judges/local_validators.py` | Offline output validators (competitor wordlist + toxicity regex) replacing the Guardrails Hub on the default path. |
| `tests/unit/test_complexity.py` | Scorer unit + property tests. |
| `tests/unit/test_providers.py` | Dispatch, fallback, and mock-path tests with `litellm` monkeypatched. |
| `tests/unit/test_local_validators.py` | Offline validator tests. |

**Modified files**

| Path | Change |
|---|---|
| `pyproject.toml` | Six-package required floor; everything else into extras; drop `faiss-cpu`, `portkey-ai`, `pyrit`, `garak`. |
| `app/config.py` | Remove all `PORTKEY_*`; add `SLM_MODEL` / `FRONTIER_MODEL`; local `EMBEDDING_MODEL`; blank `LANGFUSE_HOST`; sidecars default off. |
| `app/router/model_router.py` | Delete Portkey block + dead configs + the RouteLLM mock branch; delegate to `providers.py` and `complexity.py`. |
| `app/ingress/streaming_router.py` | Replace the Portkey streaming block with `providers.astream`. |
| `app/judges/output_validator.py` | Delete `_hub_install`; chain local validators first, Guardrails only as opt-in. |
| `app/config_health/router.py` | `portkey` → `llm_gateway`. |
| `app/models.py` | `ConfigHealthResponse.portkey` → `.llm_gateway`. |
| `app/groundedness/auditor.py` | Honest `technique` naming for the stub path (P6). |
| `mcp_servers/redteam/requirements.txt` | Gains `pyrit`, `garak`. |
| `.env.example`, `current_state.md` | Rewritten to match reality. |
| `tests/unit/test_config_health.py`, `tests/unit/test_model_router_provider.py` | Updated for the renames. |

---

### Task 0: Bootstrap a working environment and capture the true baseline

Closes **P1**, **P9**. Nothing else in this plan can be verified until this exists — right now `import app.main` fails on `httpx`, and the "245 tests pass" claim is unconfirmed.

**Files:**
- Create: `.venv/` (git-ignored — confirm `.gitignore` covers it)
- Create: `docs/superpowers/plans/baseline-2026-08-29.txt` (scratch record, deleted in Task 9)

**Interfaces:**
- Consumes: nothing.
- Produces: a `.venv` with the current `pyproject.toml` dev extras installed, and a recorded pass/fail count that every later task compares against.

- [ ] **Step 1: Create the venv and install the package as it stands today**

Install the *current* dependency set, unchanged. We need the real baseline before Task 1 alters it.

```bash
cd /home/vashu/workspace/darpan/ControlPlane.ai
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]' 2>&1 | tail -20
```

If `pyrit` or `garak` fail to build (both are heavy and frequently break on new Python), do **not** fight it — record the failure verbatim in the baseline file and continue with Step 2. Their removal is Task 1's job, and a failure here is direct evidence for it.

- [ ] **Step 2: Confirm `.gitignore` covers the venv**

```bash
grep -qE '^\.venv' .gitignore && echo "covered" || echo ".venv" >> .gitignore
```

- [ ] **Step 3: Record the honest baseline**

```bash
.venv/bin/python -m pytest tests/unit -q 2>&1 | tail -15 | tee docs/superpowers/plans/baseline-2026-08-29.txt
.venv/bin/python -c "import app.main; print('IMPORT OK')" 2>&1 | tail -5 | tee -a docs/superpowers/plans/baseline-2026-08-29.txt
```

Expected: a concrete number of passes/failures. Whatever it is, that number — not the 245 in `current_state.md` — is the regression bar for Tasks 1–9. If tests fail here, they were already failing; note which, and do not treat them as regressions you caused.

- [ ] **Step 4: Confirm the server actually boots**

```bash
.venv/bin/python -m uvicorn app.main:app --port 8099 &
sleep 8
curl -s localhost:8099/v1/config/health | head -20
kill %1
```

Expected: JSON with five integration keys. If startup hangs ~4s before responding, that is **P7** (the two MCP probes) — confirmed, and fixed in Task 7.

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore .venv"
```

---

### Task 1: Collapse the dependency floor to six packages

Closes **P2**, **P3**, **P4**, **P5**. This is the single highest-value change: it makes `pip install .` viable, and it is what actually makes the project standalone. Do it before the code changes so every later task installs fast.

**Files:**
- Modify: `pyproject.toml:10-30` (the `dependencies` and `optional-dependencies` arrays)
- Modify: `mcp_servers/redteam/requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: extras named `llm`, `safety`, `cache`, `grounded`, `observe`, `dev`, `all`. `llm` carries `litellm`, which Task 3 imports; Task 7 relies on `grounded` carrying `sentence-transformers`.

- [ ] **Step 1: Write the failing test that pins the required floor**

Create `tests/unit/test_packaging.py`. This test is the enforcement mechanism for the Global Constraints allowlist — it will outlive this plan.

```python
"""Packaging contract: the required dependency floor stays minimal.

Every package in [project].dependencies must correspond to an unguarded
top-level import somewhere in app/.  Everything else belongs in an extra,
because the code already guards it with try/except ImportError.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

REQUIRED_FLOOR = {
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic-settings",
    "httpx",
    "watchdog",
}

# Packages the code imports inside try/except ImportError, or does not
# import at all.  None of these may appear in [project].dependencies.
MUST_BE_OPTIONAL = {
    "faiss-cpu",      # never imported anywhere (P2)
    "portkey-ai",     # never imported anywhere (P3)
    "pyrit",          # redteam sidecar only (P5)
    "garak",          # redteam sidecar only (P5)
    "llm-guard",
    "routellm",
    "spacy",
    "langfuse",
    "guardrails-ai",
}


def _dep_names(specs: list[str]) -> set[str]:
    """Strip version specifiers and extras: 'uvicorn[standard]>=0.29' -> 'uvicorn'."""
    names = set()
    for spec in specs:
        name = spec.split(";")[0].strip()
        for sep in (">=", "<=", "==", "~=", ">", "<", "!="):
            name = name.split(sep)[0]
        names.add(name.split("[")[0].strip().lower())
    return names


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def test_required_dependencies_are_exactly_the_floor(pyproject: dict) -> None:
    assert _dep_names(pyproject["project"]["dependencies"]) == REQUIRED_FLOOR


def test_no_optional_package_is_a_hard_requirement(pyproject: dict) -> None:
    required = _dep_names(pyproject["project"]["dependencies"])
    leaked = required & MUST_BE_OPTIONAL
    assert leaked == set(), f"these belong in extras, not dependencies: {sorted(leaked)}"


def test_expected_extras_exist(pyproject: dict) -> None:
    extras = pyproject["project"]["optional-dependencies"]
    for group in ("llm", "safety", "cache", "grounded", "observe", "dev", "all"):
        assert group in extras, f"missing extra: {group}"


def test_no_commercial_only_packages_anywhere(pyproject: dict) -> None:
    """faiss-cpu and portkey-ai are dead code — they must be gone entirely."""
    all_specs = list(pyproject["project"]["dependencies"])
    for group_specs in pyproject["project"]["optional-dependencies"].values():
        all_specs.extend(group_specs)
    names = _dep_names(all_specs)
    assert "portkey-ai" not in names, "portkey-ai is never imported (P3)"
    assert "faiss-cpu" not in names, "faiss-cpu is never imported (P2)"
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
.venv/bin/python -m pytest tests/unit/test_packaging.py -v
```

Expected: FAIL — `test_required_dependencies_are_exactly_the_floor` reports the current 19-package set, and `test_no_commercial_only_packages_anywhere` flags `portkey-ai` and `faiss-cpu`.

- [ ] **Step 3: Rewrite the dependency arrays**

Replace `pyproject.toml` lines 10–30 (the `dependencies` array through the closing bracket of `[project.optional-dependencies]`) with:

```toml
# Required floor: every entry here has an unguarded top-level import in app/.
#   fastapi / uvicorn / pydantic / pydantic-settings — the service itself
#   httpx    — app/oversight/worldsense_oversight.py:23, app/redteam/runner.py:31,
#              app/ingress/streaming_router.py:27
#   watchdog — app/policy/loader.py:26-27 (policy hot-reload)
# Everything else is guarded by try/except ImportError and lives in an extra.
dependencies = [
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.7.1",
    "pydantic-settings>=2.2.1",
    "httpx>=0.27.0",
    "watchdog>=4.0.0",
]

[project.optional-dependencies]
# Real LLM dispatch. Without this the gateway serves contextual mock
# responses — fine for CI and local dev, useless in production.
llm = [
    "litellm>=1.44.0",
    "tiktoken>=0.7.0",
]
# Input-side safety upgrades: NLP PII masking, dependency parsing,
# zero-shot custom-entity NER, output validation.
safety = [
    "llm-guard",
    "spacy>=3.8.0",
    "gliner",
    "guardrails-ai",
]
# Semantic cache (Stage 0).
cache = [
    "gptcache",
    "numpy>=1.26.4",
]
# Groundedness: local embeddings + NLI cross-encoder (Stage 3).
grounded = [
    "sentence-transformers>=3.0.0",
]
# Optional observability. Langfuse's core is MIT and self-hostable;
# routellm is retained only for operators who already depend on it.
observe = [
    "langfuse",
    "routellm>=0.2.0",
]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.6",
    "pytest-timeout>=2.3.1",
    "hypothesis>=6.100.2",
    "anyio>=4.3.0",
]
all = [
    "controlplane-ai-gateway[llm,safety,cache,grounded,observe]",
]
```

Note `python-multipart` is dropped from the floor: no endpoint accepts multipart form data (all routers take JSON bodies). If a file-upload endpoint is ever added, it returns as that feature's dependency.

- [ ] **Step 4: Move the red-team tooling to its sidecar**

```bash
cat >> mcp_servers/redteam/requirements.txt <<'EOF'

# Adversarial frameworks — sidecar-only. The Gateway's in-process
# fallback library (5 built-in categories) needs neither of these.
pyrit
garak
EOF
```

- [ ] **Step 5: Verify the floor installs clean and fast in a throwaway venv**

```bash
python3 -m venv /tmp/cp-floor && /tmp/cp-floor/bin/python -m pip install -q -e . 2>&1 | tail -5
/tmp/cp-floor/bin/python -c "import app.main; print('MINIMAL IMPORT OK')"
du -sh /tmp/cp-floor
```

Expected: `MINIMAL IMPORT OK`, and a venv in the tens of megabytes rather than gigabytes. This is the proof the project is standalone. If the import fails, an unguarded import was missed — add that package to `REQUIRED_FLOOR` in the test *and* to `dependencies`, with a comment naming the file and line.

```bash
rm -rf /tmp/cp-floor
```

- [ ] **Step 6: Run the packaging test and the full suite**

```bash
.venv/bin/python -m pytest tests/unit/test_packaging.py -v
.venv/bin/python -m pytest tests/unit -q 2>&1 | tail -5
```

Expected: packaging tests PASS; suite total matches the Task 0 baseline.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml mcp_servers/redteam/requirements.txt tests/unit/test_packaging.py
git commit -m "chore: reduce required deps to six; drop unused faiss-cpu and portkey-ai"
```

---

### Task 2: Build a real local complexity scorer

Closes **D3**. The SLM/FRONTIER decision currently comes from `score = 0.5` hardcoded at `model_router.py:181`. This task supplies the genuine article: a pure function, no dependencies, no I/O, no vendor.

**Files:**
- Create: `app/router/complexity.py`
- Test: `tests/unit/test_complexity.py`

**Interfaces:**
- Consumes: nothing (deliberately dependency-free so it works on the six-package floor).
- Produces: `score_complexity(prompt: str) -> float` returning a value in `[0.0, 1.0]`, and `classify(prompt: str, threshold: float) -> tuple[Literal["ROUTINE","COMPLEX"], Literal["SLM","FRONTIER"], float]`. Task 4 calls both.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_complexity.py`:

```python
"""Local complexity scorer — replaces the hardcoded score=0.5 stub."""
from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from app.router.complexity import classify, score_complexity


class TestScoreRange:
    @given(st.text(max_size=2000))
    def test_score_always_in_unit_interval(self, prompt: str) -> None:
        """P-CX-1: the score is a probability-like value for ANY input."""
        assert 0.0 <= score_complexity(prompt) <= 1.0

    def test_empty_prompt_scores_zero(self) -> None:
        assert score_complexity("") == 0.0
        assert score_complexity("   ") == 0.0

    def test_deterministic(self) -> None:
        """P-CX-2: same input, same score — routing must be reproducible."""
        p = "Compare the trade-offs between optimistic and pessimistic locking."
        assert score_complexity(p) == score_complexity(p)


class TestOrdering:
    """P-CX-3: genuinely harder prompts must outscore trivial ones."""

    def test_reasoning_prompt_beats_lookup_prompt(self) -> None:
        simple = "What is your return policy?"
        complex_ = (
            "Analyse why our p99 latency regressed after the shard migration, "
            "compare it against the pre-migration baseline, and explain the "
            "trade-offs of rolling back versus re-sharding."
        )
        assert score_complexity(complex_) > score_complexity(simple)

    def test_length_increases_score_monotonically(self) -> None:
        short = "Explain caching."
        long = "Explain caching. " + ("It must handle eviction and staleness. " * 20)
        assert score_complexity(long) > score_complexity(short)

    def test_code_block_raises_score(self) -> None:
        plain = "Why does this fail?"
        with_code = "Why does this fail?\n```python\nfor i in x:\n    y(i)\n```"
        assert score_complexity(with_code) > score_complexity(plain)

    def test_multiple_questions_raise_score(self) -> None:
        one = "Is the cache enabled?"
        many = "Is the cache enabled? What is the TTL? How is it evicted?"
        assert score_complexity(many) > score_complexity(one)


class TestClassify:
    def test_below_threshold_routes_to_slm(self) -> None:
        cls, tier, score = classify("Hi", threshold=0.7)
        assert cls == "ROUTINE"
        assert tier == "SLM"
        assert score < 0.7

    def test_at_or_above_threshold_routes_to_frontier(self) -> None:
        """Boundary is inclusive: score >= threshold means COMPLEX."""
        cls, tier, _ = classify("anything", threshold=0.0)
        assert cls == "COMPLEX"
        assert tier == "FRONTIER"

    @given(st.floats(min_value=0.0, max_value=1.0), st.text(max_size=500))
    def test_tier_and_classification_never_disagree(
        self, threshold: float, prompt: str
    ) -> None:
        """P-CX-4: SLM<->ROUTINE and FRONTIER<->COMPLEX are locked together."""
        cls, tier, score = classify(prompt, threshold=threshold)
        assert (cls == "COMPLEX") == (tier == "FRONTIER")
        assert (score >= threshold) == (cls == "COMPLEX")
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_complexity.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.router.complexity'`.

- [ ] **Step 3: Implement the scorer**

Create `app/router/complexity.py`:

```python
"""Local prompt-complexity scorer for SLM/FRONTIER tier selection.

Replaces the hardcoded ``score = 0.5`` stub that RouteLLM's branch used to
return.  Deliberately dependency-free: it must work on the six-package
required floor, so no tiktoken, no transformers, no network.

The score is a weighted sum of four cheap signals, each independently
normalised to [0, 1] and then clamped.  It is a heuristic, not a learned
model — the point is that it is *real, deterministic and local*, and that
harder prompts reliably score higher than trivial ones.
"""

from __future__ import annotations

import re
from typing import Literal

# Words that signal multi-step reasoning rather than fact lookup.
_REASONING_TERMS = frozenset(
    {
        "analyse", "analyze", "compare", "contrast", "evaluate", "explain",
        "justify", "derive", "prove", "optimise", "optimize", "refactor",
        "design", "architect", "debug", "diagnose", "trade-off", "tradeoff",
        "implication", "consequence", "why", "how", "strategy", "root cause",
    }
)

# Signal weights. They sum to 1.0 so the raw score lands in [0, 1]
# before clamping.
_W_LENGTH = 0.35
_W_REASONING = 0.30
_W_STRUCTURE = 0.20
_W_QUESTIONS = 0.15

# A prompt at or above this many words saturates the length signal.
_LENGTH_SATURATION_WORDS = 200
# Reasoning-term hits at or above this count saturate that signal.
_REASONING_SATURATION = 4
# Question marks at or above this count saturate that signal.
_QUESTION_SATURATION = 3

_CODE_FENCE = re.compile(r"```|\n\s{4}\S|;\s*$", re.MULTILINE)
_WORD = re.compile(r"\b\w+\b")


def score_complexity(prompt: str) -> float:
    """Return a complexity score in [0.0, 1.0] for *prompt*.

    0.0 means trivial (or empty); 1.0 means maximally complex. The value is
    deterministic — identical input always yields an identical score, which
    routing reproducibility depends on.
    """
    if not prompt or not prompt.strip():
        return 0.0

    lowered = prompt.lower()
    words = _WORD.findall(lowered)
    word_count = len(words)

    # 1. Length — longer prompts carry more context to reason over.
    length_signal = min(word_count / _LENGTH_SATURATION_WORDS, 1.0)

    # 2. Reasoning vocabulary — counted over the raw text so multi-word
    #    terms like "root cause" are matched too.
    hits = sum(1 for term in _REASONING_TERMS if term in lowered)
    reasoning_signal = min(hits / _REASONING_SATURATION, 1.0)

    # 3. Structure — code fences, indented blocks or statement terminators
    #    mean the model must parse as well as answer.
    structure_signal = 1.0 if _CODE_FENCE.search(prompt) else 0.0

    # 4. Question density — several questions in one prompt means several
    #    sub-answers.
    question_signal = min(prompt.count("?") / _QUESTION_SATURATION, 1.0)

    raw = (
        _W_LENGTH * length_signal
        + _W_REASONING * reasoning_signal
        + _W_STRUCTURE * structure_signal
        + _W_QUESTIONS * question_signal
    )
    # Clamp defensively: weights are trusted to sum to 1.0, but a future
    # weight edit must never be able to emit an out-of-range score.
    return max(0.0, min(raw, 1.0))


def classify(
    prompt: str, threshold: float
) -> tuple[Literal["ROUTINE", "COMPLEX"], Literal["SLM", "FRONTIER"], float]:
    """Score *prompt* and map it to a classification and a model tier.

    The boundary is inclusive: ``score >= threshold`` is COMPLEX, matching
    the existing ``classification = "COMPLEX" if score >= threshold`` logic
    that model_router.py used before this module existed.

    Returns ``(classification, selected_tier, score)``.
    """
    score = score_complexity(prompt)
    if score >= threshold:
        return "COMPLEX", "FRONTIER", score
    return "ROUTINE", "SLM", score
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/test_complexity.py -v
```

Expected: PASS, all cases including the four Hypothesis properties.

- [ ] **Step 5: Commit**

```bash
git add app/router/complexity.py tests/unit/test_complexity.py
git commit -m "feat: add local dependency-free prompt complexity scorer"
```

---

### Task 3: Build the LiteLLM egress layer

Closes **C1** (partly — Task 4 removes the last Portkey preference) and gives Tasks 4 and 5 a single shared call path. LiteLLM is BSD-3, pip-installable, needs no account and no running proxy: it is a library, not a service.

This task also adds the config fields the layer needs, because `providers.py` cannot import settings that do not exist yet.

**Files:**
- Create: `app/router/providers.py`
- Modify: `app/config.py` (section 1 — add tier model fields)
- Test: `tests/unit/test_providers.py`

**Interfaces:**
- Consumes: `settings` from `app/config.py`; `UseCaseProfile` from `app/models.py`.
- Produces — Tasks 4 and 5 call exactly these:
  - `async acomplete(prompt: str, tier: Literal["SLM","FRONTIER"], system: str | None = None) -> tuple[str, str]` → `(response_text, model_actually_used)`
  - `async astream(prompt: str, tier: Literal["SLM","FRONTIER"], system: str | None = None) -> AsyncGenerator[str, None]`
  - `generate_contextual_response(prompt: str) -> str` (moved here from `model_router.py`)
  - `is_live() -> bool`

- [ ] **Step 1: Add the tier model settings**

In `app/config.py`, replace section 1's body (currently `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_FALLBACK_MODEL` at lines 41–48) with:

```python
    # -------------------------------------------------------------------------
    # 1. LLM Provider — dispatched via LiteLLM (BSD-3, no gateway service)
    # -------------------------------------------------------------------------
    # Model strings may be bare ("gemini-2.5-flash") or fully qualified
    # ("gemini/gemini-2.5-flash").  Bare names are prefixed with the LiteLLM
    # provider derived from LLM_PROVIDER at call time.
    # If LLM_API_KEY is absent/dummy, the gateway serves safe contextual mock
    # responses — useful for local dev, CI and the test suite.
    LLM_PROVIDER: Literal["openai", "anthropic", "google", "grok", "generic"] = "openai"
    LLM_API_KEY: str = ""

    # Two-tier routing.  SLM handles ROUTINE prompts cheaply; FRONTIER handles
    # COMPLEX ones.  Each is the other's fallback on dispatch failure.
    SLM_MODEL: str = "gpt-4o-mini"
    FRONTIER_MODEL: str = "gpt-4o"

    # Retained as the alias older configs use for the SLM tier.  When
    # SLM_MODEL is left at its default and this is set, this wins.
    LLM_FALLBACK_MODEL: str = "gpt-4o-mini"

    # Per-call egress budget.  LiteLLM performs the retries internally.
    LLM_TIMEOUT_S: float = 30.0
    LLM_MAX_RETRIES: int = 2

    # Optional explicit base URL — set this for self-hosted or
    # OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, LiteLLM proxy).
    LLM_API_BASE: str = ""
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_providers.py`. Every test monkeypatches `litellm` so the suite never makes a network call.

```python
"""LiteLLM egress layer — the single LLM call path.

All tests stub the transport: the suite must never touch the network.
"""
from __future__ import annotations

import pytest

from app.router import providers


# --------------------------------------------------------------------------
# Model resolution


class TestModelResolution:
    def test_bare_model_name_gets_provider_prefix(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_PROVIDER", "google")
        monkeypatch.setattr(providers.settings, "SLM_MODEL", "gemini-2.5-flash")
        assert providers._model_for_tier("SLM") == "gemini/gemini-2.5-flash"

    def test_already_qualified_model_is_left_alone(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_PROVIDER", "google")
        monkeypatch.setattr(providers.settings, "SLM_MODEL", "openai/gpt-4o-mini")
        assert providers._model_for_tier("SLM") == "openai/gpt-4o-mini"

    def test_grok_maps_to_litellm_xai_prefix(self, monkeypatch) -> None:
        """LiteLLM calls xAI 'xai', not 'grok' — the mapping must translate."""
        monkeypatch.setattr(providers.settings, "LLM_PROVIDER", "grok")
        monkeypatch.setattr(providers.settings, "FRONTIER_MODEL", "grok-2")
        assert providers._model_for_tier("FRONTIER") == "xai/grok-2"

    def test_frontier_and_slm_are_distinct(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_PROVIDER", "openai")
        monkeypatch.setattr(providers.settings, "SLM_MODEL", "gpt-4o-mini")
        monkeypatch.setattr(providers.settings, "FRONTIER_MODEL", "gpt-4o")
        assert providers._model_for_tier("SLM") != providers._model_for_tier("FRONTIER")

    def test_legacy_fallback_model_supplies_slm_when_slm_left_default(
        self, monkeypatch
    ) -> None:
        """An existing .env with only LLM_FALLBACK_MODEL keeps working."""
        monkeypatch.setattr(providers.settings, "LLM_PROVIDER", "google")
        monkeypatch.setattr(providers.settings, "SLM_MODEL", "gpt-4o-mini")  # default
        monkeypatch.setattr(providers.settings, "LLM_FALLBACK_MODEL", "gemini-2.5-flash")
        assert providers._model_for_tier("SLM") == "gemini/gemini-2.5-flash"


# --------------------------------------------------------------------------
# Mock path (no key configured)


class TestMockPath:
    def test_is_live_false_without_key(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "")
        assert providers.is_live() is False

    def test_is_live_false_for_dummy_key(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "dummy-key")
        assert providers.is_live() is False

    def test_is_live_false_when_litellm_absent(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "sk-real-abc")
        monkeypatch.setattr(providers, "_HAS_LITELLM", False)
        assert providers.is_live() is False

    async def test_acomplete_returns_mock_without_key(self, monkeypatch) -> None:
        """P-PR-1: absent credentials degrade to mock text, never an exception."""
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "")
        text, model = await providers.acomplete("What is your return policy?", "SLM")
        assert "30 calendar days" in text
        assert model == "mock"

    async def test_astream_yields_mock_chunks_without_key(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "")
        chunks = [c async for c in providers.astream("refund please", "SLM")]
        assert len(chunks) > 1                      # genuinely chunked
        assert "30 calendar days" in "".join(chunks)


# --------------------------------------------------------------------------
# Live path, with litellm stubbed


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)
        self.delta = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


@pytest.fixture
def live(monkeypatch):
    """Put the module on its live path with a stubbed litellm."""
    monkeypatch.setattr(providers.settings, "LLM_API_KEY", "sk-real-abc")
    monkeypatch.setattr(providers.settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(providers.settings, "SLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(providers.settings, "FRONTIER_MODEL", "gpt-4o")
    monkeypatch.setattr(providers, "_HAS_LITELLM", True)
    return monkeypatch


class TestLivePath:
    async def test_acomplete_passes_resolved_model_and_returns_text(
        self, live
    ) -> None:
        seen: dict = {}

        async def fake_acompletion(**kwargs):
            seen.update(kwargs)
            return _FakeResponse("live answer")

        live.setattr(providers, "_acompletion", fake_acompletion)
        text, model = await providers.acomplete("hello", "FRONTIER")

        assert text == "live answer"
        assert model == "openai/gpt-4o"
        assert seen["model"] == "openai/gpt-4o"
        assert seen["messages"] == [{"role": "user", "content": "hello"}]
        assert seen["num_retries"] == providers.settings.LLM_MAX_RETRIES

    async def test_system_prompt_is_prepended(self, live) -> None:
        seen: dict = {}

        async def fake_acompletion(**kwargs):
            seen.update(kwargs)
            return _FakeResponse("ok")

        live.setattr(providers, "_acompletion", fake_acompletion)
        await providers.acomplete("q", "SLM", system="be careful")

        assert seen["messages"][0] == {"role": "system", "content": "be careful"}
        assert seen["messages"][1] == {"role": "user", "content": "q"}

    async def test_falls_back_to_other_tier_on_failure(self, live) -> None:
        """P-PR-2: SLM failure retries on FRONTIER — the fallback Portkey gave us."""
        attempts: list[str] = []

        async def flaky(**kwargs):
            attempts.append(kwargs["model"])
            if kwargs["model"] == "openai/gpt-4o-mini":
                raise RuntimeError("upstream 503")
            return _FakeResponse("frontier saved it")

        live.setattr(providers, "_acompletion", flaky)
        text, model = await providers.acomplete("hello", "SLM")

        assert attempts == ["openai/gpt-4o-mini", "openai/gpt-4o"]
        assert text == "frontier saved it"
        assert model == "openai/gpt-4o"

    async def test_mock_used_when_both_tiers_fail(self, live) -> None:
        """P-PR-3: total upstream failure still returns a response, never raises."""

        async def always_fails(**kwargs):
            raise RuntimeError("everything is down")

        live.setattr(providers, "_acompletion", always_fails)
        text, model = await providers.acomplete("refund", "SLM")

        assert model == "mock"
        assert "30 calendar days" in text

    async def test_astream_yields_deltas(self, live) -> None:
        async def fake_stream(**kwargs):
            for piece in ("Hel", "lo ", "world"):
                yield _FakeResponse(piece)

        live.setattr(providers, "_acompletion_stream", fake_stream)
        chunks = [c async for c in providers.astream("hi", "SLM")]
        assert "".join(chunks) == "Hello world"

    async def test_astream_raises_on_failure_so_caller_can_emit_stream_error(
        self, live
    ) -> None:
        """Streaming cannot silently fall back mid-response: the SSE layer
        needs the exception so it can emit [STREAM_ERROR]."""

        async def broken(**kwargs):
            raise RuntimeError("stream died")
            yield  # pragma: no cover

        live.setattr(providers, "_acompletion_stream", broken)
        with pytest.raises(RuntimeError):
            [c async for c in providers.astream("hi", "SLM")]
```

- [ ] **Step 3: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_providers.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.router.providers'`.

- [ ] **Step 4: Implement the egress layer**

Create `app/router/providers.py`:

```python
"""LiteLLM egress layer — the single LLM call path for the whole gateway.

Replaces the previous Portkey coupling.  LiteLLM (BSD-3) is a *library*, not
a hosted service: it needs no account, no API gateway and no running proxy,
and it speaks 100+ providers through one call signature.  It supplies the
retries, timeouts and provider abstraction that Portkey used to, without the
commercial dependency.

Nothing else in the application may call an LLM directly.  Both
``app/router/model_router.py`` and ``app/ingress/streaming_router.py``
route through here.

Degradation ladder, in order:
  1. Requested tier via LiteLLM.
  2. The other tier (SLM <-> FRONTIER) on any dispatch failure.
  3. Contextual mock text — always available, never raises.

Streaming is the one exception to step 3: a failure mid-stream must
propagate so the SSE layer can emit ``[STREAM_ERROR]`` rather than splice
fake tokens into a partially delivered response.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Literal

from app.config import settings, _is_real_key

logger = logging.getLogger(__name__)

Tier = Literal["SLM", "FRONTIER"]

# Optional litellm import — the gateway runs on mock responses without it.
try:
    import litellm  # type: ignore[import-untyped]

    # Quieten the library: it otherwise prints provider banners and
    # cost tables to stdout, which corrupts our structured logs.
    litellm.suppress_debug_info = True
    litellm.drop_params = True  # ignore kwargs a given provider rejects
    _acompletion = litellm.acompletion
    _HAS_LITELLM = True
except ImportError:  # pragma: no cover - exercised via monkeypatch
    litellm = None  # type: ignore[assignment]
    _acompletion = None  # type: ignore[assignment]
    _HAS_LITELLM = False
    logger.info(
        "LITELLM_ABSENT — install with `pip install '.[llm]'` for real LLM "
        "dispatch; contextual mock responses active"
    )


# LLM_PROVIDER values -> LiteLLM provider prefixes.  These differ: LiteLLM
# calls Google "gemini" and xAI "xai".
_PROVIDER_PREFIX: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "gemini",
    "grok": "xai",
    "generic": "openai",  # any OpenAI-compatible endpoint via LLM_API_BASE
}

_DEFAULT_SLM = "gpt-4o-mini"


def _model_for_tier(tier: Tier) -> str:
    """Return the fully qualified LiteLLM model string for *tier*.

    Bare names gain a provider prefix; already-qualified names
    (containing "/") are passed through untouched so an operator can mix
    providers across tiers.
    """
    if tier == "FRONTIER":
        model = settings.FRONTIER_MODEL
    else:
        model = settings.SLM_MODEL
        # Back-compat: an existing .env may set only LLM_FALLBACK_MODEL.
        if model == _DEFAULT_SLM and settings.LLM_FALLBACK_MODEL != _DEFAULT_SLM:
            model = settings.LLM_FALLBACK_MODEL

    if "/" in model:
        return model
    prefix = _PROVIDER_PREFIX.get(settings.LLM_PROVIDER, "openai")
    return f"{prefix}/{model}"


def _other(tier: Tier) -> Tier:
    return "FRONTIER" if tier == "SLM" else "SLM"


def is_live() -> bool:
    """True when a real LLM call is possible (litellm present + real key)."""
    return _HAS_LITELLM and _is_real_key(settings.LLM_API_KEY)


def generate_contextual_response(prompt: str) -> str:
    """Deterministic canned answers for local dev, CI and total-failure paths.

    Moved here from model_router.py so that every mock path — buffered,
    streaming, and post-failure — produces identical text.
    """
    p = prompt.lower()
    if "return" in p or "refund" in p:
        return (
            "Our standard enterprise policy permits hardware returns within 30 "
            "calendar days of delivery, provided items are returned in original "
            "packaging with valid RMA authorization."
        )
    if "balance" in p or "account" in p or "ssn" in p or "email" in p:
        return (
            "Your corporate account identity has been verified. Current balance: "
            "$2,450.00 with active enterprise tier access."
        )
    if "phoenix" in p or "apollo" in p or "checklist" in p:
        return (
            "The deployment verification checklist requires passing all integration "
            "regression suites, validating IAM least-privilege policies, and "
            "ensuring multi-region redundancy."
        )
    if "dan" in p or "bypass" in p:
        return "I am unable to fulfill instructions that request disabling enterprise security guardrails."
    return (
        f"Request processed: {prompt}. Completed safely in accordance with "
        "enterprise safety standards."
    )


def _build_messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _call_kwargs(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "api_key": settings.LLM_API_KEY,
        "timeout": settings.LLM_TIMEOUT_S,
        "num_retries": settings.LLM_MAX_RETRIES,
        "max_tokens": 512,
    }
    if settings.LLM_API_BASE:
        kwargs["api_base"] = settings.LLM_API_BASE
    return kwargs


async def acomplete(
    prompt: str,
    tier: Tier,
    system: str | None = None,
) -> tuple[str, str]:
    """Dispatch *prompt* at *tier* and return ``(text, model_used)``.

    Never raises.  On dispatch failure it retries the opposite tier, then
    falls back to contextual mock text with ``model_used == "mock"``.
    """
    if not is_live():
        return generate_contextual_response(prompt), "mock"

    messages = _build_messages(prompt, system)

    for attempt_tier in (tier, _other(tier)):
        model = _model_for_tier(attempt_tier)
        try:
            response = await _acompletion(**_call_kwargs(model, messages))
            text = response.choices[0].message.content or ""
            logger.info("LLM_DISPATCH_OK model=%s tier=%s", model, attempt_tier)
            return text, model
        except Exception as exc:
            logger.warning(
                "LLM_DISPATCH_FAILED model=%s tier=%s error=%s",
                model,
                attempt_tier,
                exc,
            )

    logger.error("LLM_DISPATCH_EXHAUSTED — falling back to mock response")
    return generate_contextual_response(prompt), "mock"


async def _acompletion_stream(**kwargs: Any) -> AsyncGenerator[Any, None]:
    """Thin seam over litellm streaming so tests can substitute it."""
    stream = await _acompletion(stream=True, **kwargs)
    async for chunk in stream:
        yield chunk


async def astream(
    prompt: str,
    tier: Tier,
    system: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield content deltas for *prompt*.

    Unlike :func:`acomplete` this **propagates** dispatch errors: the SSE
    endpoint must be able to emit ``[STREAM_ERROR]`` rather than splice mock
    tokens into a half-delivered response.  When no credentials are
    configured it yields a chunked mock stream instead.
    """
    if not is_live():
        for word in generate_contextual_response(prompt).split():
            yield word + " "
            await asyncio.sleep(0)
        return

    model = _model_for_tier(tier)
    messages = _build_messages(prompt, system)

    async for chunk in _acompletion_stream(**_call_kwargs(model, messages)):
        try:
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None)
        except (AttributeError, IndexError):
            continue
        if token:
            yield token
```

- [ ] **Step 5: Install litellm and run the tests**

```bash
.venv/bin/python -m pip install -q 'litellm>=1.44.0'
.venv/bin/python -m pytest tests/unit/test_providers.py -v
```

Expected: PASS. If LiteLLM's import emits noise despite `suppress_debug_info`, add `litellm.set_verbose = False` beside it — do not remove the guard.

- [ ] **Step 6: Commit**

```bash
git add app/router/providers.py app/config.py tests/unit/test_providers.py
git commit -m "feat: add LiteLLM egress layer with two-tier fallback"
```

---

### Task 4: Rewire the model router onto LiteLLM and delete the dead Portkey code

Closes **C1**, **D1**, **D2**, **D3**. After this task `model_router.py` contains no vendor-specific code and no mock-response branch that overrides real LLM calls.

**Files:**
- Modify: `app/router/model_router.py` (near-total rewrite — 201 lines to roughly 70)
- Modify: `tests/unit/test_model_router_provider.py`

**Interfaces:**
- Consumes: `providers.acomplete`, `providers.generate_contextual_response`, `providers.is_live` (Task 3); `complexity.classify` (Task 2).
- Produces: `route_and_call(prompt, profile, p3_clarity=None) -> RoutingDecision` — same signature as today, so `app/ingress/router.py` needs no change. `init_router() -> None` stays for the lifespan Step 6 call. `_generate_contextual_response` remains importable as a backward-compatible alias.

- [ ] **Step 1: Write the failing tests**

Replace the contents of `tests/unit/test_model_router_provider.py`:

```python
"""Model router — LiteLLM dispatch, real tiering, no Portkey."""
from __future__ import annotations

import inspect

import pytest

from app.models import UseCaseProfile
from app.router import model_router, providers


@pytest.fixture
def profile() -> UseCaseProfile:
    return UseCaseProfile(
        name="test_profile",
        complexity_threshold=0.3,
        token_compression_threshold=500,
        groundedness_pass_threshold=0.9,
    )


class TestPortkeyIsGone:
    """C1/D1/D2: no vendor coupling survives anywhere in the module."""

    def test_no_portkey_reference_in_source(self) -> None:
        src = inspect.getsource(model_router).lower()
        assert "portkey" not in src

    def test_dead_tier_configs_removed(self) -> None:
        assert not hasattr(model_router, "FRONTIER_CONFIG")
        assert not hasattr(model_router, "SLM_CONFIG")

    def test_no_hardcoded_mock_response_string(self) -> None:
        """D3: 'This is a mock LLM response.' must no longer exist."""
        assert "This is a mock LLM response" not in inspect.getsource(model_router)


class TestRouting:
    async def test_simple_prompt_selects_slm(self, profile, monkeypatch) -> None:
        captured: dict = {}

        async def fake_acomplete(prompt, tier, system=None):
            captured["tier"] = tier
            return "answer", "openai/gpt-4o-mini"

        monkeypatch.setattr(model_router.providers, "acomplete", fake_acomplete)
        decision = await model_router.route_and_call("Hi", profile)

        assert captured["tier"] == "SLM"
        assert decision.selected_tier == "SLM"
        assert decision.classification == "ROUTINE"
        assert decision.response == "answer"

    async def test_complex_prompt_selects_frontier(self, profile, monkeypatch) -> None:
        captured: dict = {}

        async def fake_acomplete(prompt, tier, system=None):
            captured["tier"] = tier
            return "deep answer", "openai/gpt-4o"

        monkeypatch.setattr(model_router.providers, "acomplete", fake_acomplete)
        decision = await model_router.route_and_call(
            "Analyse and compare the trade-offs, then explain why the p99 "
            "regressed and justify a rollback strategy.",
            profile,
        )

        assert captured["tier"] == "FRONTIER"
        assert decision.selected_tier == "FRONTIER"
        assert decision.classification == "COMPLEX"

    async def test_score_is_real_not_hardcoded(self, profile, monkeypatch) -> None:
        """D3: two different prompts must produce two different scores."""

        async def fake_acomplete(prompt, tier, system=None):
            return "x", "m"

        monkeypatch.setattr(model_router.providers, "acomplete", fake_acomplete)
        a = await model_router.route_and_call("Hi", profile)
        b = await model_router.route_and_call(
            "Explain, analyse and compare why this design fails under load. " * 8,
            profile,
        )
        assert a.routellm_score != b.routellm_score
        assert a.routellm_score != 0.5 or b.routellm_score != 0.5

    async def test_ambiguous_clarity_biases_to_frontier(
        self, profile, monkeypatch
    ) -> None:
        """An AMBIGUOUS P3 verdict escalates the tier regardless of score."""
        captured: dict = {}

        async def fake_acomplete(prompt, tier, system=None):
            captured["tier"] = tier
            captured["system"] = system
            return "careful answer", "openai/gpt-4o"

        monkeypatch.setattr(model_router.providers, "acomplete", fake_acomplete)
        decision = await model_router.route_and_call(
            "Hi", profile, p3_clarity="AMBIGUOUS"
        )

        assert captured["tier"] == "FRONTIER"
        assert captured["system"] is not None
        assert decision.selected_tier == "FRONTIER"

    async def test_mock_path_returns_contextual_response(
        self, profile, monkeypatch
    ) -> None:
        """No credentials: still a usable RoutingDecision, no exception."""
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "")
        decision = await model_router.route_and_call(
            "What is your return policy?", profile
        )
        assert decision.response is not None
        assert "30 calendar days" in decision.response
        assert decision.triage_state is None

    async def test_dispatch_exception_hard_blocks(self, profile, monkeypatch) -> None:
        """A defect inside dispatch fails closed, not open."""

        async def boom(prompt, tier, system=None):
            raise RuntimeError("unexpected internal error")

        monkeypatch.setattr(model_router.providers, "acomplete", boom)
        decision = await model_router.route_and_call("Hi", profile)

        assert decision.triage_state == "HARD_BLOCK"
        assert decision.response is None


class TestBackCompat:
    def test_contextual_response_alias_still_importable(self) -> None:
        """streaming_router and older tests import this name from here."""
        from app.router.model_router import _generate_contextual_response

        assert "30 calendar days" in _generate_contextual_response("refund")

    def test_init_router_is_safe_to_call(self) -> None:
        model_router.init_router()  # lifespan Step 6 — must not raise
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_model_router_provider.py -v
```

Expected: FAIL — `test_no_portkey_reference_in_source` finds Portkey, `test_dead_tier_configs_removed` finds both dicts, and the routing tests fail because `model_router.providers` does not exist.

- [ ] **Step 3: Rewrite the module**

Replace the entire contents of `app/router/model_router.py`:

```python
"""Model Router — local complexity scoring + LiteLLM dispatch.

Selects a model tier for each prompt and dispatches it.  Both halves are
vendor-neutral: scoring is a local pure function
(:mod:`app.router.complexity`) and dispatch goes through the LiteLLM egress
layer (:mod:`app.router.providers`).

Replaces the previous RouteLLM + Portkey implementation, in which the
"routing" was a hardcoded ``score = 0.5`` and the SLM/FRONTIER virtual-key
configs were never read.
"""

from __future__ import annotations

import logging
from typing import Literal

from app.config import settings
from app.models import RoutingDecision, UseCaseProfile
from app.router import providers
from app.router.complexity import classify

logger = logging.getLogger(__name__)

# System-message bias applied when the P3 judge reports an ambiguous query.
_AMBIGUITY_SYSTEM_PROMPT = (
    "You are a highly capable frontier model. Please handle this ambiguous or "
    "complex query carefully."
)

# Backward-compatible alias.  app/ingress/streaming_router.py and parts of the
# test suite import this name from this module.
_generate_contextual_response = providers.generate_contextual_response


def init_router() -> None:
    """Log the active dispatch configuration once at startup.

    Retained for lifespan Step 6.  There is no client to construct: LiteLLM
    is stateless and configured per call.
    """
    if providers.is_live():
        logger.info(
            "LLM_DISPATCH_ACTIVE provider=%s slm=%s frontier=%s",
            settings.LLM_PROVIDER,
            providers._model_for_tier("SLM"),
            providers._model_for_tier("FRONTIER"),
        )
    else:
        logger.info("LLM_DISPATCH_DEGRADED — contextual mock responses active")


async def route_and_call(
    prompt: str,
    profile: UseCaseProfile,
    p3_clarity: Literal["CLEAR", "AMBIGUOUS"] | None = None,
) -> RoutingDecision:
    """Score *prompt*, pick a tier, dispatch, and return a RoutingDecision.

    An AMBIGUOUS P3 clarity verdict escalates to FRONTIER regardless of
    score, on the grounds that an unclear query needs the stronger model.
    Any unexpected exception fails closed with ``triage_state="HARD_BLOCK"``.
    """
    try:
        classification, tier, score = classify(
            prompt, threshold=profile.complexity_threshold
        )

        system: str | None = None
        if p3_clarity == "AMBIGUOUS":
            classification, tier = "COMPLEX", "FRONTIER"
            system = _AMBIGUITY_SYSTEM_PROMPT

        text, model_used = await providers.acomplete(prompt, tier, system=system)

        logger.info(
            "ROUTE_DECISION tier=%s classification=%s score=%.3f model=%s",
            tier,
            classification,
            score,
            model_used,
        )
        return RoutingDecision(
            classification=classification,
            selected_tier=tier,
            # Field name retained for telemetry and test-suite stability even
            # though RouteLLM is no longer the scorer.
            routellm_score=score,
            response=text,
            triage_state=None,
        )
    except Exception as exc:
        logger.error("ROUTE_DISPATCH_ERROR %s", exc, exc_info=True)
        return RoutingDecision(
            classification="COMPLEX",
            selected_tier="FRONTIER",
            routellm_score=1.0,
            response=None,
            triage_state="HARD_BLOCK",
        )
```

- [ ] **Step 4: Run the router tests, then the full suite**

```bash
.venv/bin/python -m pytest tests/unit/test_model_router_provider.py -v
.venv/bin/python -m pytest tests/unit -q 2>&1 | tail -8
```

Expected: router tests PASS. Suite total at or above the Task 0 baseline. `app/ingress/router.py` needs no edit because `route_and_call`'s signature is unchanged — if an ingress test fails, that assumption broke and must be investigated before proceeding.

- [ ] **Step 5: Commit**

```bash
git add app/router/model_router.py tests/unit/test_model_router_provider.py
git commit -m "refactor: replace Portkey and RouteLLM stub with LiteLLM dispatch"
```

---

### Task 5: Route SSE streaming through LiteLLM

Closes **C2**, **D4**. This is the most user-visible defect in the repo: today, an operator with a valid Gemini key calling `/v1/chat/stream` gets canned mock text, because the only real streaming path requires a Portkey key. The hardcoded `"gpt-3.5-turbo"` compounds it.

**Files:**
- Modify: `app/ingress/streaming_router.py:250-306` (the `_stream_tokens_from_llm` function) and its `httpx` import at line 27
- Test: `tests/unit/test_streaming_endpoint_properties.py` (extend — do not rewrite; SSE-1..4 must keep passing)

**Interfaces:**
- Consumes: `providers.astream`, `providers.is_live` (Task 3); `complexity.classify` (Task 2).
- Produces: `_stream_tokens_from_llm(prompt, profile) -> AsyncGenerator[str, None]` — same name and signature, so the surrounding SSE machinery (per-chunk validation, `flush_remaining()`, `[DONE]`/`[REDACTED DUE TO POLICY]`/`[STREAM_ERROR]` framing) is untouched.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_streaming_endpoint_properties.py`:

```python
# ---------------------------------------------------------------------------
# C2/D4: streaming must work on a plain LLM key, with no Portkey involved.


class TestStreamingIsVendorNeutral:
    def test_no_portkey_reference_in_streaming_source(self) -> None:
        import inspect
        from app.ingress import streaming_router

        assert "portkey" not in inspect.getsource(streaming_router).lower()

    def test_no_hardcoded_model_name(self) -> None:
        """D4: the model comes from settings, never a literal."""
        import inspect
        from app.ingress import streaming_router

        src = inspect.getsource(streaming_router)
        assert "gpt-3.5-turbo" not in src

    async def test_streams_from_providers_when_key_present(self, monkeypatch) -> None:
        """C2: a real LLM key now produces a real stream — the bug this fixes."""
        from app.ingress import streaming_router
        from app.models import UseCaseProfile

        async def fake_astream(prompt, tier, system=None):
            for tok in ("alpha ", "beta ", "gamma"):
                yield tok

        monkeypatch.setattr(streaming_router.providers, "is_live", lambda: True)
        monkeypatch.setattr(streaming_router.providers, "astream", fake_astream)

        profile = UseCaseProfile(
            name="p",
            latency_budget_ms=5000,
            token_compression_threshold=500,
            inspection_timeout_ms=1000,
        )
        chunks = [
            c async for c in streaming_router._stream_tokens_from_llm("hi", profile)
        ]
        assert "".join(chunks) == "alpha beta gamma"

    async def test_falls_back_to_simulated_stream_without_key(
        self, monkeypatch
    ) -> None:
        from app.ingress import streaming_router
        from app.models import UseCaseProfile

        monkeypatch.setattr(streaming_router.providers, "is_live", lambda: False)
        profile = UseCaseProfile(
            name="p",
            latency_budget_ms=5000,
            token_compression_threshold=500,
            inspection_timeout_ms=1000,
        )
        chunks = [
            c
            async for c in streaming_router._stream_tokens_from_llm("refund", profile)
        ]
        assert len(chunks) > 1
        assert "30 calendar days" in "".join(chunks)

    async def test_dispatch_failure_raises_runtime_error(self, monkeypatch) -> None:
        """The SSE layer relies on an exception here to emit [STREAM_ERROR]."""
        from app.ingress import streaming_router
        from app.models import UseCaseProfile

        async def broken(prompt, tier, system=None):
            raise RuntimeError("upstream gone")
            yield  # pragma: no cover

        monkeypatch.setattr(streaming_router.providers, "is_live", lambda: True)
        monkeypatch.setattr(streaming_router.providers, "astream", broken)

        profile = UseCaseProfile(
            name="p",
            latency_budget_ms=5000,
            token_compression_threshold=500,
            inspection_timeout_ms=1000,
        )
        with pytest.raises(RuntimeError):
            [c async for c in streaming_router._stream_tokens_from_llm("hi", profile)]
```

If `pytest` is not already imported at the top of that file, add `import pytest`.

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_streaming_endpoint_properties.py -v -k "VendorNeutral or portkey or providers"
```

Expected: FAIL — the source still contains "portkey" and `gpt-3.5-turbo`, and `streaming_router.providers` does not exist.

- [ ] **Step 3: Replace the streaming function**

In `app/ingress/streaming_router.py`, add to the imports near line 27:

```python
from app.router import providers
from app.router.complexity import classify
```

Then replace the whole `_stream_tokens_from_llm` function (lines 250–306, from its `async def` through the `raise RuntimeError(f"Portkey streaming failed: {exc}") from exc` line) with:

```python
async def _stream_tokens_from_llm(
    prompt: str,
    profile: UseCaseProfile,
) -> AsyncGenerator[str, None]:
    """Stream content tokens for *prompt* via the LiteLLM egress layer.

    Tier selection uses the same local complexity scorer as the non-streaming
    path, so a prompt routes identically whether or not the caller streams.

    When no credentials are configured, yields a word-by-word simulated
    stream so the rest of the SSE pipeline (per-chunk validation, sliding
    window, framing) is still exercised in dev and CI.

    Dispatch failures propagate as ``RuntimeError`` — the caller converts
    that into a ``[STREAM_ERROR]`` frame.  We must not splice mock tokens
    into a partially delivered response.
    """
    if not providers.is_live():
        for word in providers.generate_contextual_response(prompt).split():
            yield word + " "
            await asyncio.sleep(0)
        return

    _classification, tier, _score = classify(
        prompt, threshold=profile.complexity_threshold
    )

    try:
        async for token in providers.astream(prompt, tier):
            yield token
    except Exception as exc:
        raise RuntimeError(f"LLM streaming failed: {exc}") from exc
```

Then check whether `httpx` is still used elsewhere in the file:

```bash
grep -n "httpx" app/ingress/streaming_router.py
```

If line 27's `import httpx` is now the only occurrence, delete it. (`httpx` stays in the required floor regardless — `worldsense_oversight.py` and `redteam/runner.py` still use it.)

- [ ] **Step 4: Run the streaming suite**

```bash
.venv/bin/python -m pytest tests/unit/test_streaming_endpoint_properties.py -v
```

Expected: PASS, including the pre-existing SSE-1 through SSE-4 properties. Those cover the framing contract; if one breaks, the replacement changed generator semantics — most likely by swallowing an exception that SSE-4 expects to surface.

- [ ] **Step 5: Commit**

```bash
git add app/ingress/streaming_router.py tests/unit/test_streaming_endpoint_properties.py
git commit -m "fix: stream via LiteLLM so SSE works without a Portkey key"
```

---

### Task 6: Replace the Guardrails Hub with local validators

Closes **C4**. Today `load_validators()` shells out to `python -m guardrails hub install <id>` at startup — a subprocess that pip-installs from a remote registry during boot. That is both a commercial-registry dependency and a startup network call, and it violates the Global Constraint against install-time I/O.

Local validators become the default chain. Guardrails stays supported as an opt-in extra, but is never auto-installed.

**Files:**
- Create: `app/judges/local_validators.py`
- Modify: `app/judges/output_validator.py` (delete `_hub_install`, rework `load_validators` and `_run_validation_sync`)
- Modify: `app/config.py` (section 4)
- Test: `tests/unit/test_local_validators.py`

**Interfaces:**
- Consumes: `GuardrailsVerdict` from `app/models.py` — fields `passed: bool`, `action: Literal["exception","filter","fix"] | None`, `triggered_validator: str | None`, `fixed_output: str | None`.
- Produces: `run_local_validators(text: str) -> GuardrailsVerdict`. `validate_output(text)` keeps its exact signature so neither caller changes. `_LOADED_VALIDATORS` keeps its name (imported by `app/config_health/router.py:47`).

- [ ] **Step 1: Add the local-validator settings**

Replace section 4 of `app/config.py`:

```python
    # -------------------------------------------------------------------------
    # 4. Output Validation
    # -------------------------------------------------------------------------
    # Local validators run by default: no registry, no network, no account.
    # Comma-separated terms whose appearance in an LLM response is blocked.
    LOCAL_BLOCKED_TERMS: str = ""
    # Local toxicity screen over the response text.
    LOCAL_TOXICITY_ENABLED: bool = True

    # Optional Guardrails AI chain, layered on top of the local validators.
    # Validators must be installed by the operator beforehand:
    #     guardrails hub install hub://guardrails/toxic_language
    # The gateway never installs them itself.
    GUARDRAILS_ENABLED: bool = False
    GUARDRAILS_VALIDATORS: str = ""
```

`GUARDRAILS_HUB_TOKEN` is deleted: nothing may reach the hub at runtime.

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_local_validators.py`:

```python
"""Offline output validators — the default, hub-free validation chain."""
from __future__ import annotations

import inspect

import pytest

from app.judges import local_validators, output_validator
from app.judges.local_validators import run_local_validators


class TestNoNetworkAtStartup:
    """C4: the boot-time subprocess install is gone for good."""

    def test_hub_install_helper_removed(self) -> None:
        assert not hasattr(output_validator, "_hub_install")

    def test_no_subprocess_in_output_validator(self) -> None:
        assert "subprocess" not in inspect.getsource(output_validator)

    def test_no_hub_token_setting(self) -> None:
        from app.config import settings

        assert not hasattr(settings, "GUARDRAILS_HUB_TOKEN")


class TestBlockedTerms:
    def test_blocked_term_fails_validation(self, monkeypatch) -> None:
        monkeypatch.setattr(
            local_validators.settings, "LOCAL_BLOCKED_TERMS", "acme,globex"
        )
        verdict = run_local_validators("You should switch to Acme instead.")
        assert verdict.passed is False
        assert verdict.action == "filter"
        assert verdict.triggered_validator == "local-blocked-terms"

    def test_match_is_case_insensitive(self, monkeypatch) -> None:
        monkeypatch.setattr(local_validators.settings, "LOCAL_BLOCKED_TERMS", "acme")
        assert run_local_validators("ACME is better").passed is False

    def test_whole_word_matching_only(self, monkeypatch) -> None:
        """'acme' must not fire on 'acmestic' — substring hits are false positives."""
        monkeypatch.setattr(local_validators.settings, "LOCAL_BLOCKED_TERMS", "acme")
        assert run_local_validators("an acmestic reading").passed is True

    def test_clean_text_passes(self, monkeypatch) -> None:
        monkeypatch.setattr(local_validators.settings, "LOCAL_BLOCKED_TERMS", "acme")
        verdict = run_local_validators("Our return window is 30 days.")
        assert verdict.passed is True
        assert verdict.action is None

    def test_empty_term_list_disables_the_check(self, monkeypatch) -> None:
        monkeypatch.setattr(local_validators.settings, "LOCAL_BLOCKED_TERMS", "")
        assert run_local_validators("acme globex whatever").passed is True

    def test_blank_entries_are_ignored(self, monkeypatch) -> None:
        """A trailing comma must not create an empty term that matches everything."""
        monkeypatch.setattr(local_validators.settings, "LOCAL_BLOCKED_TERMS", "acme,,")
        assert run_local_validators("perfectly fine text").passed is True


class TestToxicity:
    def test_toxic_output_is_blocked(self, monkeypatch) -> None:
        monkeypatch.setattr(local_validators.settings, "LOCAL_BLOCKED_TERMS", "")
        monkeypatch.setattr(
            local_validators.settings, "LOCAL_TOXICITY_ENABLED", True
        )
        verdict = run_local_validators("you are an idiot and I hate you")
        assert verdict.passed is False
        assert verdict.triggered_validator == "local-toxicity"

    def test_toxicity_can_be_disabled(self, monkeypatch) -> None:
        monkeypatch.setattr(local_validators.settings, "LOCAL_BLOCKED_TERMS", "")
        monkeypatch.setattr(
            local_validators.settings, "LOCAL_TOXICITY_ENABLED", False
        )
        assert run_local_validators("you are an idiot").passed is True

    def test_benign_text_is_not_flagged(self, monkeypatch) -> None:
        monkeypatch.setattr(local_validators.settings, "LOCAL_BLOCKED_TERMS", "")
        monkeypatch.setattr(
            local_validators.settings, "LOCAL_TOXICITY_ENABLED", True
        )
        assert run_local_validators("I would be happy to help with that.").passed is True


class TestNeverRaises:
    @pytest.mark.parametrize("text", ["", "   ", "\n", "x" * 50_000, "🙂🙂🙂"])
    def test_pathological_input_returns_a_verdict(self, text: str) -> None:
        """P-LV-1: validation is on the response path — it must never throw."""
        verdict = run_local_validators(text)
        assert isinstance(verdict.passed, bool)


class TestValidateOutputIntegration:
    async def test_local_chain_runs_without_guardrails(self, monkeypatch) -> None:
        monkeypatch.setattr(output_validator, "_GUARDRAILS_AVAILABLE", False)
        monkeypatch.setattr(
            local_validators.settings, "LOCAL_BLOCKED_TERMS", "acme"
        )
        verdict = await output_validator.validate_output("try acme today")
        assert verdict.passed is False

    async def test_signature_unchanged(self) -> None:
        """Both callers pass a single positional string — do not break that."""
        sig = inspect.signature(output_validator.validate_output)
        assert list(sig.parameters) == ["text"]
```

- [ ] **Step 3: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_local_validators.py -v
```

Expected: FAIL — `app.judges.local_validators` does not exist, and `_hub_install` still does.

- [ ] **Step 4: Implement the local validators**

Create `app/judges/local_validators.py`:

```python
"""Offline output validators — the default output-validation chain.

Replaces the Guardrails Hub on the required path.  The hub is a remote
registry and the previous implementation pip-installed from it inside a
subprocess *at startup*; these validators are pure local computation with no
network, no registry and no account.

Guardrails AI remains supported as an opt-in layer (see
:mod:`app.judges.output_validator`) for operators who already run it, but it
is never installed automatically.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from app.config import settings
from app.models import GuardrailsVerdict

logger = logging.getLogger(__name__)

# Deliberately narrow: this screens *model output*, where a false positive
# blocks a legitimate answer.  Slurs and unambiguous abuse only — the
# heavier semantic screen is llm-guard's Toxicity scanner, installed via
# the `safety` extra and applied to input by the P1 judge.
_TOXIC_PATTERNS: tuple[str, ...] = (
    r"\bidiot\b",
    r"\bmoron\b",
    r"\bstupid\b",
    r"\bshut up\b",
    r"\bi hate you\b",
    r"\bkill yourself\b",
    r"\bworthless\b",
)


@lru_cache(maxsize=1)
def _toxicity_regex() -> re.Pattern[str]:
    return re.compile("|".join(_TOXIC_PATTERNS), re.IGNORECASE)


def _blocked_terms() -> list[str]:
    """Parse LOCAL_BLOCKED_TERMS, discarding blank entries.

    A blank entry would compile to a pattern matching everything, so
    filtering them is a correctness requirement, not tidiness.
    """
    raw = settings.LOCAL_BLOCKED_TERMS or ""
    return [term.strip() for term in raw.split(",") if term.strip()]


def _blocked_term_hit(text: str) -> str | None:
    """Return the first blocked term present in *text* as a whole word."""
    for term in _blocked_terms():
        # Whole-word match: 'acme' must not fire inside 'acmestic'.
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
            return term
    return None


def run_local_validators(text: str) -> GuardrailsVerdict:
    """Validate *text* with the local chain.

    Returns a passing verdict when everything is clean.  Never raises: this
    sits on the response path, so an internal defect here must not turn a
    good answer into a 500.
    """
    try:
        hit = _blocked_term_hit(text)
        if hit:
            logger.info("LOCAL_VALIDATOR_BLOCKED validator=blocked-terms term=%s", hit)
            return GuardrailsVerdict(
                passed=False,
                action="filter",
                triggered_validator="local-blocked-terms",
            )

        if settings.LOCAL_TOXICITY_ENABLED and _toxicity_regex().search(text):
            logger.info("LOCAL_VALIDATOR_BLOCKED validator=toxicity")
            return GuardrailsVerdict(
                passed=False,
                action="filter",
                triggered_validator="local-toxicity",
            )

        return GuardrailsVerdict(passed=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LOCAL_VALIDATOR_ERROR %s — passing through", exc)
        return GuardrailsVerdict(passed=True)
```

- [ ] **Step 5: Rework `output_validator.py`**

Three edits:

**5a.** Delete the entire `_hub_install` function (lines 47–60) and the `import subprocess` / `import sys` lines inside it.

**5b.** Replace the body of `load_validators()` with a version that never installs:

```python
def load_validators() -> None:
    """Load the optional Guardrails AI validator chain.

    Only runs when GUARDRAILS_ENABLED is true and the package is importable.
    Validators must already be installed by the operator — we never install
    anything at startup, which the previous `guardrails hub install`
    subprocess did.  The local chain in app.judges.local_validators always
    runs regardless and needs no setup.
    """
    _LOADED_VALIDATORS.clear()

    if not settings.GUARDRAILS_ENABLED:
        logger.info("GUARDRAILS_DISABLED — local validators active")
        return
    if not _GUARDRAILS_AVAILABLE:
        logger.info(
            "GUARDRAILS_ABSENT — install with `pip install '.[safety]'`; "
            "local validators active"
        )
        return

    for vid in (v.strip() for v in settings.GUARDRAILS_VALIDATORS.split(",")):
        if not vid:
            continue
        try:
            validator = gd.hub.load(vid)  # type: ignore[attr-defined]
            _LOADED_VALIDATORS.append((vid, validator))
            logger.info("GUARDRAILS_LOADED validator=%s", vid)
        except Exception as exc:  # noqa: BLE001
            # Not installed, or incompatible. Skip it — never auto-install.
            logger.warning(
                "GUARDRAILS_SKIPPED validator=%s error=%s — install it with "
                "`guardrails hub install %s`",
                vid,
                exc,
                vid,
            )
```

**5c.** Make `_run_validation_sync` run the local chain first, then the optional Guardrails chain. Insert at the top of the function, replacing its current early return:

```python
def _run_validation_sync(text: str) -> GuardrailsVerdict:
    """Run the local chain, then any loaded Guardrails validators."""
    # Local validators always run — no dependency, no configuration needed.
    local = run_local_validators(text)
    if not local.passed:
        return local

    if not _GUARDRAILS_AVAILABLE or not _LOADED_VALIDATORS:
        return local  # passing verdict from the local chain
    # ... existing per-validator loop unchanged below ...
```

Add the import at the top of the file:

```python
from app.judges.local_validators import run_local_validators
```

- [ ] **Step 6: Run the validator tests, then the full suite**

```bash
.venv/bin/python -m pytest tests/unit/test_local_validators.py tests/unit/test_output_validator_properties.py -v
.venv/bin/python -m pytest tests/unit -q 2>&1 | tail -8
```

Expected: PASS. The pre-existing P-OV-1..P-OV-7 properties cover the fix/filter/exception contract; if one fails, the local chain is returning an `action` value those properties do not expect — check that `action="filter"` (not `"exception"`) is used for a policy hit, since `filter` is what maps to a clean HARD_BLOCK downstream.

- [ ] **Step 7: Commit**

```bash
git add app/judges/local_validators.py app/judges/output_validator.py app/config.py tests/unit/test_local_validators.py
git commit -m "feat: replace Guardrails Hub auto-install with local offline validators"
```

---

### Task 7: Purge the remaining vendor defaults and rename the health field

Closes **C3**, **C5**, **P7**, **P8**. After this task no default configuration value points at a commercial endpoint or names a paid model.

The `PORTKEY_*` settings removal and the `config_health` rename **must land in the same commit**: `app/config_health/router.py:24` reads `settings.PORTKEY_API_KEY`, so deleting the setting without renaming the field leaves a broken tree between tasks.

**Files:**
- Modify: `app/config.py` (sections 2, 3, 5, 6, 8)
- Modify: `app/models.py:325` (`ConfigHealthResponse.portkey` → `.llm_gateway`)
- Modify: `app/config_health/router.py:23-32,80-86`
- Modify: `app/router/semantic_cache.py:82`
- Modify: `app/main.py:196-201` (startup summary references Portkey)
- Modify: `tests/unit/test_config_health.py`

**Interfaces:**
- Consumes: `providers.is_live`, `providers._model_for_tier` (Task 3).
- Produces: `ConfigHealthResponse` with fields `llm_gateway`, `langfuse`, `guardrails`, `worldsense`, `llm_direct`. No `portkey` field anywhere in the codebase.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_config_health.py`, and delete that file's existing Portkey-specific assertions (the block asserting a `portkey` key in the response, and any test setting `PORTKEY_API_KEY`):

```python
class TestNoVendorDefaults:
    """C3/C5/P8: no default may point at a paid endpoint or model."""

    def test_portkey_settings_are_gone(self) -> None:
        from app.config import settings

        for attr in (
            "PORTKEY_API_KEY",
            "PORTKEY_SLM_VIRTUAL_KEY",
            "PORTKEY_FRONTIER_VIRTUAL_KEY",
        ):
            assert not hasattr(settings, attr), f"{attr} should be removed"

    def test_langfuse_host_defaults_blank(self) -> None:
        """C3: no vendor cloud URL as a default."""
        from app.config import Settings

        assert Settings.model_fields["LANGFUSE_HOST"].default == ""

    def test_embedding_model_defaults_local(self) -> None:
        """C5: default embeddings run locally, not against a paid API."""
        from app.config import Settings

        default = Settings.model_fields["EMBEDDING_MODEL"].default
        assert "text-embedding-3" not in default
        assert default == "all-MiniLM-L6-v2"

    def test_mcp_sidecars_default_off(self) -> None:
        """P7: a fresh clone must not spend ~4s probing absent sidecars."""
        from app.config import Settings

        assert Settings.model_fields["WORLDSENSE_ENABLED"].default is False
        assert Settings.model_fields["REDTEAM_ENABLED"].default is False


class TestLLMGatewayField:
    async def test_response_has_llm_gateway_not_portkey(self, client) -> None:
        resp = await client.get("/v1/config/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "llm_gateway" in body
        assert "portkey" not in body

    async def test_llm_gateway_degraded_without_key(self, client, monkeypatch) -> None:
        from app.router import providers

        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "")
        resp = await client.get("/v1/config/health")
        assert resp.json()["llm_gateway"]["status"] == "degraded"

    def test_no_portkey_reference_in_health_source(self) -> None:
        import inspect
        from app.config_health import router as health_router

        assert "portkey" not in inspect.getsource(health_router).lower()
```

Reuse whatever `client` fixture that file already defines. If it constructs its own `AsyncClient`, follow the existing pattern rather than introducing a new one.

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_config_health.py -v
```

Expected: FAIL on every new test — the Portkey settings still exist and the response still carries a `portkey` key.

- [ ] **Step 3: Edit `app/config.py`**

**3a.** Delete section 2 (`PORTKEY_API_KEY`, `PORTKEY_SLM_VIRTUAL_KEY`, `PORTKEY_FRONTIER_VIRTUAL_KEY`) in full. Replace the section header with:

```python
    # -------------------------------------------------------------------------
    # 2. (removed) Portkey Gateway
    # -------------------------------------------------------------------------
    # Portkey was a commercial SaaS proxy. LiteLLM (section 1) replaces it as
    # an in-process library — no account, no gateway service. Any leftover
    # PORTKEY_* entries in an existing .env are ignored (extra="ignore").
```

**3b.** In section 3, blank the vendor host:

```python
    # Leave blank to disable Langfuse. For self-hosted Langfuse (its core is
    # MIT-licensed), set this to your own instance, e.g. http://localhost:3000.
    # No vendor cloud endpoint is assumed.
    LANGFUSE_HOST: str = ""
```

**3c.** Flip both sidecars off (they are opt-in; each costs a 2s startup probe when absent):

```python
    WORLDSENSE_ENABLED: bool = False
```
```python
    REDTEAM_ENABLED: bool = False
```

**3d.** In section 8, make embeddings local:

```python
    # Local sentence-transformers model (Apache-2.0) — runs on CPU, no API
    # key, no per-token cost. Install via `pip install '.[grounded]'`.
    # The previous default (text-embedding-3-small) was a paid OpenAI endpoint.
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
```

- [ ] **Step 4: Rename the model field**

In `app/models.py`, change `ConfigHealthResponse.portkey` (line 325) to:

```python
    llm_gateway: IntegrationStatus
```

- [ ] **Step 5: Rewrite the health endpoint's first block**

In `app/config_health/router.py`, replace the `# --- Portkey ---` block (lines 23–32) with:

```python
    # --- LLM gateway (LiteLLM, in-process) ---
    from app.router import providers  # local import avoids a cycle at startup

    llm_live = providers.is_live()
    llm_gateway = IntegrationStatus(
        status="active" if llm_live else "degraded",
        detail=(
            f"LiteLLM dispatch active; provider={settings.LLM_PROVIDER} "
            f"slm={providers._model_for_tier('SLM')} "
            f"frontier={providers._model_for_tier('FRONTIER')}"
            if llm_live
            else "LLM_API_KEY not set or litellm not installed — mock responses active"
        ),
    )
```

And in the return statement (lines 80–86) change `portkey=portkey,` to `llm_gateway=llm_gateway,`.

- [ ] **Step 6: Fix the startup summary and the cache default**

In `app/main.py`, replace the `"  Portkey        : %s\n"` line and its matching argument (lines ~196 and ~201) with:

```python
        "  LLM dispatch   : %s\n"
```
```python
        f"ACTIVE (provider={settings.LLM_PROVIDER})" if providers.is_live() else "DEGRADED (mock)",
```

Add `from app.router import providers` to `main.py`'s imports if it is not already present, and remove the now-unused `_is_real_key` import if nothing else in the file uses it (`grep -n "_is_real_key" app/main.py` to check).

In `app/router/semantic_cache.py:82`, change the parameter default:

```python
        embedding_model: str = "all-MiniLM-L6-v2",
```

- [ ] **Step 7: Verify no Portkey reference survives anywhere in the app**

```bash
grep -rn -i "portkey" app/ tests/ | grep -v "removed) Portkey" | grep -v "Portkey was a commercial"
```

Expected: no output beyond the two explanatory comments. Any other hit is a missed reference — fix it before committing.

- [ ] **Step 8: Run the suite**

```bash
.venv/bin/python -m pytest tests/unit/test_config_health.py -v
.venv/bin/python -m pytest tests/unit -q 2>&1 | tail -8
```

Expected: PASS, suite at or above baseline.

- [ ] **Step 9: Confirm the boot delay is gone (P7)**

```bash
time (.venv/bin/python -c "
import asyncio
from app.main import app
async def boot():
    async with app.router.lifespan_context(app):
        print('LIFESPAN OK')
asyncio.run(boot())
")
```

Expected: `LIFESPAN OK` with no ~4s stall. The two MCP probes are now skipped by default.

- [ ] **Step 10: Commit**

```bash
git add app/config.py app/models.py app/config_health/router.py app/main.py app/router/semantic_cache.py tests/unit/test_config_health.py
git commit -m "refactor: remove Portkey settings, local embeddings, opt-in sidecars"
```

---

### Task 8: Make the Stage 3 groundedness stub honest

Closes **P6**. `current_state.md` describes Stage 3 as a working FAISS embedding pipeline. It is not: `vector_store.py` is 22 lines returning `Document("mock document content")` and `auditor.py:59` uses `embedding = [0.1] * 128`. A reported `technique="embedding_similarity"` from a placeholder is worse than no score, because triage at `gateway.py` makes block/pass decisions on it.

This task does **not** build a real vector store — that is a feature, out of scope here. It makes the stub declare itself, so no operator mistakes a placeholder for a measurement.

**Files:**
- Modify: `app/groundedness/auditor.py:50-66`
- Modify: `app/groundedness/vector_store.py` (docstring)
- Test: `tests/unit/test_groundedness_auditor_properties.py` (extend)

**Interfaces:**
- Consumes: `AuditResult` from `app/models.py` — fields `groundedness_score: float`, `technique: str`, `is_unverified: bool`, `nli_label`.
- Produces: `technique` values `"stub_placeholder"`, `"embedding_similarity"`, `"nli_stub_placeholder"`, `"nli_embedding_similarity"`. The NLI aggregation path is unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_groundedness_auditor_properties.py`:

```python
class TestStubHonesty:
    """P6: a placeholder score must not masquerade as a real measurement."""

    async def test_stub_vector_store_is_labelled_stub(self) -> None:
        from app.groundedness.auditor import audit_groundedness
        from app.groundedness.vector_store import FAISSVectorStore

        result = await audit_groundedness("some response", FAISSVectorStore(), None)
        assert "stub" in result.technique
        assert result.is_unverified is True

    async def test_stub_result_never_claims_plain_embedding_similarity(self) -> None:
        from app.groundedness.auditor import audit_groundedness
        from app.groundedness.vector_store import FAISSVectorStore

        result = await audit_groundedness("x", FAISSVectorStore(), None)
        assert result.technique != "embedding_similarity"

    def test_vector_store_documents_its_stub_status(self) -> None:
        import app.groundedness.vector_store as vs

        assert "stub" in (vs.__doc__ or "").lower()
```

Match the real call signature of `audit_groundedness` — check it with `grep -n "async def audit_groundedness" -A6 app/groundedness/auditor.py` and adjust the argument list if it differs.

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_groundedness_auditor_properties.py -v -k StubHonesty
```

Expected: FAIL — `technique` is currently the unqualified `"embedding_similarity"`.

- [ ] **Step 3: Label the stub path**

In `app/groundedness/auditor.py`, at the placeholder-embedding site (~lines 57–66), replace the embedding block and `technique` assignment with:

```python
        # STUB: this is a fixed placeholder vector, not an embedding of
        # ``response``. A real implementation embeds the response with
        # settings.EMBEDDING_MODEL and queries a populated vector store.
        # Until then the score below is not a measurement, and ``technique``
        # says so — triage reads this field to decide whether to trust it.
        embedding = [0.1] * 128
        docs = await vector_store.similarity_search(embedding, top_k=5)
        is_stub = True
```

and:

```python
    technique = "stub_placeholder" if is_stub else "embedding_similarity"
```

Where the NLI branch currently sets `technique = "nli_embedding_similarity"`, make it:

```python
            technique = "nli_stub_placeholder" if is_stub else "nli_embedding_similarity"
```

Set `is_unverified=True` on the stub path if it is not already.

Add to the top of `app/groundedness/vector_store.py`:

```python
"""Vector Store abstraction — FAISS adapter is currently a STUB.

``FAISSVectorStore.similarity_search`` returns a fixed mock document; it does
not build or query a FAISS index, and no faiss dependency is installed.
Stage 3 groundedness therefore reports ``technique="stub_placeholder"`` so
callers can tell a placeholder from a measurement.

Implementing this for real means: embedding documents with
settings.EMBEDDING_MODEL (local sentence-transformers, see the `grounded`
extra), persisting an index, and returning true top-K neighbours.
"""
```

- [ ] **Step 4: Run the groundedness and triage suites**

```bash
.venv/bin/python -m pytest tests/unit/test_groundedness_auditor_properties.py tests/unit/test_triage_gateway_properties.py -v
```

Expected: PASS. If a triage property asserts on `technique == "embedding_similarity"`, update that assertion — triage decisions key off `groundedness_score` and `nli_label`, not the technique string, so the change is safe; verify that is true in `app/triage/gateway.py` before editing the test.

- [ ] **Step 5: Commit**

```bash
git add app/groundedness/auditor.py app/groundedness/vector_store.py tests/unit/test_groundedness_auditor_properties.py
git commit -m "fix: label Stage 3 groundedness stub so scores are not mistaken for real"
```

---

### Task 9: Rewrite the documentation to match reality, and verify end to end

Closes **P9** and the documentation half of every finding. `current_state.md` currently asserts several things that are false (FAISS required and powering Stage 3, Portkey as the recommended production path, 245 passing tests). Leaving it stale would re-introduce the confusion this plan exists to remove.

**Files:**
- Modify: `.env.example` (full rewrite of the affected sections)
- Modify: `current_state.md`
- Modify: `README.md` (currently 17 bytes — give it a real quickstart)
- Delete: `docs/superpowers/plans/baseline-2026-08-29.txt` (Task 0 scratch)

**Interfaces:**
- Consumes: the final state of `app/config.py` after Task 7.
- Produces: documentation with no vendor-cloud defaults and no false capability claims.

- [ ] **Step 1: Regenerate `.env.example` from the real settings**

Every variable in `.env.example` must exist in `Settings`, and every `Settings` field should appear. Check both directions:

```bash
.venv/bin/python - <<'EOF'
import re
from pathlib import Path
from app.config import Settings

declared = set(Settings.model_fields)
in_example = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", Path(".env.example").read_text(), re.M))
print("in .env.example but NOT a setting (stale):", sorted(in_example - declared))
print("a setting but NOT documented        :", sorted(declared - in_example))
EOF
```

Fix both lists. Expected stale entries: `PORTKEY_API_KEY`, `PORTKEY_SLM_VIRTUAL_KEY`, `PORTKEY_FRONTIER_VIRTUAL_KEY`, `GUARDRAILS_HUB_TOKEN`. Expected undocumented new ones: `SLM_MODEL`, `FRONTIER_MODEL`, `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`, `LLM_API_BASE`, `LOCAL_BLOCKED_TERMS`, `LOCAL_TOXICITY_ENABLED`, `GUARDRAILS_ENABLED`.

The LLM section should read:

```bash
# -----------------------------------------------------------------------------
# 1. LLM Provider — dispatched via LiteLLM (BSD-3, in-process, no gateway)
# -----------------------------------------------------------------------------
# The ONLY commercial service this gateway needs is the LLM you choose. Leave
# LLM_API_KEY blank and the gateway serves contextual mock responses — the
# full safety pipeline still runs, which is what CI and local dev use.
#
# Free options: Google Gemini has a no-cost tier (~1,500 req/day). For a
# fully local setup, run Ollama or vLLM and set:
#     LLM_PROVIDER=generic
#     LLM_API_BASE=http://localhost:11434/v1
#     LLM_API_KEY=ollama
#
LLM_PROVIDER=openai
LLM_API_KEY=

# Two-tier routing. A local complexity scorer picks the tier per prompt;
# each tier is the other's fallback on dispatch failure.
SLM_MODEL=gpt-4o-mini
FRONTIER_MODEL=gpt-4o

# Optional: explicit base URL for self-hosted / OpenAI-compatible endpoints.
LLM_API_BASE=
LLM_TIMEOUT_S=30.0
LLM_MAX_RETRIES=2
```

Add a header comment near the top recording the removal, so an operator with an old `.env` understands why their keys stopped mattering:

```bash
# NOTE: PORTKEY_* and GUARDRAILS_HUB_TOKEN were removed. Portkey was a
# commercial proxy; LiteLLM replaces it in-process. Leftover entries in your
# .env are ignored, not an error — delete them at your convenience.
```

- [ ] **Step 2: Correct `current_state.md`**

Make these specific edits — each one is currently a false statement:

1. **Dependency table** — remove the `FAISS (faiss-cpu)` and `portkey-ai` rows entirely (never imported). Change the "Required?" column to name the extra that carries each package (`llm`, `safety`, `cache`, `grounded`, `observe`), and state the six-package required floor explicitly.
2. **Stage 2 section** — delete the "Tier A — Portkey Gateway (recommended for production)" subsection and the Portkey header table. Replace the three-tier description with: LiteLLM dispatch → opposite-tier fallback → contextual mock.
3. **Stage 3 section** — replace "FAISS cosine similarity (always)" with an explicit statement that the vector store is a stub returning a fixed document, that the score is a placeholder reported as `technique="stub_placeholder"`, and that implementing it is outstanding work.
4. **Model Router section** — record that `FRONTIER_CONFIG`/`SLM_CONFIG` were dead code, that `PORTKEY_FRONTIER_VIRTUAL_KEY` was never sent, and that the RouteLLM branch returned a hardcoded score of 0.5 and the literal `"This is a mock LLM response."`. Describe the local scorer that replaced it.
5. **Settings table** — drop the Portkey row; add SLM/FRONTIER model settings and the local-validator settings; note `LANGFUSE_HOST` and both `*_ENABLED` sidecar defaults changed.
6. **Test suite section** — replace the unverified "245 tests pass, 4 skipped" with the count from the final Step 5 run below, and add the new test files (`test_packaging.py`, `test_complexity.py`, `test_providers.py`, `test_local_validators.py`).
7. **Lifespan table** — Steps 10 and 11 are now skipped unless the sidecars are explicitly enabled.
8. Add a short "Vendor independence" section stating the invariant: the only commercial dependency is the operator's chosen LLM, and everything else is OSI-licensed and self-hostable.

- [ ] **Step 3: Write a real README**

```bash
cat > README.md <<'EOF'
# ControlPlane.ai

An enterprise AI proxy gateway. Applications call this service instead of
calling an LLM directly; every request passes a five-stage safety pipeline
that enforces input safety, cost-aware model routing, groundedness checks and
output triage before a response is returned.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env          # optional — runs on mock responses without it
.venv/bin/python -m uvicorn app.main:app --reload
curl localhost:8000/v1/config/health
```

That runs with **no API keys and no external services**: regex PII masking,
local output validators, contextual mock LLM responses, file-based telemetry.

## Installing more capability

| Command | Adds |
|---|---|
| `pip install '.[llm]'` | Real LLM dispatch via LiteLLM (100+ providers) |
| `pip install '.[safety]'` | NLP PII masking, spaCy parsing, custom-entity NER |
| `pip install '.[cache]'` | Semantic cache (Stage 0) |
| `pip install '.[grounded]'` | Local embeddings + NLI groundedness |
| `pip install '.[observe]'` | Langfuse tracing (self-hosted or cloud) |
| `pip install '.[all]'` | Everything |

## Vendor independence

The only commercial service required is the LLM you choose — and even that is
optional for development. Every other dependency is OSI-licensed and
self-hostable. For a fully local stack, point `LLM_API_BASE` at Ollama or
vLLM.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat` | Primary request path |
| `POST /v1/chat/stream` | SSE streaming |
| `GET /v1/config/health` | Which integrations are live vs degraded |
| `GET /v1/metrics` | Aggregate telemetry |
| `POST /v1/redteam/run` | Adversarial test run |

## Tests

```bash
.venv/bin/python -m pytest tests/unit -q
```

See `current_state.md` for full architecture.
EOF
```

- [ ] **Step 4: Prove the minimal install still works from scratch**

The headline claim of this plan. Verify it in a clean venv with **no extras**:

```bash
python3 -m venv /tmp/cp-verify
/tmp/cp-verify/bin/python -m pip install -q -e .
du -sh /tmp/cp-verify
/tmp/cp-verify/bin/python -m uvicorn app.main:app --port 8098 &
sleep 6
echo "--- health ---"
curl -s localhost:8098/v1/config/health
echo "--- chat ---"
curl -s -X POST localhost:8098/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is your return policy?","profile_name":"customer_chatbot"}'
kill %1; rm -rf /tmp/cp-verify
```

Expected: all five integrations report `degraded` with explanatory detail, and `/v1/chat` returns the contextual mock answer about 30 calendar days. Zero network calls, zero keys. If `/v1/chat`'s body shape differs, read `app/ingress/router.py`'s request model and adjust the payload — do not change the endpoint.

- [ ] **Step 5: Full suite and final counts**

```bash
.venv/bin/python -m pytest tests/unit -q 2>&1 | tail -6
grep -rn -i "portkey\|faiss\|hub install\|cloud.langfuse\|text-embedding-3" \
  app/ pyproject.toml .env.example | grep -v "removed) Portkey" | grep -v "was a commercial"
```

Expected: suite green at or above the Task 0 baseline. The grep should return nothing — every commercial reference gone from code and config. Put the real pass/skip count into `current_state.md` per Step 2 item 6.

- [ ] **Step 6: Optional live check, if a key is available**

Only if the operator has a key to hand (a free Gemini key suffices):

```bash
LLM_PROVIDER=google LLM_API_KEY="$GEMINI_KEY" \
  SLM_MODEL=gemini-2.5-flash FRONTIER_MODEL=gemini-2.5-pro \
  .venv/bin/python -m uvicorn app.main:app --port 8097 &
sleep 6
curl -s localhost:8097/v1/config/health | python3 -m json.tool | head -8
curl -sN -X POST localhost:8097/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Name three caching strategies.","profile_name":"customer_chatbot"}' | head -20
kill %1
```

Expected: `llm_gateway` reports `active` with both tier models named, and the stream emits real tokens ending in `[DONE]`. This is the direct confirmation of **C2** — the same call serves canned text on `main` today.

- [ ] **Step 7: Clean up and commit**

```bash
rm -f docs/superpowers/plans/baseline-2026-08-29.txt
git add README.md current_state.md .env.example
git rm --cached --ignore-unmatch docs/superpowers/plans/baseline-2026-08-29.txt
git commit -m "docs: correct architecture docs, document six-package floor and vendor independence"
```

---

## Self-Review

Checked after writing, against the Findings table.

**Coverage.** Every finding maps to a task: C1 → Tasks 3, 4, 7; C2 → Task 5; C3, C5 → Task 7; C4 → Task 6; D1, D2 → Task 4; D3 → Tasks 2, 4; D4 → Task 5; P1 → Task 0; P2, P3, P4, P5 → Task 1; P6 → Task 8; P7, P8 → Task 7; P9 → Tasks 0, 9. No finding is unaddressed.

**Deliberate non-goals.** Two things this plan does *not* do, so no one expects them: it does not build a real FAISS vector store (Task 8 labels the stub honestly instead — building it is a feature, not a vendor-independence fix), and it does not remove `langfuse`, `routellm`, `llm-guard`, `gliner`, `guardrails-ai` or `gptcache` as *capabilities* — each stays available as an opt-in extra, since the goal is removing commercial coupling from the required path, not shrinking function.

**Ordering.** Task 0 must precede everything (nothing is verifiable without a venv). Task 1 precedes the code tasks so later installs are fast. Task 3 precedes Tasks 4, 5, 7, which all import `providers`. Task 7 bundles the `PORTKEY_*` deletion with the `config_health` rename because splitting them would leave `config_health/router.py:24` reading a deleted setting — the one place in this plan where a smaller commit would break the tree.

**Type consistency.** `providers.acomplete` returns `tuple[str, str]` in Task 3 and is unpacked as `text, model_used` in Task 4. `complexity.classify` returns a 3-tuple in Task 2 and is unpacked as `classification, tier, score` in Task 4 and `_classification, tier, _score` in Task 5. `GuardrailsVerdict` field names in Task 6 (`passed`, `action`, `triggered_validator`, `fixed_output`) match `app/models.py:134-143` as read. `RoutingDecision.routellm_score` keeps its name per the Global Constraints, with the reason recorded in a code comment.

**Risks worth stating.** LiteLLM is a large-ish pure-Python package and pins its own `openai`/`httpx` floors — if its `httpx` requirement conflicts with the pinned floor, resolve by raising ours, never by removing the constraint test. And `test_no_portkey_reference_in_source` is a source-text assertion, so it also fires on the word appearing in a comment; that is intended, and the two explanatory comments live in `config.py`, which that test does not read.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-29-vendor-independence.md`. Two execution options:

**1. Subagent-Driven (recommended)** — a fresh subagent per task, with review between tasks. Fast iteration, and each task's context stays clean.

**2. Inline Execution** — execute the tasks in this session with checkpoints for review.
