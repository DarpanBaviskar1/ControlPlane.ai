---
inclusion: auto
---

# ControlPlane.ai — Performance Budget Rules

## The 300 ms Hard Ceiling

`WORLDSENSE_TIMEOUT_MS` is set to **300 ms** by default (`app/config.py`).
The Heuristic Evaluator (`app/oversight/worldsense_oversight.py`) and every
judge that runs before the LLM call — P1, P3, and the PII Masking Engine —
must individually complete **well under 300 ms** so that the entire
Worldsense + Orchestrator budget is never exhausted by a single component.

---

## Mandatory Profiling Rule

**Before committing any of the following changes, you must profile the
affected code path and confirm it stays under the ceiling documented below.**

| Component | File(s) | Ceiling |
|---|---|---|
| P1 Judge (Toxicity + Injection) | `app/judges/p1_judge.py` | 150 ms per call |
| P3 Judge (spaCy clarity check) | `app/judges/p3_judge.py` | 50 ms per call |
| Heuristic Evaluator (keyword/regex scan) | `app/oversight/worldsense_oversight.py` | 20 ms per call |
| PII Masking Engine (regex fallback) | `app/judges/pii_masking.py` — `_FallbackScanner.scan` | 30 ms per call |
| Full Orchestrator (P1 + P2 + P3 concurrent) | `app/judges/orchestrator.py` | 200 ms wall-clock |

---

## What Triggers a Profiling Requirement

You **must** profile whenever you:

1. **Add or change a regex pattern** anywhere in `app/judges/pii_masking.py`,
   `app/oversight/worldsense_oversight.py`, `app/judges/p1_judge.py`, or
   `app/judges/p3_judge.py`.  Catastrophic backtracking on long inputs is the
   most common source of budget overruns.

2. **Add a new loop** that iterates over prompt tokens, conversation turns,
   or entity lists inside any of the five components in the table above.

3. **Import a new synchronous library** inside one of the above files.
   Even a one-time lazy import can add tens of milliseconds to the first
   request and must be moved to the FastAPI `lifespan` startup block.

4. **Change the spaCy pipeline** in `app/judges/p3_judge.py` (e.g. adding a
   component like `ner` or `textcat` to `en_core_web_sm`).  Each additional
   spaCy pipeline component adds ~5–20 ms per call.

5. **Extend `_RISK_KEYWORDS` or `_CONSEQUENCE_PATTERNS`** in
   `app/oversight/worldsense_oversight.py` beyond 30 total entries.

---

## How to Profile

Use the built-in `timeit` snippet; paste it in a scratch script and run
against the **worst-case input** for that component (longest valid prompt,
maximum conversation history depth, etc.):

```python
import timeit, statistics

# Example: profile _FallbackScanner on a 32 768-char prompt
from app.judges.pii_masking import _FallbackScanner
scanner = _FallbackScanner()
worst_case = "a" * 32_768  # adjust to the component under test

times_ms = [
    timeit.timeit(lambda: scanner.scan(worst_case), number=1) * 1000
    for _ in range(100)
]
print(f"p50={statistics.median(times_ms):.1f}ms  p99={sorted(times_ms)[99]:.1f}ms")
```

The **p99 value must be below the ceiling** in the table above.  If it is
not, the change must be optimised (compiled regex, early-exit, caching) or
the timeout budget must be raised via a deliberate config change with a
corresponding PR note explaining the tradeoff.

---

## Async Safety Requirement

All synchronous CPU-bound operations in the five components above **must**
be wrapped in `asyncio.to_thread(...)`.  Never call `.scan()`, `_nlp()`, or
any regex-heavy function directly on the event loop coroutine.  Blocking the
event loop for more than **5 ms** is considered a latency regression.

Correct pattern:
```python
result = await asyncio.to_thread(scanner.scan, prompt)
```

Incorrect (blocks event loop):
```python
result = scanner.scan(prompt)   # ← never do this inside an async function
```

---

## Enforcement

These rules are enforced by the property-based tests in
`tests/unit/test_output_validator_properties.py` (P-OV-7) and by the
`inspection_timeout_ms` budget in every `UseCaseProfile`.  A CI failure
caused by a timeout in those tests is a direct signal that this budget has
been violated.
