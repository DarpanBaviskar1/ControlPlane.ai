"""Property-based tests for NLIScorer.

Tasks: 4.2
Properties:
  NLI-3 — Aggregation priority: any label list containing at least one CONTRADICTION
           must return CONTRADICTION regardless of other labels.

Requirements: 2.9
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.groundedness.nli_scorer import NLIScorer, NLILabel


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

_ALL_LABELS = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]

# A non-empty list of labels that contains at least one CONTRADICTION
_list_with_contradiction = st.lists(
    st.sampled_from(_ALL_LABELS),
    min_size=1,
    max_size=20,
).filter(lambda labels: "CONTRADICTION" in labels)

# A non-empty list of labels that contains NO CONTRADICTION
_list_without_contradiction = st.lists(
    st.sampled_from(["ENTAILMENT", "NEUTRAL"]),
    min_size=1,
    max_size=20,
)

# A non-empty list containing at least one ENTAILMENT but no CONTRADICTION
_list_with_entailment_no_contradiction = st.lists(
    st.sampled_from(["ENTAILMENT", "NEUTRAL"]),
    min_size=1,
    max_size=20,
).filter(lambda labels: "ENTAILMENT" in labels)


# ---------------------------------------------------------------------------
# NLI-3: Aggregation priority
# ---------------------------------------------------------------------------

class TestNLI3AggregationPriority:
    """Property NLI-3: CONTRADICTION > ENTAILMENT > NEUTRAL priority rule."""

    @given(_list_with_contradiction)
    @settings(max_examples=200)
    def test_contradiction_wins_over_any_other_labels(self, labels: list[NLILabel]) -> None:
        """
        Property NLI-3: For any label list containing at least one CONTRADICTION,
        aggregate() must return CONTRADICTION regardless of other labels present.
        """
        result = NLIScorer.aggregate(labels)
        assert result == "CONTRADICTION", (
            f"Expected CONTRADICTION for labels={labels!r}, got {result!r}"
        )

    @given(_list_with_entailment_no_contradiction)
    @settings(max_examples=200)
    def test_entailment_wins_over_neutral(self, labels: list[NLILabel]) -> None:
        """When there is no CONTRADICTION but at least one ENTAILMENT, result is ENTAILMENT."""
        result = NLIScorer.aggregate(labels)
        assert result == "ENTAILMENT", (
            f"Expected ENTAILMENT for labels={labels!r}, got {result!r}"
        )

    @given(st.lists(st.just("NEUTRAL"), min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_all_neutral_returns_neutral(self, labels: list[NLILabel]) -> None:
        """All-NEUTRAL list must return NEUTRAL."""
        result = NLIScorer.aggregate(labels)
        assert result == "NEUTRAL"

    def test_empty_list_returns_neutral(self) -> None:
        """Empty label list must return NEUTRAL (safe default)."""
        assert NLIScorer.aggregate([]) == "NEUTRAL"

    def test_single_contradiction(self) -> None:
        assert NLIScorer.aggregate(["CONTRADICTION"]) == "CONTRADICTION"

    def test_single_entailment(self) -> None:
        assert NLIScorer.aggregate(["ENTAILMENT"]) == "ENTAILMENT"

    def test_single_neutral(self) -> None:
        assert NLIScorer.aggregate(["NEUTRAL"]) == "NEUTRAL"

    def test_contradiction_beats_entailment_and_neutral(self) -> None:
        assert NLIScorer.aggregate(["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]) == "CONTRADICTION"

    def test_entailment_beats_neutral(self) -> None:
        assert NLIScorer.aggregate(["NEUTRAL", "ENTAILMENT", "NEUTRAL"]) == "ENTAILMENT"

    @given(
        prefix=st.lists(st.sampled_from(["ENTAILMENT", "NEUTRAL"]), max_size=10),
        suffix=st.lists(st.sampled_from(["ENTAILMENT", "NEUTRAL"]), max_size=10),
    )
    @settings(max_examples=100)
    def test_contradiction_injected_anywhere_still_wins(
        self, prefix: list[NLILabel], suffix: list[NLILabel]
    ) -> None:
        """CONTRADICTION at any position in the list must always produce CONTRADICTION."""
        labels = prefix + ["CONTRADICTION"] + suffix
        assert NLIScorer.aggregate(labels) == "CONTRADICTION"
