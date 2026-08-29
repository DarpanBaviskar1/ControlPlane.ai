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


class TestAbsoluteCalibration:
    """Ordering tests alone cannot catch a scorer whose whole range is
    compressed — pin the two ends of the scale against realistic prompts."""

    def test_trivial_prompt_scores_low(self) -> None:
        assert score_complexity("Hi") < 0.2

    def test_hard_multisignal_prompt_scores_high(self) -> None:
        hard = (
            "Analyse why our p99 latency regressed after the shard migration, "
            "compare it against the pre-migration baseline, evaluate the "
            "trade-offs of rolling back versus re-sharding, and justify your "
            "recommendation. Here is the relevant code:\n"
            "```python\n"
            "for shard in shards:\n"
            "    rebalance(shard)\n"
            "```\n"
            "What caused the regression? How should we roll out the fix? "
            "What is the risk of a partial rollback?"
        )
        assert score_complexity(hard) >= 0.6


class TestReasoningTermWordBoundary:
    """A naive substring check on reasoning terms fires on ordinary prose
    like 'however' or 'shower' (both contain 'how'). Word-boundary matching
    must not treat these as reasoning-term hits."""

    def test_however_and_shower_are_not_false_positives(self) -> None:
        # Same word count (9) and no code/question marks in either, so if
        # word-boundary matching works, "how"/"however" substrings inside
        # "however"/"shower" contribute zero reasoning signal and both
        # prompts score identically.
        false_positive = "However, I took a shower this morning before work."
        baseline = "Nevertheless, I ate breakfast this early morning before work."
        assert score_complexity(false_positive) == score_complexity(baseline)

    def test_real_how_term_still_counts(self) -> None:
        with_how = "How does this algorithm work?"
        without = "This algorithm works well."
        assert score_complexity(with_how) > score_complexity(without)


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
