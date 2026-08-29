"""Local complexity scorer — replaces the hardcoded score=0.5 stub."""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from app.router.complexity import (
    _W_LENGTH,
    _W_QUESTIONS,
    _W_REASONING,
    _W_STRUCTURE,
    classify,
    score_complexity,
)


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
    """P-CX-3: genuinely harder prompts must outscore trivial ones.

    Word count is held EQUAL within each pair below (except the length
    test, whose whole point is to vary length) so that the signal under
    test is what actually decides the comparison, not a length confound
    riding along for free.
    """

    def test_reasoning_prompt_beats_lookup_prompt(self) -> None:
        # 11 words each; neither has a "?", code fence, or structure hint.
        simple = "Please describe your monthly pricing plans and support options available today"
        complex_ = "Please analyze and compare your monthly pricing plans support options available"
        assert score_complexity(complex_) > score_complexity(simple)

    def test_length_increases_score_monotonically(self) -> None:
        short = "Explain caching."
        long = "Explain caching. " + ("It must handle eviction and staleness. " * 20)
        assert score_complexity(long) > score_complexity(short)

    def test_code_block_raises_score(self) -> None:
        # 17 words each; neither has a "?" or a reasoning-term difference
        # beyond the shared "why".
        plain = (
            "Why does this particular function actually fail when given a "
            "large odd strange input list here today"
        )
        with_code = (
            "Why does this fail\n```\nfor i in x:\n    y(i)\n```\nwhen given "
            "a large odd input list"
        )
        assert score_complexity(with_code) > score_complexity(plain)

    def test_multiple_questions_raise_score(self) -> None:
        # 11 words each; neither has a code fence or a reasoning term.
        one = "Is the cache currently enabled for all requests right there now"
        many = "Is the cache enabled? Is the TTL set? Is it evicted"
        assert score_complexity(many) > score_complexity(one)


class TestAbsoluteCalibration:
    """Ordering tests alone cannot catch a scorer whose whole range is
    compressed — pin the two ends of the scale against realistic prompts."""

    def test_trivial_prompt_scores_low(self) -> None:
        assert score_complexity("Hi") < 0.2

    def test_hard_multisignal_prompt_scores_high(self) -> None:
        # Saturates reasoning (6 distinct terms), structure (code fence)
        # and questions (3 "?") on its own — length is not the deciding
        # signal here. See test_length_decisive_prompt_scores_high below
        # for the pair where length alone must carry the score.
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

    def test_length_decisive_prompt_scores_high(self) -> None:
        """Length must be able to carry a prompt over the bar on its own.

        This prompt has no code fence and only one "?", and its 4
        reasoning-term hits are well short of saturation (which needs 6+
        distinct terms to hit 1.0, and even 4/4 only contributes
        _W_REASONING=0.30 fully). Its 103 words are what push it to 0.700
        at _LENGTH_SATURATION_WORDS=60 (vs. 0.530 at the brief's original
        200) — asserting >= 0.6, not >= 0.7, since 0.700 is a float
        equality boundary this test must not sit on.
        """
        prompt = (
            "Our nightly batch reconciliation job used to finish in about twenty minutes and "
            "now takes close to three hours, and the finance team needs the output before the "
            "markets open, so the delay has become a business problem rather than merely an "
            "engineering annoyance. Nothing changed in the job itself during that window, "
            "although the upstream ledger table has roughly doubled in row count and we also "
            "moved the warehouse onto a new instance class at the same time. Why did the "
            "regression appear only now, and how would you compare and evaluate the two "
            "suspected causes before we commit to a fix?"
        )
        assert score_complexity(prompt) >= 0.6


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
        # 9 words each; equal word count so the "how" hit is what decides,
        # not a length confound.
        with_how = "How does this particular caching algorithm actually work internally"
        without = "This particular caching algorithm actually works well internally today"
        assert score_complexity(with_how) > score_complexity(without)


class TestClassify:
    def test_below_threshold_routes_to_slm(self) -> None:
        cls, tier, score = classify("Hi", threshold=0.7)
        assert cls == "ROUTINE"
        assert tier == "SLM"
        assert score < 0.7

    def test_at_or_above_threshold_routes_to_frontier(self) -> None:
        """Boundary is inclusive: score >= threshold means COMPLEX.

        Using threshold=0.0 against any nonzero-scoring prompt does not
        exercise inclusivity: a strict ``>`` implementation would also
        return COMPLEX there, since the score is already above zero. To
        actually pin the ``>=`` boundary, score a prompt first and then
        feed that exact score back in as the threshold, so score ==
        threshold precisely.
        """
        prompt = "Explain the trade-offs of sharding versus replication for this system"
        exact_score = score_complexity(prompt)
        cls, tier, score = classify(prompt, threshold=exact_score)
        assert score == exact_score
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


class TestWeightCalibration:
    def test_signal_weights_sum_to_one(self) -> None:
        """The [0, 1] range depends on the four weights summing to 1.0 —
        score_complexity's clamp protects the range even if this drifts,
        but a drifted sum silently recalibrates every score. Nothing but
        this test enforces the invariant the module's own comment claims."""
        total = _W_LENGTH + _W_REASONING + _W_STRUCTURE + _W_QUESTIONS
        assert math.isclose(total, 1.0, rel_tol=1e-9)
