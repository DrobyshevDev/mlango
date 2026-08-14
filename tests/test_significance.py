"""McNemar's test, as it is used before a promotion.

Only the rows the two versions disagree about carry information: rows both got
right, and rows both got wrong, are silent. So the question is whether a coin
that came up ``fixed`` heads in ``fixed + broke`` tosses was fair.

The values below are checked against the exact binomial, computed by hand where
that is short enough to be obvious, so a wrong implementation cannot agree with
a wrong expectation.
"""

from __future__ import annotations

import math

import pytest

from mlango.training.comparison import DEFAULT_ALPHA, significance


def exact_two_sided(fixed: int, broke: int) -> float:
    """The definition, written out separately from the implementation."""
    n = fixed + broke
    k = min(fixed, broke)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


class TestTheStatistic:
    @pytest.mark.parametrize(
        ("fixed", "broke"),
        [(0, 1), (1, 0), (3, 0), (5, 1), (10, 2), (20, 8), (40, 38), (200, 3), (1, 7)],
    )
    def test_it_matches_the_exact_binomial(self, fixed, broke):
        assert significance(fixed, broke)["p_value"] == pytest.approx(exact_two_sided(fixed, broke))

    def test_a_p_value_is_a_probability(self):
        for fixed in range(0, 12):
            for broke in range(0, 12):
                p = significance(fixed, broke)["p_value"]
                assert 0.0 <= p <= 1.0

    def test_an_even_split_is_certainly_not_evidence(self):
        # Ten fixed against ten broken is the fairest coin there is.
        assert significance(10, 10)["p_value"] == 1.0

    def test_it_is_symmetric_in_its_arguments(self):
        # The strength of the evidence does not depend on which way it points.
        assert significance(3, 17)["p_value"] == significance(17, 3)["p_value"]

    def test_more_lopsided_is_stronger_evidence(self):
        assert significance(18, 2)["p_value"] < significance(14, 6)["p_value"]

    def test_the_same_ratio_on_more_rows_is_stronger_evidence(self):
        # Six out of eight could be chance. Sixty out of eighty could not.
        assert significance(60, 20)["p_value"] < significance(6, 2)["p_value"]


class TestWhatItConcludes:
    def test_no_disagreement_is_no_evidence_rather_than_agreement(self):
        stats = significance(0, 0)
        assert stats["direction"] == "identical"
        assert stats["p_value"] == 1.0
        assert stats["discordant"] == 0

    def test_a_landslide_of_fixes_is_an_improvement(self):
        stats = significance(200, 3)
        assert stats["direction"] == "improvement"
        assert stats["p_value"] < DEFAULT_ALPHA
        assert "real improvement" in stats["verdict"]

    def test_a_landslide_of_breakage_is_a_regression(self):
        stats = significance(3, 200)
        assert stats["direction"] == "regression"
        assert stats["p_value"] < DEFAULT_ALPHA
        assert "real regression" in stats["verdict"]

    def test_the_case_the_whole_thing_exists_for(self):
        # Thirty-eight fixed against forty broken is a coin. A rule that counts
        # broken rows calls it a regression and blocks the promotion.
        stats = significance(38, 40)
        assert stats["direction"] == "regression"
        assert stats["p_value"] > DEFAULT_ALPHA
        assert "not distinguishable from noise" in stats["verdict"]

    def test_a_dead_heat_says_so_plainly(self):
        stats = significance(7, 7)
        assert stats["direction"] == "tie"
        assert "coin" in stats["verdict"]

    def test_one_broken_row_and_nothing_else_is_not_yet_evidence(self):
        # A single discordant row cannot reach 0.05 no matter which way it goes:
        # the smallest two-sided p-value available is 1.0.
        stats = significance(0, 1)
        assert stats["direction"] == "regression"
        assert stats["p_value"] > DEFAULT_ALPHA

    def test_five_broken_and_none_fixed_is_evidence(self):
        # 2 * (1/32) = 0.0625 is not; five against nought needs one more row.
        assert significance(0, 5)["p_value"] == pytest.approx(0.0625)
        assert significance(0, 6)["p_value"] == pytest.approx(0.03125)

    def test_the_verdict_carries_the_number_behind_it(self):
        # So a reader who disagrees with 0.05 can see what they are disagreeing
        # with, rather than being handed a word.
        assert "p=" in significance(200, 3)["verdict"]
        assert "p=" in significance(38, 40)["verdict"]
