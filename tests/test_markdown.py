"""Rendering a comparison as Markdown, for a pull request rather than a tty.

The renderer is a pure function of the report dictionary, so these tests build
report dictionaries directly. That is the point of the seam: four different
comparisons produce the same shape, and one renderer covers all of them without
knowing which one it was handed.
"""

from __future__ import annotations

import re

import pytest

from mlango.core.markdown import marker_for, render


def model_report(**overrides):
    report = {
        "label": "reviews.Sentiment",
        "left": 1,
        "right": 2,
        "dataset": "reviews.Reviews",
        "rows": 500,
        "task": "classification",
        "labelled": True,
        "agreement": 0.92,
        "changed": 40,
        "transitions": {"pos → neg": 22, "neg → pos": 18},
        "fixed": 29,
        "broke": 11,
        "metrics": {
            "key": "accuracy",
            "left": {"accuracy": 0.77},
            "right": {"accuracy": 0.806},
            "delta": 0.036,
        },
        "significance": {
            "discordant": 40,
            "verdict": "a real improvement: 29 fixed against 11 broken (p=0.006)",
            "direction": "improvement",
            "p_value": 0.006,
        },
    }
    report.update(overrides)
    return report


def eval_report(**overrides):
    report = {
        "kind": "eval",
        "label": "support.AnswerQuality",
        "left": "a1b2c3",
        "right": "d4e5f6",
        "cases": 40,
        "only_left": [],
        "only_right": [],
        "pass_rate": {"left": 0.8, "right": 0.925, "delta": 0.125},
        "fixed": 7,
        "broke": 2,
        "changed": 12,
        "agreement": 0.925,
        "significance": {"discordant": 9, "verdict": "a real improvement (p=0.03)"},
        "config": {"changed": {}},
    }
    report.update(overrides)
    return report


class TestWhatItLeadsWith:
    def test_the_broken_count_is_in_the_first_sentence(self):
        """A pull request comment is usually first read as a notification."""
        first = render(model_report()).split("\n")[2]
        assert first.startswith("###")

        headline = render(model_report()).split("\n")[4]
        assert "11 rows broken" in headline
        assert "29 fixed" in headline

    def test_nothing_broken_says_so_rather_than_saying_zero(self):
        assert "Nothing broken" in render(model_report(broke=0))

    @pytest.mark.parametrize(
        ("broke", "glyph"),
        [(11, "⚠️"), (0, "✅")],
    )
    def test_the_heading_carries_a_status_glyph(self, broke, glyph):
        assert render(model_report(broke=broke)).split("\n")[2].startswith(f"### {glyph}")

    def test_unlabelled_data_claims_nothing_about_better(self):
        """Without labels there is no fixed and no broke, only movement."""
        # A real unlabelled report has no metrics and no verdict either:
        # there is nothing to be significant about.
        out = render(
            model_report(labelled=False, significance=None, metrics=None, fixed=None, broke=None)
        )
        assert "ℹ️" in out
        assert "broken" not in out
        assert "says what changed, not what improved" in out

    def test_one_broken_row_is_not_pluralised(self):
        assert "1 row broken" in render(model_report(broke=1))


class TestTheNumbers:
    def test_the_metric_moves_inside_one_cell(self):
        assert "| `accuracy` | 0.7700 → **0.8060** (+0.0360) |" in render(model_report())

    def test_the_broken_count_is_bold_and_the_fixed_count_is_not(self):
        out = render(model_report())
        assert "| broke | **11 rows** |" in out
        assert "| fixed | 29 rows |" in out

    def test_zero_broken_is_not_shouted(self):
        assert "| broke | 0 rows |" in render(model_report(broke=0))

    def test_the_verdict_is_quoted_rather_than_asserted(self):
        assert "> a real improvement: 29 fixed against 11 broken (p=0.006)" in render(
            model_report()
        )

    def test_a_coin_flip_has_no_verdict_to_report(self):
        out = render(model_report(significance={"discordant": 0, "verdict": "no evidence"}))
        assert "no evidence" not in out

    def test_movement_between_classes_is_named(self):
        assert "Movement: `pos → neg` 22 · `neg → pos` 18" in render(model_report())

    def test_a_regression_task_reports_distance_not_rightness(self):
        out = render(
            model_report(
                task="regression",
                closer=300,
                further=40,
                mean_delta=-0.02,
                largest_delta=1.4,
                metrics={"key": "mae", "left": {"mae": 3.1}, "right": {"mae": 2.9}, "delta": -0.2},
                significance={"discordant": 340, "verdict": "a real improvement (p=0.001)"},
                transitions={},
            )
        )
        assert "| closer | 300 rows |" in out
        assert "| further | **40 rows** |" in out
        assert "mean delta" in out
        assert "fixed" not in out, "right and wrong are not categories here"


class TestTheRows:
    def test_they_are_folded_away(self):
        """The verdict is above the fold; the evidence is one click below it."""
        changes = [{"text": "a film", "left": "neg", "right": "pos", "expected": "pos"}]
        out = render(model_report(changes=changes))
        assert "<details>" in out
        assert "| text | left | right | expected |" in out
        assert out.index("11 rows broken") < out.index("<details>")

    def test_the_summary_counts_the_real_total_not_what_was_asked_for(self):
        """`--show-changes 4` of forty disagreements is four of forty."""
        changes = [{"id": n} for n in range(4)]
        assert "<summary>4 of 40 rows where they disagree</summary>" in render(
            model_report(changes=changes)
        )

    def test_a_very_long_list_is_capped(self):
        changes = [{"id": n} for n in range(500)]
        out = render(model_report(changes=changes, changed=500))
        assert "<summary>20 of 500 rows where they disagree</summary>" in out
        assert out.count("\n| 4") == 1, "only ids 4 and 40-something would repeat, not 400"

    def test_a_pipe_in_the_data_cannot_break_the_table(self):
        """Otherwise one row of user text silently mangles the whole report."""
        changes = [{"text": "a | b | c", "left": "neg", "right": "pos"}]
        row = [
            line for line in render(model_report(changes=changes)).split("\n") if "a \\|" in line
        ]
        assert len(row) == 1
        assert row[0].count("|") - row[0].count("\\|") == 4, "three cells, four delimiters"

    def test_a_newline_in_the_data_cannot_break_the_table(self):
        changes = [{"text": "first\nsecond", "left": "neg", "right": "pos"}]
        assert "| first second | neg | pos |" in render(model_report(changes=changes))

    def test_a_long_value_is_truncated(self):
        changes = [{"text": "x" * 500, "left": "neg", "right": "pos"}]
        assert "x" * 200 not in render(model_report(changes=changes))

    def test_no_changes_means_no_disclosure(self):
        assert "<details>" not in render(model_report())


class TestEvaluationRuns:
    def test_it_reads_as_cases_rather_than_rows(self):
        out = render(eval_report())
        assert "2 cases broken" in out
        assert "rows" not in out

    def test_runs_are_named_by_their_ids(self):
        assert "support.AnswerQuality a1b2c3 → d4e5f6" in render(eval_report())

    def test_rewording_is_reported_separately(self):
        """Nothing for a classifier, half the product for something a person reads."""
        assert "| reworded | 3 cases |" in render(eval_report())

    def test_a_suite_that_grew_is_named_rather_than_averaged(self):
        out = render(eval_report(only_right=["billing-04", "refunds-01"]))
        assert "2 case(s) only in the newer run" in out
        assert "`billing-04`" in out

    def test_what_changed_about_the_target_is_put_beside_the_effect(self):
        out = render(
            eval_report(
                config={
                    "changed": {
                        "system": {"was": "a", "now": "b", "long": True},
                        "model": {"was": "haiku", "now": "opus", "long": False},
                    }
                }
            )
        )
        assert "**What changed:** `system` rewritten, `model` haiku → opus" in out

    def test_an_unchanged_target_says_nothing(self):
        assert "What changed" not in render(eval_report())


class TestTheMarker:
    def test_it_leads_the_document(self):
        assert render(model_report()).startswith("<!-- mlango:diff:model:reviews.Sentiment -->")

    def test_it_is_stable_across_runs_of_the_same_comparison(self):
        """A CI job edits its own last comment; it cannot if the key moves."""
        assert marker_for(model_report(left=4, right=9)) == marker_for(model_report())

    def test_two_models_do_not_share_one_comment(self):
        assert marker_for(model_report(label="a.B")) != marker_for(model_report(label="c.D"))

    def test_a_model_and_an_eval_of_the_same_name_do_not_collide(self):
        assert marker_for(model_report(label="x.Y")) != marker_for(eval_report(label="x.Y"))

    def test_artefacts_have_no_label_and_still_get_a_marker(self):
        assert "artefact" in marker_for(model_report(label=""))


class TestItIsValidMarkdown:
    @pytest.mark.parametrize("report", [model_report(), eval_report()])
    def test_every_table_row_has_the_same_number_of_cells(self, report):
        report["changes"] = [{"a": 1, "b": 2, "c": 3}]
        rows = [line for line in render(report).split("\n") if line.startswith("|")]
        by_width: dict[int, int] = {}
        for row in rows:
            by_width[row.count("|")] = by_width.get(row.count("|"), 0) + 1
        # Two tables: two columns of facts, three columns of changes — and a
        # row of N cells carries N+1 delimiters.
        assert sorted(by_width) == [3, 4]

    @pytest.mark.parametrize("report", [model_report(), eval_report()])
    def test_it_ends_with_the_attribution(self, report):
        assert render(report).rstrip().endswith("</sub>")

    @pytest.mark.parametrize("report", [model_report(), eval_report()])
    def test_no_ansi_escapes_survive(self, report):
        """The terminal renderer's colours would be literal noise here."""
        assert not re.search(r"\033\[", render(report))

    def test_two_artefacts_read_as_their_paths(self):
        out = render(model_report(label="", left="models/a.pkl", right="models/b.pkl"))
        assert "### ⚠️ models/a.pkl → models/b.pkl" in out
