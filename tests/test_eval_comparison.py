"""Diffing two evaluation runs.

An agent has no version number, so "did my prompt change break anything" is a
question about two runs of the same suite rather than two registered artifacts.
The arithmetic is the one a model diff does; only where the pass and fail come
from is different.
"""

from __future__ import annotations

import pytest

from mlango.evals.comparison import compare_runs, recent_runs


def make_run(label: str, cases: dict[str, bool], outputs: dict[str, str] | None = None):
    """A finished evaluation run with the given per-case verdicts."""
    from mlango.metastore.models import EvalResult, Run, RunKind, RunStatus
    from mlango.metastore.session import session_scope

    outputs = outputs or {}
    with session_scope() as session:
        run = Run(kind=RunKind.EVAL, target=label, status=RunStatus.FINISHED)
        session.add(run)
        session.flush()
        for case_id, passed in cases.items():
            session.add(
                EvalResult(
                    run_id=run.id,
                    eval_label=label,
                    case_id=case_id,
                    passed=passed,
                    output=outputs.get(case_id, "pass" if passed else "fail"),
                    expected="pass",
                )
            )
        return run


class TestTheArithmetic:
    def test_fixed_and_broke_are_counted_separately(self, project):
        """The number a pass rate hides: cases that used to work and now do not."""
        older = make_run("app.Suite", {"a": False, "b": False, "c": True, "d": True})
        newer = make_run("app.Suite", {"a": True, "b": True, "c": True, "d": False})

        report = compare_runs(older, newer)

        assert report["cases"] == 4
        assert report["fixed"] == 2
        assert report["broke"] == 1
        assert report["pass_rate"]["left"] == 0.5
        assert report["pass_rate"]["right"] == 0.75
        assert report["pass_rate"]["delta"] == 0.25

    def test_a_better_pass_rate_can_still_have_broken_something(self, project):
        older = make_run("app.Suite", {f"c{i}": i < 8 for i in range(10)})
        newer = make_run("app.Suite", {f"c{i}": i != 0 for i in range(10)})

        report = compare_runs(older, newer)

        assert report["pass_rate"]["delta"] > 0, "the suite got better overall"
        assert report["broke"] == 1, "and still lost a case that used to pass"

    def test_agreement_counts_cases_whose_verdict_held(self, project):
        older = make_run("app.Suite", {"a": True, "b": True, "c": False, "d": False})
        newer = make_run("app.Suite", {"a": True, "b": False, "c": False, "d": False})

        assert compare_runs(older, newer)["agreement"] == 0.75

    def test_two_identical_runs_report_no_evidence(self, project):
        cases = {"a": True, "b": False}
        report = compare_runs(make_run("app.Suite", cases), make_run("app.Suite", cases))

        assert report["fixed"] == report["broke"] == 0
        assert report["significance"]["direction"] == "identical"
        assert report["agreement"] == 1.0

    def test_the_verdict_says_whether_the_change_is_a_coin(self, project):
        """Twenty fixed against nineteen broken is noise, and must read as noise."""
        older = make_run("app.Suite", {f"c{i}": i >= 20 for i in range(60)})
        newer = make_run("app.Suite", {f"c{i}": i < 20 or i >= 39 for i in range(60)})

        stats = compare_runs(older, newer)["significance"]
        assert stats["discordant"] == 39
        assert stats["p_value"] > 0.05
        assert "not distinguishable from noise" in stats["verdict"]


class TestCasesThatDoNotLineUp:
    def test_a_suite_that_grew_is_reported_not_absorbed(self, project):
        """Adding easy cases is how a pass rate improves without anything improving."""
        older = make_run("app.Suite", {"a": True, "b": False})
        newer = make_run("app.Suite", {"a": True, "b": False, "c": True, "d": True})

        report = compare_runs(older, newer)

        assert report["cases"] == 2, "only the shared cases are compared"
        assert report["only_right"] == ["c", "d"]
        assert report["only_left"] == []
        assert report["pass_rate"]["right"] == 0.5, "not 0.75 — the new cases do not count"

    def test_a_removed_case_is_named(self, project):
        older = make_run("app.Suite", {"a": True, "gone": False})
        newer = make_run("app.Suite", {"a": True})

        assert compare_runs(older, newer)["only_left"] == ["gone"]

    def test_no_shared_case_is_an_error_that_explains_case_ids(self, project):
        older = make_run("app.Suite", {"a": True})
        newer = make_run("app.Suite", {"z": True})

        with pytest.raises(LookupError, match="case_id"):
            compare_runs(older, newer)


class TestWhatChanged:
    def test_changes_lead_with_the_cases_that_moved(self, project):
        older = make_run(
            "app.Suite",
            {"held": True, "lost": True, "won": False},
            outputs={"held": "same", "lost": "was right", "won": "was wrong"},
        )
        newer = make_run(
            "app.Suite",
            {"held": True, "lost": False, "won": True},
            outputs={"held": "same", "lost": "now wrong", "won": "now right"},
        )

        rows = compare_runs(older, newer, max_changes=5)["changes"]

        assert [row["case"] for row in rows] == ["lost", "won"]
        assert rows[0]["was"] == "pass" and rows[0]["now"] == "fail"
        assert rows[0]["left"] == "was right" and rows[0]["right"] == "now wrong"
        assert rows[0]["expected"] == "pass"

    def test_a_reworded_answer_that_still_passes_is_counted(self, project):
        """For an agent the wording is half the product, so a silent rewrite matters."""
        older = make_run("app.Suite", {"a": True}, outputs={"a": "Yes, certainly."})
        newer = make_run("app.Suite", {"a": True}, outputs={"a": "Yep."})

        report = compare_runs(older, newer, max_changes=5)

        assert report["fixed"] == report["broke"] == 0
        assert report["changed"] == 1
        assert report["changes"][0]["case"] == "a"

    def test_changes_are_not_collected_unless_asked(self, project):
        older = make_run("app.Suite", {"a": True})
        newer = make_run("app.Suite", {"a": False})

        assert "changes" not in compare_runs(older, newer)

    def test_an_agent_envelope_is_unwrapped_before_comparing(self, project):
        """A trace id differs on every run and is not what a changed answer means."""
        from sqlalchemy import update

        from mlango.metastore.models import EvalResult
        from mlango.metastore.session import session_scope

        older = make_run("app.Suite", {"a": True})
        newer = make_run("app.Suite", {"a": True})

        with session_scope() as session:
            session.execute(
                update(EvalResult)
                .where(EvalResult.run_id == older.id)
                .values(output={"output": "same answer", "trace": "aaaa", "steps": 2})
            )
            session.execute(
                update(EvalResult)
                .where(EvalResult.run_id == newer.id)
                .values(output={"output": "same answer", "trace": "bbbb", "steps": 3})
            )

        assert compare_runs(older, newer)["changed"] == 0


class TestFindingTheRuns:
    def test_recent_runs_are_newest_first(self, project):
        first = make_run("app.Suite", {"a": True})
        second = make_run("app.Suite", {"a": False})

        found = recent_runs("app.Suite", limit=2)
        assert [run.id for run in found] == [second.id, first.id]

    def test_only_finished_runs_of_that_suite_are_offered(self, project):
        from mlango.metastore.models import Run, RunKind, RunStatus
        from mlango.metastore.session import session_scope

        make_run("app.Suite", {"a": True})
        with session_scope() as session:
            session.add(Run(kind=RunKind.EVAL, target="app.Suite", status=RunStatus.FAILED))
            session.add(Run(kind=RunKind.EVAL, target="app.Other", status=RunStatus.FINISHED))
            session.add(Run(kind=RunKind.TRAIN, target="app.Suite", status=RunStatus.FINISHED))

        assert len(recent_runs("app.Suite", limit=10)) == 1
