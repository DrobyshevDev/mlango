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


class TestWhatChangedAboutTheTarget:
    """Results without a cause leave the user to remember what they changed."""

    def _run(self, label, cases, *, config=None, version=None, fingerprint="fp"):
        from mlango.metastore.models import EvalResult, Run, RunKind, RunStatus
        from mlango.metastore.session import session_scope

        params = {"_eval": label, "_target_fingerprint": fingerprint}
        if config is not None:
            params["_target_config"] = config
        if version is not None:
            params["_target_version"] = version

        with session_scope() as session:
            run = Run(kind=RunKind.EVAL, target=label, status=RunStatus.FINISHED, params=params)
            session.add(run)
            session.flush()
            for case_id, passed in cases.items():
                session.add(
                    EvalResult(
                        run_id=run.id,
                        eval_label=label,
                        case_id=case_id,
                        passed=passed,
                        output="pass" if passed else "fail",
                        expected="pass",
                    )
                )
            return run

    def test_a_changed_option_is_named_with_both_values(self, project):
        older = self._run("app.Suite", {"a": True}, config={"model": "sonnet", "max_steps": 8})
        newer = self._run("app.Suite", {"a": False}, config={"model": "opus", "max_steps": 8})

        delta = compare_runs(older, newer)["config"]["changed"]

        assert set(delta) == {"model"}, "an option that held is not a change"
        assert delta["model"]["was"] == "sonnet"
        assert delta["model"]["now"] == "opus"
        assert delta["model"]["long"] is False

    def test_a_long_prompt_is_flagged_rather_than_inlined(self, project):
        """A system prompt is a page; printing it in a report helps nobody."""
        older = self._run("app.Suite", {"a": True}, config={"system": "be helpful " * 20})
        newer = self._run("app.Suite", {"a": True}, config={"system": "be terse " * 20})

        entry = compare_runs(older, newer)["config"]["changed"]["system"]
        assert entry["long"] is True

    def test_an_added_option_shows_what_it_was_not(self, project):
        older = self._run("app.Suite", {"a": True}, config={})
        newer = self._run("app.Suite", {"a": True}, config={"thinking": "adaptive"})

        entry = compare_runs(older, newer)["config"]["changed"]["thinking"]
        assert entry["was"] is None
        assert entry["now"] == "adaptive"

    def test_a_retrained_model_is_a_change_even_with_the_same_declaration(self, project):
        """The class did not move; the artifact did, and that is what ran."""
        older = self._run("app.Suite", {"a": True}, config={"C": 1.0}, version=3)
        newer = self._run("app.Suite", {"a": False}, config={"C": 1.0}, version=4)

        config = compare_runs(older, newer)["config"]
        assert config["version"] == {"was": 3, "now": 4}
        assert config["identical"] is False, "a new version is not an unchanged target"

    def test_an_untouched_target_says_so(self, project):
        """Then the movement is the target's own — sampling, temperature, a tool."""
        config = {"model": "opus", "system": "same"}
        older = self._run("app.Suite", {"a": True}, config=config, version=2)
        newer = self._run("app.Suite", {"a": False}, config=config, version=2)

        delta = compare_runs(older, newer)["config"]
        assert delta["changed"] == {}
        assert delta["identical"] is True

    def test_runs_from_before_this_existed_claim_nothing(self, project):
        """An old run has no recorded config, and absence is not sameness."""
        from mlango.metastore.models import Run, RunKind, RunStatus
        from mlango.metastore.session import session_scope

        older = self._run("app.Suite", {"a": True})
        with session_scope() as session:
            run = Run(kind=RunKind.EVAL, target="app.Suite", status=RunStatus.FINISHED, params={})
            session.add(run)
            session.flush()
            from mlango.metastore.models import EvalResult

            session.add(
                EvalResult(
                    run_id=run.id, eval_label="app.Suite", case_id="a", passed=True, expected="pass"
                )
            )
            newer = run

        config = compare_runs(older, newer)["config"]
        assert config["changed"] == {}
        assert config["identical"] is False, "no fingerprint on one side proves nothing"


class TestWhatEvaluateRecords:
    def test_an_agents_declaration_is_recorded(self, project, isolated_registry):
        """The prompt and the model are what a prompt change changes."""
        from mlango.agents import Agent
        from mlango.core import fields
        from mlango.data import Dataset, InMemorySource
        from mlango.evals import Eval, exact_match

        class Cases(Dataset):
            id = fields.IntegerField()
            question = fields.TextField()
            answer = fields.TextField()

            class Meta:
                source = InMemorySource([{"id": 1, "question": "hi", "answer": "hi"}])
                primary_key = "id"

        class Bot(Agent):
            class Meta:
                system = "Be brief."
                max_steps = 3

        class Quality(Eval):
            class Meta:
                dataset = Cases
                target = Bot
                input_field = "question"
                expected_field = "answer"
                case_id_field = "id"
                scorers = {"correct": exact_match}

        from mlango.training.run import get_run

        report = Quality.evaluate()
        params = get_run(report.run.uuid).params

        assert params["_target_fingerprint"]
        assert params["_target_config"]["system"] == "Be brief."
        assert params["_target_config"]["max_steps"] == 3

    def test_a_models_version_is_recorded_because_the_artifact_is_the_behaviour(
        self, project, reviews, isolated_registry
    ):
        pytest.importorskip("sklearn")

        from mlango.core import fields
        from mlango.evals import Eval, exact_match
        from mlango.training import Model

        class Guess(Model):
            C = fields.FloatField(default=1.0)

            class Meta:
                dataset = reviews
                trainer = "sklearn"
                task = "classification"
                features = ["text"]

            def build(self):
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline

                return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=200))

        class Checked(Eval):
            class Meta:
                dataset = reviews
                target = Guess
                input_field = "text"
                expected_field = "label"
                case_id_field = "id"
                scorers = {"correct": exact_match}
                max_cases = 5

        from mlango.training.run import get_run

        Guess.fit(C=2.0)
        params = get_run(Checked.evaluate().run.uuid).params

        assert params["_target_version"] == 1
        assert params["_target_config"]["C"] == 2.0

    def test_recording_it_never_breaks_the_evaluation(self, project):
        """Bookkeeping about a run must not be able to fail the run."""
        from mlango.evals.base import _target_state

        class Broken:
            @property
            def _meta(self):
                raise RuntimeError("not a declarative at all")

        assert _target_state(object()) == {}
        assert isinstance(_target_state(Broken()), dict)
