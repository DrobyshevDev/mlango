"""Hyperparameter sweeps: search space expansion, run bookkeeping, promotion."""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.core.exceptions import ImproperlyConfigured, RunError
from mlango.data import Dataset, InMemorySource
from mlango.metastore.models import RunStatus
from mlango.training.sweep import Trial, expand_grid, run_sweep, sample_space

pytestmark = pytest.mark.usefixtures("isolated_registry")

ROWS = [
    {
        "id": index,
        "text": ("great movie " if index % 2 else "awful movie ") + str(index),
        "label": "pos" if index % 2 else "neg",
    }
    for index in range(40)
]


@pytest.fixture
def sweepable(project):
    pytest.importorskip("sklearn")

    from mlango.training import Model

    class Rows(Dataset):
        id = fields.IntegerField()
        text = fields.TextField()
        label = fields.LabelField(["neg", "pos"])

        class Meta:
            source = InMemorySource(ROWS)
            primary_key = "id"

    class Tuned(Model):
        """A model with two tunable hyperparameters."""

        C = fields.FloatField(default=1.0, tunable=True)
        max_features = fields.IntegerField(default=1000, tunable=True)
        note = fields.CharField(default="", required=False)

        class Meta:
            dataset = Rows
            trainer = "sklearn"
            task = "classification"
            features = ["text"]
            monitor = "accuracy"
            monitor_mode = "max"

        def build(self):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline

            return make_pipeline(
                TfidfVectorizer(max_features=self.max_features),
                LogisticRegression(C=self.C, max_iter=300),
            )

    return Rows, Tuned


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #


class TestSearchSpace:
    def test_the_grid_is_every_combination_in_a_stable_order(self):
        points = expand_grid({"a": [1, 2], "b": ["x", "y"]})
        assert points == [
            {"a": 1, "b": "x"},
            {"a": 1, "b": "y"},
            {"a": 2, "b": "x"},
            {"a": 2, "b": "y"},
        ]

    def test_an_empty_space_is_one_trial_with_defaults(self):
        assert expand_grid({}) == [{}]

    def test_a_single_axis(self):
        assert expand_grid({"C": [0.5, 1.0]}) == [{"C": 0.5}, {"C": 1.0}]

    def test_sampling_is_reproducible(self):
        space = {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}
        assert sample_space(space, 5, seed=7) == sample_space(space, 5, seed=7)

    def test_sampling_does_not_repeat_a_combination(self):
        points = sample_space({"a": [1, 2, 3, 4, 5]}, 3, seed=1)
        assert len(points) == 3
        assert len({tuple(sorted(p.items())) for p in points}) == 3

    def test_asking_for_more_than_the_grid_returns_the_grid(self):
        assert sample_space({"a": [1, 2]}, 99, seed=1) == expand_grid({"a": [1, 2]})


# --------------------------------------------------------------------------- #
# Declaration errors
# --------------------------------------------------------------------------- #


class TestValidation:
    def test_an_unknown_hyperparameter_lists_the_real_ones(self, sweepable):
        _rows, tuned = sweepable
        with pytest.raises(ImproperlyConfigured, match="Available: C, max_features, note"):
            run_sweep(tuned, {"nope": [1]})

    def test_a_bad_mode_is_rejected_before_anything_runs(self, sweepable):
        _rows, tuned = sweepable
        with pytest.raises(ValueError, match="mode must be"):
            run_sweep(tuned, {"C": [1.0]}, mode="highest")

    def test_a_random_sweep_needs_a_trial_count(self, sweepable):
        _rows, tuned = sweepable
        with pytest.raises(ImproperlyConfigured, match="--trials"):
            run_sweep(tuned, {"C": [1.0, 2.0]}, strategy="random")

    def test_an_unknown_strategy_names_the_valid_ones(self, sweepable):
        _rows, tuned = sweepable
        with pytest.raises(ImproperlyConfigured, match="'grid' or 'random'"):
            run_sweep(tuned, {"C": [1.0]}, strategy="bayesian")

    def test_the_default_metric_comes_from_the_declaration(self, sweepable):
        _rows, tuned = sweepable
        result = run_sweep(tuned, {"C": [1.0]})
        assert result.metric == "accuracy"
        assert result.mode == "max"


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


class TestGridSweep:
    def test_one_child_run_per_point(self, sweepable):
        _rows, tuned = sweepable
        result = run_sweep(tuned, {"C": [0.5, 2.0]})

        assert len(result.trials) == 2
        assert all(t.status == RunStatus.FINISHED for t in result.trials)
        assert all(t.run_uuid for t in result.trials)
        assert len({t.run_uuid for t in result.trials}) == 2

    def test_the_parent_run_ties_the_search_together(self, sweepable):
        from mlango.training import get_run

        _rows, tuned = sweepable
        result = run_sweep(tuned, {"C": [0.5, 2.0]}, name="search")

        parent = get_run(result.run_uuid)
        assert parent.kind == "sweep"
        assert parent.status == RunStatus.FINISHED
        assert parent.params["_strategy"] == "grid"
        assert parent.params["_space"] == {"C": [0.5, 2.0]}
        assert parent.params["_trials"] == 2

    def test_children_are_tagged_with_the_parent(self, sweepable):
        from mlango.training import get_run, recent_runs

        _rows, tuned = sweepable
        result = run_sweep(tuned, {"C": [0.5, 2.0]})
        short = get_run(result.run_uuid).uuid[:8]

        children = [r for r in recent_runs(limit=20) if r.kind == "train"]
        assert children
        assert all(f"sweep:{short}" in (r.tags or []) for r in children)

    def test_trials_can_be_capped(self, sweepable):
        _rows, tuned = sweepable
        result = run_sweep(tuned, {"C": [0.5, 1.0, 2.0, 4.0]}, trials=2)
        assert len(result.trials) == 2

    def test_two_axes_multiply(self, sweepable):
        _rows, tuned = sweepable
        result = run_sweep(tuned, {"C": [0.5, 2.0], "max_features": [100, 500]})
        assert len(result.trials) == 4
        assert {t.params["max_features"] for t in result.trials} == {100, 500}

    def test_a_random_sweep_honours_the_trial_count(self, sweepable):
        _rows, tuned = sweepable
        result = run_sweep(tuned, {"C": [0.5, 1.0, 2.0, 4.0]}, strategy="random", trials=2, seed=3)
        assert len(result.trials) == 2

    def test_the_progress_hook_sees_every_trial(self, sweepable):
        _rows, tuned = sweepable
        seen: list[int] = []
        run_sweep(
            tuned, {"C": [0.5, 2.0]}, on_trial=lambda trial, _result: seen.append(trial.index)
        )
        assert seen == [1, 2]

    def test_a_looked_up_run_is_usable_outside_its_session(self, sweepable):
        """get_run() hands back a detached object; its collections must work.

        Reaching for run.artifacts used to raise DetachedInstanceError — a wall of
        SQLAlchemy internals for what reads like ordinary attribute access.
        """
        from mlango.training import get_run

        _rows, tuned = sweepable
        run = tuned().train()

        record = get_run(run.uuid)
        assert isinstance(record.artifacts, list)
        assert isinstance(record.metrics, list)
        assert isinstance(record.model_versions, list)
        assert isinstance(record.eval_results, list)

        # Lookup by numeric id takes a different branch, so it needs its own check.
        by_id = get_run(str(record.id))
        assert [a.name for a in by_id.artifacts] == [a.name for a in record.artifacts]

    def test_the_sweep_json_artifact_records_every_trial(self, sweepable):
        import json

        from mlango.storage import default_storage
        from mlango.training.run import get_run

        _rows, tuned = sweepable
        result = run_sweep(tuned, {"C": [0.5, 2.0]})

        parent = get_run(result.run_uuid)
        artifact = next(a for a in parent.artifacts if a.name == "sweep.json")
        payload = json.loads(default_storage().read_text(artifact.path))

        assert len(payload["trials"]) == 2
        assert payload["summary"]["completed"] == 2


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


class TestResults:
    def test_the_best_trial_maximises_the_metric(self, sweepable):
        _rows, tuned = sweepable
        result = run_sweep(tuned, {"C": [0.1, 10.0]})

        assert result.best is not None
        assert result.best.score == max(t.score for t in result.completed)

    def test_min_mode_picks_the_lowest(self):
        result = _fake_result("min", [(1, 0.9), (2, 0.1), (3, 0.5)])
        assert result.best.index == 2
        assert [t.index for t in result.ranked()] == [2, 3, 1]

    def test_max_mode_ranks_descending(self):
        result = _fake_result("max", [(1, 0.9), (2, 0.1), (3, 0.5)])
        assert result.best.index == 1
        assert [t.index for t in result.ranked()] == [1, 3, 2]

    def test_a_sweep_with_nothing_completed_has_no_best(self):
        result = _fake_result("max", [])
        assert result.best is None
        assert result.ranked() == []
        assert "no completed trials" in repr(result)

    def test_the_summary_counts_failures(self):
        result = _fake_result("max", [(1, 0.5)])
        result.trials.append(Trial(index=2, params={}, status=RunStatus.FAILED))
        summary = result.summary()
        assert summary == {
            "trials": 2,
            "completed": 1,
            "failed": 1,
            "metric": "accuracy",
            "mode": "max",
            "best_accuracy": 0.5,
        }

    def test_repr_shows_the_best_score(self):
        assert "best accuracy=0.5000" in repr(_fake_result("max", [(1, 0.5)]))

    def test_a_trial_is_only_ok_when_it_finished_with_a_score(self):
        assert Trial(1, {}, score=0.5, status=RunStatus.FINISHED).ok is True
        assert Trial(1, {}, score=None, status=RunStatus.FINISHED).ok is False
        assert Trial(1, {}, score=0.5, status=RunStatus.FAILED).ok is False


class TestFailureHandling:
    def test_one_bad_trial_does_not_throw_away_the_good_ones(self, project, sweepable):
        """A failing corner of the space must not lose the completed trials."""
        rows, tuned = sweepable
        from mlango.training import Model

        class Fragile(Model):
            C = fields.FloatField(default=1.0, tunable=True)

            class Meta:
                dataset = rows
                trainer = "sklearn"
                task = "classification"
                features = ["text"]
                monitor = "accuracy"
                monitor_mode = "max"

            def build(self):
                if self.C < 0:
                    raise ValueError("C must be positive")
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline

                return make_pipeline(TfidfVectorizer(), LogisticRegression(C=self.C, max_iter=300))

        result = run_sweep(Fragile, {"C": [-1.0, 1.0]})

        assert len(result.trials) == 2
        assert len(result.completed) == 1
        failed = next(t for t in result.trials if not t.ok)
        assert failed.status == RunStatus.FAILED
        assert "C must be positive" in failed.error
        assert result.best is not None

    def test_a_trial_that_records_no_such_metric_explains_itself(self, sweepable):
        _rows, tuned = sweepable
        result = run_sweep(tuned, {"C": [1.0]}, metric="nonexistent", mode="max")

        trial = result.trials[0]
        assert trial.score is None
        assert "recorded no 'nonexistent'" in trial.error
        assert "Available:" in trial.error
        assert result.best is None


class TestPromotion:
    def test_the_winner_can_be_promoted(self, sweepable):
        _rows, tuned = sweepable
        run_sweep(tuned, {"C": [0.1, 10.0]}, promote_best="production")

        production = [v for v in tuned.versions() if v.stage == "production"]
        assert len(production) == 1
        assert production[0].run_id is not None

    def test_promoting_nothing_is_an_error_not_a_silent_no_op(self, sweepable):
        _rows, tuned = sweepable
        with pytest.raises(RunError, match="nothing to promote"):
            run_sweep(tuned, {"C": [1.0]}, metric="nonexistent", promote_best="production")

    def test_a_trial_without_a_registered_version_cannot_be_promoted(self, sweepable):
        from mlango.training.sweep import _promote

        _rows, tuned = sweepable
        with pytest.raises(RunError, match="registered no model version"):
            _promote(tuned, Trial(index=1, params={}, run_uuid="does-not-exist"), "production")


# --------------------------------------------------------------------------- #
# Model.sweep()
# --------------------------------------------------------------------------- #


class TestModelSweep:
    def test_the_method_on_the_model_is_the_same_search(self, sweepable):
        _rows, tuned = sweepable
        result = tuned.sweep({"C": [0.5, 2.0]})
        assert len(result.trials) == 2

    def test_the_default_space_comes_from_tunable_fields(self, sweepable):
        _rows, tuned = sweepable
        space = tuned.default_space()
        assert set(space) == {"C", "max_features"}
        assert "note" not in space  # declared, but not tunable
        assert all(len(values) > 1 for values in space.values())

    def test_sweeping_with_no_space_uses_the_default(self, sweepable):
        _rows, tuned = sweepable
        result = tuned.sweep(trials=2, strategy="random", seed=1)
        assert len(result.trials) == 2
        assert set(result.trials[0].params) <= {"C", "max_features"}


def _fake_result(mode: str, scored: list[tuple[int, float]]):
    """A SweepResult built without training, to test the ranking logic alone."""
    from mlango.training.sweep import SweepResult

    result = SweepResult(model_label="a.B", metric="accuracy", mode=mode)
    result.trials = [
        Trial(index=index, params={}, score=score, status=RunStatus.FINISHED)
        for index, score in scored
    ]
    return result


class TestParallelTrials:
    """``workers`` above 1 runs trials concurrently. Same results, different order."""

    def test_every_trial_still_runs(self, sweepable):
        from mlango.training.sweep import run_sweep

        _rows, sweepable = sweepable

        result = run_sweep(sweepable, {"C": [0.1, 0.5, 1.0, 2.0]}, workers=3)

        assert len(result.trials) == 4
        assert all(trial.status == "finished" for trial in result.trials)

    def test_trials_are_reported_in_order_however_they_finished(self, sweepable):
        """They complete out of order; a report whose rows jump around is worse."""
        from mlango.training.sweep import run_sweep

        _rows, sweepable = sweepable

        result = run_sweep(sweepable, {"C": [0.1, 0.5, 1.0, 2.0]}, workers=3)

        assert [trial.index for trial in result.trials] == [1, 2, 3, 4]

    def test_the_same_space_finds_the_same_winner(self, sweepable):
        from mlango.training.sweep import run_sweep

        _rows, sweepable = sweepable

        space = {"C": [0.01, 1.0]}
        serial = run_sweep(sweepable, space, workers=1)
        parallel = run_sweep(sweepable, space, workers=3)

        assert serial.best is not None and parallel.best is not None
        assert serial.best.params == parallel.best.params

    def test_each_trial_registers_its_own_version(self, sweepable):
        """Concurrent writers to the version table must not collide or overwrite."""
        from mlango.training.sweep import run_sweep

        _rows, sweepable = sweepable

        run_sweep(sweepable, {"C": [0.1, 0.5, 1.0, 2.0]}, workers=4)

        versions = sweepable.versions()
        assert len({v.version for v in versions}) == len(versions) == 4

    def test_a_failing_trial_does_not_take_the_pool_down(self, sweepable):
        """One raising thread would otherwise lose the trials already collected."""
        from mlango.training.sweep import run_sweep

        _rows, sweepable = sweepable

        original = sweepable.train
        calls = []

        def train(self, **kwargs):
            calls.append(self.C)
            if self.C == 0.5:
                raise RuntimeError("this corner of the space is bad")
            return original(self, **kwargs)

        sweepable.train = train
        try:
            result = run_sweep(sweepable, {"C": [0.1, 0.5, 1.0]}, workers=3)
        finally:
            sweepable.train = original

        statuses = {trial.index: trial.status for trial in result.trials}
        assert sum(1 for s in statuses.values() if s == "failed") == 1
        assert sum(1 for s in statuses.values() if s == "finished") == 2
        assert result.best is not None, "the good trials still produced a winner"

    def test_the_callback_fires_once_per_trial(self, sweepable):
        from mlango.training.sweep import run_sweep

        _rows, sweepable = sweepable

        seen = []
        run_sweep(
            sweepable, {"C": [0.1, 0.5, 1.0]}, workers=3, on_trial=lambda t, r: seen.append(t.index)
        )

        assert sorted(seen) == [1, 2, 3]

    def test_the_parent_run_records_a_point_per_trial(self, sweepable):
        from mlango.training.run import get_run, metric_history
        from mlango.training.sweep import run_sweep

        _rows, sweepable = sweepable

        result = run_sweep(sweepable, {"C": [0.1, 0.5, 1.0]}, workers=3)

        parent = get_run(result.run_uuid)
        points = metric_history(parent.id, result.metric)
        assert [step for step, _ in points] == [1, 2, 3], "in trial order, not completion order"
