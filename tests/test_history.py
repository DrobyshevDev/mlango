"""The promotion log — what moved, when, and on the strength of what.

``stage`` on a version row is a mutable column: promoting v3 overwrites what v2
was. So a registry that has been in use for a year could say what is live and
nothing about how it got there, which is the question a post-mortem opens with.
"""

from __future__ import annotations

import datetime as dt

import pytest

from mlango.core import fields
from mlango.data import Dataset, InMemorySource
from mlango.management.manager import load_command
from mlango.metastore.history import actor_name, history, stage_at
from mlango.metastore.models import Stage
from mlango.training import Model

ROWS = [
    {"id": n, "text": "great warm" if n % 2 else "awful dull", "label": "pos" if n % 2 else "neg"}
    for n in range(40)
]


def run(*argv: str) -> int:
    command = load_command("promote", "mlango.management.commands.promote")
    return command.run_from_argv(list(argv))


@pytest.fixture
def versions(project, isolated_registry):
    pytest.importorskip("sklearn")

    class Rows(Dataset):
        id = fields.IntegerField()
        text = fields.TextField()
        label = fields.LabelField(["neg", "pos"])

        class Meta:
            source = InMemorySource(ROWS)
            primary_key = "id"

    class Classifier(Model):
        max_features = fields.IntegerField(default=5000)

        class Meta:
            dataset = Rows
            trainer = "sklearn"
            task = "classification"
            features = ["text"]

        def build(self):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline

            return make_pipeline(
                TfidfVectorizer(max_features=self.max_features),
                LogisticRegression(max_iter=500),
            )

    Classifier.fit(max_features=2)
    Classifier.fit(max_features=5000)
    return Classifier


class TestWhatGetsRecorded:
    def test_a_promotion_leaves_a_row(self, versions):
        versions.promote(1)

        moves = history(versions._meta.label)
        assert len(moves) == 1
        assert (moves[0].version, moves[0].from_stage, moves[0].to_stage) == (
            1,
            "none",
            "production",
        )

    def test_the_demotion_it_causes_is_recorded_too(self, versions):
        """Otherwise the log is a list of winners, not a history."""
        versions.promote(1)
        versions.promote(2)

        moves = history(versions._meta.label)
        assert [(m.version, m.to_stage) for m in moves] == [
            (2, Stage.PRODUCTION),
            (1, Stage.ARCHIVED),
            (1, Stage.PRODUCTION),
        ]

    def test_a_demotion_says_what_displaced_it(self, versions):
        """A demotion is a consequence; the useful fact is what caused it."""
        versions.promote(1)
        versions.promote(2)

        demotion = next(m for m in history(versions._meta.label) if m.to_stage == Stage.ARCHIVED)
        assert demotion.evidence == {"superseded_by": 2}

    def test_promoting_to_the_stage_it_already_holds_records_nothing(self, versions):
        """It did not happen, so the log should not claim it did."""
        versions.promote(1)
        versions.promote(1)

        assert len(history(versions._meta.label)) == 1

    def test_evidence_is_kept_beside_the_move(self, versions):
        verdict = {"fixed": 9, "broke": 2, "verdict": "a real improvement"}
        versions.promote(2, evidence=verdict, notes="ship it")

        move = history(versions._meta.label)[0]
        assert move.evidence == verdict
        assert move.notes == "ship it"

    def test_an_unchecked_promotion_records_no_evidence(self, versions):
        """Null means nobody looked, which is worth being able to see."""
        versions.promote(1)
        assert history(versions._meta.label)[0].evidence is None

    def test_the_actor_is_recorded(self, versions, monkeypatch):
        monkeypatch.setenv("MLANGO_ACTOR", "ci-bot")
        versions.promote(1)
        assert history(versions._meta.label)[0].actor == "ci-bot"

    def test_the_log_lands_in_the_same_transaction(self, versions):
        """A history that can disagree with the stage column is worse than none."""
        versions.promote(2)

        live = next(v for v in versions.versions() if v.stage == Stage.PRODUCTION)
        assert history(versions._meta.label)[0].version == live.version


class TestActorName:
    def test_the_environment_wins(self, monkeypatch):
        monkeypatch.setenv("MLANGO_ACTOR", "deploy@ci")
        assert actor_name() == "deploy@ci"

    def test_a_blank_override_falls_through(self, monkeypatch):
        monkeypatch.setenv("MLANGO_ACTOR", "   ")
        assert actor_name() != "   "

    def test_it_never_raises(self, monkeypatch):
        """A promotion must not fail because the container has no passwd entry."""
        import getpass

        monkeypatch.delenv("MLANGO_ACTOR", raising=False)
        monkeypatch.setattr(getpass, "getuser", lambda: (_ for _ in ()).throw(OSError("no user")))
        assert actor_name() == ""


class TestReadingItBack:
    def test_newest_first(self, versions):
        versions.promote(1)
        versions.promote(2)
        assert history(versions._meta.label)[0].version == 2

    def test_the_whole_registry_is_the_default(self, versions):
        """Nobody asks what happened to one model before learning something did."""
        versions.promote(1)
        assert len(history()) == 1

    def test_it_can_be_narrowed_to_a_family(self, versions):
        versions.promote(1)
        assert history(kind="model")
        assert not history(kind="agent")

    def test_it_can_be_narrowed_to_a_stage(self, versions):
        versions.promote(1)
        versions.promote(2)
        assert [m.version for m in history(stage=Stage.PRODUCTION)] == [2, 1]

    def test_the_limit_is_honoured(self, versions):
        versions.promote(1)
        versions.promote(2)
        assert len(history(limit=1)) == 1


class TestWhatWasLiveThen:
    def test_it_replays_the_log(self, versions):
        """The version rows only know about now; the log knows about then."""
        versions.promote(1)
        first = history(versions._meta.label)[0].at

        versions.promote(2)
        later = history(versions._meta.label)[0].at

        assert stage_at(versions._meta.label, first) == 1
        assert stage_at(versions._meta.label, later) == 2

    def test_before_anything_was_promoted_nothing_held_it(self, versions):
        versions.promote(1)
        long_ago = dt.datetime(2020, 1, 1)
        assert stage_at(versions._meta.label, long_ago) is None

    def test_archiving_the_holder_empties_the_stage(self, versions):
        versions.promote(1)
        versions.promote(1, Stage.ARCHIVED)

        now = history(versions._meta.label)[0].at
        assert stage_at(versions._meta.label, now) is None


class TestTheCommand:
    def test_history_lists_the_moves(self, versions, capsys):
        run(versions._meta.label, "1")
        run(versions._meta.label, "2")
        capsys.readouterr()

        assert run(versions._meta.label, "--history") == 0
        out = capsys.readouterr().out
        assert "none → production" in out
        assert "superseded by v1" not in out, "v1 was displaced by v2, not the reverse"
        assert "superseded by v2" in out

    def test_history_without_a_label_covers_everything(self, versions, capsys):
        run(versions._meta.label, "1")
        capsys.readouterr()

        assert run("--history") == 0
        out = capsys.readouterr().out
        assert versions._meta.label in out, "the label becomes a column when it is not the heading"

    def test_an_empty_log_says_so(self, versions, capsys):
        assert run("--history") == 0
        assert "Nothing has been promoted" in capsys.readouterr().out

    def test_a_check_writes_its_verdict_into_the_log(self, versions, capsys):
        """The whole point: what a promotion was decided on, kept beside it."""
        run(versions._meta.label, "1")
        run(versions._meta.label, "2", "--check", "significant")

        move = history(versions._meta.label)[0]
        assert move.evidence is not None
        assert move.evidence["against"] == 1
        assert move.evidence["mode"] == "significant"
        assert move.evidence["metric"] == "accuracy"

    def test_notes_are_recorded(self, versions, capsys):
        run(versions._meta.label, "1", "--notes", "quarterly refresh")
        assert history(versions._meta.label)[0].notes == "quarterly refresh"

    def test_a_promotion_nobody_checked_says_so(self, versions, capsys):
        run(versions._meta.label, "1")
        capsys.readouterr()

        run(versions._meta.label, "--history")
        assert "not checked" in capsys.readouterr().out

    def test_no_label_and_no_history_is_an_error_that_says_what_to_do(self, versions, capsys):
        assert run() == 1
        assert "--history" in capsys.readouterr().err


class TestAgents:
    def test_an_agent_promotion_lands_in_the_same_log(self, project, isolated_registry):
        """One question — what changed last week — so one table."""
        from mlango.agents import Agent

        class Helper(Agent):
            class Meta:
                system = "Be helpful."

        Helper.register_version()
        Helper.promote(1)

        moves = history(Helper._meta.label)
        assert len(moves) == 1
        assert moves[0].kind == "agent"
