"""``manage.py promote`` — the step the project's own pitch used to end before.

The framework could say what a new version broke and could promote one from
Python or the admin, but not from a terminal. `--check` closes the loop the
other way: compare with the incumbent first, and refuse a promotion that lost
rows.
"""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.data import Dataset, InMemorySource
from mlango.management.manager import load_command
from mlango.metastore.models import Stage
from mlango.training import Model

# Real label noise, so neither version is perfect and they can disagree in both
# directions — the only shape in which "broke" means anything.
ROWS = []
for _index in range(200):
    _truth = "pos" if _index % 2 else "neg"
    _words = "great fine warm" if _truth == "pos" else "awful dull bleak"
    _label = _truth if _index % 7 else ("neg" if _truth == "pos" else "pos")
    ROWS.append({"id": _index, "text": f"{_words} film {_index}", "label": _label})


def run(*argv: str) -> int:
    command = load_command("promote", "mlango.management.commands.promote")
    return command.run_from_argv(list(argv))


@pytest.fixture
def versions(project, isolated_registry):
    """Two registered versions of one model, neither of them perfect."""
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

    # v1 has a one-word vocabulary and predicts the majority class; v2 and v3
    # are both good and agree with each other, which is the third case the
    # check has to handle — a candidate that loses nothing.
    Classifier.fit(max_features=1)
    Classifier.fit(max_features=2)
    Classifier.fit(max_features=5000)
    return Classifier


class TestPromoting:
    def test_a_named_version_moves_to_production(self, versions, capsys):
        assert run(versions._meta.label, "1") == 0
        assert versions.production()._version.version == 1
        assert "production" in capsys.readouterr().out

    def test_the_newest_is_the_default(self, versions, capsys):
        """The version you just trained is the one you mean, almost always."""
        assert run(versions._meta.label) == 0
        assert versions.production()._version.version == 3

    def test_another_stage_can_be_named(self, versions, capsys):
        assert run(versions._meta.label, "1", "--stage", "staging") == 0
        assert [v.stage for v in versions.versions() if v.version == 1] == [Stage.STAGING]

    def test_promoting_demotes_the_incumbent(self, versions, capsys):
        run(versions._meta.label, "1")
        run(versions._meta.label, "2")

        stages = {v.version: v.stage for v in versions.versions()}
        assert stages == {3: Stage.NONE, 2: Stage.PRODUCTION, 1: Stage.ARCHIVED}

    def test_an_invented_stage_lists_the_real_ones(self, versions, capsys):
        assert run(versions._meta.label, "1", "--stage", "live") == 1
        assert "production" in capsys.readouterr().err

    def test_an_unknown_version_is_reported(self, versions, capsys):
        assert run(versions._meta.label, "99") == 1
        assert "99" in capsys.readouterr().err

    def test_something_that_has_no_versions_says_what_to_do(
        self, project, isolated_registry, capsys
    ):
        assert run("nope.Nothing") == 1
        err = capsys.readouterr().err
        assert "model or agent" in err

    def test_promoting_without_checking_suggests_checking(self, versions, capsys):
        """Said once and quietly — the comparison is the point of the framework."""
        run(versions._meta.label, "1")
        assert "--check" in capsys.readouterr().out


class TestTheCheck:
    def test_an_empty_stage_has_nothing_to_compare_with(self, versions, capsys):
        assert run(versions._meta.label, "1", "--check") == 0
        assert "nothing to compare" in capsys.readouterr().out

    def test_a_candidate_that_lost_rows_is_refused(self, versions, capsys):
        """The whole argument for having the comparison at all.

        v3 fixes far more than it breaks, and the strict rule still refuses it:
        that rule is for a curated suite where nothing may be lost.
        """
        run(versions._meta.label, "1")  # the weak one holds production

        assert run(versions._meta.label, "3", "--check") == 1
        err = capsys.readouterr().err
        assert "Refusing to promote" in err
        assert "--show-changes" in err, "it should say how to look at them"
        assert versions.production()._version.version == 1, "the incumbent must not have moved"

    def test_a_candidate_that_lost_nothing_goes_through(self, versions, capsys):
        """v2 and v3 agree on every row, so there is nothing to refuse."""
        run(versions._meta.label, "2")

        assert run(versions._meta.label, "3", "--check") == 0
        assert versions.production()._version.version == 3
        assert "keeps everything" in capsys.readouterr().out

    def test_significant_mode_allows_a_loss_the_evidence_cannot_call(self, versions, capsys):
        """Fixing many and losing few is a promotion, not a regression."""
        run(versions._meta.label, "1")

        assert run(versions._meta.label, "3", "--check", "significant") == 0
        assert versions.production()._version.version == 3
        assert "improvement" in capsys.readouterr().out

    def test_significant_mode_still_refuses_a_real_regression(self, versions, capsys):
        """Going back to the weak version loses far more than it gains."""
        run(versions._meta.label, "3")

        assert run(versions._meta.label, "1", "--check", "significant") == 1
        assert "Refusing to promote" in capsys.readouterr().err
        assert versions.production()._version.version == 3

    def test_promoting_over_itself_is_refused(self, versions, capsys):
        run(versions._meta.label, "2")

        assert run(versions._meta.label, "2", "--check") == 1
        assert "already holds" in capsys.readouterr().err

    def test_an_agent_is_pointed_at_the_eval_diff(self, project, isolated_registry, capsys):
        """--check compares predictions, and an agent does not make those."""
        from mlango.agents import Agent

        class Helper(Agent):
            class Meta:
                system = "Be helpful."

        Helper.register_version()

        assert run(Helper._meta.label, "1", "--check") == 1
        assert "diff --eval" in capsys.readouterr().err


class TestAgents:
    def test_an_agent_version_promotes_through_the_same_verb(
        self, project, isolated_registry, capsys
    ):
        """One command for both families; remembering which owns which is a seam."""
        from mlango.agents import Agent

        class Helper(Agent):
            class Meta:
                system = "Be helpful."

        Helper.register_version()

        assert run(Helper._meta.label, "1") == 0
        assert Helper.production().version == 1

    def test_it_says_what_a_version_cannot_pin(self, project, isolated_registry, capsys):
        from mlango.agents import Agent

        class Helper(Agent):
            class Meta:
                system = "Be helpful."

        Helper.register_version()
        run(Helper._meta.label, "1")

        assert "tools come from" in capsys.readouterr().out
