"""Comparing models mlango did not train.

The door: a team with two pickled models and a CSV can get the one answer no
registry gives them — which rows the new one lost — without adopting anything
first. So the failure modes worth testing are the ones a stranger hits in their
first five minutes, and every message they see has to name the fix.
"""

from __future__ import annotations

import pickle
import sys

import pytest

from mlango.core.exceptions import MlangoError
from mlango.training.external import columns_for, installed_loaders, load_predictor


class AlwaysSays:
    """A predictor small enough to pickle and obvious enough to assert on."""

    def __init__(self, answer):
        self.answer = answer

    def predict(self, inputs):
        return [self.answer for _ in inputs]


class NotAPredictor:
    pass


def dump(obj, path):
    with path.open("wb") as handle:
        pickle.dump(obj, handle)
    return path


class TestLoadingAnArtefact:
    def test_a_plain_path_loads(self, tmp_path):
        # The common case: somebody's saved model, no scheme, no registry.
        path = dump(AlwaysSays("yes"), tmp_path / "model.pkl")
        assert load_predictor(str(path)).predict([1, 2]) == ["yes", "yes"]

    def test_a_file_scheme_loads(self, tmp_path):
        path = dump(AlwaysSays("no"), tmp_path / "model.pkl")
        assert load_predictor(f"file:{path}").predict([1]) == ["no"]

    @pytest.mark.skipif(sys.platform != "win32", reason="drive letters are a Windows thing")
    def test_a_windows_drive_letter_is_not_a_scheme(self, tmp_path):
        # `C:\...` would otherwise be read as scheme "C".
        path = dump(AlwaysSays("yes"), tmp_path / "model.pkl")
        assert load_predictor(str(path.resolve())).predict([1]) == ["yes"]

    def test_a_missing_file_says_what_to_pass(self, tmp_path):
        with pytest.raises(MlangoError) as caught:
            load_predictor(str(tmp_path / "nope.pkl"))
        assert "No such file" in str(caught.value)

    def test_something_that_cannot_predict_is_refused(self, tmp_path):
        path = dump(NotAPredictor(), tmp_path / "thing.pkl")
        with pytest.raises(MlangoError) as caught:
            load_predictor(str(path))
        assert "no predict()" in str(caught.value)
        assert "NotAPredictor" in str(caught.value)

    def test_an_unknown_scheme_names_the_ones_that_exist(self):
        with pytest.raises(MlangoError) as caught:
            load_predictor("mlflow:models:/Sentiment/3")
        message = str(caught.value)
        assert "No loader for 'mlflow'" in message
        # And how to make one, since the answer is "install a package".
        assert "entry-point group" in message

    def test_a_registered_scheme_is_used(self, tmp_path, monkeypatch):
        # Written to a real module on the path, so the import that a third-party
        # loader would go through is the one exercised here.
        (tmp_path / "toyloader.py").write_text(
            "class Toy:\n"
            "    def __init__(self, reference):\n"
            "        self.reference = reference\n"
            "    def predict(self, inputs):\n"
            "        return [self.reference for _ in inputs]\n"
            "\n"
            "def load(reference):\n"
            "    return Toy(reference)\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        import mlango.training.external as external

        monkeypatch.setattr(external, "discover", lambda group: {"toy": "toyloader:load"})
        assert load_predictor("toy:abc").predict([1, 2]) == ["abc", "abc"]

    def test_the_reference_after_the_scheme_reaches_the_loader(self, tmp_path, monkeypatch):
        # `mlflow:models:/Name/3` must hand over `models:/Name/3`, colons and all.
        seen = []

        import mlango.training.external as external

        monkeypatch.setattr(external, "discover", lambda group: {"probe": "probe:load"})
        (tmp_path / "probe.py").write_text(
            "REFERENCES = []\n"
            "class P:\n"
            "    def predict(self, inputs):\n"
            "        return list(inputs)\n"
            "def load(reference):\n"
            "    REFERENCES.append(reference)\n"
            "    return P()\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        load_predictor("probe:models:/Sentiment/3")
        import probe

        seen = probe.REFERENCES
        assert seen == ["models:/Sentiment/3"]

    def test_installed_loaders_reports_what_is_registered(self):
        # Nothing ships one, so the honest answer out of the box is nothing.
        assert isinstance(installed_loaders(), dict)


class Field:
    def __init__(self, name, is_target=False):
        self.name = name
        self.is_target = is_target


class Meta:
    def __init__(self, fields, extras=None):
        self.fields = fields
        self.extras = extras or {}
        self.label = "demo.Rows"

    @property
    def target_fields(self):
        return [f for f in self.fields if f.is_target]

    def has_field(self, name):
        return any(f.name == name for f in self.fields)


class FakeDataset:
    def __init__(self, fields, extras=None):
        self._meta = Meta(fields, extras)


class TestWorkingOutTheColumns:
    def test_the_declared_target_is_used(self):
        dataset = FakeDataset([Field("text"), Field("label", is_target=True)])
        features, target = columns_for(dataset)
        assert target == "label"
        assert features == ["text"]

    def test_the_primary_key_is_not_a_feature(self):
        # Scoring on an id is the same leak it would be during training.
        dataset = FakeDataset(
            [Field("id"), Field("text"), Field("label", is_target=True)],
            extras={"primary_key": "id"},
        )
        features, _ = columns_for(dataset)
        assert features == ["text"]

    def test_an_explicit_target_wins(self):
        dataset = FakeDataset([Field("a"), Field("b"), Field("label", is_target=True)])
        features, target = columns_for(dataset, target="b")
        assert target == "b"
        assert "b" not in features

    def test_explicit_features_win(self):
        dataset = FakeDataset([Field("a"), Field("b"), Field("label", is_target=True)])
        features, _ = columns_for(dataset, features=["a"])
        assert features == ["a"]

    def test_a_feature_the_dataset_does_not_have_is_refused(self):
        dataset = FakeDataset([Field("a"), Field("label", is_target=True)])
        with pytest.raises(MlangoError) as caught:
            columns_for(dataset, features=["nope"])
        assert "no field(s) nope" in str(caught.value)

    def test_two_targets_ask_which_one(self):
        dataset = FakeDataset([Field("x"), Field("a", is_target=True), Field("b", is_target=True)])
        with pytest.raises(MlangoError) as caught:
            columns_for(dataset)
        assert "--target" in str(caught.value)

    def test_no_targets_asks_the_same_thing(self):
        dataset = FakeDataset([Field("x"), Field("y")])
        with pytest.raises(MlangoError) as caught:
            columns_for(dataset)
        assert "--target" in str(caught.value)

    def test_nothing_left_to_feed_the_model_says_so(self):
        dataset = FakeDataset([Field("label", is_target=True)])
        with pytest.raises(MlangoError) as caught:
            columns_for(dataset)
        assert "--features" in str(caught.value)
