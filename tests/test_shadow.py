"""Shadow deployment: a candidate answering the same requests as production.

A dataset says how a candidate does on rows you curated. A shadow says what it
would have told the people who actually asked, which is the question a
promotion is really about — and the only one available before labels exist.
"""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.data import Dataset, InMemorySource
from mlango.metastore.models import Stage
from mlango.serve import path
from mlango.training import Model

ROWS = [
    {
        "id": index,
        "text": ("great movie " if index % 2 else "terrible movie ") + str(index),
        "label": "pos" if index % 2 else "neg",
    }
    for index in range(60)
]


@pytest.fixture
def served(project, isolated_registry):
    """A good version in production and a deliberately weak one in staging."""
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

    good = Classifier.fit(max_features=5000)._version.version
    weak = Classifier.fit(max_features=1)._version.version
    Classifier.promote(good, Stage.PRODUCTION)
    Classifier.promote(weak, Stage.STAGING)
    return Classifier, good, weak


@pytest.fixture
def client(served):
    from fastapi.testclient import TestClient

    from mlango.serve.api import create_app

    classifier, _good, _weak = served
    routes = [path("predict/", classifier.as_endpoint(stage=Stage.PRODUCTION))]
    with TestClient(create_app(include_admin=False, routes=routes)) as test_client:
        yield test_client


def turn_on(shadow=True, log=True):
    from mlango.conf import settings

    settings.PREDICTION_LOG = {"ENABLED": log, "SAMPLE": 1.0, "MAX_ROWS": 0}
    settings.SHADOW = {"ENABLED": shadow, "STAGE": Stage.STAGING, "SAMPLE": 1.0}


def logged(label):
    import sqlalchemy as sa

    from mlango.metastore.models import Prediction
    from mlango.metastore.session import session_scope

    with session_scope() as session:
        return list(
            session.execute(sa.select(Prediction).where(Prediction.label == label)).scalars()
        )


class TestServingIsUnaffected:
    def test_the_response_comes_from_production_not_the_shadow(self, client, served):
        turn_on()
        classifier, good, _weak = served

        body = client.post("/api/predict/", json={"input": "great movie"}).json()

        assert body["version"] == good, "the candidate must not answer the caller"
        assert body["predictions"] == ["pos"]

    def test_a_failing_candidate_does_not_fail_the_request(self, client, served):
        """A shadow informs a promotion; it must not be able to cause an outage."""
        turn_on()
        classifier, good, _weak = served

        # The candidate itself breaks — not the code that is supposed to
        # contain the breakage, which would test nothing.
        original = classifier.predict
        broken_calls = []

        def predict(self, inputs):
            if getattr(self._version, "stage", None) == Stage.STAGING:
                broken_calls.append(1)
                raise RuntimeError("the candidate is broken")
            return original(self, inputs)

        classifier.predict = predict
        try:
            response = client.post("/api/predict/", json={"input": "great movie"})
        finally:
            classifier.predict = original

        assert response.status_code == 200
        assert response.json()["version"] == good
        assert broken_calls, "the shadow really was attempted and really did fail"

    def test_nothing_shadows_when_it_is_switched_off(self, client, served):
        turn_on(shadow=False)
        classifier, good, _weak = served

        client.post("/api/predict/", json={"input": "great movie"})

        versions = {row.version for row in logged(classifier._meta.label)}
        assert versions == {good}


class TestWhatGetsLogged:
    def test_both_versions_answer_and_both_are_recorded(self, client, served):
        turn_on()
        classifier, good, weak = served

        client.post("/api/predict/", json={"input": "great movie"})

        rows = logged(classifier._meta.label)
        assert {row.version for row in rows} == {good, weak}

    def test_the_pair_shares_one_request_id(self, client, served):
        """Matching on inputs instead would fuse two callers asking the same thing."""
        turn_on()
        classifier, _good, _weak = served

        client.post("/api/predict/", json={"input": "great movie"})

        rows = logged(classifier._meta.label)
        ids = {row.request_id for row in rows}
        assert len(ids) == 1
        assert None not in ids

    def test_separate_requests_get_separate_ids(self, client, served):
        turn_on()
        classifier, _good, _weak = served

        client.post("/api/predict/", json={"input": "great movie"})
        client.post("/api/predict/", json={"input": "terrible movie"})

        rows = logged(classifier._meta.label)
        assert len({row.request_id for row in rows}) == 2

    def test_a_batch_shares_the_id_of_its_request(self, client, served):
        turn_on()
        classifier, _good, _weak = served

        client.post("/api/predict/", json={"inputs": ["great movie", "terrible movie"]})

        rows = logged(classifier._meta.label)
        assert len(rows) == 4, "two inputs, two versions"
        assert len({row.request_id for row in rows}) == 1

    def test_sampling_at_zero_shadows_nothing(self, client, served):
        from mlango.conf import settings

        turn_on()
        settings.SHADOW = {"ENABLED": True, "STAGE": Stage.STAGING, "SAMPLE": 0.0}
        classifier, good, _weak = served

        client.post("/api/predict/", json={"input": "great movie"})

        assert {row.version for row in logged(classifier._meta.label)} == {good}


class TestComparingFromTheLog:
    def test_two_versions_are_compared_on_what_they_answered(self, client, served):
        from mlango.training.comparison import compare_from_log

        turn_on()
        classifier, good, weak = served
        for text in ("great movie", "terrible movie", "great movie again"):
            client.post("/api/predict/", json={"input": text})

        report = compare_from_log(classifier, good, weak)

        assert report["rows"] == 3
        assert report["source"] == "log"
        assert report["dataset"] == "the prediction log"
        assert report["changed"] > 0, "a one-word vocabulary disagrees with a real one"

    def test_production_traffic_carries_no_truth_and_says_so(self, client, served):
        """The absence of labels is the reason a shadow is worth running."""
        from mlango.training.comparison import compare_from_log

        turn_on()
        classifier, good, weak = served
        client.post("/api/predict/", json={"input": "great movie"})

        report = compare_from_log(classifier, good, weak)
        assert report["labelled"] is False
        assert "metrics" not in report
        assert "fixed" not in report

    def test_changed_requests_show_both_answers(self, client, served):
        from mlango.training.comparison import compare_from_log

        turn_on()
        classifier, good, weak = served
        for text in ("great movie", "wonderful film", "terrible movie"):
            client.post("/api/predict/", json={"input": text})

        report = compare_from_log(classifier, good, weak, max_changes=5)

        assert report["changes"]
        row = report["changes"][0]
        assert row["left"] != row["right"]
        assert row["input"]

    def test_no_shared_request_explains_what_to_turn_on(self, client, served):
        from mlango.training.comparison import compare_from_log

        turn_on(shadow=False)
        classifier, good, weak = served
        client.post("/api/predict/", json={"input": "great movie"})

        with pytest.raises(LookupError, match="SHADOW"):
            compare_from_log(classifier, good, weak)


class TestNotShadowingItself:
    def test_a_candidate_equal_to_the_served_version_is_skipped(self, project, served):
        """Serving "latest" often *is* the staging version; that pair is not a comparison."""
        from fastapi.testclient import TestClient

        from mlango.serve.api import create_app

        turn_on()
        classifier, _good, weak = served

        # No stage: this resolves to the newest version, which is the candidate.
        routes = [path("predict/", classifier.as_endpoint())]
        with TestClient(create_app(include_admin=False, routes=routes)) as client:
            body = client.post("/api/predict/", json={"input": "great movie"}).json()

        assert body["version"] == weak
        rows = logged(classifier._meta.label)
        assert {row.version for row in rows} == {weak}
        assert len(rows) == 1, "one row, because there was nothing to shadow with"
