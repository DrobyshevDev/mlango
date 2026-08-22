"""The OpenTelemetry bridge.

mlango keeps its own record in the metastore; this exists so a training run and
an agent's tool loop land in the same trace view as the HTTP request that
started them. It has to be invisible when the dependency is absent, silent when
the setting is off, and incapable of failing the work it describes.
"""

from __future__ import annotations

import pytest

from mlango.core import telemetry


@pytest.fixture(autouse=True)
def clean():
    telemetry.reset()
    yield
    telemetry.reset()


@pytest.fixture
def collected(project):
    """Turn telemetry on and capture the spans an in-memory exporter receives."""
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    export = pytest.importorskip("opentelemetry.sdk.trace.export")
    memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")

    from mlango.conf import settings

    exporter = memory.InMemorySpanExporter()
    provider = sdk.TracerProvider()
    provider.add_span_processor(export.SimpleSpanProcessor(exporter))

    # set_tracer_provider only takes once per process, so the provider is used
    # directly rather than installed globally — otherwise the second test in a
    # run would silently collect nothing.
    original = telemetry.get_tracer
    settings.TELEMETRY = {"ENABLED": True, "SERVICE_NAME": "mlango-tests"}
    telemetry.get_tracer = lambda: provider.get_tracer("mlango-tests")
    try:
        yield exporter
    finally:
        telemetry.get_tracer = original
        settings.TELEMETRY = {"ENABLED": False, "SERVICE_NAME": "mlango"}


def names(exporter):
    return [span.name for span in exporter.get_finished_spans()]


class TestWhenItIsOff:
    def test_nothing_is_emitted_by_default(self, project):
        assert telemetry.configured() is False
        assert telemetry.get_tracer() is None

    def test_a_span_is_a_no_op_that_still_yields(self, project):
        with telemetry.span("mlango.test", answer=42) as current:
            assert current is None

    def test_annotate_on_nothing_does_nothing(self, project):
        telemetry.annotate(None, answer=42)

    def test_an_error_inside_a_no_op_span_still_propagates(self, project):
        """Swallowing the caller's exception would be a bug factory."""
        with pytest.raises(ValueError, match="boom"):
            with telemetry.span("mlango.test"):
                raise ValueError("boom")


class TestWhenItIsOn:
    def test_a_span_reaches_the_exporter(self, collected):
        with telemetry.span("mlango.test", target="reviews.Sentiment"):
            pass

        assert names(collected) == ["mlango.test"]

    def test_attributes_are_namespaced(self, collected):
        """A trace view holds spans from a dozen libraries; ours must be findable."""
        with telemetry.span("mlango.test", target="reviews.Sentiment", step=3):
            pass

        attributes = collected.get_finished_spans()[0].attributes
        assert attributes["mlango.target"] == "reviews.Sentiment"
        assert attributes["mlango.step"] == 3

    def test_a_value_otel_cannot_hold_is_stringified_not_dropped(self, collected):
        with telemetry.span("mlango.test", config={"a": 1}):
            pass

        assert collected.get_finished_spans()[0].attributes["mlango.config"] == "{'a': 1}"

    def test_none_attributes_are_omitted(self, collected):
        with telemetry.span("mlango.test", missing=None, present="yes"):
            pass

        attributes = collected.get_finished_spans()[0].attributes
        assert "mlango.missing" not in attributes
        assert attributes["mlango.present"] == "yes"

    def test_annotate_adds_to_an_open_span(self, collected):
        with telemetry.span("mlango.test") as current:
            telemetry.annotate(current, output="hello")

        assert collected.get_finished_spans()[0].attributes["mlango.output"] == "hello"

    def test_an_exception_is_recorded_and_re_raised(self, collected):
        with pytest.raises(ValueError, match="boom"):
            with telemetry.span("mlango.test"):
                raise ValueError("boom")

        span = collected.get_finished_spans()[0]
        assert span.events, "the exception should be on the span"


class TestItNeverCostsTheWork:
    def test_a_training_run_emits_a_span(self, collected, reviews):
        pytest.importorskip("sklearn")

        from mlango.core import fields
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

        Guess.fit()

        spans = [s for s in collected.get_finished_spans() if s.name == "mlango.train"]
        assert spans, "a run is a unit of work with a start, an end and a status"
        assert spans[0].attributes["mlango.target"].endswith("Guess")
        assert spans[0].attributes["mlango.status"] == "finished"

    def test_a_failed_run_is_recorded_as_failed(self, collected, reviews):
        from mlango.core.exceptions import RunError
        from mlango.training import Model

        class Broken(Model):
            class Meta:
                dataset = reviews
                trainer = "sklearn"
                task = "classification"
                features = ["text"]

            def build(self):
                raise ValueError("no estimator here")

        with pytest.raises(RunError):
            Broken.fit()

        spans = [s for s in collected.get_finished_spans() if s.name == "mlango.train"]
        assert spans[0].attributes["mlango.status"] == "failed"

    def test_an_agent_emits_a_span_per_step(self, collected, isolated_registry):
        from mlango.agents import Agent

        class Helper(Agent):
            class Meta:
                system = "Be helpful."

        Helper().run("hello")

        emitted = names(collected)
        assert any(name.startswith("mlango.") for name in emitted)

    def test_a_broken_tracer_does_not_break_the_work(self, project):
        """Telemetry describes the work; it does not get to fail it."""

        class Exploding:
            def start_as_current_span(self, *args, **kwargs):
                raise RuntimeError("the collector is on fire")

        original = telemetry.get_tracer
        telemetry.get_tracer = lambda: Exploding()
        try:
            with telemetry.span("mlango.test") as current:
                assert current is None
        finally:
            telemetry.get_tracer = original


class TestWithoutThePackage:
    def test_a_missing_dependency_warns_once_and_stays_quiet(self, project, monkeypatch, caplog):
        import builtins
        import logging

        from mlango.conf import settings

        settings.TELEMETRY = {"ENABLED": True, "SERVICE_NAME": "mlango"}
        telemetry.reset()

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError("no opentelemetry here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        try:
            with caplog.at_level(logging.WARNING, logger="mlango.telemetry"):
                assert telemetry.get_tracer() is None
            assert "mlango[otel]" in caplog.text

            # Resolved once: a hot path must not retry a failing import per span.
            with telemetry.span("mlango.test") as current:
                assert current is None
        finally:
            settings.TELEMETRY = {"ENABLED": False, "SERVICE_NAME": "mlango"}
