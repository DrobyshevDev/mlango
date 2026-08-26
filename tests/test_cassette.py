"""Recording an agent's model calls, and replaying them.

The point of a cassette is that a test with an agent in it stops being slow,
paid for, and different every time. So the guarantees worth holding are about
fidelity: what comes back on replay has to be what the provider said, including
the parts the loop feeds straight back to the API.
"""

from __future__ import annotations

import json

import pytest

from mlango.agents.cassette import (
    FORMAT_VERSION,
    Cassette,
    CassetteError,
    RecordingProvider,
    ReplayProvider,
    request_key,
)
from mlango.agents.providers.base import Completion, Provider, ToolCall, Usage


class Scripted(Provider):
    """A provider that says what it was told to, and counts being asked."""

    name = "scripted"
    offline = True

    def __init__(self, *completions: Completion) -> None:
        super().__init__()
        self.completions = list(completions)
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.completions[len(self.calls) - 1]


class Block:
    """Stands in for an SDK content block: an object, not a dictionary."""

    def __init__(self, text):
        self.text = text

    def model_dump(self):
        return {"type": "text", "text": self.text}


def completion(text="hi", **kwargs):
    return Completion(
        text=text, model="test-model", usage=Usage(**kwargs.pop("usage", {})), **kwargs
    )


def call(**overrides):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "system": "be brief",
        "tools": [],
        "max_tokens": 1024,
    }
    payload.update(overrides)
    return payload


class TestTheKey:
    def test_the_same_request_has_the_same_key(self):
        assert request_key(call()) == request_key(call())

    def test_a_different_request_has_a_different_key(self):
        assert request_key(call()) != request_key(call(system="be verbose"))

    def test_key_order_does_not_matter(self):
        # A refactor that builds the same request in another order must not
        # invalidate every cassette in the repository.
        one = {"model": "m", "system": "s", "messages": []}
        other = {"messages": [], "system": "s", "model": "m"}
        assert request_key(one) == request_key(other)

    def test_objects_in_the_request_do_not_break_it(self):
        assert request_key(call(tools=[Block("x")])) == request_key(call(tools=[Block("x")]))


class TestRecording:
    def test_it_passes_the_call_through(self, tmp_path):
        inner = Scripted(completion("from the model"))
        recorder = RecordingProvider(inner, tmp_path / "c.json")
        assert recorder.complete(**call()).text == "from the model"
        assert len(inner.calls) == 1

    def test_it_writes_after_every_call_not_at_the_end(self, tmp_path):
        # A run that dies on step three has still recorded steps one and two,
        # which is usually the half worth having.
        path = tmp_path / "c.json"
        recorder = RecordingProvider(Scripted(completion("a"), completion("b")), path)
        recorder.complete(**call())
        assert len(Cassette.load(path)) == 1
        recorder.complete(**call(system="second"))
        assert len(Cassette.load(path)) == 2

    def test_sdk_objects_survive_as_dictionaries(self, tmp_path):
        # Provider-native content is objects; every API that hands them over
        # also takes them back as plain dictionaries.
        path = tmp_path / "c.json"
        recorder = RecordingProvider(Scripted(completion(raw_content=[Block("hello")])), path)
        recorder.complete(**call())
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["interactions"][0]["response"]["raw_content"] == [
            {"type": "text", "text": "hello"}
        ]

    def test_it_reports_what_it_is_wrapping(self, tmp_path):
        recorder = RecordingProvider(Scripted(completion()), tmp_path / "c.json")
        assert recorder.describe()["inner"] == "scripted"


class TestReplay:
    def record(self, tmp_path, *completions, calls=None):
        path = tmp_path / "c.json"
        recorder = RecordingProvider(Scripted(*completions), path)
        for payload in calls or [call()]:
            recorder.complete(**payload)
        return path

    def test_it_answers_without_the_model(self, tmp_path):
        path = self.record(tmp_path, completion("recorded answer"))
        assert ReplayProvider(path).complete(**call()).text == "recorded answer"

    def test_it_never_reaches_the_network(self, tmp_path):
        assert ReplayProvider(self.record(tmp_path, completion())).offline is True

    def test_tool_calls_come_back_whole(self, tmp_path):
        original = Completion(
            text="",
            tool_calls=[ToolCall(id="t1", name="lookup", arguments={"q": "rain"})],
            stop_reason=Completion.TOOL_USE,
        )
        replayed = ReplayProvider(self.record(tmp_path, original)).complete(**call())
        assert replayed.wants_tools
        assert replayed.tool_calls[0].name == "lookup"
        assert replayed.tool_calls[0].arguments == {"q": "rain"}
        assert replayed.stop_reason == Completion.TOOL_USE

    def test_the_turn_the_loop_echoes_back_survives(self, tmp_path):
        # This is the field glia's format has nowhere to put, and the reason
        # mlango records its own. Losing it makes a replay that passes here and
        # fails against the real API.
        path = self.record(tmp_path, completion(raw_content=[Block("echoed")]))
        assert ReplayProvider(path).complete(**call()).raw_content == [
            {"type": "text", "text": "echoed"}
        ]

    def test_usage_survives(self, tmp_path):
        path = self.record(tmp_path, completion(usage={"input_tokens": 11, "output_tokens": 7}))
        usage = ReplayProvider(path).complete(**call()).usage
        assert usage.input_tokens == 11
        assert usage.total_tokens == 18

    def test_a_recorded_call_is_matched_by_content_not_by_order(self, tmp_path):
        first, second = call(system="one"), call(system="two")
        path = self.record(
            tmp_path, completion("answer one"), completion("answer two"), calls=[first, second]
        )
        # Asked in the other order, each still gets its own answer.
        player = ReplayProvider(path)
        assert player.complete(**second).text == "answer two"
        assert player.complete(**first).text == "answer one"

    def test_a_recording_is_used_once(self, tmp_path):
        path = self.record(
            tmp_path, completion("first"), completion("second"), calls=[call(), call()]
        )
        player = ReplayProvider(path)
        assert player.complete(**call()).text == "first"
        assert player.complete(**call()).text == "second"

    def test_an_unrecorded_call_falls_back_to_order(self, tmp_path):
        # An edited prompt that did not change what was asked should not force a
        # re-record.
        path = self.record(tmp_path, completion("recorded"))
        assert ReplayProvider(path).complete(**call(system="reworded")).text == "recorded"

    def test_strict_refuses_to_guess(self, tmp_path):
        path = self.record(tmp_path, completion("recorded"))
        with pytest.raises(CassetteError) as caught:
            ReplayProvider(path, strict=True).complete(**call(system="reworded"))
        assert "will not guess" in str(caught.value)

    def test_running_past_the_end_says_so(self, tmp_path):
        path = self.record(tmp_path, completion("only one"))
        player = ReplayProvider(path)
        player.complete(**call())
        with pytest.raises(CassetteError) as caught:
            player.complete(**call())
        assert "Re-record" in str(caught.value)


class TestTheFileItself:
    def test_a_missing_cassette_says_how_to_make_one(self, tmp_path):
        with pytest.raises(CassetteError) as caught:
            Cassette.load(tmp_path / "nope.json")
        assert "RecordingProvider" in str(caught.value)

    def test_broken_json_says_so(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("{ not json", encoding="utf-8")
        with pytest.raises(CassetteError) as caught:
            Cassette.load(path)
        assert "not valid JSON" in str(caught.value)

    def test_a_future_format_is_refused_rather_than_misread(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"version": 99, "interactions": []}), encoding="utf-8")
        with pytest.raises(CassetteError) as caught:
            Cassette.load(path)
        assert "version 99" in str(caught.value)

    def test_it_round_trips(self, tmp_path):
        cassette = Cassette()
        cassette.append("k", call(), completion("saved"))
        path = tmp_path / "c.json"
        cassette.save(path)
        assert len(Cassette.load(path)) == 1
        assert Cassette.load(path).to_dict()["version"] == FORMAT_VERSION

    def test_it_creates_the_directory(self, tmp_path):
        Cassette().save(tmp_path / "deep" / "nested" / "c.json")
        assert (tmp_path / "deep" / "nested" / "c.json").is_file()


class TestAThoroughlyRealRun:
    """The guarantee the rest of this file is in aid of.

    A recorded agent run replays without a provider, and produces the same
    answer through the same loop. If this passes and everything above fails,
    the cassette still works; if this fails, none of it matters.
    """

    def agent(self):
        from mlango.agents import Agent

        class Recorded(Agent):
            class Meta:
                system = "You are under test."
                max_steps = 3
                tracing = False

        return Recorded

    def test_a_run_records_and_replays_to_the_same_answer(self, project, tmp_path):
        from mlango.agents.providers.base import get_provider

        path = tmp_path / "run.json"
        agent = self.agent()

        live = agent().run("hello", provider=RecordingProvider(get_provider(), path))
        assert live.ok
        assert path.is_file()

        replayed = agent().run("hello", provider=ReplayProvider(path))
        assert replayed.output == live.output
        assert replayed.steps == live.steps
        assert replayed.ok

    def test_replaying_needs_no_provider_of_its_own(self, project, tmp_path):
        # The whole point: this run is offline, free, and the same every time.
        from mlango.agents.providers.base import get_provider

        path = tmp_path / "run.json"
        agent = self.agent()
        agent().run("hello", provider=RecordingProvider(get_provider(), path))

        player = ReplayProvider(path)
        assert player.offline is True
        assert agent().run("hello", provider=player).ok

    def test_the_declared_provider_is_untouched_by_the_override(self, project, tmp_path):
        # An override must not leak into the next run, which is why it is an
        # argument rather than a setting.
        from mlango.agents.providers.base import get_provider

        agent = self.agent()
        agent().run("hello", provider=RecordingProvider(get_provider(), tmp_path / "r.json"))
        assert agent.get_provider().name != "recording"

    def test_streaming_replays_too(self, project, tmp_path):
        # run() and stream() share one loop with one call to the provider, so a
        # cassette covers both without knowing there are two entry points.
        from mlango.agents.events import Finished
        from mlango.agents.providers.base import get_provider

        path = tmp_path / "run.json"
        agent = self.agent()
        recorded = agent().run("hello", provider=RecordingProvider(get_provider(), path))

        events = list(agent().stream("hello", provider=ReplayProvider(path)))
        finished = [event for event in events if isinstance(event, Finished)]
        assert finished, "the stream ended without a Finished event"
        assert finished[-1].result.output == recorded.output
