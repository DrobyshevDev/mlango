"""An agent's version registry.

A model version is an artifact; an agent version is a declaration. That makes
the registry cheaper — the row *is* the version — and narrower, because a
prompt can be recorded and a tool's implementation cannot.
"""

from __future__ import annotations

import pytest

from mlango.agents import Agent, tool
from mlango.metastore.models import Stage


@tool
def lookup(query: str) -> str:
    """Look something up.

    Args:
        query: What to look for.
    """
    return f"result for {query}"


@pytest.fixture
def agent_class(project, isolated_registry):
    class Helper(Agent):
        """An agent whose declaration we edit between assertions."""

        class Meta:
            system = "Be helpful."
            model = "claude-opus-5"
            max_steps = 4
            tools = [lookup]

    return Helper


def edit(agent_class, **options):
    """Change the declaration the way editing the source would."""
    agent_class._meta.extras.update(options)
    agent_class._version_cache = None
    return agent_class


class TestRecording:
    def test_the_first_registration_is_version_one(self, agent_class):
        version = agent_class.register_version()

        assert version.version == 1
        assert version.label == agent_class._meta.label
        assert version.stage == Stage.NONE

    def test_the_declaration_is_what_gets_recorded(self, agent_class):
        config = agent_class.register_version().config

        assert config["system"] == "Be helpful."
        assert config["model"] == "claude-opus-5"
        assert config["max_steps"] == 4

    def test_tools_are_recorded_by_name_only(self, agent_class):
        """The code behind a tool cannot go in a row, and a name at least shows a loss."""
        assert agent_class.register_version().tools == ["lookup"]

    def test_registering_an_unchanged_declaration_adds_nothing(self, agent_class):
        """A served agent answers a thousand times against one prompt."""
        first = agent_class.register_version()
        again = agent_class.register_version()

        assert again.version == first.version
        assert len(agent_class.versions()) == 1

    def test_an_edited_prompt_is_a_new_version(self, agent_class):
        agent_class.register_version()
        edit(agent_class, system="Be terse.")

        assert agent_class.register_version().version == 2
        assert [v.version for v in agent_class.versions()] == [2, 1]

    def test_versions_come_back_newest_first(self, agent_class):
        agent_class.register_version()
        edit(agent_class, system="Second.")
        agent_class.register_version()
        edit(agent_class, system="Third.")
        agent_class.register_version()

        assert [v.version for v in agent_class.versions()] == [3, 2, 1]

    def test_reverting_a_prompt_records_a_new_version(self, agent_class):
        """Versions are a log of what the declaration was, in order.

        Reusing the earlier row would lose the fact that it changed back, and
        when. Ping-ponging between two prompts really does produce a row each
        time, which is what happened.
        """
        first = agent_class.register_version()
        edit(agent_class, system="Different.")
        agent_class.register_version()
        edit(agent_class, system="Be helpful.")
        reverted = agent_class.register_version()

        assert reverted.version == 3
        assert reverted.fingerprint == first.fingerprint
        assert agent_class.current_version().version == 3

    def test_a_tool_object_does_not_break_the_row(self, agent_class):
        """`tools` is a list of live objects, and a list is not made of strings
        by being a list — the recorded config has to exclude it."""
        assert "tools" not in agent_class.register_version().config


class TestWhichOneIsLive:
    def test_current_version_matches_the_declaration_in_the_code(self, agent_class):
        recorded = agent_class.register_version()

        current = agent_class.current_version()
        assert current is not None
        assert current.version == recorded.version

    def test_an_unrecorded_edit_has_no_current_version(self, agent_class):
        """What is written down and what would run have parted company."""
        agent_class.register_version()
        edit(agent_class, system="Edited but never recorded.")

        assert agent_class.current_version() is None

    def test_nothing_recorded_means_no_current_version(self, agent_class):
        assert agent_class.current_version() is None


class TestStages:
    def test_promoting_moves_the_version(self, agent_class):
        agent_class.register_version()

        promoted = agent_class.promote(1, Stage.PRODUCTION)
        assert promoted.stage == Stage.PRODUCTION
        assert agent_class.production().version == 1

    def test_promoting_demotes_the_incumbent(self, agent_class):
        """Two versions in production would make the word mean nothing."""
        agent_class.register_version()
        edit(agent_class, system="Second.")
        agent_class.register_version()

        agent_class.promote(1, Stage.PRODUCTION)
        agent_class.promote(2, Stage.PRODUCTION)

        stages = {v.version: v.stage for v in agent_class.versions()}
        assert stages == {2: Stage.PRODUCTION, 1: Stage.ARCHIVED}

    def test_nothing_promoted_means_no_production(self, agent_class):
        agent_class.register_version()
        assert agent_class.production() is None

    def test_an_unknown_version_is_reported(self, agent_class):
        agent_class.register_version()
        with pytest.raises(LookupError, match="no version 9"):
            agent_class.promote(9)

    def test_an_invented_stage_lists_the_real_ones(self, agent_class):
        from mlango.core.exceptions import ImproperlyConfigured

        agent_class.register_version()
        with pytest.raises(ImproperlyConfigured, match="production"):
            agent_class.promote(1, "live")


class TestRunningAnAgent:
    def test_running_records_a_version_and_stamps_the_trace(self, agent_class):
        """A trace read next month must say which declaration answered."""
        from mlango.agents.tracing import recent_traces

        agent_class().run("hello")

        assert agent_class.versions(), "running registered the declaration"
        trace = recent_traces(limit=1, agent=agent_class._meta.label)[0]
        assert trace.meta["version"] == 1

    def test_the_version_is_resolved_once_per_declaration(self, agent_class):
        """A served agent must not pay a query per request."""
        calls = []
        original = agent_class.register_version.__func__

        def counting(cls, **kwargs):
            calls.append(1)
            return original(cls, **kwargs)

        agent_class.register_version = classmethod(counting)
        try:
            agent = agent_class()
            for _ in range(5):
                agent.run("hello")
        finally:
            del agent_class.register_version

        assert len(calls) == 1

    def test_an_edited_prompt_is_picked_up_without_a_restart(self, agent_class):
        agent_class().run("hello")
        edit(agent_class, system="Now different.")
        agent_class().run("hello again")

        assert [v.version for v in agent_class.versions()] == [2, 1]

    def test_a_broken_metastore_does_not_stop_the_agent(self, agent_class):
        """Bookkeeping about a run may not cost the run."""
        from mlango.agents import agent as agent_module

        def refuse(cls, **kwargs):
            raise RuntimeError("metastore is gone")

        agent_class.register_version = classmethod(refuse)
        try:
            result = agent_class().run("hello")
            # Asserted while the metastore is still broken: outside the patch
            # the real method works, and the check would prove nothing.
            assert agent_module._version_for(agent_class) is None
        finally:
            del agent_class.register_version

        assert result.output, "the agent still answered"
        assert agent_class.versions() == [], "and recorded nothing it could not record"
