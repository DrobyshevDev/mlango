"""The ``Agent`` declarative class.

    class SupportAgent(Agent):
        \"\"\"Answers product questions from the docs.\"\"\"

        class Meta:
            model = "claude-opus-5"
            system = "You are a support engineer. Cite the docs you used."
            tools = [search_docs, create_ticket]
            memory = BufferMemory(k=20)

    SupportAgent().run("How do I rotate an API key?")

The tool-use loop, tracing, memory and usage accounting are the framework's.
An agent declares *what* it is; the loop that makes it work is written once.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from mlango.agents.events import (
    AgentEvent,
    Failed,
    Finished,
    Started,
    StepFinished,
    TextChunk,
    Thinking,
    ToolCalled,
    ToolFinished,
)
from mlango.agents.memory import Memory, NullMemory
from mlango.agents.providers.base import Completion, Provider, ToolCall, Usage, get_provider
from mlango.agents.tools import Tool, Toolbox, ToolResult
from mlango.agents.tracing import Tracer
from mlango.core.base import Declarative
from mlango.core.exceptions import ProviderError, RunError
from mlango.core.signals import agent_finished, agent_started, agent_step, tool_called
from mlango.metastore.models import RunStatus, Span

logger = logging.getLogger("mlango.agents")


@dataclass
class AgentRun:
    """The outcome of one agent invocation."""

    output: str = ""
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = Completion.END_TURN
    trace_uuid: str = ""
    session_id: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def tools_used(self) -> list[str]:
        return [r.name for r in self.tool_results]

    def describe(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "steps": self.steps,
            "usage": self.usage.describe(),
            "stop_reason": self.stop_reason,
            "trace": self.trace_uuid,
            "tools_used": self.tools_used,
            "error": self.error,
        }

    def __str__(self) -> str:
        return self.output


class Agent(Declarative):
    """A declared LLM agent."""

    _kind = "agent"
    _meta_options = (
        "model",
        "system",
        "tools",
        "provider",
        "max_steps",
        "max_tokens",
        "thinking",
        "effort",
        "memory",
        "stop_sequences",
        "tracing",
        "input_field",
    )

    class Meta:
        abstract = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._toolbox: Toolbox | None = None

    # -- configuration -------------------------------------------------------

    @classmethod
    def get_provider(cls) -> Provider:
        return get_provider(cls._meta.extras.get("provider"))

    @classmethod
    def get_model(cls) -> str:
        from mlango.conf import settings

        return str(cls._meta.extras.get("model") or settings.DEFAULT_AGENT_MODEL)

    def get_system(self) -> str:
        """The system prompt. Override to build it from the agent's fields."""
        return str(type(self)._meta.extras.get("system", "") or "")

    def get_tools(self) -> Toolbox:
        if self._toolbox is None:
            declared = type(self)._meta.extras.get("tools") or []
            self._toolbox = Toolbox([_as_tool(item) for item in declared])
        return self._toolbox

    @classmethod
    def get_memory(cls) -> Memory:
        memory = cls._meta.extras.get("memory")
        if memory is None:
            return NullMemory()
        return memory() if isinstance(memory, type) else memory

    @classmethod
    def get_max_steps(cls) -> int:
        from mlango.conf import settings

        return int(cls._meta.extras.get("max_steps") or settings.AGENT_MAX_STEPS)

    @classmethod
    def tracing_enabled(cls) -> bool:
        from mlango.conf import settings

        declared = cls._meta.extras.get("tracing")
        return bool(settings.TRACING if declared is None else declared)

    # -- running -------------------------------------------------------------

    @classmethod
    def chat(cls, message: str, **kwargs: Any) -> AgentRun:
        """One-liner for the common case: ``SupportAgent.chat("hi")``."""
        return cls().run(message, **kwargs)

    def run(
        self,
        message: str,
        *,
        session_id: str = "",
        memory: Memory | None = None,
        history: list[dict[str, Any]] | None = None,
        max_steps: int | None = None,
        run_id: int | None = None,
        **provider_kwargs: Any,
    ) -> AgentRun:
        """Run the tool-use loop until the model stops asking for tools."""
        opts = type(self)._meta
        provider = self.get_provider()
        toolbox = self.get_tools()
        store = memory if memory is not None else self.get_memory()
        limit = max_steps or self.get_max_steps()

        prior = list(history) if history is not None else store.load(session_id)
        user_turn = {"role": "user", "content": message}
        messages: list[dict[str, Any]] = [*prior, user_turn]

        tracer = Tracer(
            opts.label,
            session_id=session_id,
            run_id=run_id,
            enabled=self.tracing_enabled(),
            meta={
                "model": self.get_model(),
                "provider": provider.name,
                # Which declaration answered. Without it a trace from last month
                # is read against today's prompt, which is how people conclude
                # the model changed when the prompt did.
                "version": _version_for(type(self)),
            },
        ).start(message)

        result = AgentRun(session_id=session_id, trace_uuid=tracer.uuid, messages=messages)
        agent_started.send(sender=type(self), agent=self, trace=tracer)

        try:
            for _event in self._loop(
                provider, toolbox, messages, tracer, result, limit, provider_kwargs
            ):
                pass
        except ProviderError as exc:
            result.error = str(exc)
            tracer.fail(exc, steps=result.steps)
            agent_finished.send(sender=type(self), agent=self, trace=tracer)
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as RunError below
            result.error = f"{type(exc).__name__}: {exc}"
            tracer.fail(exc, steps=result.steps)
            agent_finished.send(sender=type(self), agent=self, trace=tracer)
            raise RunError(f"{opts.label} failed: {exc}") from exc

        self._settle(result, tracer, store, session_id, user_turn)
        return result

    def stream(
        self,
        message: str,
        *,
        session_id: str = "",
        memory: Memory | None = None,
        history: list[dict[str, Any]] | None = None,
        max_steps: int | None = None,
        run_id: int | None = None,
        **provider_kwargs: Any,
    ) -> Iterator[AgentEvent]:
        """Run the loop, yielding events as they happen.

        Exactly the same loop as :meth:`run` — one implementation, so the two can
        never disagree about what the agent did. The final :class:`Finished`
        event carries the identical :class:`AgentRun`.

            for event in agent.stream("hello"):
                if isinstance(event, TextChunk):
                    print(event.text, end="", flush=True)
        """
        opts = type(self)._meta
        provider = self.get_provider()
        toolbox = self.get_tools()
        store = memory if memory is not None else self.get_memory()
        limit = max_steps or self.get_max_steps()

        prior = list(history) if history is not None else store.load(session_id)
        user_turn = {"role": "user", "content": message}
        messages: list[dict[str, Any]] = [*prior, user_turn]

        tracer = Tracer(
            opts.label,
            session_id=session_id,
            run_id=run_id,
            enabled=self.tracing_enabled(),
            meta={
                "model": self.get_model(),
                "provider": provider.name,
                # Which declaration answered. Without it a trace from last month
                # is read against today's prompt, which is how people conclude
                # the model changed when the prompt did.
                "version": _version_for(type(self)),
            },
        ).start(message)

        result = AgentRun(session_id=session_id, trace_uuid=tracer.uuid, messages=messages)
        agent_started.send(sender=type(self), agent=self, trace=tracer)

        yield Started(
            step=0,
            agent=opts.label,
            trace=tracer.uuid,
            session_id=session_id,
            message=message,
        )

        try:
            yield from self._loop(
                provider, toolbox, messages, tracer, result, limit, provider_kwargs
            )
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            tracer.fail(exc, steps=result.steps)
            agent_finished.send(sender=type(self), agent=self, trace=tracer)
            yield Failed(step=result.steps, error=str(exc), exception_type=type(exc).__name__)
            if isinstance(exc, ProviderError):
                raise
            raise RunError(f"{opts.label} failed: {exc}") from exc

        self._settle(result, tracer, store, session_id, user_turn)

        yield Finished(
            step=result.steps,
            output=result.output,
            trace=result.trace_uuid,
            tools_used=result.tools_used,
            usage=result.usage.describe(),
            error=result.error,
            result=result,
        )

    def _settle(
        self,
        result: AgentRun,
        tracer: Tracer,
        store: Memory,
        session_id: str,
        user_turn: dict[str, Any],
    ) -> None:
        """Close the trace and record the turn. Shared by run() and stream()."""
        tracer.finish(
            result.output,
            steps=result.steps,
            usage=result.usage,
            status=RunStatus.FAILED if result.error else RunStatus.FINISHED,
            error=result.error,
        )
        result.trace_uuid = tracer.uuid

        if session_id and not result.error:
            store.append(session_id, [user_turn, {"role": "assistant", "content": result.output}])

        agent_finished.send(sender=type(self), agent=self, trace=tracer)

    # -- the loop ------------------------------------------------------------

    def _loop(
        self,
        provider: Provider,
        toolbox: Toolbox,
        messages: list[dict[str, Any]],
        tracer: Tracer,
        result: AgentRun,
        limit: int,
        provider_kwargs: dict[str, Any],
    ) -> Iterator[AgentEvent]:
        """The one and only agent loop, as a generator of events.

        ``run()`` drains it and ignores the events; ``stream()`` forwards them.
        Having a single implementation is what keeps the two honest.
        """
        opts = type(self)._meta
        system = self.get_system()
        schemas = toolbox.schemas() or None

        for step in range(1, limit + 1):
            result.steps = step
            yield Thinking(step=step, model=self.get_model())

            with tracer.span(
                f"llm:{self.get_model()}", Span.LLM, {"messages": len(messages)}
            ) as span:
                completion = provider.complete(
                    model=self.get_model(),
                    messages=messages,
                    system=system,
                    tools=schemas,
                    max_tokens=int(opts.extras.get("max_tokens", 4096)),
                    thinking=opts.extras.get("thinking", "adaptive"),
                    effort=opts.extras.get("effort"),
                    stop_sequences=opts.extras.get("stop_sequences"),
                    **provider_kwargs,
                )
                span["output"] = {"text": completion.text[:2000], "stop": completion.stop_reason}
                span["usage"] = completion.usage.describe()

            result.usage = result.usage.add(completion.usage)
            result.stop_reason = completion.stop_reason
            agent_step.send(sender=type(self), agent=self, trace=tracer, step=step)

            if completion.text:
                yield TextChunk(step=step, text=completion.text)
            yield StepFinished(
                step=step,
                stop_reason=completion.stop_reason,
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
            )

            if completion.stop_reason == Completion.REFUSAL:
                detail = completion.refusal or {}
                result.error = (
                    "The model declined this request"
                    + (f" ({detail.get('category')})" if detail.get("category") else "")
                    + "."
                )
                result.output = completion.text
                return

            messages.append(_assistant_turn(completion))

            if completion.stop_reason == Completion.PAUSE_TURN:
                # A server-side tool hit its iteration cap. Re-send as-is; the
                # server resumes where it left off — do not inject a nudge.
                continue

            if not completion.wants_tools:
                result.output = completion.text
                return

            results = []
            for call in completion.tool_calls:
                yield ToolCalled(
                    step=step,
                    name=call.name,
                    arguments=call.arguments,
                    tool_call_id=call.id,
                )
                outcome = self._run_one_tool(toolbox, call, tracer)
                results.append(outcome)
                yield ToolFinished(
                    step=step,
                    name=outcome.name,
                    content=outcome.as_text()[:4000],
                    is_error=outcome.is_error,
                    duration_s=outcome.duration_s,
                    tool_call_id=outcome.tool_call_id,
                )
            result.tool_results.extend(results)
            # Every tool_result for one assistant turn goes back in a single
            # user message; splitting them teaches the model to stop batching.
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.tool_call_id,
                            "content": r.as_text(),
                            **({"is_error": True} if r.is_error else {}),
                        }
                        for r in results
                    ],
                }
            )

        result.error = (
            f"{opts.label} stopped after {limit} steps without finishing. "
            f"Raise max_steps, or narrow the task."
        )
        result.output = _last_text(messages)

    def _run_one_tool(self, toolbox: Toolbox, call: ToolCall, tracer: Tracer) -> ToolResult:
        """Execute one requested tool, reporting an unknown name to the model."""
        found = toolbox.get(call.name)
        if found is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=(
                    f"No tool named {call.name!r} is available. "
                    f"Available tools: {', '.join(toolbox.names()) or '(none)'}."
                ),
                is_error=True,
            )

        tool_called.send(sender=type(self), agent=self, tool=found, arguments=call.arguments)
        with tracer.span(f"tool:{call.name}", Span.TOOL, call.arguments) as span:
            outcome = found.run(call_id=call.id, **call.arguments)
            span["output"] = {"content": outcome.as_text()[:4000], "is_error": outcome.is_error}
        return outcome

    def _dispatch(
        self, toolbox: Toolbox, completion: Completion, tracer: Tracer
    ) -> list[ToolResult]:
        """Every tool the model asked for, in order."""
        return [self._run_one_tool(toolbox, call, tracer) for call in completion.tool_calls]

    # -- serving -------------------------------------------------------------

    # -- versions ------------------------------------------------------------

    @classmethod
    def register_version(cls, *, notes: str = "") -> Any:
        """Record the current declaration, if it is not already the newest.

        Idempotent by fingerprint: running an agent a thousand times against an
        unchanged prompt leaves one row. A version exists exactly when the
        declaration differs from the last one recorded, which is what makes
        "when did this prompt change" answerable at all.
        """
        from sqlalchemy import func, select

        from mlango.metastore.models import AgentVersion, Stage
        from mlango.metastore.session import session_scope

        opts = cls._meta
        fingerprint = opts.fingerprint()

        with session_scope() as session:
            newest = session.execute(
                select(AgentVersion)
                .where(AgentVersion.label == opts.label)
                .order_by(AgentVersion.version.desc())
                .limit(1)
            ).scalar_one_or_none()
            if newest is not None and newest.fingerprint == fingerprint:
                return newest

            highest = session.execute(
                select(func.max(AgentVersion.version)).where(AgentVersion.label == opts.label)
            ).scalar()
            version = AgentVersion(
                label=opts.label,
                version=(highest or 0) + 1,
                fingerprint=fingerprint,
                config=opts.recordable(),
                # Names only. The code behind a tool is not recoverable from a
                # row, and a list of names at least makes a removed tool visible.
                tools=sorted(_declared_tool_names(cls)),
                stage=Stage.NONE,
                notes=notes,
            )
            session.add(version)
            session.flush()
            logger.info("Registered %s", version.ref)
            return version

    @classmethod
    def versions(cls) -> list[Any]:
        """Every recorded version, newest first."""
        from sqlalchemy import select

        from mlango.metastore.models import AgentVersion
        from mlango.metastore.session import session_scope

        with session_scope() as session:
            return list(
                session.execute(
                    select(AgentVersion)
                    .where(AgentVersion.label == cls._meta.label)
                    .order_by(AgentVersion.version.desc())
                ).scalars()
            )

    @classmethod
    def current_version(cls) -> Any | None:
        """The recorded version matching the declaration as it stands now.

        None when the declaration has changed since anything was recorded —
        which is the interesting case, because it means what is deployed and
        what is written down have parted company.
        """
        fingerprint = cls._meta.fingerprint()
        return next((v for v in cls.versions() if v.fingerprint == fingerprint), None)

    @classmethod
    def promote(cls, version: int, stage: str = "production") -> Any:
        """Move a version to a stage, demoting whatever held it."""
        from sqlalchemy import select

        from mlango.core.exceptions import ImproperlyConfigured
        from mlango.metastore.models import AgentVersion, Stage
        from mlango.metastore.session import session_scope

        if stage not in Stage.ALL:
            raise ImproperlyConfigured(f"{stage!r} is not a stage. One of: {', '.join(Stage.ALL)}.")

        with session_scope() as session:
            rows = list(
                session.execute(
                    select(AgentVersion).where(AgentVersion.label == cls._meta.label)
                ).scalars()
            )
            target = next((row for row in rows if row.version == version), None)
            if target is None:
                raise LookupError(f"{cls._meta.label} has no version {version}.")

            if stage in (Stage.PRODUCTION, Stage.STAGING):
                # One version per stage, or "production" stops meaning anything.
                for row in rows:
                    if row.stage == stage and row is not target:
                        row.stage = Stage.ARCHIVED
            target.stage = stage
            session.flush()
            return target

    @classmethod
    def production(cls) -> Any | None:
        """The version at stage ``production``, if one has been promoted."""
        from mlango.metastore.models import Stage

        return next((v for v in cls.versions() if v.stage == Stage.PRODUCTION), None)

    @classmethod
    def as_endpoint(cls, **agent_kwargs: Any) -> Any:
        """A chat endpoint for ``routes.py``, à la Django's ``as_view()``."""
        from mlango.serve.endpoints import agent_endpoint

        return agent_endpoint(cls, **agent_kwargs)

    @classmethod
    def as_stream_endpoint(cls, **agent_kwargs: Any) -> Any:
        """A Server-Sent Events endpoint, for a UI that shows progress."""
        from mlango.serve.endpoints import agent_stream_endpoint

        return agent_stream_endpoint(cls, **agent_kwargs)

    # -- introspection -------------------------------------------------------

    @classmethod
    def summary(cls) -> dict[str, Any]:
        instance = cls()
        return {
            "label": cls._meta.label,
            "model": cls.get_model(),
            "provider": cls._meta.extras.get("provider"),
            "tools": instance.get_tools().names(),
            "max_steps": cls.get_max_steps(),
            "memory": cls.get_memory().describe(),
            "system": instance.get_system(),
            "config": {f.name: f.get_default() for f in cls._meta.fields},
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _version_for(agent_class: Any) -> int | None:
    """The recorded version of this declaration, registering it if it is new.

    Cached on the class by fingerprint, so a served agent answers a thousand
    requests with one query rather than a thousand. The cache key is the
    fingerprint itself, so editing a prompt in a live reload is picked up.

    Best effort throughout: an agent that can answer must not stop answering
    because the metastore is unreachable.
    """
    try:
        fingerprint = agent_class._meta.fingerprint()
    except Exception:  # noqa: BLE001 - see above
        return None

    cached = getattr(agent_class, "_version_cache", None)
    if cached and cached[0] == fingerprint:
        return cached[1]

    try:
        version = agent_class.register_version().version
    except Exception:  # noqa: BLE001 - see above
        logger.debug("Could not register a version for %s", agent_class, exc_info=True)
        return None

    agent_class._version_cache = (fingerprint, version)
    return version


def _declared_tool_names(agent_class: Any) -> list[str]:
    """Tool names from the declaration, without instantiating the agent.

    ``get_tools`` builds a Toolbox on an instance because it caches one;
    recording a version should not require constructing an agent, and a name
    is all the row can honestly keep anyway.
    """
    names = []
    for item in agent_class._meta.extras.get("tools") or []:
        name = getattr(item, "name", None) or getattr(item, "__name__", None)
        if name:
            names.append(str(name))
    return names


def _as_tool(item: Any) -> Tool:
    from mlango.agents.tools import tool as make_tool

    if isinstance(item, Tool):
        return item
    if callable(item):
        return make_tool(item)
    raise TypeError(f"{item!r} is not a tool: pass a @tool function or a Tool instance.")


def _assistant_turn(completion: Completion) -> dict[str, Any]:
    """The assistant turn to echo back, preserving provider-native blocks."""
    if completion.raw_content is not None:
        return {"role": "assistant", "content": completion.raw_content}

    content: list[dict[str, Any]] = []
    if completion.text:
        content.append({"type": "text", "text": completion.text})
    for call in completion.tool_calls:
        content.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        )
    return {"role": "assistant", "content": content or [{"type": "text", "text": ""}]}


def _last_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else None)
                for b in content
            ]
            joined = "".join(p for p in parts if p)
            if joined:
                return joined
    return ""


__all__ = ["Agent", "AgentRun"]
