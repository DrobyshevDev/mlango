"""Agent tracing.

Every LLM call and every tool call becomes a span under one trace, so "why did
the agent say that?" is answerable after the fact from the admin rather than by
re-running with print statements. Tracing is best-effort by design: a metastore
hiccup must degrade observability, never break the agent.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from mlango.core import telemetry
from mlango.core.serialization import jsonable
from mlango.metastore.models import RunStatus, Span, Trace, utcnow
from mlango.metastore.session import session_scope

logger = logging.getLogger("mlango.agents.tracing")


class Tracer:
    """Records one agent invocation."""

    def __init__(
        self,
        agent_label: str,
        *,
        session_id: str = "",
        run_id: int | None = None,
        enabled: bool = True,
        meta: dict[str, Any] | None = None,
    ):
        self.agent_label = agent_label
        self.session_id = session_id
        self.run_id = run_id
        self.enabled = enabled
        self.meta = dict(meta or {})

        self.trace_id: int | None = None
        self.uuid: str = ""
        self._ordering = 0
        self._started = time.perf_counter()

    # -- lifecycle -----------------------------------------------------------

    def start(self, user_input: str) -> Tracer:
        if not self.enabled:
            return self
        try:
            with session_scope() as session:
                trace = Trace(
                    agent=self.agent_label,
                    session_id=self.session_id,
                    run_id=self.run_id,
                    input=user_input[:100_000],
                    status=RunStatus.RUNNING,
                    meta=self.meta,
                )
                session.add(trace)
                session.flush()
                self.trace_id, self.uuid = trace.id, trace.uuid
        except Exception:
            logger.exception(
                "Could not open a trace for %s; continuing untraced.", self.agent_label
            )
            self.enabled = False
        return self

    def finish(
        self,
        output: str,
        *,
        steps: int,
        usage: Any = None,
        status: str = RunStatus.FINISHED,
        error: str = "",
    ) -> None:
        if not self.enabled or self.trace_id is None:
            return
        try:
            with session_scope() as session:
                trace = session.get(Trace, self.trace_id)
                if trace is None:
                    return
                trace.output = (output or "")[:100_000]
                trace.status = status
                trace.steps = steps
                trace.error = error[:8000]
                trace.ended_at = utcnow()
                trace.duration_s = time.perf_counter() - self._started
                if usage is not None:
                    trace.input_tokens = usage.input_tokens
                    trace.output_tokens = usage.output_tokens
        except Exception:
            logger.exception("Could not close trace %s.", self.uuid)

    def fail(self, exc: BaseException, *, steps: int = 0) -> None:
        self.finish("", steps=steps, status=RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}")

    # -- spans ---------------------------------------------------------------

    @contextmanager
    def span(self, name: str, kind: str, payload: Any = None) -> Iterator[dict[str, Any]]:
        """Record one step. The yielded dict is the span's mutable output slot."""
        result: dict[str, Any] = {}
        started = time.perf_counter()
        self._ordering += 1
        ordering = self._ordering

        # Emitted whether or not the metastore trace is on: they answer to
        # different audiences, and turning off mlango's own history is not a
        # request to go dark in somebody's Grafana.
        with telemetry.span(
            f"mlango.{kind}", agent=self.agent_label, step=ordering, span_kind=kind
        ) as external:
            if not self.enabled or self.trace_id is None:
                yield result
                telemetry.annotate(external, output=_short(result.get("output")))
                return

            error = ""
            try:
                yield result
                telemetry.annotate(external, output=_short(result.get("output")))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                try:
                    with session_scope() as session:
                        session.add(
                            Span(
                                trace_id=self.trace_id,
                                ordering=ordering,
                                name=name,
                                kind=kind,
                                input=_jsonable(payload),
                                output=_jsonable(result.get("output")),
                                usage=_jsonable(result.get("usage")) or {},
                                error=error,
                                started_at=utcnow(),
                                ended_at=utcnow(),
                                duration_s=time.perf_counter() - started,
                            )
                        )
                except Exception:
                    logger.exception("Could not record span %r on trace %s.", name, self.uuid)

    @property
    def short_id(self) -> str:
        return self.uuid[:8]

    def __repr__(self) -> str:
        state = "on" if self.enabled else "off"
        return f"<Tracer {self.agent_label} [{state}] {self.short_id}>"


# --------------------------------------------------------------------------- #
# Reading traces back
# --------------------------------------------------------------------------- #


def recent_traces(limit: int = 20, *, agent: str | None = None) -> list[Trace]:
    from sqlalchemy import select

    with session_scope() as session:
        statement = select(Trace).order_by(Trace.started_at.desc()).limit(limit)
        if agent:
            statement = statement.where(Trace.agent == agent)
        return list(session.execute(statement).scalars())


def get_trace(reference: str) -> Trace | None:
    """Look a trace up by uuid prefix or numeric id, with its spans loaded."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    with session_scope() as session:
        statement = select(Trace).options(selectinload(Trace.spans))
        if reference.isdigit():
            found = session.execute(statement.where(Trace.id == int(reference))).scalars().first()
            if found is not None:
                return found
        return (
            session.execute(
                statement.where(Trace.uuid.startswith(reference)).order_by(Trace.started_at.desc())
            )
            .scalars()
            .first()
        )


def _jsonable(payload: Any) -> Any:
    return jsonable(payload) if payload is not None else None


__all__ = ["Tracer", "recent_traces", "get_trace"]


def _short(value: object, length: int = 200) -> str | None:
    """A span attribute, not a transcript: the metastore keeps the full text."""
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= length else text[: length - 1] + "…"
