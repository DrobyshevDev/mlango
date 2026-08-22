"""Turning declared models and agents into HTTP endpoints.

The request and response shapes are pydantic models, so FastAPI's ``/api/docs``
describes each endpoint without anyone writing OpenAPI by hand — the schema is
derived from the same declaration that trains the model.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any

from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from mlango.core.exceptions import RunError
from mlango.core.typing import AgentClass, ModelClass
from mlango.serve.routing import Endpoint
from mlango.training.model import current_request

logger = logging.getLogger("mlango.serve")


class PredictRequest(BaseModel):
    """One input, or a batch of them."""

    input: Any | None = Field(default=None, description="A single input record or value.")
    inputs: list[Any] | None = Field(default=None, description="A batch of input records.")
    proba: bool = Field(default=False, description="Also return class probabilities.")


class PredictResponse(BaseModel):
    model: str
    version: int | None = None
    predictions: list[Any]
    probabilities: list[Any] | None = None


class ChatRequest(BaseModel):
    message: str = Field(description="The user's message.")
    session_id: str = Field(default="", description="Conversation id for memory continuity.")


class ChatResponse(BaseModel):
    agent: str
    output: str
    steps: int
    trace: str = ""
    tools_used: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    error: str = ""


class _LazyModel:
    """Loads a registered model version once, on first request."""

    def __init__(self, model_class: ModelClass, version: int | None, stage: str | None):
        self.model_class = model_class
        self.version = version
        self.stage = stage
        self._instance: Any = None
        self._lock = threading.Lock()

    def get(self) -> Any:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self.model_class.load(version=self.version, stage=self.stage)
                    logger.info(
                        "Loaded %s for serving", getattr(self._instance._version, "ref", "?")
                    )
        return self._instance

    def reset(self) -> None:
        with self._lock:
            self._instance = None


class _Shadow:
    """The candidate version, loaded lazily and only if one is configured.

    Kept beside the served model rather than inside it: a shadow is a property
    of the deployment, not of the model, and the served path has to keep
    working identically when there is no candidate.
    """

    def __init__(self, model_class: ModelClass, served_stage: str | None):
        self.model_class = model_class
        self.served_stage = served_stage
        self._loader: _LazyModel | None = None
        self._checked = False
        self._lock = threading.Lock()

    def config(self) -> dict[str, Any]:
        from mlango.conf import settings

        return dict(getattr(settings, "SHADOW", {}) or {})

    def get(self, served: Any = None) -> Any:
        """The candidate, or None when there is nothing to shadow with."""
        config = self.config()
        if not config.get("ENABLED", False):
            return None

        stage = str(config.get("STAGE", "staging"))
        with self._lock:
            if not self._checked:
                self._checked = True
                self._loader = _LazyModel(self.model_class, None, stage)
        if self._loader is None:
            return None

        try:
            candidate = self._loader.get()
        except Exception:  # noqa: BLE001 - nothing promoted to that stage yet
            logger.debug("No %s version to shadow with", stage, exc_info=True)
            self._loader = None
            return None

        # Compared by resolved version, not by stage name. An endpoint serving
        # "latest" has no stage to compare against, and the latest version is
        # frequently the one just promoted to staging — the pair would be a
        # version shadowing itself, which costs double and compares nothing.
        if served is not None and _version_of(candidate) == _version_of(served):
            return None
        return candidate


def _version_of(model: Any) -> Any:
    return getattr(getattr(model, "_version", None), "version", None)


def _run_shadow(shadow: _Shadow, batch: list[Any], served: Any = None) -> None:
    """Score the candidate on the same input, for the log only.

    Never touches the response and never raises: a shadow exists to inform a
    promotion, and an endpoint that fell over because the candidate did would
    make the whole idea unusable.
    """
    import random

    config = shadow.config()
    sample = float(config.get("SAMPLE", 1.0) or 0.0)
    if sample <= 0 or (sample < 1.0 and random.random() >= sample):
        return

    try:
        candidate = shadow.get(served)
        if candidate is None:
            return
        candidate.predict(batch)
    except Exception:  # noqa: BLE001 - see above
        logger.warning("Shadow prediction failed; the response is unaffected", exc_info=True)


def model_endpoint(
    model_class: ModelClass, *, version: int | None = None, stage: str | None = None
) -> Endpoint:
    """Build a prediction endpoint for a declared model."""
    loader = _LazyModel(model_class, version, stage)
    shadow = _Shadow(model_class, stage)
    label = model_class._meta.label

    def handler(payload: PredictRequest) -> PredictResponse:
        if payload.inputs is None and payload.input is None:
            raise RunError("Send either `input` (one record) or `inputs` (a batch).")

        batch = payload.inputs if payload.inputs is not None else [payload.input]
        model = loader.get()

        # One id for the whole request, so the shadow's rows can be paired with
        # the served ones later. Set even when nothing is shadowing: it costs a
        # uuid and makes the log answer "which rows were one call".
        token = current_request.set(uuid.uuid4().hex)
        try:
            predictions = model.predict(batch)
            probabilities = model.predict_proba(batch) if payload.proba else None
            _run_shadow(shadow, batch, model)
        finally:
            current_request.reset(token)

        return PredictResponse(
            model=label,
            version=getattr(model._version, "version", None),
            predictions=list(predictions),
            probabilities=list(probabilities) if probabilities is not None else None,
        )

    handler.__name__ = f"predict_{label.replace('.', '_')}"
    return Endpoint(
        kind="model",
        label=label,
        handler=handler,
        summary=f"Predict with {label}",
        description=model_class._meta.description or f"Run inference with {label}.",
        meta={
            "task": model_class.get_task(),
            "version": version,
            "stage": stage,
            "features": _safe(model_class.get_features),
        },
    )


def agent_endpoint(agent_class: AgentClass, **agent_kwargs: Any) -> Endpoint:
    """Build a chat endpoint for a declared agent."""
    label = agent_class._meta.label

    def handler(payload: ChatRequest) -> ChatResponse:
        agent = agent_class(**agent_kwargs)
        result = agent.run(payload.message, session_id=payload.session_id)
        return ChatResponse(
            agent=label,
            output=result.output,
            steps=result.steps,
            trace=result.trace_uuid,
            tools_used=result.tools_used,
            usage=result.usage.describe(),
            error=result.error,
        )

    handler.__name__ = f"chat_{label.replace('.', '_')}"
    instance = agent_class()
    return Endpoint(
        kind="agent",
        label=label,
        handler=handler,
        summary=f"Chat with {label}",
        description=agent_class._meta.description or f"Send a message to {label}.",
        meta={
            "model": agent_class.get_model(),
            "tools": instance.get_tools().names(),
            "max_steps": agent_class.get_max_steps(),
        },
    )


def agent_stream_endpoint(agent_class: AgentClass, **agent_kwargs: Any) -> Endpoint:
    """Build a Server-Sent Events endpoint for a declared agent.

    A multi-step agent can take a minute, and a blank screen for a minute reads
    as broken. This streams each step as it happens, in the ``text/event-stream``
    format every browser understands natively through ``EventSource``.

    ``StreamingResponse`` is imported at module scope on purpose: ``from
    __future__ import annotations`` turns the handler's return annotation into a
    string, and FastAPI has to resolve it to build the schema. A function-local
    import leaves it unresolvable and breaks ``/api/openapi.json`` for the whole
    project, not just this route.
    """
    label = agent_class._meta.label

    def handler(payload: ChatRequest) -> StreamingResponse:
        def events():
            agent = agent_class(**agent_kwargs)
            try:
                for event in agent.stream(payload.message, session_id=payload.session_id):
                    body = json.dumps(event.describe(), ensure_ascii=False)
                    yield f"event: {event.kind}\ndata: {body}\n\n"
            except Exception as exc:  # noqa: BLE001 - reported on the stream, not as a 500
                # The response has already started, so an exception cannot become
                # a status code; the client is told on the stream instead.
                logger.exception("Streaming %s failed", label)
                body = json.dumps(
                    {"event": "failed", "error": str(exc), "exception_type": type(exc).__name__}
                )
                yield f"event: failed\ndata: {body}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Tells nginx not to buffer, which would defeat the point.
                "X-Accel-Buffering": "no",
            },
        )

    handler.__name__ = f"stream_{label.replace('.', '_')}"
    instance = agent_class()
    return Endpoint(
        kind="agent",
        label=label,
        handler=handler,
        summary=f"Stream a conversation with {label}",
        description=(
            f"Server-Sent Events from {label}. Event names: started, thinking, "
            f"text_chunk, tool_called, tool_finished, step_finished, finished, failed."
        ),
        meta={
            "model": agent_class.get_model(),
            "tools": instance.get_tools().names(),
            "streaming": True,
        },
    )


def _safe(fn: Any) -> Any:
    """Call an introspection helper, tolerating an incomplete declaration."""
    try:
        return fn()
    except Exception:
        return None


__all__ = [
    "model_endpoint",
    "agent_endpoint",
    "agent_stream_endpoint",
    "PredictRequest",
    "PredictResponse",
    "ChatRequest",
    "ChatResponse",
]
