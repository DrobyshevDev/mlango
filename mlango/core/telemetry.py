"""OpenTelemetry, when the surrounding organisation speaks it.

mlango already records what happened, in its own metastore, and that record is
what the admin and every comparison command read. This is not a replacement for
it: it is a bridge, so a training run and an agent's tool loop show up in the
same Grafana or Datadog view as the HTTP request that started them.

Three rules keep it from becoming a liability:

* **Optional.** ``opentelemetry-api`` need not be installed. Everything here
  degrades to a no-op, and no import of mlango pulls it in.
* **Silent by default.** Even installed, nothing is emitted until ``TELEMETRY``
  is turned on. A framework that started exporting spans to whatever collector
  happened to be configured would be a surprise nobody wants.
* **Not ours to configure.** mlango never sets up an exporter, an endpoint or a
  sampler. The process does that, the way every other OpenTelemetry-instrumented
  library expects. Owning that configuration would mean owning a second, worse
  copy of the OTel SDK's own settings.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("mlango.telemetry")

#: Attribute prefix, so mlango's own attributes are recognisable in a trace
#: view holding spans from a dozen libraries.
NAMESPACE = "mlango"

_tracer: Any = None
_looked = False


def configured() -> bool:
    """Whether the project asked for spans to be emitted."""
    try:
        from mlango.conf import settings

        return bool(dict(getattr(settings, "TELEMETRY", {}) or {}).get("ENABLED", False))
    except Exception:  # noqa: BLE001 - settings may not be configured yet
        return False


def get_tracer() -> Any:
    """The OpenTelemetry tracer, or None when there is nothing to emit to.

    Resolved once. If the process has no SDK installed, the API hands back a
    no-op tracer and spans cost a function call — which is why this does not
    also have to check whether an exporter exists.
    """
    global _tracer, _looked

    if not configured():
        return None
    if _looked:
        return _tracer

    _looked = True
    try:
        from opentelemetry import trace

        from mlango.conf import settings

        name = str(dict(settings.TELEMETRY).get("SERVICE_NAME", "mlango"))
        _tracer = trace.get_tracer(name)
    except ImportError:
        logger.warning(
            "TELEMETRY is on but opentelemetry-api is not installed; "
            "no spans will be emitted. Install it with: pip install 'mlango[otel]'"
        )
        _tracer = None
    return _tracer


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Emit one span, or do nothing at all.

    Never raises. Telemetry describes the work; it does not get to fail it, and
    a collector being down must not stop a training run.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    # Opening the span and running the caller's block are guarded separately.
    # Wrapping both in one try would let a failure *inside* the block fall into
    # the telemetry handler, which would then yield a second time — a generator
    # context manager cannot do that, and the caller sees a baffling
    # "generator didn't stop after throw()" instead of their own exception.
    try:
        scope = tracer.start_as_current_span(name)
        current = scope.__enter__()
    except Exception:  # noqa: BLE001 - only the telemetry path, never the work
        logger.debug("Could not start span %r", name, exc_info=True)
        yield None
        return

    try:
        _set(current, attributes)
        yield current
    except Exception as exc:
        # Recorded and re-raised: the caller's error handling is the caller's,
        # and swallowing it here would be a bug factory.
        _quietly(current.record_exception, exc)
        _quietly(scope.__exit__, type(exc), exc, exc.__traceback__)
        raise
    else:
        _quietly(scope.__exit__, None, None, None)


def annotate(current: Any, **attributes: Any) -> None:
    """Add attributes to an open span, if there is one."""
    if current is not None:
        _set(current, attributes)


def reset() -> None:
    """Forget the resolved tracer. Used by tests and by settings changes."""
    global _tracer, _looked
    _tracer = None
    _looked = False


def _quietly(call: Any, *args: Any) -> None:
    """Run a telemetry call whose failure must not reach the caller."""
    try:
        call(*args)
    except Exception:  # noqa: BLE001 - see the module docstring
        logger.debug("Telemetry call failed", exc_info=True)


def _set(current: Any, attributes: dict[str, Any]) -> None:
    try:
        for key, value in attributes.items():
            if value is None:
                continue
            # OTel accepts str, bool, int, float and sequences of those. Anything
            # else is stringified here rather than dropped by the SDK in silence.
            if not isinstance(value, (str, bool, int, float)):
                value = str(value)
            current.set_attribute(f"{NAMESPACE}.{key}", value)
    except Exception:  # noqa: BLE001 - never at the cost of the work
        logger.debug("Could not set span attributes", exc_info=True)


__all__ = ["span", "annotate", "get_tracer", "configured", "reset", "NAMESPACE"]
