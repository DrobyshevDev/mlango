"""Run tracking.

Every training, evaluation and agent invocation happens inside a ``RunContext``.
It owns one row in the metastore and is the only place that writes metrics,
artifacts and status, which is what makes "what produced this number?" always
answerable.

Metrics are buffered and flushed in batches: a tight training loop logging per
step should not pay a database round trip per scalar.
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator
from typing import Any

from mlango.core import telemetry
from mlango.core.serialization import jsonable
from mlango.core.signals import metric_logged, run_failed, run_finished, run_started
from mlango.metastore.models import Artifact, Metric, Run, RunStatus, utcnow
from mlango.metastore.session import session_scope

logger = logging.getLogger("mlango.run")

#: Metric rows held in memory before a flush.
FLUSH_EVERY = 200


class RunContext:
    """A live run. Use :meth:`start` or the ``with`` form."""

    def __init__(
        self,
        run_id: int,
        uuid: str,
        *,
        autoflush: int = FLUSH_EVERY,
        kind: str = "",
        target: str = "",
    ):
        self.run_id = run_id
        self.uuid = uuid
        #: Kept on the context as well as the row so telemetry can name the
        #: span without a query, and so status is readable after finish().
        self.kind = kind
        self.target = target
        self.status = RunStatus.RUNNING
        self.autoflush = autoflush
        self._buffer: list[dict[str, Any]] = []
        self._step = 0
        self._started = time.perf_counter()
        self._finished = False
        #: Set by callbacks such as EarlyStopping to ask the loop to stop.
        self.should_stop = False

    # -- construction --------------------------------------------------------

    @classmethod
    def start(
        cls,
        *,
        kind: str,
        target: str,
        name: str = "",
        params: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        dataset_version_id: int | None = None,
        seed: int | None = None,
        device: str = "",
        notes: str = "",
    ) -> RunContext:
        environment = _environment()
        with session_scope() as session:
            run = Run(
                name=name or f"{target}-{utcnow():%Y%m%d-%H%M%S}",
                kind=kind,
                target=target,
                status=RunStatus.RUNNING,
                params=_jsonable(params or {}),
                tags=list(tags or []),
                dataset_version_id=dataset_version_id,
                seed=seed,
                device=device,
                notes=notes,
                **environment,
            )
            session.add(run)
            session.flush()
            run_id, run_uuid = run.id, run.uuid

        context = cls(run_id, run_uuid, kind=kind, target=target)
        run_started.send(sender=None, run=context)
        logger.info("Run %s started (%s: %s)", run_uuid[:8], kind, target)
        return context

    # -- logging -------------------------------------------------------------

    def log_metric(
        self,
        key: str,
        value: float,
        *,
        step: int | None = None,
        epoch: int | None = None,
        split: str = "",
    ) -> None:
        if value is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            logger.warning("Metric %r is not numeric (%r); skipping.", key, value)
            return
        if step is None:
            step = self._step
        else:
            self._step = max(self._step, step)

        self._buffer.append(
            {
                "run_id": self.run_id,
                "key": key,
                "value": numeric,
                "step": step,
                "epoch": epoch,
                "split": split,
            }
        )
        metric_logged.send(sender=None, run=self, key=key, value=numeric, step=step)
        if len(self._buffer) >= self.autoflush:
            self.flush()

    def log_metrics(self, metrics: dict[str, Any], **kwargs: Any) -> None:
        for key, value in metrics.items():
            self.log_metric(key, value, **kwargs)

    def advance(self, by: int = 1) -> int:
        self._step += by
        return self._step

    def flush(self) -> None:
        if not self._buffer:
            return
        pending, self._buffer = self._buffer, []
        with session_scope() as session:
            session.bulk_insert_mappings(Metric, pending)  # type: ignore[arg-type]

    # -- artifacts -----------------------------------------------------------

    def log_artifact(
        self,
        name: str,
        path: str,
        *,
        kind: str = "file",
        meta: dict[str, Any] | None = None,
    ) -> None:
        size, sha = _measure(path)
        with session_scope() as session:
            session.add(
                Artifact(
                    run_id=self.run_id,
                    name=name,
                    kind=kind,
                    path=path,
                    size_bytes=size,
                    sha256=sha,
                    meta=_jsonable(meta or {}),
                )
            )

    def log_text(self, name: str, text: str, *, kind: str = "text") -> str:
        from mlango.storage import default_storage

        storage = default_storage()
        # The storage-relative name, not an absolute path: a run on a GPU box
        # writing "/home/gpu/artifacts/..." into the metastore leaves a row the
        # laptop reading it cannot resolve, even when both see the same bucket.
        target = f"runs/{self.uuid}/{name}"
        storage.save_text(target, text)
        self.log_artifact(name, target, kind=kind)
        return target

    def log_json(self, name: str, payload: Any) -> str:
        import json

        return self.log_text(name, json.dumps(_jsonable(payload), indent=2, ensure_ascii=False))

    # -- mutation ------------------------------------------------------------

    def update(self, **changes: Any) -> None:
        with session_scope() as session:
            run = session.get(Run, self.run_id)
            if run is None:
                return
            for key, value in changes.items():
                if key in {"params", "tags", "summary"}:
                    value = _jsonable(value)
                setattr(run, key, value)

    def set_params(self, params: dict[str, Any]) -> None:
        self.update(params=params)

    def set_summary(self, summary: dict[str, Any]) -> None:
        self.update(summary=summary)

    def add_tag(self, tag: str) -> None:
        with session_scope() as session:
            run = session.get(Run, self.run_id)
            if run is not None and tag not in run.tags:
                run.tags = [*run.tags, tag]

    # -- completion ----------------------------------------------------------

    def finish(self, status: str = RunStatus.FINISHED, *, error: str = "") -> None:
        if self._finished:
            return
        self.flush()
        duration = time.perf_counter() - self._started
        with session_scope() as session:
            run = session.get(Run, self.run_id)
            if run is not None:
                run.status = status
                run.ended_at = utcnow()
                run.duration_s = duration
                if error:
                    run.error = error
        self._finished = True
        self.status = status
        run_finished.send(sender=None, run=self, status=status)
        logger.info("Run %s %s in %.2fs", self.uuid[:8], status, duration)

    def fail(self, exc: BaseException) -> None:
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.finish(RunStatus.FAILED, error=detail[-8000:])
        run_failed.send(sender=None, run=self, exception=exc)

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> RunContext:
        # A training run is a unit of work with a start, an end and a status,
        # which is exactly what a span is. Entered here so it closes on the way
        # out whichever way the run ends.
        span = telemetry.span(
            f"mlango.{self.kind or 'run'}", target=self.target, run=self.uuid, kind=self.kind
        )
        self._span: Any = span
        self._span_scope: Any = span.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Returning None (never True) so the exception keeps propagating: the run
        # is recorded as failed, but the caller still sees what went wrong.
        try:
            if exc is not None:
                if isinstance(exc, KeyboardInterrupt):
                    self.finish(RunStatus.KILLED, error="Interrupted by user.")
                else:
                    self.fail(exc)
                return
            self.finish()
        finally:
            span = getattr(self, "_span", None)
            if span is not None:
                telemetry.annotate(getattr(self, "_span_scope", None), status=self.status)
                span.__exit__(exc_type, exc, tb)
                self._span = None

    # -- reading back --------------------------------------------------------

    def refresh(self) -> Run | None:
        with session_scope() as session:
            return session.get(Run, self.run_id)

    @property
    def short_id(self) -> str:
        return self.uuid[:8]

    def __repr__(self) -> str:
        return f"<RunContext {self.short_id}>"


# --------------------------------------------------------------------------- #
# Queries used by the CLI and admin
# --------------------------------------------------------------------------- #


def recent_runs(
    limit: int = 20, *, kind: str | None = None, target: str | None = None
) -> list[Run]:
    from sqlalchemy import select

    with session_scope() as session:
        statement = select(Run).order_by(Run.started_at.desc()).limit(limit)
        if kind:
            statement = statement.where(Run.kind == kind)
        if target:
            statement = statement.where(Run.target == target)
        return list(session.execute(statement).scalars())


def get_run(reference: str) -> Run | None:
    """Look a run up by uuid prefix, name or numeric id.

    The returned object outlives its session, so its collections are loaded up
    front: ``get_run(x).artifacts`` in a shell or a template has to return a list
    rather than a DetachedInstanceError.
    """
    from sqlalchemy import or_, select
    from sqlalchemy.orm import selectinload

    eager = (
        selectinload(Run.artifacts),
        selectinload(Run.metrics),
        selectinload(Run.model_versions),
        selectinload(Run.eval_results),
    )

    with session_scope() as session:
        if reference.isdigit():
            found = (
                session.execute(select(Run).where(Run.id == int(reference)).options(*eager))
                .scalars()
                .first()
            )
            if found is not None:
                return found
        return (
            session.execute(
                select(Run)
                .where(or_(Run.uuid.startswith(reference), Run.name == reference))
                .order_by(Run.started_at.desc())
                .options(*eager)
            )
            .scalars()
            .first()
        )


def metric_history(run_id: int, key: str) -> list[tuple[int, float]]:
    from sqlalchemy import select

    with session_scope() as session:
        rows = session.execute(
            select(Metric.step, Metric.value)
            .where(Metric.run_id == run_id, Metric.key == key)
            .order_by(Metric.step)
        ).all()
    return [(step, value) for step, value in rows]


def metric_keys(run_id: int) -> list[str]:
    from sqlalchemy import select

    with session_scope() as session:
        rows = session.execute(
            select(Metric.key).where(Metric.run_id == run_id).distinct()
        ).scalars()
        return sorted(rows)


def iter_runs(**filters: Any) -> Iterator[Run]:
    from sqlalchemy import select

    with session_scope() as session:
        statement = select(Run).order_by(Run.started_at.desc())
        for key, value in filters.items():
            statement = statement.where(getattr(Run, key) == value)
        yield from session.execute(statement).scalars()


# --------------------------------------------------------------------------- #
# Environment capture
# --------------------------------------------------------------------------- #


def _measure(path: str) -> tuple[int, str]:
    """Size and SHA-256 of a stored artifact, as far as the backend allows.

    Artifacts are recorded by storage-relative name so a run on one machine can
    be read on another, which means this has to ask storage rather than the
    filesystem. On a backend with no local path the hash is left empty instead
    of downloading the object to produce one — the size still lands, and a
    checkpoint is not worth a round trip to fingerprint at write time.
    """
    from mlango.core.hashing import file_digest
    from mlango.storage import default_storage

    if os.path.isabs(path) and os.path.exists(path):
        return (os.path.getsize(path), file_digest(path) if os.path.isfile(path) else "")

    storage = default_storage()
    try:
        local = storage.locate(path)
    except Exception:  # noqa: BLE001 - a remote backend says so by raising
        local = ""

    if local and os.path.exists(local):
        return (os.path.getsize(local), file_digest(local) if os.path.isfile(local) else "")

    try:
        return (storage.size(path), "")
    except Exception:  # noqa: BLE001 - a directory artifact, or already gone
        return (0, "")


def _environment() -> dict[str, Any]:
    commit, dirty = _git_state()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "host": socket.gethostname()[:255],
        "python_version": platform.python_version(),
    }


def _git_state() -> tuple[str, bool]:
    """Best-effort commit id. A project outside git is perfectly normal."""
    try:
        from mlango.conf import settings

        cwd = str(settings.BASE_DIR)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if commit.returncode != 0:
            return "", False
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return commit.stdout.strip()[:64], bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return "", False


#: Kept as a module-level alias so existing imports keep working; the
#: implementation lives in core so no layer has to depend on training for it.
_jsonable = jsonable


def set_global_seed(seed: int | None) -> None:
    """Seed python, numpy and torch together so runs are comparable."""
    if seed is None:
        return
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    if "torch" in sys.modules:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not seed torch", exc_info=True)


__all__ = [
    "RunContext",
    "recent_runs",
    "get_run",
    "metric_history",
    "metric_keys",
    "iter_runs",
    "set_global_seed",
]
