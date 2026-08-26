"""Hyperparameter sweeps.

Fields already carry ``tunable=True``; this is what makes that declaration pay
off. A sweep is one parent run plus a child run per trial, so the admin shows
the whole search as a unit and every trial keeps its own full record.

    python manage.py sweep reviews.Sentiment -p C=0.5,1,2 -p max_features=500,5000
"""

from __future__ import annotations

import concurrent.futures
import functools
import itertools
import logging
import random
import threading
from dataclasses import dataclass, field
from typing import Any

from mlango.core.exceptions import ImproperlyConfigured, RunError
from mlango.core.typing import ModelClass
from mlango.metastore.models import RunKind, RunStatus
from mlango.training.run import RunContext

logger = logging.getLogger("mlango.sweep")

GRID = "grid"
RANDOM = "random"


@dataclass
class Trial:
    """One point in the search space and what it scored."""

    index: int
    params: dict[str, Any]
    score: float | None = None
    run_uuid: str = ""
    status: str = RunStatus.PENDING
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.FINISHED and self.score is not None


@dataclass
class SweepResult:
    """The outcome of a whole sweep."""

    model_label: str
    metric: str
    mode: str
    trials: list[Trial] = field(default_factory=list)
    run_uuid: str = ""

    @property
    def completed(self) -> list[Trial]:
        return [t for t in self.trials if t.ok]

    @property
    def best(self) -> Trial | None:
        if not self.completed:
            return None
        pick = max if self.mode == "max" else min
        return pick(self.completed, key=lambda t: t.score)  # type: ignore[arg-type,return-value]

    def ranked(self) -> list[Trial]:
        return sorted(
            self.completed,
            key=lambda t: t.score,  # type: ignore[arg-type,return-value]
            reverse=self.mode == "max",
        )

    def summary(self) -> dict[str, Any]:
        best = self.best
        return {
            "trials": len(self.trials),
            "completed": len(self.completed),
            "failed": len(self.trials) - len(self.completed),
            "metric": self.metric,
            "mode": self.mode,
            f"best_{self.metric}": best.score if best else None,
        }

    def __repr__(self) -> str:
        best = self.best
        detail = f"best {self.metric}={best.score:.4f}" if best else "no completed trials"
        return f"<SweepResult {self.model_label}: {len(self.trials)} trials, {detail}>"


def expand_grid(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Every combination, in a stable order."""
    if not space:
        return [{}]
    names = list(space)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(space[n] for n in names))
    ]


def sample_space(
    space: dict[str, list[Any]], n: int, *, seed: int | None = None
) -> list[dict[str, Any]]:
    """``n`` random draws, without repeating a combination when possible."""
    grid = expand_grid(space)
    rng = random.Random(seed)
    if n >= len(grid):
        return grid
    return rng.sample(grid, n)


def _run_trial(
    model_class: ModelClass,
    index: int,
    params: dict[str, Any],
    *,
    metric: str,
    name: str,
    tags: list[str],
    train_kwargs: dict[str, Any],
) -> Trial:
    """One point in the space, with its failure recorded rather than raised.

    A bad corner of a search space should cost that corner and nothing else,
    which matters more when trials run concurrently: one raising thread would
    otherwise take the pool down with the results already collected.
    """
    trial = Trial(index=index, params=params)
    opts = model_class._meta

    try:
        model = model_class(**params)
        child = model.train(
            name=f"{name or opts.object_name}-trial{index}", tags=tags, **train_kwargs
        )
        trial.run_uuid = child.uuid
        record = child.refresh()
        trial.status = record.status if record else RunStatus.FAILED
        summary = (record.summary if record else None) or {}
        value = summary.get(metric)
        trial.score = float(value) if isinstance(value, (int, float)) else None

        if trial.score is None:
            trial.error = (
                f"Trial finished but recorded no {metric!r}. "
                f"Available: {', '.join(sorted(summary)) or '(none)'}."
            )
    except Exception as exc:  # noqa: BLE001 - recorded, then the sweep continues
        trial.status = RunStatus.FAILED
        trial.error = f"{type(exc).__name__}: {exc}"
        logger.warning("Trial %s of %s failed: %s", index, opts.label, exc)

    return trial


def run_sweep(
    model_class: ModelClass,
    space: dict[str, list[Any]],
    *,
    strategy: str = GRID,
    trials: int | None = None,
    metric: str | None = None,
    mode: str | None = None,
    seed: int | None = None,
    name: str = "",
    tags: list[str] | None = None,
    notes: str = "",
    promote_best: str | None = None,
    workers: int = 1,
    on_trial: Any = None,
    **train_kwargs: Any,
) -> SweepResult:
    """Train ``model_class`` once per point in ``space`` and report the best.

    A failing trial is recorded and the sweep continues — one bad corner of the
    search space should not throw away the trials that already succeeded.

    ``workers`` above 1 runs trials concurrently in threads. That is a real
    speed-up for sklearn and torch, whose numeric work releases the GIL, and it
    carries one honest cost: the RNG seed is process-global, so concurrent
    trials no longer each begin from the same state. A sweep is a search rather
    than a result to reproduce, which is why the option exists — but re-run the
    winning point on its own if you need its exact number back.
    """
    opts = model_class._meta

    unknown = [name_ for name_ in space if not opts.has_field(name_)]
    if unknown:
        available = ", ".join(opts.field_names) or "(none)"
        raise ImproperlyConfigured(
            f"{opts.label} has no hyperparameter(s) {', '.join(unknown)}. Available: {available}."
        )

    if metric is None or mode is None:
        default_metric, default_mode = model_class.monitor()
        metric = metric or default_metric
        mode = mode or default_mode
    if mode not in {"min", "max"}:
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")

    if strategy == RANDOM:
        if not trials:
            raise ImproperlyConfigured("A random sweep needs --trials to say how many to run.")
        points = sample_space(space, trials, seed=seed)
    elif strategy == GRID:
        points = expand_grid(space)
        if trials:
            points = points[:trials]
    else:
        raise ImproperlyConfigured(f"Unknown sweep strategy {strategy!r}; use 'grid' or 'random'.")

    result = SweepResult(model_label=opts.label, metric=metric, mode=mode)

    parent = RunContext.start(
        kind=RunKind.SWEEP,
        target=opts.label,
        name=name,
        params={
            "_strategy": strategy,
            "_space": {k: list(v) for k, v in space.items()},
            "_metric": metric,
            "_mode": mode,
            "_trials": len(points),
        },
        tags=tags,
        notes=notes,
        seed=seed,
    )
    result.run_uuid = parent.uuid
    sweep_tag = f"sweep:{parent.short_id}"

    run_one = functools.partial(
        _run_trial,
        model_class,
        metric=metric,
        name=name,
        tags=[*(tags or []), sweep_tag],
        train_kwargs=train_kwargs,
    )

    with parent:
        if workers > 1:
            done = threading.Lock()
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(run_one, index, params): index
                    for index, params in enumerate(points, start=1)
                }
                for future in concurrent.futures.as_completed(futures):
                    trial = future.result()
                    # Serialised so a callback that prints does not interleave
                    # two trials mid-line.
                    with done:
                        result.trials.append(trial)
                        if on_trial is not None:
                            on_trial(trial, result)
            # Sorted afterwards: trials finish out of order, and a report whose
            # rows jump around is harder to read than one that waits.
            result.trials.sort(key=lambda t: t.index)
        else:
            for index, params in enumerate(points, start=1):
                trial = run_one(index, params)
                result.trials.append(trial)
                if on_trial is not None:
                    on_trial(trial, result)

        # Logged after the pool rather than inside it: the parent's metric
        # buffer is not built for concurrent writers, and a sweep's curve is
        # only meaningful in trial order anyway.
        for trial in result.trials:
            if trial.score is not None:
                parent.log_metric(metric, trial.score, step=trial.index)

        best = result.best
        if best is not None:
            parent.set_summary({**result.summary(), "best_params": best.params})
            parent.log_json(
                "sweep.json",
                {
                    "summary": result.summary(),
                    "trials": [
                        {
                            "index": t.index,
                            "params": t.params,
                            "score": t.score,
                            "run": t.run_uuid,
                            "status": t.status,
                            "error": t.error,
                        }
                        for t in result.trials
                    ],
                },
            )
        else:
            parent.set_summary(result.summary())

    if promote_best:
        best = result.best
        if best is None:
            raise RunError("No trial completed, so there is nothing to promote.")
        _promote(model_class, best, promote_best)

    return result


def _promote(model_class: ModelClass, trial: Trial, stage: str) -> None:
    """Promote the version registered by the winning trial."""
    from sqlalchemy import select

    from mlango.metastore.models import ModelVersion, Run
    from mlango.metastore.session import session_scope

    with session_scope() as session:
        run = session.execute(select(Run).where(Run.uuid == trial.run_uuid)).scalar_one_or_none()
        version = None
        if run is not None:
            version = (
                session.execute(select(ModelVersion).where(ModelVersion.run_id == run.id))
                .scalars()
                .first()
            )
        number = version.version if version else None

    if number is None:
        raise RunError(
            f"Trial {trial.index} registered no model version, so it cannot be promoted."
        )
    model_class.promote(number, stage)


__all__ = ["run_sweep", "SweepResult", "Trial", "expand_grid", "sample_space", "GRID", "RANDOM"]
