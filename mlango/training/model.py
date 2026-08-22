"""The ``Model`` declarative class.

    class Sentiment(Model):
        max_features = IntegerField(default=20_000, tunable=True)
        C = FloatField(default=1.0, tunable=True)

        class Meta:
            dataset = Reviews
            trainer = "sklearn"
            task = "classification"

        def build(self):
            return make_pipeline(TfidfVectorizer(max_features=self.max_features),
                                 LogisticRegression(C=self.C))

Hyperparameters are fields, so they are validated, defaulted, introspectable,
sweepable and recorded on every run without the user writing any of that.
"""

from __future__ import annotations

import contextvars
import logging
import random
import time
from typing import Any

from mlango.core.base import Declarative
from mlango.core.exceptions import ImproperlyConfigured, RunError
from mlango.core.signals import model_registered, post_predict, pre_predict
from mlango.core.typing import DatasetClass
from mlango.metastore.models import RunKind, Stage
from mlango.training import metrics as metric_lib
from mlango.training.callbacks import Callback, CallbackList, build_callbacks
from mlango.training.run import RunContext, set_global_seed
from mlango.training.trainer import Trainer, get_trainer

logger = logging.getLogger("mlango.model")

#: The request a prediction belongs to, set by the serving layer so a shadow
#: row and the row it shadowed can be paired later. A ContextVar rather than an
#: argument because predict() is called by user code that has no idea a shadow
#: exists, and rather than a global because one worker serves many requests at
#: once.
current_request: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mlango_request_id", default=None
)


class Model(Declarative):
    """A declared, trainable model."""

    _kind = "model"
    _meta_options = (
        "dataset",
        "trainer",
        "task",
        "target",
        "features",
        "exclude",
        "metrics",
        "monitor",
        "monitor_mode",
        "splits",
        "callbacks",
        "serve_input",
    )

    class Meta:
        abstract = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._fitted: Any = None
        self._version: Any = None

    # -- declaration ---------------------------------------------------------

    def build(self) -> Any:
        """Return the underlying estimator or module. Must be overridden."""
        raise NotImplementedError(
            f"{type(self)._meta.label} must implement build() and return the object "
            f"its trainer knows how to fit."
        )

    @property
    def params(self) -> dict[str, Any]:
        """The hyperparameters, cleaned and complete."""
        return self.full_clean()

    # -- configuration -------------------------------------------------------

    @classmethod
    def get_trainer(cls) -> Trainer:
        name = cls._meta.extras.get("trainer")
        if not name:
            raise ImproperlyConfigured(
                f"{cls._meta.label}.Meta.trainer is not set. Pick one of the keys in the "
                f"TRAINERS setting, e.g. trainer = 'sklearn'."
            )
        return get_trainer(str(name))

    @classmethod
    def get_dataset(cls) -> DatasetClass:
        dataset = cls._meta.extras.get("dataset")
        if dataset is None:
            raise ImproperlyConfigured(
                f"{cls._meta.label}.Meta.dataset is not set, so there is nothing to train on. "
                f"Set it, or pass dataset= to train()."
            )
        if isinstance(dataset, str):
            from mlango.core.registry import apps

            return apps.get_dataset(dataset)
        return dataset

    @classmethod
    def get_task(cls) -> str:
        return str(cls._meta.extras.get("task", "classification"))

    @classmethod
    def get_target(cls, dataset: DatasetClass | None = None) -> str:
        target = cls._meta.extras.get("target")
        if target:
            return str(target)
        dataset = dataset or cls.get_dataset()
        targets = dataset._meta.target_fields
        if len(targets) != 1:
            raise ImproperlyConfigured(
                f"{dataset._meta.label} declares {len(targets)} target fields; set "
                f"`target` in {cls._meta.label}.Meta to say which one to predict."
            )
        return targets[0].name or ""

    @classmethod
    def get_features(cls, dataset: DatasetClass | None = None) -> list[str]:
        """Which dataset fields this model consumes.

        ``Meta.features`` is explicit and wins. Otherwise every non-target field
        is a feature, minus the dataset's primary key — an id is bookkeeping,
        and quietly training on it is a classic way to leak.
        """
        dataset = dataset or cls.get_dataset()
        declared = cls._meta.extras.get("features")
        if declared:
            names = [str(n) for n in declared]
            unknown = [n for n in names if not dataset._meta.has_field(n)]
            if unknown:
                raise ImproperlyConfigured(
                    f"{cls._meta.label}.Meta.features names field(s) {', '.join(unknown)} that "
                    f"{dataset._meta.label} does not declare."
                )
            return names

        excluded = {str(n) for n in (cls._meta.extras.get("exclude") or ())}
        primary_key = dataset._meta.extras.get("primary_key")
        if primary_key:
            excluded.add(str(primary_key))
        target = cls.get_target(dataset)
        excluded.add(target)
        return [f.name or "" for f in dataset._meta.fields if (f.name or "") not in excluded]

    @classmethod
    def get_splits(cls) -> dict[str, float]:
        splits = cls._meta.extras.get("splits")
        if splits:
            return dict(splits)
        return {"train": 0.8, "val": 0.2}

    @classmethod
    def monitor(cls) -> tuple[str, str]:
        default = ("val_loss", "min") if cls.get_task() == "regression" else ("accuracy", "max")
        key = str(cls._meta.extras.get("monitor", default[0]))
        mode = str(cls._meta.extras.get("monitor_mode", default[1]))
        return key, mode

    # -- training ------------------------------------------------------------

    @classmethod
    def fit(cls, **kwargs: Any) -> Model:
        """Instantiate with hyperparameters, train, and return the model."""
        hyperparams = {k: v for k, v in kwargs.items() if cls._meta.has_field(k)}
        options = {k: v for k, v in kwargs.items() if not cls._meta.has_field(k)}
        model = cls(**hyperparams)
        model.train(**options)
        return model

    def train(
        self,
        *,
        dataset: DatasetClass | None = None,
        queryset: Any = None,
        splits: dict[str, float] | None = None,
        callbacks: list[Callback] | None = None,
        name: str = "",
        tags: list[str] | None = None,
        notes: str = "",
        seed: int | None = None,
        materialize: bool = False,
        register: bool = True,
        **trainer_kwargs: Any,
    ) -> RunContext:
        """Train the model, recording everything about the run.

        ``materialize=True`` freezes the training view into a dataset version
        first, which costs a full pass but makes the run exactly reproducible
        even if the upstream source changes.
        """
        from mlango.conf import settings

        opts = type(self)._meta
        dataset_class: DatasetClass = dataset or (
            queryset.dataset if queryset is not None else self.get_dataset()
        )
        trainer = self.get_trainer()
        params = self.params
        target = self.get_target(dataset_class)
        features = self.get_features(dataset_class)

        base_query = queryset if queryset is not None else dataset_class.objects.get_queryset()

        effective_seed = settings.SEED if seed is None else seed
        set_global_seed(effective_seed)

        dataset_version_id = None
        if materialize:
            version = dataset_class.materialize(base_query, notes=f"training {opts.label}")
            dataset_version_id = version.id
            base_query = dataset_class.load_version(version.version)

        split_ratios = splits or self.get_splits()
        parts = base_query.split(**split_ratios) if split_ratios else {"train": base_query}
        train_query = parts.get("train", base_query).cache()
        validation_query = parts.get("val") or parts.get("validation")
        if validation_query is not None:
            validation_query = validation_query.cache()

        callback_list: CallbackList = build_callbacks(callbacks)
        device = trainer.resolve_device()

        run = RunContext.start(
            kind=RunKind.TRAIN,
            target=opts.label,
            name=name,
            params={
                **params,
                "_dataset": dataset_class._meta.label,
                "_target": target,
                "_features": features,
                "_trainer": trainer.name,
                "_task": self.get_task(),
                "_splits": split_ratios,
                "_data_fingerprint": base_query.fingerprint(),
            },
            tags=tags,
            dataset_version_id=dataset_version_id,
            seed=effective_seed,
            device=device,
            notes=notes,
        )

        try:
            with run:
                callback_list.emit("on_train_begin", run, self, trainer=trainer)
                fitted = trainer.fit(
                    self,
                    train_query,
                    validation_query,
                    run,
                    callback_list,
                    target=target,
                    features=features,
                    **trainer_kwargs,
                )
                self._fitted = fitted

                report: dict[str, Any] = {}
                if validation_query is not None:
                    report = self.evaluate(validation_query, target=target, run=run)
                    run.log_metrics(metric_lib.flatten_report(report), split="eval")
                    callback_list.emit("on_evaluate_end", run, report, model=self)

                callback_list.emit(
                    "on_train_end", run, self, trainer=trainer, fitted=fitted, metrics=report
                )

                summary = metric_lib.flatten_report(report)
                run.set_summary({**summary, **trainer.describe(self, fitted)})

                if register:
                    self._register_version(
                        run,
                        trainer,
                        summary,
                        params,
                        baseline=_baseline(train_query, features, target),
                    )
        except RunError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise RunError(f"Training {opts.label} failed: {exc}") from exc

        return run

    def _register_version(
        self,
        run: RunContext,
        trainer: Trainer,
        summary: dict[str, float],
        params: dict[str, Any],
        baseline: dict[str, Any] | None = None,
    ) -> None:
        from sqlalchemy import func, select

        from mlango.metastore.models import ModelVersion
        from mlango.metastore.session import session_scope
        from mlango.training.run import _jsonable

        opts = type(self)._meta
        path = trainer.save(self, self._fitted, f"models/{opts.label.replace('.', '/')}/{run.uuid}")
        run.log_artifact("model", path, kind="model")

        with session_scope() as session:
            highest = session.execute(
                select(func.max(ModelVersion.version)).where(ModelVersion.label == opts.label)
            ).scalar()
            version = ModelVersion(
                label=opts.label,
                version=(highest or 0) + 1,
                fingerprint=opts.fingerprint(),
                run_id=run.run_id,
                path=path,
                params=_jsonable(params),
                metrics=_jsonable(summary),
                importances=_importances(self, trainer),
                baseline=baseline,
                stage=Stage.NONE,
            )
            session.add(version)
            session.flush()
            self._version = version

        model_registered.send(sender=type(self), model=self, version=self._version)
        logger.info("Registered %s", self._version.ref)

    # -- inference -----------------------------------------------------------

    @property
    def fitted(self) -> Any:
        if self._fitted is None:
            raise RunError(
                f"{type(self)._meta.label} has no fitted weights. Call train(), or load a "
                f"registered version with {type(self).__name__}.load()."
            )
        return self._fitted

    def predict(self, inputs: Any) -> Any:
        """Predict for one input or a list of them."""
        single = not isinstance(inputs, (list, tuple))
        batch = [inputs] if single else list(inputs)
        pre_predict.send(sender=type(self), model=self, inputs=batch)
        started = time.perf_counter()
        outputs = self.get_trainer().predict(self, self.fitted, batch)
        elapsed_ms = (time.perf_counter() - started) * 1000
        post_predict.send(sender=type(self), model=self, inputs=batch, outputs=outputs)
        log_predictions(self, batch, outputs, elapsed_ms)
        return outputs[0] if single else outputs

    def predict_proba(self, inputs: Any) -> Any:
        single = not isinstance(inputs, (list, tuple))
        batch = [inputs] if single else list(inputs)
        outputs = self.get_trainer().predict_proba(self, self.fitted, batch)
        if outputs is None:
            return None
        return outputs[0] if single else outputs

    def evaluate(
        self, queryset: Any = None, *, target: str | None = None, run: RunContext | None = None
    ) -> dict[str, Any]:
        """Score the fitted model over a queryset."""
        dataset_class = queryset.dataset if queryset is not None else self.get_dataset()
        query = queryset if queryset is not None else dataset_class.objects.get_queryset()
        target = target or self.get_target(dataset_class)
        features = self.get_features(dataset_class)

        records = list(query)
        if not records:
            return {"support": 0}

        inputs = [
            record.get(features[0]) if len(features) == 1 else {n: record.get(n) for n in features}
            for record in records
        ]
        truth = [record.get(target) for record in records]
        predictions = self.get_trainer().predict(self, self.fitted, inputs)

        report = metric_lib.report_for_task(self.get_task(), truth, predictions)
        if run is not None:
            run.log_json("evaluation.json", report)
        return report

    # -- versions ------------------------------------------------------------

    @classmethod
    def versions(cls) -> list[Any]:
        from sqlalchemy import select

        from mlango.metastore.models import ModelVersion
        from mlango.metastore.session import session_scope

        with session_scope() as session:
            return list(
                session.execute(
                    select(ModelVersion)
                    .where(ModelVersion.label == cls._meta.label)
                    .order_by(ModelVersion.version.desc())
                ).scalars()
            )

    @classmethod
    def load(cls, version: int | None = None, *, stage: str | None = None) -> Model:
        """Rebuild a model from a registered version, ready to predict."""
        from sqlalchemy import select

        from mlango.metastore.models import ModelVersion
        from mlango.metastore.session import session_scope

        with session_scope() as session:
            statement = select(ModelVersion).where(ModelVersion.label == cls._meta.label)
            if version is not None:
                statement = statement.where(ModelVersion.version == version)
            elif stage is not None:
                statement = statement.where(ModelVersion.stage == stage)
            row = session.execute(statement.order_by(ModelVersion.version.desc())).scalars().first()

        if row is None:
            detail = f" at stage {stage!r}" if stage else (f" v{version}" if version else "")
            raise LookupError(f"{cls._meta.label} has no registered version{detail}.")

        instance = cls(**{k: v for k, v in (row.params or {}).items() if cls._meta.has_field(k)})
        instance._fitted = cls.get_trainer().load(instance, row.path or "")
        instance._version = row
        return instance

    @classmethod
    def production(cls) -> Model:
        return cls.load(stage=Stage.PRODUCTION)

    @classmethod
    def promote(cls, version: int, stage: str = Stage.PRODUCTION) -> Any:
        """Move a version to a stage, demoting whoever held it."""
        from sqlalchemy import select

        from mlango.metastore.models import ModelVersion
        from mlango.metastore.session import session_scope

        if stage not in Stage.ALL:
            raise ValueError(f"Unknown stage {stage!r}; use one of {Stage.ALL}.")

        with session_scope() as session:
            rows = list(
                session.execute(
                    select(ModelVersion).where(ModelVersion.label == cls._meta.label)
                ).scalars()
            )
            target = next((r for r in rows if r.version == version), None)
            if target is None:
                raise LookupError(f"{cls._meta.label} has no version {version}.")
            if stage in (Stage.PRODUCTION, Stage.STAGING):
                for row in rows:
                    if row.stage == stage and row is not target:
                        row.stage = Stage.ARCHIVED
            target.stage = stage
            session.flush()
            return target

    # -- sweeps --------------------------------------------------------------

    @classmethod
    def sweep(cls, space: dict[str, list[Any]] | None = None, **options: Any) -> Any:
        """Train once per point in a search space and report the best trial.

        With no explicit ``space``, every field marked ``tunable=True`` is
        varied around its default.
        """
        from mlango.training.sweep import run_sweep

        if space is None:
            space = cls.default_space()
            if not space:
                raise ImproperlyConfigured(
                    f"{cls._meta.label} declares no tunable fields, so there is nothing to "
                    f"sweep. Mark a field with tunable=True, or pass an explicit space."
                )
        return run_sweep(cls, space, **options)

    @classmethod
    def default_space(cls) -> dict[str, list[Any]]:
        """A small search space around each tunable field's default."""
        space: dict[str, list[Any]] = {}
        for field in cls._meta.tunable_fields:
            default = field.get_default()
            if isinstance(default, bool):
                space[field.name or ""] = [True, False]
            elif isinstance(default, int):
                space[field.name or ""] = sorted({max(1, default // 2), default, default * 2})
            elif isinstance(default, float):
                space[field.name or ""] = sorted({default / 2, default, default * 2})
            elif field.choices:
                space[field.name or ""] = list(field.choices)
        return space

    # -- serving -------------------------------------------------------------

    @classmethod
    def as_endpoint(cls, *, version: int | None = None, stage: str | None = None) -> Any:
        """A prediction endpoint for ``routes.py``, à la Django's ``as_view()``."""
        from mlango.serve.endpoints import model_endpoint

        return model_endpoint(cls, version=version, stage=stage)

    # -- introspection -------------------------------------------------------

    @classmethod
    def summary(cls) -> dict[str, Any]:
        extras = cls._meta.extras
        dataset = extras.get("dataset")
        dataset_meta = getattr(dataset, "_meta", None)
        return {
            "label": cls._meta.label,
            "task": cls.get_task(),
            "trainer": extras.get("trainer"),
            # Meta.dataset may still be the string form, or absent entirely.
            "dataset": dataset_meta.label if dataset_meta is not None else dataset,
            "hyperparameters": {f.name: f.get_default() for f in cls._meta.fields},
            "tunable": [f.name for f in cls._meta.tunable_fields],
        }


def _baseline(queryset: Any, features: list[str], target: str) -> dict[str, Any] | None:
    """Summarise the training split so drift has something to compare against.

    The queryset is already cached by the time this runs, so the extra pass is
    over memory. Like the importances, a failure here loses a diagnostic and
    must not lose the model that was just trained.
    """
    from mlango.training import drift

    columns = [*features, target] if target and target not in features else list(features)
    if not columns:
        return None
    try:
        return drift.profile(queryset, columns) or None
    except Exception:  # noqa: BLE001 - see above
        logger.debug("Could not profile the training split", exc_info=True)
        return None


def log_predictions(
    model: Model, inputs: list[Any], outputs: list[Any], elapsed_ms: float | None = None
) -> int:
    """Record what a model was asked, when ``PREDICTION_LOG`` is on.

    Best effort throughout. A prediction that succeeded must be returned even
    if the metastore is unreachable, read-only, or locked by another writer —
    an observability feature that can take an endpoint down is worse than no
    observability feature.
    """
    from mlango.conf import settings

    config = settings.PREDICTION_LOG
    if not config.get("ENABLED", False) or not inputs:
        return 0

    sample = float(config.get("SAMPLE", 1.0) or 0.0)
    if sample <= 0:
        return 0
    if sample < 1.0:
        # Sampled per row rather than per batch: a batch endpoint would
        # otherwise record all of a request or none of it, and the log would
        # follow request sizes instead of the input distribution.
        keep = [index for index in range(len(inputs)) if random.random() < sample]
    else:
        keep = list(range(len(inputs)))
    if not keep:
        return 0

    try:
        from mlango.core.serialization import jsonable
        from mlango.metastore.models import Prediction
        from mlango.metastore.session import session_scope

        version = getattr(model, "_version", None)
        label = type(model)._meta.label
        request_id = current_request.get()
        # Divided out so a batch of 500 does not report each row as taking as
        # long as the whole call.
        per_row = elapsed_ms / len(inputs) if elapsed_ms is not None and inputs else None

        with session_scope() as session:
            for index in keep:
                session.add(
                    Prediction(
                        label=label,
                        version=getattr(version, "version", None),
                        inputs=jsonable(inputs[index]),
                        output=jsonable(outputs[index]) if index < len(outputs) else None,
                        latency_ms=per_row,
                        request_id=request_id,
                    )
                )
        _trim_predictions(label, int(config.get("MAX_ROWS", 0) or 0))
        return len(keep)
    except Exception:  # noqa: BLE001 - see above
        logger.debug("Could not record predictions for %s", type(model)._meta.label, exc_info=True)
        return 0


def _trim_predictions(label: str, max_rows: int) -> None:
    """Keep the newest ``max_rows`` for this model and drop the rest."""
    if max_rows <= 0:
        return
    from sqlalchemy import delete, func, select

    from mlango.metastore.models import Prediction
    from mlango.metastore.session import session_scope

    with session_scope() as session:
        total = session.execute(
            select(func.count()).select_from(Prediction).where(Prediction.label == label)
        ).scalar_one()
        if total <= max_rows:
            return
        cutoff = session.execute(
            select(Prediction.id)
            .where(Prediction.label == label)
            .order_by(Prediction.id.desc())
            .offset(max_rows - 1)
            .limit(1)
        ).scalar_one_or_none()
        if cutoff is not None:
            session.execute(
                delete(Prediction).where(Prediction.label == label, Prediction.id < cutoff)
            )


def _importances(model: Model, trainer: Trainer) -> dict[str, float] | None:
    """Ask the backend to explain the fit, and never let the answer break it.

    Explanations are a read of the fitted object, so a backend that raises here
    — an estimator whose weights are not where the attribute names suggest, a
    vectoriser that was never fitted — has failed at describing a run that
    otherwise succeeded. Losing the chart is acceptable. Losing the trained
    model because the chart raised is not.
    """
    try:
        return trainer.importances(model, model._fitted)
    except Exception:  # noqa: BLE001 - see above
        logger.debug("Could not extract importances for %s", type(model)._meta.label, exc_info=True)
        return None


__all__ = ["Model", "log_predictions", "current_request"]
