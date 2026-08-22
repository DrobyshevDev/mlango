"""The ``Eval`` declarative class.

    class AnswerQuality(Eval):
        \"\"\"Does the support agent answer from the docs?\"\"\"

        class Meta:
            dataset = SupportCases
            target = SupportAgent
            input_field = "question"
            expected_field = "answer"
            scorers = {"overlap": token_f1, "cited": contains_all("docs/")}
            threshold = 0.6

Running it produces a ``Run`` plus one ``EvalResult`` per case, so a
regression shows up as a diff between two runs rather than as a number someone
remembers from last week.
"""

from __future__ import annotations

import logging
from typing import Any

from mlango.core.base import Declarative
from mlango.core.exceptions import ImproperlyConfigured
from mlango.core.serialization import jsonable as _jsonable
from mlango.core.typing import DatasetClass
from mlango.evals.scorers import Scorer
from mlango.metastore.models import RunKind
from mlango.training.run import RunContext

logger = logging.getLogger("mlango.evals")


class EvalReport:
    """Aggregate outcome of one evaluation run."""

    def __init__(self, label: str, run: RunContext):
        self.label = label
        self.run = run
        self.cases: list[dict[str, Any]] = []

    def add(self, case: dict[str, Any]) -> None:
        self.cases.append(case)

    # -- aggregates ----------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.get("passed"))

    @property
    def failed(self) -> int:
        return sum(1 for c in self.cases if c.get("passed") is False)

    @property
    def errored(self) -> int:
        return sum(1 for c in self.cases if c.get("error"))

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def mean_scores(self) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for case in self.cases:
            for key, value in (case.get("scores") or {}).items():
                if isinstance(value, (int, float, bool)):
                    totals.setdefault(key, []).append(float(value))
        return {key: sum(values) / len(values) for key, values in totals.items() if values}

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "pass_rate": self.pass_rate,
            **{f"score_{k}": v for k, v in self.mean_scores().items()},
        }

    def failures(self) -> list[dict[str, Any]]:
        return [c for c in self.cases if not c.get("passed")]

    def __repr__(self) -> str:
        return f"<EvalReport {self.label}: {self.passed}/{self.total} passed>"


class Eval(Declarative):
    """A declared evaluation suite."""

    _kind = "eval"
    _meta_options = (
        "dataset",
        "target",
        "input_field",
        "expected_field",
        "case_id_field",
        "scorers",
        "threshold",
        "max_cases",
        "fail_fast",
    )

    class Meta:
        abstract = True

    # -- configuration -------------------------------------------------------

    @classmethod
    def get_dataset(cls) -> DatasetClass:
        dataset = cls._meta.extras.get("dataset")
        if dataset is None:
            raise ImproperlyConfigured(
                f"{cls._meta.label}.Meta.dataset is not set, so there are no cases to run."
            )
        if isinstance(dataset, str):
            from mlango.core.registry import apps

            return apps.get_dataset(dataset)
        return dataset

    @classmethod
    def get_target(cls) -> Any:
        target = cls._meta.extras.get("target")
        if target is None:
            raise ImproperlyConfigured(
                f"{cls._meta.label}.Meta.target is not set. Point it at an Agent or a Model, "
                f"or override predict()."
            )
        if isinstance(target, str):
            from mlango.core.registry import apps

            _kind, resolved = apps.find(target)
            return resolved
        return target

    @classmethod
    def get_scorers(cls) -> dict[str, Scorer]:
        return dict(cls._meta.extras.get("scorers") or {})

    @classmethod
    def get_threshold(cls) -> float | None:
        threshold = cls._meta.extras.get("threshold")
        return float(threshold) if threshold is not None else None

    @classmethod
    def input_field(cls) -> str:
        return str(cls._meta.extras.get("input_field") or "input")

    @classmethod
    def expected_field(cls) -> str | None:
        declared = cls._meta.extras.get("expected_field")
        return str(declared) if declared else None

    # -- hooks ---------------------------------------------------------------

    def cases(self) -> Any:
        """The cases to evaluate. Override to filter or sample."""
        dataset = self.get_dataset()
        query = dataset.objects.get_queryset()
        limit = self._meta.extras.get("max_cases")
        return query.take(int(limit)) if limit else query

    def predict(self, case: Any) -> Any:
        """Run the target on one case.

        The default understands agents and models; override for anything else
        — a pipeline, an HTTP endpoint, a human-in-the-loop stub.
        """
        runnable = _resolve_target(self.get_target())
        value = case.get(self.input_field())

        if hasattr(runnable, "run") and hasattr(runnable, "get_tools"):
            return runnable.run(value)
        if hasattr(runnable, "predict"):
            return runnable.predict(value)
        if callable(runnable):
            return runnable(value)
        raise ImproperlyConfigured(
            f"{self._meta.label}: don't know how to run {runnable!r}. Override predict()."
        )

    def score(self, case: Any, output: Any) -> dict[str, Any] | float | bool:
        """Score one case. The default applies ``Meta.scorers``."""
        scorers = self.get_scorers()
        if not scorers:
            raise ImproperlyConfigured(
                f"{self._meta.label} declares no scorers and does not override score()."
            )
        expected_name = self.expected_field()
        expected = case.get(expected_name) if expected_name else None
        text = _as_text(output)
        return {
            name: fn(output if _wants_run(fn) else text, expected) for name, fn in scorers.items()
        }

    def decide(self, scores: dict[str, Any]) -> bool | None:
        """Turn per-scorer values into pass/fail. ``None`` means "no verdict"."""
        threshold = self.get_threshold()
        numeric = [float(v) for v in scores.values() if isinstance(v, (int, float, bool))]
        if not numeric:
            return None
        if threshold is None:
            return all(bool(v) for v in scores.values())
        return sum(numeric) / len(numeric) >= threshold

    # -- running -------------------------------------------------------------

    @classmethod
    def evaluate(cls, **kwargs: Any) -> EvalReport:
        """Shortcut: ``AnswerQuality.evaluate()``."""
        return cls().run(**kwargs)

    def run(
        self,
        *,
        name: str = "",
        tags: list[str] | None = None,
        notes: str = "",
        progress: bool = False,
    ) -> EvalReport:
        from mlango.metastore.models import EvalResult
        from mlango.metastore.session import session_scope

        opts = type(self)._meta
        target = self.get_target()
        target_label = getattr(getattr(target, "_meta", None), "label", str(target))
        fail_fast = bool(opts.extras.get("fail_fast", False))

        run = RunContext.start(
            kind=RunKind.EVAL,
            target=opts.label,
            name=name,
            params={
                "_eval": opts.label,
                "_target": target_label,
                "_dataset": self.get_dataset()._meta.label,
                "_scorers": sorted(self.get_scorers()),
                "_threshold": self.get_threshold(),
                # What the target was configured like when this ran. Without
                # it a comparison between two runs can say the answers moved
                # and not why, which leaves the user to remember what they
                # changed — exactly the thing a framework should not ask.
                **_target_state(target),
            },
            tags=tags,
            notes=notes,
        )
        report = EvalReport(opts.label, run)

        with run:
            case_id_field = opts.extras.get("case_id_field")
            for index, case in enumerate(self.cases()):
                case_id = str(case.get(case_id_field)) if case_id_field else str(index)
                entry: dict[str, Any] = {"case_id": case_id, "inputs": dict(case)}

                try:
                    output = self.predict(case)
                    raw = self.score(case, output)
                    scores = raw if isinstance(raw, dict) else {"score": raw}
                    entry["output"] = _as_text(output)
                    entry["scores"] = scores
                    entry["passed"] = self.decide(scores)
                    entry["trace_uuid"] = getattr(output, "trace_uuid", "") or ""
                except Exception as exc:  # noqa: BLE001 - recorded per case
                    logger.exception("Case %s of %s failed", case_id, opts.label)
                    entry.update(
                        {"error": f"{type(exc).__name__}: {exc}", "passed": False, "scores": {}}
                    )
                    if fail_fast:
                        report.add(entry)
                        self._persist(run, opts.label, [entry], EvalResult, session_scope)
                        raise

                expected_name = self.expected_field()
                if expected_name:
                    entry["expected"] = case.get(expected_name)

                report.add(entry)
                if progress:
                    mark = "ok" if entry.get("passed") else "FAIL"
                    print(f"[{run.short_id}] case {case_id}: {mark}")

            self._persist(run, opts.label, report.cases, EvalResult, session_scope)

            summary = report.summary()
            run.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
            run.set_summary(summary)
            run.log_json("eval_report.json", {"summary": summary, "cases": report.cases})

        return report

    @staticmethod
    def _persist(run, label, cases, EvalResult, session_scope) -> None:
        if not cases:
            return
        with session_scope() as session:
            session.add_all(
                EvalResult(
                    run_id=run.run_id,
                    eval_label=label,
                    case_id=case.get("case_id", ""),
                    passed=case.get("passed"),
                    scores=_jsonable(case.get("scores") or {}),
                    inputs=_jsonable(case.get("inputs") or {}),
                    output=_jsonable(case.get("output")),
                    expected=_jsonable(case.get("expected")),
                    trace_uuid=case.get("trace_uuid", "") or "",
                    error=case.get("error", ""),
                )
                for case in cases
            )

    # -- introspection -------------------------------------------------------

    @classmethod
    def summary(cls) -> dict[str, Any]:
        extras = cls._meta.extras
        return {
            "label": cls._meta.label,
            "dataset": _label_of(extras.get("dataset")),
            "target": _label_of(extras.get("target")),
            "scorers": sorted(cls.get_scorers()),
            "threshold": cls.get_threshold(),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _target_state(target: Any) -> dict[str, Any]:
    """The evaluated thing's identity and configuration, as far as it has one.

    An agent's is its declaration: the prompt, the model, the step limit. A
    model's is the registered version it will load, because its behaviour is
    the artifact rather than the class. Anything else contributes nothing and
    says so by omission rather than by a null.
    """
    try:
        opts = getattr(target, "_meta", None)
        if opts is None:
            return {}

        state: dict[str, Any] = {"_target_fingerprint": opts.fingerprint()}
        if opts.kind == "agent":
            state["_target_config"] = opts.recordable()
            current = target.current_version()
            if current is not None:
                state["_target_version"] = current.version
            return state

        if opts.kind == "model":
            versions = target.versions()
            if versions:
                newest = versions[0]
                state["_target_version"] = newest.version
                state["_target_config"] = dict(newest.params or {})
        return state
    except Exception:  # noqa: BLE001 - the whole body, deliberately
        # Every line here describes the run rather than performing it, so any
        # failure costs a note in the report and must never cost the run.
        # getattr with a default does not help: a property that raises anything
        # other than AttributeError propagates straight through it.
        logger.debug("Could not record the target's state", exc_info=True)
        return {}


def _resolve_target(target: Any) -> Any:
    """Turn a declared class into something ready to be called once per case.

    ``Meta.target`` is normally a class, and the two families need different
    treatment: an agent is cheap to construct, while a model has to come back
    from its registered version — otherwise the eval would score an untrained
    object and report a number that means nothing. Anything already
    instantiated, or a plain callable, is passed straight through.
    """
    if not isinstance(target, type):
        return target
    declared: Any = target
    if hasattr(declared, "get_tools"):
        return declared()
    if hasattr(declared, "load"):
        return declared.load()
    return declared


def _as_text(output: Any) -> str:
    text = getattr(output, "output", None)
    if isinstance(text, str):
        return text
    return output if isinstance(output, str) else str(output)


def _wants_run(fn: Any) -> bool:
    """True for scorers that opted into receiving the whole run object.

    Scorers mark themselves with ``fn.wants_run = True`` (see
    :func:`~mlango.evals.scorers.used_tool`); everything else gets the text.
    """
    return bool(getattr(fn, "wants_run", False))


def _label_of(value: Any) -> Any:
    meta = getattr(value, "_meta", None)
    return meta.label if meta is not None else value


__all__ = ["Eval", "EvalReport"]
