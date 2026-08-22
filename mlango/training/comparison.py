"""Comparing what two registered versions actually predict.

Metrics answer "is the new one better" in aggregate, and aggregate is exactly
where the answer you are afraid of hides: a version two points more accurate
overall can still have broken forty rows that used to work, and accuracy will
not mention it. Promotion is the moment that matters, and the only honest way
to decide is to score both versions on the same data and look at the rows where
they disagree.

This is the model's equivalent of reading a diff. It needs nothing new — both
versions are registered, the dataset is declared — only for someone to ask.
"""

from __future__ import annotations

from typing import Any

# From core, not from here: evaluations ask the same question about prompts that
# this module asks about model versions, and evals may not import training.
from mlango.core.stats import DEFAULT_ALPHA, significance
from mlango.training import metrics as metric_lib

#: Regression predictions are floats, so "did the answer change" is not a
#: question equality can answer. Anything closer than this counts as the same
#: prediction, which keeps the changed-row list about real movement.
DEFAULT_TOLERANCE = 1e-6


def compare_versions(
    model_class: Any,
    left: int,
    right: int,
    *,
    queryset: Any = None,
    limit: int | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_changes: int = 0,
) -> dict[str, Any]:
    """Score two registered versions on the same rows and diff the answers.

    Both are loaded and run over identical input, in the same order, so a
    disagreement is a disagreement about the data rather than about which rows
    each one happened to see.
    """
    dataset_class = model_class.get_dataset()
    query = queryset if queryset is not None else dataset_class.objects.get_queryset()
    if limit:
        query = query.take(limit)

    records = [dict(record) for record in query]
    if not records:
        raise LookupError(f"{dataset_class._meta.label} produced no rows to compare.")

    return compare_predictors(
        model_class.load(version=left),
        model_class.load(version=right),
        records=records,
        features=model_class.get_features(),
        task=model_class.get_task(),
        target=model_class.get_target(dataset_class),
        label=model_class._meta.label,
        dataset=dataset_class._meta.label,
        left=left,
        right=right,
        tolerance=tolerance,
        max_changes=max_changes,
    )


def compare_predictors(
    older: Any,
    newer: Any,
    *,
    records: list[dict[str, Any]],
    features: list[str],
    task: str,
    target: str | None,
    label: str,
    dataset: str,
    left: Any,
    right: Any,
    tolerance: float = DEFAULT_TOLERANCE,
    max_changes: int = 0,
) -> dict[str, Any]:
    """The comparison itself, given two things that can ``predict``.

    Split out from :func:`compare_versions` so that where the two models came
    from stops being part of the question. Two registered versions are the usual
    case; a pair of artefacts on disk, or two versions in somebody else's
    registry, are the same comparison once something can load them.

    Both are run over identical input in the same order, so a disagreement is a
    disagreement about the data rather than about which rows each one saw.
    """
    inputs = [_as_input(record, features) for record in records]

    left_out = list(older.predict(inputs))
    right_out = list(newer.predict(inputs))

    truth = [record.get(target) for record in records] if target else []
    labelled = bool(truth) and all(value is not None for value in truth)

    report: dict[str, Any] = {
        "label": label,
        "left": left,
        "right": right,
        "task": task,
        "dataset": dataset,
        "rows": len(records),
        "labelled": labelled,
    }

    if task == "regression":
        report.update(_numeric_diff(left_out, right_out, tolerance))
    else:
        report.update(_categorical_diff(left_out, right_out))

    changed = [
        index
        for index, (a, b) in enumerate(zip(left_out, right_out, strict=True))
        if _differs(a, b, task, tolerance)
    ]

    if labelled:
        report.update(_against_truth(truth, left_out, right_out, task, tolerance))

    if max_changes:
        report["changes"] = [
            {
                **{name: records[index].get(name) for name in features},
                "left": left_out[index],
                "right": right_out[index],
                **({"expected": truth[index]} if labelled else {}),
            }
            for index in changed[:max_changes]
        ]

    return report


def compare_from_log(
    model_class: Any,
    left: int,
    right: int,
    *,
    since: Any = None,
    limit: int = 10_000,
    tolerance: float = DEFAULT_TOLERANCE,
    max_changes: int = 0,
) -> dict[str, Any]:
    """Diff two versions using predictions they already made, not new ones.

    A shadow deployment answers the question a dataset cannot: not "how would
    the candidate do on the rows I curated" but "what would it have said to the
    people who actually asked". Both versions answered the same request, so
    they are paired by request id rather than by input — two callers who happen
    to ask the same question are two requests, and fusing them would invent
    agreement that was never observed.

    There is no ground truth here. Production traffic arrives unlabelled, which
    is the whole reason a shadow is worth running before the labels exist.
    """
    from sqlalchemy import select

    from mlango.metastore.models import Prediction, utcnow
    from mlango.metastore.session import session_scope

    label = model_class._meta.label
    statement = select(Prediction).where(
        Prediction.label == label,
        Prediction.version.in_([left, right]),
        Prediction.request_id.is_not(None),
    )
    if since is not None:
        statement = statement.where(Prediction.created_at >= utcnow() - since)

    with session_scope() as session:
        rows = list(session.execute(statement.order_by(Prediction.id.desc())).scalars())

    paired: dict[str, dict[int, Any]] = {}
    for row in rows:
        paired.setdefault(str(row.request_id), {})[int(row.version or 0)] = row

    both = [r for r, sides in paired.items() if left in sides and right in sides]
    if not both:
        raise LookupError(
            f"No request was answered by both v{left} and v{right} of {label}. "
            f"Shadowing writes a row per version per request — check that "
            f"SHADOW and PREDICTION_LOG are both on, and that traffic has arrived since."
        )

    # Newest first out of the query; oldest first reads better in a report.
    both.reverse()
    left_out = [paired[r][left].output for r in both]
    right_out = [paired[r][right].output for r in both]
    records = [{"input": paired[r][left].inputs} for r in both]

    task = model_class.get_task()
    report: dict[str, Any] = {
        "label": label,
        "left": left,
        "right": right,
        "task": task,
        "dataset": "the prediction log",
        "rows": len(both),
        # Nothing in a request says what the right answer was.
        "labelled": False,
        "source": "log",
    }
    if task == "regression":
        report.update(_numeric_diff(left_out, right_out, tolerance))
    else:
        report.update(_categorical_diff(left_out, right_out))

    if max_changes:
        changed = [
            index
            for index, (a, b) in enumerate(zip(left_out, right_out, strict=True))
            if _differs(a, b, task, tolerance)
        ]
        report["changes"] = [
            {
                "input": records[index]["input"],
                "left": left_out[index],
                "right": right_out[index],
            }
            for index in changed[:max_changes]
        ]
    return report


# --------------------------------------------------------------------------- #
# The diff itself
# --------------------------------------------------------------------------- #


def _categorical_diff(left: list[Any], right: list[Any]) -> dict[str, Any]:
    transitions: dict[str, int] = {}
    same = 0
    for a, b in zip(left, right, strict=True):
        if a == b:
            same += 1
        else:
            transitions[f"{a} → {b}"] = transitions.get(f"{a} → {b}", 0) + 1
    total = len(left) or 1
    return {
        "agreement": same / total,
        "changed": total - same,
        # Sorted by frequency: the biggest movement between two classes is the
        # thing worth looking at, and it is rarely the alphabetically first one.
        "transitions": dict(sorted(transitions.items(), key=lambda pair: -pair[1])),
    }


def _numeric_diff(left: list[Any], right: list[Any], tolerance: float) -> dict[str, Any]:
    deltas = [float(b) - float(a) for a, b in zip(left, right, strict=True)]
    moved = [delta for delta in deltas if abs(delta) > tolerance]
    total = len(deltas) or 1
    return {
        "agreement": (total - len(moved)) / total,
        "changed": len(moved),
        "mean_delta": sum(deltas) / total,
        "mean_absolute_delta": sum(abs(delta) for delta in deltas) / total,
        "largest_delta": max(deltas, key=abs) if deltas else 0.0,
    }


def _against_truth(
    truth: list[Any], left: list[Any], right: list[Any], task: str, tolerance: float
) -> dict[str, Any]:
    """The half that matters: what the change fixed, and what it broke.

    A version can improve on every summary statistic and still be a bad
    promotion, because the rows it lost were the ones someone complained about
    last month. Counting them separately is the only way that shows up.
    """
    left_report = metric_lib.flatten_report(metric_lib.report_for_task(task, truth, left))
    right_report = metric_lib.flatten_report(metric_lib.report_for_task(task, truth, right))

    key = "accuracy" if task != "regression" else "mae"
    out: dict[str, Any] = {
        "metrics": {
            "left": left_report,
            "right": right_report,
            "key": key,
            "delta": right_report.get(key, 0.0) - left_report.get(key, 0.0),
        }
    }

    if task == "regression":
        # "Right" and "wrong" are not categories here, so closer-or-further is
        # the honest analogue.
        closer = sum(
            1
            for expected, a, b in zip(truth, left, right, strict=True)
            if abs(float(b) - float(expected)) < abs(float(a) - float(expected)) - tolerance
        )
        further = sum(
            1
            for expected, a, b in zip(truth, left, right, strict=True)
            if abs(float(b) - float(expected)) > abs(float(a) - float(expected)) + tolerance
        )
        out["closer"] = closer
        out["further"] = further
        # The same test. "Got closer" and "got further" are the discordant pairs
        # here, so this is a sign test, which is McNemar's under another name.
        out["significance"] = significance(closer, further)
        return out

    fixed = sum(1 for expected, a, b in zip(truth, left, right, strict=True) if a != expected == b)
    broke = sum(1 for expected, a, b in zip(truth, left, right, strict=True) if b != expected == a)
    out["fixed"] = fixed
    out["broke"] = broke
    out["significance"] = significance(fixed, broke)
    return out


def _differs(a: Any, b: Any, task: str, tolerance: float) -> bool:
    if task != "regression":
        return bool(a != b)
    try:
        return abs(float(b) - float(a)) > tolerance
    except (TypeError, ValueError):
        return bool(a != b)


def _as_input(record: dict[str, Any], features: list[str]) -> Any:
    """One feature is passed bare, several as a dict — matching QuerySet.xy()."""
    if len(features) == 1:
        return record.get(features[0])
    return {name: record.get(name) for name in features}


__all__ = [
    "compare_versions",
    "compare_from_log",
    "compare_predictors",
    "significance",
    "DEFAULT_TOLERANCE",
    "DEFAULT_ALPHA",
]
