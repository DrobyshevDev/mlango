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

import math
from typing import Any

from mlango.training import metrics as metric_lib

#: Regression predictions are floats, so "did the answer change" is not a
#: question equality can answer. Anything closer than this counts as the same
#: prediction, which keeps the changed-row list about real movement.
DEFAULT_TOLERANCE = 1e-6

#: Below this, a difference is reported as real rather than as noise. Nothing
#: about 0.05 is principled; it is the number everyone reads without asking, and
#: the p-value is printed beside the verdict so you can disagree with it.
DEFAULT_ALPHA = 0.05


def significance(fixed: int, broke: int) -> dict[str, Any]:
    """Is the difference between two versions distinguishable from noise?

    Rows both versions get right, and rows both get wrong, say nothing about
    which is better — only the disagreements carry information. That leaves
    ``fixed`` rows the new version rescued and ``broke`` rows it lost, and the
    question becomes whether a coin that came up ``fixed`` heads in
    ``fixed + broke`` tosses was fair.

    This is McNemar's test, computed exactly rather than through the chi-square
    approximation, because promotion decisions are often made on a few hundred
    rows where the approximation is worst.

    A version that fixes 200 rows and breaks 3 is an improvement; one that fixes
    38 and breaks 40 is a coin. Both look like "a regression" to a rule that
    counts broken rows, which is why this exists beside that rule rather than
    instead of it.
    """
    discordant = fixed + broke
    if discordant == 0:
        # The two versions are right and wrong on exactly the same rows. There
        # is no evidence either way, and no amount of data would change that.
        return {
            "discordant": 0,
            "p_value": 1.0,
            "direction": "identical",
            "verdict": "the two versions are right on exactly the same rows",
        }

    smaller = min(fixed, broke)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / (2**discordant)
    p_value = min(1.0, 2 * tail)

    if fixed > broke:
        direction = "improvement"
    elif broke > fixed:
        direction = "regression"
    else:
        direction = "tie"

    return {
        "discordant": discordant,
        "p_value": p_value,
        "direction": direction,
        "verdict": _verdict(direction, p_value, fixed, broke),
    }


def _verdict(direction: str, p_value: float, fixed: int, broke: int) -> str:
    if direction == "tie":
        return f"{fixed} fixed against {broke} broken is a coin, not a change"
    if p_value >= DEFAULT_ALPHA:
        return (
            f"{fixed} fixed against {broke} broken is not distinguishable from noise "
            f"(p={p_value:.3f})"
        )
    if direction == "improvement":
        return f"a real improvement: {fixed} fixed against {broke} broken (p={p_value:.3f})"
    return f"a real regression: {broke} broken against {fixed} fixed (p={p_value:.3f})"


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
    older = model_class.load(version=left)
    newer = model_class.load(version=right)

    dataset_class = model_class.get_dataset()
    query = queryset if queryset is not None else dataset_class.objects.get_queryset()
    if limit:
        query = query.take(limit)

    records = [dict(record) for record in query]
    if not records:
        raise LookupError(f"{dataset_class._meta.label} produced no rows to compare.")

    features = model_class.get_features()
    inputs = [_as_input(record, features) for record in records]

    left_out = list(older.predict(inputs))
    right_out = list(newer.predict(inputs))

    task = model_class.get_task()
    target = model_class.get_target(dataset_class)
    truth = [record.get(target) for record in records] if target else []
    labelled = bool(truth) and all(value is not None for value in truth)

    report: dict[str, Any] = {
        "label": model_class._meta.label,
        "left": left,
        "right": right,
        "task": task,
        "dataset": dataset_class._meta.label,
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


__all__ = ["compare_versions", "DEFAULT_TOLERANCE"]
