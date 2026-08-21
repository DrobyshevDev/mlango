"""Comparing two evaluation runs, case by case.

An agent has no version number: you change a prompt, a tool description or a
model and re-run the suite, and the only thing that moves is a pass rate. A pass
rate going from 82% to 87% hides the same thing an accuracy going from 0.88 to
0.90 hides — that some of the cases which used to pass now do not, and they are
usually the ones somebody complained about.

The per-case results are already stored. This joins two runs on ``case_id`` and
says which cases were rescued, which were lost, and whether the difference is
distinguishable from noise.
"""

from __future__ import annotations

from typing import Any

from mlango.core.stats import DEFAULT_ALPHA, significance


def compare_runs(
    left: Any,
    right: Any,
    *,
    max_changes: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Diff two finished evaluation runs of the same suite.

    ``left`` and ``right`` are :class:`~mlango.metastore.models.Run` rows. Cases
    are matched by ``case_id``, so a suite that grew between the two runs is
    reported honestly rather than silently making the newer run look different.
    """
    from sqlalchemy import select

    from mlango.metastore.models import EvalResult
    from mlango.metastore.session import session_scope

    with session_scope() as session:
        before = {
            row.case_id: row
            for row in session.execute(
                select(EvalResult).where(EvalResult.run_id == left.id)
            ).scalars()
        }
        after = {
            row.case_id: row
            for row in session.execute(
                select(EvalResult).where(EvalResult.run_id == right.id)
            ).scalars()
        }

    shared = sorted(set(before) & set(after))
    if not shared:
        raise LookupError(
            "The two runs have no case in common, so there is nothing to compare. "
            "Cases are matched by case_id — declare Meta.case_id_field if the suite "
            "does not have one, or the ids are row positions and move when the data does."
        )

    fixed: list[str] = []
    broke: list[str] = []
    changed: list[str] = []
    for case_id in shared:
        was, now = bool(before[case_id].passed), bool(after[case_id].passed)
        if was != now:
            (broke if was else fixed).append(case_id)
        if _output_of(before[case_id]) != _output_of(after[case_id]):
            changed.append(case_id)

    left_passed = sum(1 for case_id in shared if before[case_id].passed)
    right_passed = sum(1 for case_id in shared if after[case_id].passed)
    total = len(shared)

    report: dict[str, Any] = {
        "label": left.target,
        "left": left.short_id,
        "right": right.short_id,
        "kind": "eval",
        "cases": total,
        # Named rather than counted away: a suite that grew is a different
        # suite, and pretending otherwise is how a pass rate improves by
        # adding easy cases.
        "only_left": sorted(set(before) - set(after)),
        "only_right": sorted(set(after) - set(before)),
        "pass_rate": {
            "left": left_passed / total,
            "right": right_passed / total,
            "delta": (right_passed - left_passed) / total,
        },
        "fixed": len(fixed),
        "broke": len(broke),
        "changed": len(changed),
        "agreement": (total - len(fixed) - len(broke)) / total,
        "significance": significance(len(fixed), len(broke), alpha=alpha),
    }

    if max_changes:
        # Cases whose verdict moved come first: a different wording that still
        # passes is interesting, and a case that started failing is the point.
        ordered = [c for c in (broke + fixed) if c in shared]
        ordered += [c for c in changed if c not in broke and c not in fixed]
        report["changes"] = [
            {
                "case": case_id,
                "was": "pass" if before[case_id].passed else "fail",
                "now": "pass" if after[case_id].passed else "fail",
                "left": _output_of(before[case_id]),
                "right": _output_of(after[case_id]),
                "expected": before[case_id].expected,
            }
            for case_id in ordered[:max_changes]
        ]

    return report


def recent_runs(label: str, limit: int = 2) -> list[Any]:
    """The most recent finished evaluation runs of ``label``, newest first."""
    from sqlalchemy import select

    from mlango.metastore.models import Run, RunKind, RunStatus
    from mlango.metastore.session import session_scope

    with session_scope() as session:
        return list(
            session.execute(
                select(Run)
                .where(
                    Run.target == label,
                    Run.kind == RunKind.EVAL,
                    Run.status == RunStatus.FINISHED,
                )
                .order_by(Run.started_at.desc())
                .limit(limit)
            ).scalars()
        )


def _output_of(row: Any) -> Any:
    """What the agent or model actually said, as something comparable."""
    output = row.output
    if isinstance(output, dict):
        # Evaluations store the output verbatim, and an agent's is a dict with
        # a trace id and token counts in it. Those differ on every run and are
        # not what "the answer changed" means.
        for key in ("output", "text", "answer", "prediction"):
            if key in output:
                return output[key]
    return output


__all__ = ["compare_runs", "recent_runs"]
