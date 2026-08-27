"""A comparison report as Markdown, for the place a promotion is decided.

The terminal report is the one that gets used, and a terminal is a place one
person looks. Whether to promote a version is rarely one person's decision, and
where the others are is a pull request — so the same report has to survive
being posted into one, on a machine with no tty and nobody watching.

Everything here is a pure function of the dictionary ``diff`` already builds,
which is how one renderer covers four different comparisons — registered
versions, evaluation runs, files mlango never trained, and live shadow traffic
— without knowing that any of them exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["marker_for", "render"]

# Enough rows to judge a regression by eye. A comment longer than this is one
# nobody scrolls, and `--json` is there for whoever wants all of them.
MAX_ROWS = 20

_FOOTER = (
    "<sub>Posted by <code>manage.py diff</code> — "
    "<a href='https://github.com/DrobyshevDev/mlango'>mlango</a></sub>"
)


def marker_for(report: Mapping[str, Any]) -> str:
    """An invisible key, so a CI job can find and replace its own last comment.

    Without one, every push leaves another comment and the thread becomes a
    changelog nobody reads. Keyed by what was compared rather than by when, so
    the second push to a branch edits what the first push wrote.
    """
    kind = "eval" if report.get("kind") == "eval" else "model"
    return f"<!-- mlango:diff:{kind}:{report.get('label') or 'artefact'} -->"


def render(report: Mapping[str, Any]) -> str:
    """The report as Markdown, ready to be posted somewhere people read."""
    out: list[str] = [marker_for(report), ""]
    if report.get("kind") == "eval":
        _evaluation(report, out)
    else:
        _model(report, out)

    _changes(report, out)
    out += ["", _FOOTER]
    return "\n".join(out).strip() + "\n"


# -- the two report shapes ---------------------------------------------------


def _model(report: Mapping[str, Any], out: list[str]) -> None:
    older, newer = _side(report["left"]), _side(report["right"])
    labelled = report["labelled"]
    regression = report["task"] == "regression"
    lost = report.get("further" if regression else "broke", 0)
    gained = report.get("closer" if regression else "fixed", 0)

    words = ("moved further from the truth", "moved closer") if regression else ("broken", "fixed")

    out.append(f"### {_status(labelled, lost)} {_title(report, older, newer)}")
    out.append("")
    out.append(_headline(labelled, lost, gained, "row", report["rows"], report["dataset"], words))
    out.append("")

    rows: list[tuple[str, str]] = []
    if labelled:
        metrics = report["metrics"]
        key = metrics["key"]
        left = metrics["left"].get(key, 0.0)
        right = metrics["right"].get(key, 0.0)
        rows.append((f"`{key}`", _movement(left, right, metrics["delta"])))
    rows.append(("agreement", f"{report['agreement']:.1%}"))
    rows.append(("changed", f"{report['changed']} rows"))
    if regression:
        rows.append(("mean delta", f"{report['mean_delta']:+.6g}"))
        rows.append(("largest delta", f"{report['largest_delta']:+.6g}"))
    if labelled:
        rows.append(("closer" if regression else "fixed", f"{gained} rows"))
        rows.append(("further" if regression else "broke", _loud(f"{lost} rows", lost)))

    out += _table(f"{older} → {newer}", rows)
    _transitions(report, out)
    _verdict(report, out)

    if not labelled:
        out += ["", "The data carries no labels, so this says what changed, not what improved."]


def _evaluation(report: Mapping[str, Any], out: list[str]) -> None:
    older, newer = str(report["left"]), str(report["right"])
    broke, fixed = report["broke"], report["fixed"]

    out.append(f"### {_status(True, broke)} {report['label']} {older} → {newer}")
    out.append("")
    # An evaluation is labelled by construction: a case that cannot pass or
    # fail is not a case.
    out.append(_headline(True, broke, fixed, "case", report["cases"], None))
    out.append("")

    rates = report["pass_rate"]
    rows = [
        ("pass rate", _movement(rates["left"], rates["right"], rates["delta"])),
        ("agreement", f"{report['agreement']:.1%}"),
        ("changed", f"{report['changed']} cases"),
        ("fixed", f"{fixed} cases"),
        ("broke", _loud(f"{broke} cases", broke)),
    ]
    # A case that answers differently and still passes is nothing for a
    # classifier and half the product for something a person reads.
    reworded = report["changed"] - fixed - broke
    if reworded > 0:
        rows.append(("reworded", f"{reworded} cases"))

    out += _table(f"{older} → {newer}", rows)
    _verdict(report, out)
    _configuration(report, out)

    for side, cases in (("older", report["only_left"]), ("newer", report["only_right"])):
        if cases:
            # Never folded into the totals: a suite that grew is a different
            # suite, and that is how a pass rate improves by adding easy cases.
            shown = ", ".join(f"`{case}`" for case in cases[:5]) + ("…" if len(cases) > 5 else "")
            out += ["", f"⚠️ {len(cases)} case(s) only in the {side} run: {shown}"]


# -- shared pieces -----------------------------------------------------------


def _headline(
    labelled: bool,
    lost: int,
    gained: int,
    unit: str,
    total: int,
    dataset: Any,
    words: tuple[str, str] = ("broken", "fixed"),
) -> str:
    """The sentence someone reads in a notification, without opening anything.

    The words differ by task on purpose: "broken" is meaningless for a
    regression, where the two versions are not right or wrong but nearer and
    further, and borrowing the classifier's vocabulary would quietly claim
    otherwise.
    """
    lost_word, gained_word = words
    where = f"{total} {unit}s"
    if dataset:
        where += f" of `{dataset}`"
    if not labelled:
        return f"Compared over {where}."
    if lost:
        plural = "" if lost == 1 else "s"
        return f"**{lost} {unit}{plural} {lost_word}**, {gained} {gained_word}, over {where}."
    return f"Nothing {lost_word}, {gained} {gained_word}, over {where}."


def _verdict(report: Mapping[str, Any], out: list[str]) -> None:
    """Whether the balance above is a change or a coin.

    Quoted rather than stated, because the counts invite a conclusion and this
    is the sentence saying whether the conclusion is available at all.
    """
    stats = report.get("significance")
    if not stats or not stats.get("discordant"):
        return
    out += ["", f"> {stats['verdict']}"]


def _transitions(report: Mapping[str, Any], out: list[str]) -> None:
    transitions = report.get("transitions") or {}
    if not transitions:
        return
    shown = list(transitions.items())[:6]
    out += ["", "Movement: " + " · ".join(f"`{name}` {count}" for name, count in shown)]


def _configuration(report: Mapping[str, Any], out: list[str]) -> None:
    """What was different about the thing evaluated.

    Put beside the effect, because "the prompt changed" is the context in which
    "seven fixed, two broken" means anything at all.
    """
    config = report.get("config") or {}
    changed = config.get("changed") or {}
    version = config.get("version")
    if not changed and not version:
        return

    parts = []
    if version:
        parts.append(f"version {_side(version['was'])} → {_side(version['now'])}")
    for key, delta in changed.items():
        if delta.get("long"):
            parts.append(f"`{key}` rewritten")
        else:
            parts.append(f"`{key}` {_cell(delta['was'])} → {_cell(delta['now'])}")
    out += ["", "**What changed:** " + ", ".join(parts)]


def _changes(report: Mapping[str, Any], out: list[str]) -> None:
    """The disagreeing rows, folded away.

    A reviewer wants the verdict in the notification and the rows only once
    they have decided to look, so these go behind a disclosure rather than
    above the fold.
    """
    changes = report.get("changes")
    if not changes:
        return
    shown = list(changes)[:MAX_ROWS]
    noun = "cases" if report.get("kind") == "eval" else "rows"
    # Against the real total, not against how many were asked for: "4 rows
    # where they disagree" is a different and much better claim than the truth
    # when forty of them do.
    total = max(report.get("changed", 0), len(shown))
    of = f" of {total}" if total > len(shown) else ""

    columns = list(shown[0])
    out += [
        "",
        "<details>",
        f"<summary>{len(shown)}{of} {noun} where they disagree</summary>",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    out += ["| " + " | ".join(_cell(row.get(c)) for c in columns) + " |" for row in shown]
    out += ["", "</details>"]


# -- formatting --------------------------------------------------------------


def _table(heading: str, rows: list[tuple[str, str]]) -> list[str]:
    # The left column is deliberately unlabelled: the right one names the
    # comparison, and repeating "measure" beside it would say nothing.
    return [f"| | {heading} |", "|---|---:|"] + [f"| {name} | {value} |" for name, value in rows]


def _movement(left: float, right: float, delta: float) -> str:
    return f"{left:.4f} → **{right:.4f}** ({delta:+.4f})"


def _status(labelled: bool, lost: int) -> str:
    """One glyph, because a pull request is often first seen as a list of them."""
    if not labelled:
        return "ℹ️"
    return "⚠️" if lost else "✅"


def _loud(text: str, count: int) -> str:
    """The number nobody reports and everybody wants, given weight when it is not zero."""
    return f"**{text}**" if count else text


def _title(report: Mapping[str, Any], older: str, newer: str) -> str:
    # A label is absent when two artefacts are compared: they are named by
    # their paths, and there is no declaration above them to name.
    return " ".join(part for part in (report.get("label"), f"{older} → {newer}") if part)


def _side(value: Any) -> str:
    """A registered version reads as `v3`; an artefact reads as its URI."""
    return f"v{value}" if isinstance(value, int) else str(value)


def _cell(value: Any, length: int = 60) -> str:
    """A value that cannot break the table it sits in."""
    if value is None:
        return ""
    text = " ".join(str(value).split()).replace("|", "\\|")
    return text if len(text) <= length else text[: length - 1] + "…"
