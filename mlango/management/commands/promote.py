"""``manage.py promote`` — move a version to a stage, optionally checking first.

The framework could already tell you what a new version broke and could already
promote one from Python or the admin, but there was no way to do the second from
a terminal. So the workflow the project leads with ended one step short: read
the diff, decide to ship, and then go and open a shell.

``--check`` closes the loop the other way. It runs the comparison against
whoever currently holds the stage and refuses the promotion if the candidate
lost rows — which is the whole argument for having the comparison at all.
"""

from __future__ import annotations

from typing import Any

from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Move a model or agent version to a stage, and optionally check it first."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "label",
            nargs="?",
            help="Model or agent label, e.g. reviews.Sentiment.",
        )
        parser.add_argument(
            "version",
            nargs="?",
            type=int,
            help="Which version. Defaults to the newest registered one.",
        )
        parser.add_argument(
            "--stage",
            default="production",
            help="Stage to move it to: none, staging, production, archived.",
        )
        parser.add_argument(
            "--check",
            nargs="?",
            const="any",
            choices=["any", "significant"],
            default=None,
            metavar="MODE",
            help="Compare against whoever holds the stage now, and refuse the "
            "promotion on a regression. 'any' (the default when the flag is bare) "
            "refuses if a single labelled row was lost; 'significant' refuses only "
            "when the losses outweigh the gains by more than chance.",
        )
        parser.add_argument("--dataset", help="Score --check against this dataset.")
        parser.add_argument("-n", "--limit", type=int, help="Cap the rows --check scores.")
        parser.add_argument(
            "--history",
            action="store_true",
            help="List what has been promoted instead of promoting anything. "
            "With no label, the whole registry's history.",
        )
        parser.add_argument("--notes", default="", help="Why, recorded with the move.")

    def handle(self, **options: Any) -> None:
        from mlango.core.exceptions import MlangoError
        from mlango.core.registry import apps
        from mlango.metastore.models import Stage

        label = options["label"]
        if options["history"]:
            self._history(label, options)
            return
        if not label:
            raise CommandError(
                "Name a model or agent to promote, or pass --history to see what "
                "has already been promoted."
            )

        stage = options["stage"]
        if stage not in Stage.ALL:
            raise CommandError(f"{stage!r} is not a stage. One of: {', '.join(Stage.ALL)}.")

        target, kind = self._resolve(apps, label)
        version = options["version"] or self._newest(target, label)

        evidence = None
        if options["check"]:
            evidence = self._check(target, kind, version, stage, options)

        try:
            promoted = target.promote(version, stage, evidence=evidence, notes=options["notes"])
        except LookupError as exc:
            raise CommandError(str(exc)) from exc
        except (MlangoError, ValueError) as exc:
            raise CommandError(str(exc)) from exc

        self.ok(f"{promoted.ref} is now at stage {promoted.stage!r}.")

        if kind == "agent":
            self.write(
                self.style.dim(
                    "A version pins the configuration, not the code: tools come from "
                    "whatever is imported when the agent runs."
                )
            )
        elif not options["check"]:
            # Said once, quietly. The comparison is the reason this framework
            # knows anything about promotions, and skipping it silently would
            # be an odd thing for the command to encourage.
            self.write(
                self.style.dim(
                    f"Promoted without checking. To see what it changes first: "
                    f"manage.py promote {label} {version} --check"
                )
            )

    # -- resolution ----------------------------------------------------------

    def _resolve(self, apps: Any, label: str) -> tuple[Any, str]:
        """The declared object, whichever family it belongs to.

        One verb for both: a model version and an agent version are the same
        idea, and making the user remember which command owns which family is
        the kind of seam a framework exists to remove.
        """
        try:
            return apps.get_model(label), "model"
        except LookupError:
            pass
        try:
            return apps.get_agent(label), "agent"
        except LookupError as exc:
            raise CommandError(
                f"No model or agent named {label!r}. Only models and agents have "
                f"versions to promote."
            ) from exc

    def _newest(self, target: Any, label: str) -> int:
        versions = target.versions()
        if not versions:
            raise CommandError(
                f"{label} has no registered versions yet. Train it first: manage.py train {label}"
            )
        return int(versions[0].version)

    # -- history -------------------------------------------------------------

    def _history(self, label: str | None, options: dict[str, Any]) -> None:
        """What has been promoted, newest first.

        The stage column on a version row only knows about now, so without this
        a registry that has been in use for a year can say what is live and
        nothing about how it got there.
        """
        from mlango.metastore.history import history

        moves = history(label, limit=options.get("limit") or 50)
        if not moves:
            where = f" for {label}" if label else ""
            self.write(f"Nothing has been promoted{where} yet.")
            return

        self.write(
            self.style.bold(
                f"{label or 'Every model and agent'} — {len(moves)} move(s), newest first"
            )
        )
        self.write("")

        columns = ["when", "what", "version", "move", "who", "on the strength of"]
        if label:
            # Naming it once in the heading is enough.
            columns.remove("what")

        rows = []
        for move in moves:
            row = [
                move.at.strftime("%Y-%m-%d %H:%M"),
                move.label,
                f"v{move.version}",
                f"{move.from_stage} → {move.to_stage}",
                move.actor or "—",
                _evidence_line(move),
            ]
            if label:
                del row[1]
            rows.append(row)
        self.table(columns, rows)

    # -- the check -----------------------------------------------------------

    def _check(
        self, target: Any, kind: str, version: int, stage: str, options: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Compare the candidate with the incumbent, and stop if it lost rows.

        Returns what the comparison found, so the promotion can be recorded
        beside the evidence it was made on rather than as a bare fact.
        """
        if kind != "model":
            raise CommandError(
                "--check compares predictions, which needs a model. For an agent, "
                "compare two runs of its evaluation suite: manage.py diff --eval <suite>"
            )

        incumbent = next((v for v in target.versions() if v.stage == stage), None)
        if incumbent is None:
            self.write(
                self.style.dim(f"Nothing holds {stage!r} yet, so there is nothing to compare with.")
            )
            return None
        if incumbent.version == version:
            raise CommandError(f"v{version} already holds {stage!r}.")

        from mlango.core.exceptions import MlangoError
        from mlango.core.registry import apps
        from mlango.training.comparison import compare_versions

        queryset = None
        if options["dataset"]:
            queryset = apps.get_dataset(options["dataset"]).objects.get_queryset()

        try:
            report = compare_versions(
                target,
                incumbent.version,
                version,
                queryset=queryset,
                limit=options.get("limit"),
            )
        except (LookupError, MlangoError) as exc:
            raise CommandError(
                f"Could not compare v{incumbent.version} with v{version}: {exc}"
            ) from exc

        self._report(report, incumbent.version, version)

        if not report["labelled"]:
            raise CommandError(
                "--check needs labelled data to say whether anything broke, and this "
                "dataset has none. Point --dataset at one that carries the target column."
            )

        broke = report.get("broke", report.get("further", 0))
        evidence = _evidence(report, incumbent.version, options["check"])
        if not broke:
            self.ok(f"v{version} keeps everything v{incumbent.version} got right.")
            return evidence

        stats = report.get("significance") or {}
        if options["check"] == "significant":
            if stats.get("direction") == "regression" and stats.get("p_value", 1.0) < 0.05:
                raise CommandError(
                    f"Refusing to promote: {stats['verdict']}. "
                    f"Inspect it with: manage.py diff {report['label']} "
                    f"{incumbent.version} {version} --show-changes {min(broke, 20)}"
                )
            # Just the verdict. The report above already printed the count, and
            # glueing the verdict to a clause of mine produced "not a significant
            # regression: a real improvement", which reads like a contradiction.
            self.write(f"  verdict        {stats.get('verdict', '')}")
            return evidence

        raise CommandError(
            f"Refusing to promote: v{version} is wrong on {broke} row(s) that "
            f"v{incumbent.version} got right. Inspect them with: "
            f"manage.py diff {report['label']} {incumbent.version} {version} "
            f"--show-changes {min(broke, 20)}\n"
            f"To promote anyway despite that, drop --check, or use --check significant "
            f"to allow losses the evidence cannot distinguish from noise."
        )

    def _report(self, report: dict[str, Any], older: int, newer: int) -> None:
        metrics = report.get("metrics") or {}
        key = metrics.get("key", "")
        self.write(
            self.style.bold(f"v{older} → v{newer} on {report['rows']} rows of {report['dataset']}")
        )
        if key:
            left = metrics["left"].get(key, 0.0)
            right = metrics["right"].get(key, 0.0)
            self.write(f"  {key:<14} {left:.4f} → {right:.4f}   {metrics['delta']:+.4f}")
        self.write(f"  fixed          {report.get('fixed', 0)} row(s)")
        self.write(f"  broke          {report.get('broke', 0)} row(s)")


def _evidence(report: dict[str, Any], against: int, mode: str) -> dict[str, Any]:
    """What the comparison found, small enough to keep on the move forever.

    Deliberately not the whole report: the rows it disagreed on are large, and
    they can be recomputed from two versions that both still exist. The counts
    and the verdict cannot be recomputed once the dataset has moved on, which is
    exactly why they are the part worth storing.
    """
    metrics = report.get("metrics") or {}
    key = metrics.get("key", "")
    stats = report.get("significance") or {}
    return {
        "against": against,
        "mode": mode,
        "rows": report.get("rows"),
        "dataset": report.get("dataset"),
        "metric": key,
        "before": metrics.get("left", {}).get(key),
        "after": metrics.get("right", {}).get(key),
        "delta": metrics.get("delta"),
        "fixed": report.get("fixed", report.get("closer")),
        "broke": report.get("broke", report.get("further")),
        "verdict": stats.get("verdict"),
    }


def _evidence_line(move: Any) -> str:
    """One column's worth of why.

    A move nobody checked says so rather than showing a blank, because "we did
    not look" is the single most useful thing a promotion log can tell you.
    """
    evidence = move.evidence or {}
    if "superseded_by" in evidence:
        return f"superseded by v{evidence['superseded_by']}"
    if not evidence:
        return move.notes or "not checked"

    parts = []
    fixed, broke = evidence.get("fixed"), evidence.get("broke")
    if fixed is not None or broke is not None:
        parts.append(f"{fixed or 0} fixed / {broke or 0} broke")
    if evidence.get("metric") and evidence.get("delta") is not None:
        parts.append(f"{evidence['metric']} {evidence['delta']:+.4f}")
    if move.notes:
        parts.append(move.notes)
    return ", ".join(parts) or "checked"
