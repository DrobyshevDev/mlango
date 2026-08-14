"""``manage.py diff`` — what changed between two trained versions.

The question before every promotion, and the one aggregate metrics answer
badly: a version two points more accurate overall can still have broken forty
rows that used to work, and accuracy will not mention it.
"""

from __future__ import annotations

import json
from typing import Any

from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Compare what two registered model versions predict on the same data."

    def add_arguments(self, parser) -> None:
        parser.add_argument("model", help="Model label, e.g. reviews.Sentiment.")
        parser.add_argument(
            "versions",
            nargs="*",
            type=int,
            help="Two version numbers. Omit to compare production against the newest.",
        )
        parser.add_argument("--dataset", help="Score this dataset instead of the declared one.")
        parser.add_argument("-n", "--limit", type=int, help="Stop after this many rows.")
        parser.add_argument(
            "--show-changes",
            type=int,
            default=0,
            metavar="N",
            help="Print up to N rows where the two disagree.",
        )
        parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
        # Was a flag; now a flag with an optional mode, so the old spelling
        # still means what it meant.
        parser.add_argument(
            "--fail-on-regression",
            nargs="?",
            const="any",
            choices=["any", "significant"],
            default=None,
            metavar="MODE",
            help="Exit non-zero on regression. 'any' (the default when the flag is "
            "given bare) fails on a single labelled row the newer version lost. "
            "'significant' fails only when the losses outweigh the gains by more "
            "than chance, which is the question a promotion actually asks.",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=None,
            metavar="P",
            help="Significance level for --fail-on-regression significant. Default 0.05.",
        )

    def handle(self, **options: Any) -> None:
        from mlango.core.exceptions import MlangoError
        from mlango.core.registry import apps
        from mlango.training.comparison import compare_versions

        model_class = apps.get_model(options["model"])
        left, right = self._versions(model_class, options["versions"])

        queryset = None
        if options["dataset"]:
            queryset = apps.get_dataset(options["dataset"]).objects.get_queryset()

        try:
            report = compare_versions(
                model_class,
                left,
                right,
                queryset=queryset,
                limit=options.get("limit"),
                max_changes=options["show_changes"],
            )
        except LookupError as exc:
            raise CommandError(str(exc)) from exc
        except MlangoError as exc:
            raise CommandError(str(exc)) from exc

        if options["json"]:
            self.write(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        else:
            self._report(report)

        if options["fail_on_regression"]:
            from mlango.training.comparison import DEFAULT_ALPHA

            alpha = options["alpha"] if options["alpha"] is not None else DEFAULT_ALPHA
            self._check_regression(report, options["fail_on_regression"], alpha)

    # -- selection -----------------------------------------------------------

    def _versions(self, model_class: Any, given: list[int]) -> tuple[int, int]:
        """The two versions to compare, defaulting to the interesting pair.

        With no arguments the question is almost always "should I promote the
        new one over what is live", so that is what an unqualified call answers.
        """
        if len(given) == 2:
            return given[0], given[1]
        if len(given) == 1:
            raise CommandError("Give two version numbers, or none to compare production to latest.")
        if given:
            raise CommandError(f"Expected two version numbers, got {len(given)}.")

        from sqlalchemy import select

        from mlango.metastore.models import ModelVersion, Stage
        from mlango.metastore.session import session_scope

        label = model_class._meta.label
        with session_scope() as session:
            live = session.execute(
                select(ModelVersion)
                .where(ModelVersion.label == label, ModelVersion.stage == Stage.PRODUCTION)
                .order_by(ModelVersion.version.desc())
                .limit(1)
            ).scalar_one_or_none()
            newest = session.execute(
                select(ModelVersion)
                .where(ModelVersion.label == label)
                .order_by(ModelVersion.version.desc())
                .limit(1)
            ).scalar_one_or_none()

        if newest is None:
            raise CommandError(f"{label} has no registered versions. Train it first.")
        if live is None:
            raise CommandError(
                f"{label} has nothing at stage 'production', so there is no default pair. "
                f"Name two versions, e.g. manage.py diff {label} 1 {newest.version}"
            )
        if live.version == newest.version:
            raise CommandError(
                f"{label}@v{newest.version} is both the newest and the one in production, "
                f"so there is nothing to compare it against."
            )
        return live.version, newest.version

    # -- output --------------------------------------------------------------

    def _report(self, report: dict[str, Any]) -> None:
        self.write(
            self.style.bold(
                f"{report['label']} v{report['left']} → v{report['right']} "
                f"on {report['rows']} rows of {report['dataset']}"
            )
        )
        self.write("")

        agreement = report["agreement"]
        self.write(f"  agreement      {agreement:.1%}")
        self.write(f"  changed        {report['changed']} row(s)")

        for transition, count in (report.get("transitions") or {}).items():
            self.write(self.style.dim(f"    {transition:<24} {count}"))

        if report["task"] == "regression":
            self.write(f"  mean delta     {report['mean_delta']:+.6g}")
            self.write(f"  mean |delta|   {report['mean_absolute_delta']:.6g}")
            self.write(f"  largest delta  {report['largest_delta']:+.6g}")

        if report["labelled"]:
            self._truth_block(report)
        else:
            self.write("")
            self.write(
                self.style.dim(
                    "  The data carries no labels, so this says what changed, not what improved."
                )
            )

        if report.get("changes"):
            self.write("")
            self.write(self.style.bold("Rows where they disagree"))
            columns = list(report["changes"][0])
            self.table(
                columns, [[_short(row.get(c)) for c in columns] for row in report["changes"]]
            )

    def _truth_block(self, report: dict[str, Any]) -> None:
        metrics = report["metrics"]
        key = metrics["key"]
        left = metrics["left"].get(key, 0.0)
        right = metrics["right"].get(key, 0.0)
        delta = metrics["delta"]

        self.write("")
        self.write(self.style.bold("Against the labels"))
        self.write(f"  v{report['left']} {key:<12} {left:.4f}")
        self.write(f"  v{report['right']} {key:<12} {right:.4f}   {delta:+.4f}")

        if report["task"] == "regression":
            self.write(f"  closer         {report['closer']} row(s)")
            self.write(f"  further        {report['further']} row(s)")
            self._significance_line(report)
            return

        self.write(f"  fixed          {report['fixed']} row(s) wrong in v{report['left']}")
        # The number nobody reports and everybody wants: a promotion that
        # improves the average while losing rows that used to work is exactly
        # the kind that gets reverted a week later.
        broke = report["broke"]
        line = f"  broke          {broke} row(s) right in v{report['left']}"
        self.write(self.style.warn(line) if broke else line)
        self._significance_line(report)

    def _significance_line(self, report: dict[str, Any]) -> None:
        """Whether the fixed-against-broken balance is a change or a coin.

        Printed unconditionally, because the counts above invite a conclusion
        and this is the sentence that says whether the conclusion is available.
        """
        stats = report.get("significance")
        if not stats or not stats["discordant"]:
            return
        self.write(f"  verdict        {stats['verdict']}")

    def _check_regression(self, report: dict[str, Any], mode: str, alpha: float) -> None:
        if not report["labelled"]:
            raise CommandError(
                "--fail-on-regression needs labelled data, and this dataset has none. "
                "Point --dataset at one that carries the target column."
            )
        broke = report.get("broke", report.get("further", 0))

        if mode == "significant":
            stats = report.get("significance") or {}
            if stats.get("direction") == "regression" and stats.get("p_value", 1.0) < alpha:
                raise CommandError(
                    f"v{report['right']} is worse than v{report['left']} by more than chance: "
                    f"{stats['verdict']}. Inspect the rows with: "
                    f"--show-changes {min(broke, 20)}"
                )
            # Losing rows is still worth saying out loud, even when the balance
            # of the change is favourable or too close to call.
            if broke:
                self.write(
                    f"v{report['right']} lost {broke} row(s), and that is not a "
                    f"significant regression at alpha={alpha:g}: {stats.get('verdict', '')}"
                )
            else:
                self.ok(f"No regression: v{report['right']} lost nothing v{report['left']} had.")
            return

        if broke:
            raise CommandError(
                f"v{report['right']} is wrong on {broke} row(s) that v{report['left']} got "
                f"right. Inspect them with: --show-changes {min(broke, 20)}"
            )
        self.ok(f"No regression: v{report['right']} keeps everything v{report['left']} got right.")


def _short(value: Any, length: int = 40) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= length else text[: length - 1] + "…"
