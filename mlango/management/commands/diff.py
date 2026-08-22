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
        parser.add_argument(
            "model",
            nargs="?",
            help="Model label, e.g. reviews.Sentiment. Omit when comparing "
            "artefacts with --left and --right.",
        )
        parser.add_argument(
            "versions",
            nargs="*",
            type=int,
            help="Two version numbers. Omit to compare production against the newest.",
        )
        parser.add_argument("--dataset", help="Score this dataset instead of the declared one.")

        # An agent has no version number, so the pair to compare is two runs of
        # its evaluation suite. Same question, same arithmetic, different rows.
        parser.add_argument(
            "--eval",
            metavar="LABEL",
            dest="eval_label",
            help="Compare two runs of an evaluation suite instead of two model "
            "versions, e.g. support.AnswerQuality.",
        )
        parser.add_argument(
            "--runs",
            nargs=2,
            metavar=("OLDER", "NEWER"),
            help="Which two evaluation runs. Defaults to the two most recent.",
        )

        # Comparing two things mlango did not train. Same question, same report;
        # only where the models come from differs.
        parser.add_argument(
            "--left",
            metavar="URI",
            help="Path to a saved model, or scheme:reference, to compare as the "
            "older side. Requires --right and --dataset.",
        )
        parser.add_argument("--right", metavar="URI", help="The newer side. See --left.")
        parser.add_argument(
            "--task",
            choices=["classification", "regression"],
            default="classification",
            help="How to read the predictions of external models. Default classification.",
        )
        parser.add_argument(
            "--target", help="Column to score against. Defaults to the dataset's declared target."
        )
        parser.add_argument(
            "--features",
            help="Comma-separated columns to feed the models. Defaults to every "
            "field except the target and the primary key.",
        )
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

        try:
            if options["eval_label"]:
                report = self._eval_report(options)
            elif options["left"] or options["right"]:
                report = self._external_report(options)
            else:
                if not options["model"]:
                    raise CommandError(
                        "Name a model to compare its registered versions, or pass "
                        "--left and --right to compare two saved artefacts."
                    )
                model_class = apps.get_model(options["model"])
                left, right = self._versions(model_class, options["versions"])

                queryset = None
                if options["dataset"]:
                    queryset = apps.get_dataset(options["dataset"]).objects.get_queryset()

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

    def _eval_report(self, options: dict[str, Any]) -> dict[str, Any]:
        """Diff two runs of an evaluation suite.

        The per-case results were already stored by ``manage.py evaluate``; this
        only has to find the two runs and join them.
        """
        from mlango.core.registry import apps
        from mlango.evals.comparison import compare_runs, recent_runs
        from mlango.training.run import get_run

        if options["left"] or options["right"]:
            raise CommandError("--eval compares evaluation runs; --left and --right compare files.")

        label = apps.get_eval(options["eval_label"])._meta.label

        if options["runs"]:
            runs = []
            for reference in options["runs"]:
                run = get_run(reference)
                if run is None:
                    raise CommandError(f"No run matches {reference!r}.")
                if run.target != label:
                    raise CommandError(f"Run {reference} evaluated {run.target!r}, not {label!r}.")
                runs.append(run)
            older, newer = runs
        else:
            found = recent_runs(label, limit=2)
            if len(found) < 2:
                raise CommandError(
                    f"{label} has {len(found)} finished run(s); comparing needs two. "
                    f"Run it again: manage.py evaluate {label}"
                )
            # recent_runs is newest first, and the older one is the baseline.
            newer, older = found

        return compare_runs(
            older,
            newer,
            max_changes=options["show_changes"],
            alpha=options["alpha"] if options["alpha"] is not None else _default_alpha(),
        )

    def _external_report(self, options: dict[str, Any]) -> dict[str, Any]:
        """Compare two artefacts mlango did not train.

        The dataset is required here because nothing else can say what rows to
        score or which column is the answer — a saved model carries neither.
        """
        from mlango.core.registry import apps
        from mlango.training.comparison import compare_predictors
        from mlango.training.external import columns_for, load_predictor

        if not (options["left"] and options["right"]):
            raise CommandError("--left and --right go together; a diff needs two sides.")
        if not options["dataset"]:
            raise CommandError(
                "--left and --right need --dataset: a saved model does not carry "
                "the rows to score it on. Declare one, or generate it from a file "
                "with: manage.py inspectdata data/rows.csv"
            )

        dataset_class = apps.get_dataset(options["dataset"])
        declared = [name.strip() for name in (options["features"] or "").split(",") if name.strip()]
        features, target = columns_for(
            dataset_class,
            features=declared or None,
            target=options["target"],
        )

        query = dataset_class.objects.get_queryset()
        if options.get("limit"):
            query = query.take(options["limit"])
        records = [dict(record) for record in query]
        if not records:
            raise LookupError(f"{dataset_class._meta.label} produced no rows to compare.")

        return compare_predictors(
            load_predictor(options["left"]),
            load_predictor(options["right"]),
            records=records,
            features=features,
            task=options["task"],
            target=target,
            # No label: the two URIs in the heading already say what this is.
            label="",
            dataset=dataset_class._meta.label,
            left=options["left"],
            right=options["right"],
            max_changes=options["show_changes"],
        )

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
        if report.get("kind") == "eval":
            self._eval_report_lines(report)
            return

        # Registered versions are numbers and read as "v3"; artefacts are URIs
        # and read as themselves. Both are "the older side" and "the newer one".
        heading = " ".join(
            part
            for part in (
                report["label"],
                f"{_side(report['left'])} → {_side(report['right'])}",
                f"on {report['rows']} rows of {report['dataset']}",
            )
            if part
        )
        self.write(self.style.bold(heading))
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

    def _eval_report_lines(self, report: dict[str, Any]) -> None:
        """The same report, for a suite whose cases pass or fail rather than score."""
        self.write(
            self.style.bold(
                f"{report['label']} {report['left']} → {report['right']} "
                f"on {report['cases']} shared case(s)"
            )
        )
        self.write("")

        rates = report["pass_rate"]
        width = max(len(report["left"]), len(report["right"]), 6)
        self.write(f"  {report['left']:<{width}} pass rate    {rates['left']:.4f}")
        self.write(
            f"  {report['right']:<{width}} pass rate    {rates['right']:.4f}   {rates['delta']:+.4f}"
        )
        self.write(f"  fixed          {report['fixed']} case(s) failing in {report['left']}")
        broke = report["broke"]
        line = f"  broke          {broke} case(s) passing in {report['left']}"
        self.write(self.style.warn(line) if broke else line)
        self._significance_line(report)

        # A case that answers differently and still passes is worth knowing about
        # for an agent, where the wording is half the product.
        reworded = report["changed"] - report["fixed"] - report["broke"]
        if reworded > 0:
            self.write(
                self.style.dim(
                    f"  reworded       {reworded} case(s) answered differently and still passed"
                )
            )

        self._config_lines(report)

        for side, cases in (("older", report["only_left"]), ("newer", report["only_right"])):
            if cases:
                # Never folded into the totals: a suite that grew is a different
                # suite, and that is how a pass rate improves by adding easy cases.
                shown = ", ".join(cases[:5]) + ("…" if len(cases) > 5 else "")
                self.warn(f"  {len(cases)} case(s) only in the {side} run: {shown}")

        if report.get("changes"):
            self.write("")
            self.write(self.style.bold("Cases that moved"))
            columns = list(report["changes"][0])
            self.table(
                columns, [[_short(row.get(c)) for c in columns] for row in report["changes"]]
            )

    def _config_lines(self, report: dict[str, Any]) -> None:
        """What was different about the thing being evaluated.

        Printed before the verdict, because "the prompt changed" is the
        context in which "seven fixed, two broken" means anything at all.
        """
        config = report.get("config") or {}
        changed = config.get("changed") or {}
        version = config.get("version")

        if not changed and not version:
            if config.get("identical"):
                self.write(
                    self.style.dim(
                        "  config         unchanged — the difference is the target's own"
                    )
                )
            return

        self.write("")
        self.write(self.style.bold("What changed about it"))
        if version:
            self.write(f"  version        {version['was']} → {version['now']}")
        for key, entry in changed.items():
            if entry["long"]:
                # A system prompt is a page; naming it is the useful part.
                self.write(
                    f"  {key:<14} changed ({_length(entry['was'])} → {_length(entry['now'])})"
                )
            else:
                self.write(f"  {key:<14} {entry['was']!r} → {entry['now']!r}")

    def _truth_block(self, report: dict[str, Any]) -> None:
        metrics = report["metrics"]
        key = metrics["key"]
        left = metrics["left"].get(key, 0.0)
        right = metrics["right"].get(key, 0.0)
        delta = metrics["delta"]

        self.write("")
        self.write(self.style.bold("Against the labels"))
        # Width chosen so `v3` and a file path both leave the metric aligned.
        older, newer = _side(report["left"]), _side(report["right"])
        width = max(len(older), len(newer), 6)
        self.write(f"  {older:<{width}} {key:<12} {left:.4f}")
        self.write(f"  {newer:<{width}} {key:<12} {right:.4f}   {delta:+.4f}")

        if report["task"] == "regression":
            self.write(f"  closer         {report['closer']} row(s)")
            self.write(f"  further        {report['further']} row(s)")
            self._significance_line(report)
            return

        self.write(f"  fixed          {report['fixed']} row(s) wrong in {_side(report['left'])}")
        # The number nobody reports and everybody wants: a promotion that
        # improves the average while losing rows that used to work is exactly
        # the kind that gets reverted a week later.
        broke = report["broke"]
        line = f"  broke          {broke} row(s) right in {_side(report['left'])}"
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
        # An evaluation is labelled by construction — a case that cannot pass or
        # fail is not a case — so only a model diff can be missing its truth.
        evaluation = report.get("kind") == "eval"
        if not evaluation and not report["labelled"]:
            raise CommandError(
                "--fail-on-regression needs labelled data, and this dataset has none. "
                "Point --dataset at one that carries the target column."
            )
        unit = "case" if evaluation else "row"
        broke = report.get("broke", report.get("further", 0))

        if mode == "significant":
            stats = report.get("significance") or {}
            if stats.get("direction") == "regression" and stats.get("p_value", 1.0) < alpha:
                raise CommandError(
                    f"{_side(report['right'])} is worse than {_side(report['left'])} by more than chance: "
                    f"{stats['verdict']}. Inspect them with: "
                    f"--show-changes {min(broke, 20)}"
                )
            # Losing rows is still worth saying out loud, even when the balance
            # of the change is favourable or too close to call.
            if broke:
                self.write(
                    f"{_side(report['right'])} lost {broke} {unit}(s), and that is not a "
                    f"significant regression at alpha={alpha:g}: {stats.get('verdict', '')}"
                )
            else:
                self.ok(
                    f"No regression: {_side(report['right'])} lost nothing {_side(report['left'])} had."
                )
            return

        if broke:
            raise CommandError(
                f"{_side(report['right'])} is wrong on {broke} {unit}(s) that {_side(report['left'])} got "
                f"right. Inspect them with: --show-changes {min(broke, 20)}"
            )
        self.ok(
            f"No regression: {_side(report['right'])} keeps everything {_side(report['left'])} got right."
        )


def _length(value: Any) -> str:
    if value is None:
        return "unset"
    return f"{len(str(value))} chars"


def _default_alpha() -> float:
    from mlango.core.stats import DEFAULT_ALPHA

    return DEFAULT_ALPHA


def _short(value: Any, length: int = 40) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= length else text[: length - 1] + "…"


def _side(value: Any) -> str:
    """A registered version reads as `v3`; an artefact reads as its URI."""
    return f"v{value}" if isinstance(value, int) else str(value)
