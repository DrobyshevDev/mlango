"""``manage.py sweep`` — search a hyperparameter space."""

from __future__ import annotations

import json
from typing import Any

from mlango.core.typing import ModelClass
from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Train a model once per point in a hyperparameter space and report the best."

    def add_arguments(self, parser) -> None:
        parser.add_argument("model", help="Model label, e.g. reviews.Sentiment.")
        parser.add_argument(
            "-p",
            "--param",
            action="append",
            default=[],
            metavar="NAME=V1,V2,...",
            help="Values to try for one hyperparameter. Repeatable.",
        )
        parser.add_argument(
            "--strategy",
            choices=["grid", "random"],
            default="grid",
            help="Try every combination, or sample them.",
        )
        parser.add_argument("--trials", type=int, help="Cap the number of trials.")
        parser.add_argument("--metric", help="Metric to rank by. Defaults to Meta.monitor.")
        parser.add_argument("--mode", choices=["min", "max"], help="Whether higher is better.")
        parser.add_argument("--seed", type=int, help="Seed for a random sweep.")
        parser.add_argument("--name", default="", help="Name for the sweep run.")
        parser.add_argument("--tag", action="append", default=[], help="Tag every run.")
        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            metavar="N",
            help="Run N trials at once. Threads, so it helps for backends whose "
            "numeric work releases the GIL (sklearn, torch). The seed is "
            "process-global, so concurrent trials do not each start from it.",
        )
        parser.add_argument(
            "--promote-best",
            metavar="STAGE",
            help="Promote the winning version to this stage, e.g. production.",
        )

    def handle(self, **options: Any) -> None:
        from mlango.core.registry import apps

        model_class = apps.get_model(options["model"])
        space = _parse_space(model_class, options["param"]) or model_class.default_space()

        if not space:
            raise CommandError(
                f"{model_class._meta.label} declares no tunable fields. Pass -p NAME=V1,V2 "
                f"or mark a field with tunable=True."
            )

        self.write(self.style.bold(f"Sweeping {model_class._meta.label}"))
        for name, values in space.items():
            self.write(self.style.dim(f"  {name}: {', '.join(map(str, values))}"))

        def report(trial, result):
            score = f"{trial.score:.4f}" if trial.score is not None else "—"
            mark = self.style.success("ok") if trial.ok else self.style.error("fail")
            body = ", ".join(f"{k}={v}" for k, v in trial.params.items())
            self.write(f"  [{mark}] trial {trial.index}: {body} → {score}")
            if trial.error:
                self.write(self.style.dim(f"        {trial.error}"))

        result = model_class.sweep(
            space,
            strategy=options["strategy"],
            trials=options.get("trials"),
            metric=options.get("metric"),
            mode=options.get("mode"),
            seed=options.get("seed"),
            name=options["name"],
            tags=options["tag"] or None,
            promote_best=options.get("promote_best"),
            workers=max(1, int(options.get("workers") or 1)),
            on_trial=report,
        )

        self.write("")
        self.table(
            ["rank", "trial", *space, result.metric, "run"],
            [
                [
                    rank,
                    trial.index,
                    *[trial.params.get(name) for name in space],
                    f"{trial.score:.4f}",
                    trial.run_uuid[:8],
                ]
                for rank, trial in enumerate(result.ranked(), start=1)
            ],
        )

        best = result.best
        self.write("")
        self.write(f"sweep run {result.run_uuid}")
        if best is None:
            raise CommandError("No trial completed successfully.")

        self.ok(
            f"Best: {result.metric}={best.score:.4f} with "
            + ", ".join(f"{k}={v}" for k, v in best.params.items())
        )
        if options.get("promote_best"):
            self.ok(f"Promoted the winning version to {options['promote_best']}.")


def _parse_space(model_class: ModelClass, entries: list[str]) -> dict[str, list[Any]]:
    """Turn ``name=v1,v2`` strings into a cleaned search space."""
    space: dict[str, list[Any]] = {}
    for entry in entries:
        if "=" not in entry:
            raise CommandError(f"--param expects NAME=V1,V2, got {entry!r}.")
        name, _, raw = entry.partition("=")
        name = name.strip()
        if not model_class._meta.has_field(name):
            available = ", ".join(model_class._meta.field_names) or "(none)"
            raise CommandError(
                f"{model_class._meta.label} has no hyperparameter {name!r}. Available: {available}."
            )
        field = model_class._meta.get_field(name)
        values = []
        for piece in raw.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                values.append(field.clean(_coerce(piece)))
            except Exception as exc:
                raise CommandError(f"--param {name}: {exc}") from exc
        if not values:
            raise CommandError(f"--param {name} lists no values.")
        space[name] = values
    return space


def _coerce(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
