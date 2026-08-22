"""The admin web application.

Server-rendered with Jinja2 and no build step, so a project can open the admin
on a fresh checkout with nothing but ``pip install``. Everything it shows comes
from ``_meta`` and the metastore, which is why one set of templates covers
datasets, models, agents and evals.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from mlango.admin.sites import AdminSite
from mlango.admin.sites import site as default_site

logger = logging.getLogger("mlango.admin")

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def _template_dirs() -> list[str]:
    """Where to look for admin templates, project first.

    Jinja2 searches these in order, so dropping ``base.html`` into a project's
    own admin template directory replaces the shipped one and leaves the other
    nine alone. That is how Django's admin has always been customised, and it
    is the difference between a theme you can change and one you have to fork.

    The framework's directory is always last, so an override can never make a
    page disappear: whatever the project does not provide still resolves.
    """
    from mlango.conf import settings

    dirs: list[str] = []
    for entry in getattr(settings, "ADMIN_TEMPLATE_DIRS", []) or []:
        path = str(entry)
        resolved = path if os.path.isabs(path) else os.path.join(str(settings.BASE_DIR), path)
        # Normalised, so a setting written with forward slashes does not leave
        # mixed separators in the loader's search path on Windows.
        resolved = os.path.normpath(resolved)
        if os.path.isdir(resolved) and resolved not in dirs:
            dirs.append(resolved)

    dirs.append(TEMPLATE_DIR)
    return dirs


def build_admin_app(site: AdminSite | None = None) -> FastAPI:
    """Build the admin sub-application."""
    from mlango.conf import settings

    site = site or default_site
    site.autodiscover()

    app = FastAPI(title=settings.ADMIN_SITE_TITLE, docs_url=None, redoc_url=None, openapi_url=None)

    from mlango.admin.auth import BasicAuthMiddleware, auth_configured

    app.add_middleware(BasicAuthMiddleware)
    if not auth_configured() and not settings.DEBUG:
        logger.warning(
            "The admin is unauthenticated and DEBUG is off. Set ADMIN_PASSWORD, or put "
            "the admin behind your identity provider."
        )

    templates = Jinja2Templates(directory=_template_dirs())
    templates.env.filters["short"] = _short
    templates.env.filters["duration"] = _duration
    templates.env.filters["number"] = _number

    def page(
        request: Request, template: str, *, status_code: int = 200, **context: Any
    ) -> HTMLResponse:
        import mlango

        return templates.TemplateResponse(
            request=request,
            name=template,
            status_code=status_code,
            context={
                "site_header": settings.ADMIN_SITE_HEADER,
                "site_title": settings.ADMIN_SITE_TITLE,
                "version": mlango.get_version(),
                "app_list": site.app_list(),
                "admin_url": settings.ADMIN_URL.rstrip("/"),
                **context,
            },
        )

    def missing(request: Request, what: str) -> HTMLResponse:
        """A styled 404.

        The status code matters as much as the page: a bookmark to a deleted run
        should not read as success to a browser, a crawler or an uptime check.
        """
        return page(request, "missing.html", status_code=404, what=what)

    # -- dashboard -----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        from mlango.agents.tracing import recent_traces
        from mlango.core.registry import apps
        from mlango.training.run import recent_runs

        return page(
            request,
            "index.html",
            counts=apps.summary()["counts"],
            runs=recent_runs(limit=8),
            traces=recent_traces(limit=8),
            versions=_model_versions(limit=6),
            datasets=_dataset_versions(limit=6),
        )

    # -- declared objects ----------------------------------------------------

    @app.get("/o/{label}", response_class=HTMLResponse)
    def object_detail(
        request: Request,
        label: str,
        q: str = Query(default=""),
        period: str = Query(default=""),
        message: str = Query(default=""),
        page_no: int = Query(default=1, alias="page", ge=1),
    ) -> HTMLResponse:
        try:
            entry = site.get(label)
        except LookupError:
            # A mistyped URL is a 404, not a 500: the sidebar on the page lists
            # every label, which is exactly what someone who mistyped one needs.
            return missing(request, f"object {label!r}")

        filters = {
            name: request.query_params.get(f"f_{name}", "") for name in entry.get_list_filter()
        }

        context: dict[str, Any] = {
            "entry": entry,
            "opts": entry.opts,
            "schema": entry.opts.schema(),
            "summary": entry.summary(),
            "search": q,
            "filters": filters,
            "filter_values": {n: entry.filter_values(n) for n in entry.get_list_filter()},
            "columns": entry.get_list_display(),
            "page_no": page_no,
            "period": period,
            "periods": entry.date_periods(),
            "actions": entry.get_actions(),
            "message": message,
        }

        if entry.kind == "dataset":
            context.update(_dataset_page(entry, q, filters, page_no, period))
        elif entry.kind == "model":
            versions = _model_versions(label=label)
            context["versions"] = versions
            context["runs"] = _runs_for(label)
            context["importances"] = _importance_bars(versions)
            context["drift"] = _drift_summary(entry.target, versions)
        elif entry.kind == "agent":
            from mlango.agents.tracing import recent_traces

            context["traces"] = recent_traces(limit=25, agent=label)
        elif entry.kind == "eval":
            context["runs"] = _runs_for(label)
            context["eval_diff"] = _eval_diff(label)

        return page(request, "object.html", **context)

    # -- runs ----------------------------------------------------------------

    @app.get("/runs", response_class=HTMLResponse)
    def run_list(
        request: Request,
        kind: str = Query(default=""),
        status: str = Query(default=""),
        target: str = Query(default=""),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> HTMLResponse:
        from mlango.metastore.models import RunKind, RunStatus
        from mlango.training.run import recent_runs

        runs = recent_runs(limit=limit, kind=kind or None, target=target or None)
        if status:
            runs = [r for r in runs if r.status == status]
        return page(
            request,
            "runs.html",
            runs=runs,
            kinds=RunKind.ALL,
            statuses=RunStatus.ALL,
            selected={"kind": kind, "status": status, "target": target},
        )

    @app.get("/runs/{reference}", response_class=HTMLResponse)
    def run_detail(request: Request, reference: str) -> HTMLResponse:
        from mlango.training.run import get_run, metric_history, metric_keys

        run = get_run(reference)
        if run is None:
            return missing(request, f"run {reference!r}")

        charts = []
        for key in metric_keys(run.id):
            points = metric_history(run.id, key)
            if len(points) > 1:
                charts.append({"key": key, "points": points, "svg": _sparkline(points)})

        return page(
            request,
            "run.html",
            run=run,
            charts=charts,
            scalars=_final_metrics(run.id),
            artifacts=_artifacts(run.id),
            eval_results=_eval_results(run.id),
        )

    @app.get("/compare", response_class=HTMLResponse)
    def compare(request: Request, ids: str = Query(default="")) -> HTMLResponse:
        from mlango.training.run import get_run

        references = [part.strip() for part in ids.split(",") if part.strip()]
        runs = [r for r in (get_run(ref) for ref in references) if r is not None]
        keys = sorted({k for run in runs for k in (run.summary or {})})
        params = sorted({k for run in runs for k in (run.params or {}) if not k.startswith("_")})
        return page(
            request, "compare.html", runs=runs, metric_keys=keys, param_keys=params, ids=ids
        )

    # -- traces --------------------------------------------------------------

    @app.get("/traces", response_class=HTMLResponse)
    def trace_list(
        request: Request,
        agent: str = Query(default=""),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> HTMLResponse:
        from mlango.agents.tracing import recent_traces

        return page(
            request,
            "traces.html",
            traces=recent_traces(limit=limit, agent=agent or None),
            agents=[a.label for a in site.all("agent")],
            selected_agent=agent,
        )

    @app.get("/traces/{reference}", response_class=HTMLResponse)
    def trace_detail(request: Request, reference: str) -> HTMLResponse:
        from mlango.agents.tracing import get_trace

        trace = get_trace(reference)
        if trace is None:
            return missing(request, f"trace {reference!r}")
        total = trace.duration_s or 0.0
        spans = [
            {
                "span": span,
                "width": max(1.0, (span.duration_s or 0) / total * 100) if total else 1.0,
            }
            for span in trace.spans
        ]
        return page(request, "trace.html", trace=trace, spans=spans)

    # -- versions ------------------------------------------------------------

    @app.get("/versions", response_class=HTMLResponse)
    def versions(request: Request) -> HTMLResponse:
        from mlango.metastore.models import Stage

        return page(
            request,
            "versions.html",
            versions=_model_versions(limit=200),
            datasets=_dataset_versions(limit=200),
            stages=Stage.ALL,
        )

    @app.post("/o/{label}/action")
    def object_action(
        request: Request,
        label: str,
        action: str = Form(...),
        selected: list[str] = Form(default=[]),
    ) -> RedirectResponse:
        """Run one admin action over the selected rows.

        Django's changelist actions, with the same shape: pick rows, pick an
        action, apply. A redirect afterwards so a refresh does not run it twice.
        """
        from mlango.core.exceptions import MlangoError

        try:
            entry = site.get(label)
        except LookupError:
            return RedirectResponse(str(request.url_for("index")), status_code=303)

        keys = {str(value) for value in selected}
        primary_key = entry.opts.extras.get("primary_key")
        try:
            records = [
                record
                for record in entry.get_queryset().take(entry.preview_limit)
                if primary_key and str(record.get(primary_key)) in keys
            ]
            message = entry.run_action(action, records)
        except MlangoError as exc:
            message = f"Failed: {exc}"
        except Exception as exc:  # noqa: BLE001 - the admin reports, it does not 500
            logger.exception("Admin action %s on %s failed", action, label)
            message = f"Failed: {exc}"

        target = request.url_for("object_detail", label=label)
        return RedirectResponse(f"{target}?message={quote(message)}", status_code=303)

    @app.post("/versions/{version_id}/promote")
    def promote(request: Request, version_id: int, stage: str = Form(...)) -> RedirectResponse:
        from mlango.core.registry import apps
        from mlango.metastore.models import ModelVersion
        from mlango.metastore.session import session_scope

        with session_scope() as session:
            row = session.get(ModelVersion, version_id)
            label, number = (row.label, row.version) if row else (None, None)

        if label is not None and number is not None:
            apps.get_model(label).promote(number, stage)

        # url_for, not settings.ADMIN_URL: this app may be mounted anywhere (or
        # nowhere, under a test client), and the redirect has to land on the page
        # this very app serves rather than on a path only the default mount has.
        return RedirectResponse(str(request.url_for("versions")), status_code=303)

    return app


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def _dataset_page(
    entry: Any, search: str, filters: dict[str, str], page_no: int, period: str = ""
) -> dict[str, Any]:
    per_page = entry.list_per_page
    try:
        query = entry.filtered(search=search, filters=filters, period=period)
        rows = list(query.skip((page_no - 1) * per_page).take(per_page + 1))
        has_next = len(rows) > per_page
        return {
            "rows": rows[:per_page],
            "has_next": has_next,
            "per_page": per_page,
            "versions": _dataset_versions(label=entry.label),
            "preview_error": "",
        }
    except Exception as exc:
        logger.warning("Could not preview %s: %s", entry.label, exc)
        return {
            "rows": [],
            "has_next": False,
            "per_page": per_page,
            "versions": _dataset_versions(label=entry.label),
            "preview_error": str(exc),
        }


def _model_versions(*, label: str | None = None, limit: int = 50) -> list[Any]:
    from sqlalchemy import select

    from mlango.metastore.models import ModelVersion
    from mlango.metastore.session import session_scope

    with session_scope() as session:
        statement = select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(limit)
        if label:
            statement = statement.where(ModelVersion.label == label)
        return list(session.execute(statement).scalars())


def _importance_bars(versions: list[Any], limit: int = 20) -> dict[str, Any] | None:
    """Feature weights of the newest version that recorded any, as bar widths.

    Newest rather than promoted: this is a debugging view, and the version
    someone is looking at right after training is the one they have questions
    about. Which version it came from is shown, so there is no guessing.
    """
    version = next((v for v in versions if v.importances), None)
    if version is None:
        return None

    weights = {str(k): float(v) for k, v in version.importances.items()}
    ranked = sorted(weights.items(), key=lambda pair: -abs(pair[1]))[:limit]
    largest = max((abs(value) for _, value in ranked), default=0.0)
    return {
        "version": version.version,
        "total": len(weights),
        # Tree importances are all positive and need no legend; linear
        # coefficients carry a direction, and a chart of magnitudes alone would
        # show "means negative" as agreement.
        "signed": any(value < 0 for _, value in ranked),
        "rows": [
            {
                "name": name,
                "value": value,
                # A zero-weight feature would divide by zero and, more to the
                # point, a chart of nothing but zeroes should read as empty.
                "width": (abs(value) / largest * 100) if largest else 0.0,
            }
            for name, value in ranked
        ],
    }


#: Enough rows for a stable index without turning a page load into a scan.
DRIFT_SAMPLE = 2000
DRIFT_WINDOW_DAYS = 7


def _drift_summary(model_class: Any, versions: list[Any]) -> dict[str, Any] | None:
    """Recent logged traffic measured against the version's training profile.

    Returns None whenever the answer would be meaningless — the log is off, the
    version predates baselines, nothing has been served — because an empty
    drift table on every model page teaches people to ignore the drift table.
    """
    import datetime as dt

    from sqlalchemy import select

    from mlango.metastore.models import Prediction, utcnow
    from mlango.metastore.session import session_scope
    from mlango.training import drift

    version = next((v for v in versions if v.baseline), None)
    if version is None:
        return None

    label = version.label
    since = utcnow() - dt.timedelta(days=DRIFT_WINDOW_DAYS)
    with session_scope() as session:
        rows = list(
            session.execute(
                select(Prediction.inputs, Prediction.output)
                .where(Prediction.label == label, Prediction.created_at >= since)
                .order_by(Prediction.created_at.desc())
                .limit(DRIFT_SAMPLE)
            )
        )
    if not rows:
        return None

    baseline = version.baseline or {}
    try:
        features = model_class.get_features()
        target = model_class.get_target()
    except Exception:  # noqa: BLE001 - an incomplete declaration still renders
        features, target = list(baseline), ""

    scores = drift.compare(
        {k: v for k, v in baseline.items() if k in features}, [r[0] for r in rows]
    )
    outputs = [r[1] for r in rows if r[1] is not None]
    if outputs and target in baseline:
        predicted = drift.compare({target: baseline[target]}, outputs)
        if target in predicted:
            scores[f"{target} (predicted)"] = predicted[target]
    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda pair: -pair[1]["psi"])
    return {
        "version": version.version,
        "rows": len(rows),
        "days": DRIFT_WINDOW_DAYS,
        "worst": ranked[0][1]["verdict"],
        "columns": [{"name": name, **entry} for name, entry in ranked],
    }


def _eval_diff(label: str) -> dict[str, Any] | None:
    """The last run against the one before it.

    Cheap enough for a page load, unlike the model diff: nothing is loaded and
    nothing is scored, because ``evaluate`` already wrote a verdict per case.
    That is the whole difference, and it is why this one is here and that one
    is a command you choose to run.
    """
    from mlango.evals.comparison import compare_runs, recent_runs

    found = recent_runs(label, limit=2)
    if len(found) < 2:
        return None

    newer, older = found
    try:
        return compare_runs(older, newer, max_changes=10)
    except LookupError:
        # No case in common — a renumbered suite, usually. Saying nothing is
        # better than a table comparing two things that are not comparable.
        return None


def _dataset_versions(*, label: str | None = None, limit: int = 50) -> list[Any]:
    from sqlalchemy import select

    from mlango.metastore.models import DatasetVersion
    from mlango.metastore.session import session_scope

    with session_scope() as session:
        statement = select(DatasetVersion).order_by(DatasetVersion.created_at.desc()).limit(limit)
        if label:
            statement = statement.where(DatasetVersion.label == label)
        return list(session.execute(statement).scalars())


def _runs_for(target: str, limit: int = 25) -> list[Any]:
    from mlango.training.run import recent_runs

    return recent_runs(limit=limit, target=target)


def _artifacts(run_id: int) -> list[Any]:
    from sqlalchemy import select

    from mlango.metastore.models import Artifact
    from mlango.metastore.session import session_scope

    with session_scope() as session:
        return list(session.execute(select(Artifact).where(Artifact.run_id == run_id)).scalars())


def _eval_results(run_id: int, limit: int = 100) -> list[Any]:
    from sqlalchemy import select

    from mlango.metastore.models import EvalResult
    from mlango.metastore.session import session_scope

    with session_scope() as session:
        return list(
            session.execute(
                select(EvalResult).where(EvalResult.run_id == run_id).limit(limit)
            ).scalars()
        )


def _final_metrics(run_id: int) -> dict[str, float]:
    """The last recorded value of each metric key."""
    from sqlalchemy import select

    from mlango.metastore.models import Metric
    from mlango.metastore.session import session_scope

    with session_scope() as session:
        rows = session.execute(
            select(Metric.key, Metric.value, Metric.step)
            .where(Metric.run_id == run_id)
            .order_by(Metric.step)
        ).all()
    latest: dict[str, float] = {}
    for key, value, _step in rows:
        latest[key] = value
    return dict(sorted(latest.items()))


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _sparkline(points: list[tuple[int, float]], width: int = 460, height: int = 120) -> str:
    """An inline SVG line chart — no chart library, no network fetch."""
    if len(points) < 2:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = (x_max - x_min) or 1
    y_span = (y_max - y_min) or 1
    pad = 8

    coords = []
    for x, y in points:
        px = pad + (x - x_min) / x_span * (width - 2 * pad)
        py = height - pad - (y - y_min) / y_span * (height - 2 * pad)
        coords.append(f"{px:.1f},{py:.1f}")

    polyline = " ".join(coords)
    area = f"{pad},{height - pad} {polyline} {width - pad},{height - pad}"
    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        f'aria-label="metric over {len(points)} steps">'
        f'<polygon points="{area}" class="chart-area"/>'
        f'<polyline points="{polyline}" class="chart-line"/>'
        f"</svg>"
    )


def _short(value: Any, length: int = 60) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= length else text[: length - 1] + "…"


def _duration(seconds: Any) -> str:
    if seconds is None:
        return "—"
    seconds = float(seconds)
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:.0f}s"


def _number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:,}" if isinstance(value, int) else str(value)


__all__ = ["build_admin_app"]
