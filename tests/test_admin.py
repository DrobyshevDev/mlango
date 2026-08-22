"""The admin: registration, options, authentication and every page.

Pages are fetched through ``TestClient`` and asserted on their rendered HTML —
a template that raises, or one that silently drops a section, fails here.
"""

from __future__ import annotations

import os

import pytest

from mlango.admin.options import AgentAdmin, DatasetAdmin, EvalAdmin, ModelAdmin, ObjectAdmin
from mlango.admin.sites import AdminSite
from mlango.core import fields
from mlango.core.exceptions import ImproperlyConfigured
from mlango.data import Dataset, InMemorySource

pytestmark = pytest.mark.usefixtures("isolated_registry")

ROWS = [
    {
        "id": index,
        "text": ("great movie " if index % 2 else "terrible movie ") + str(index),
        "label": "pos" if index % 2 else "neg",
        "stars": (index % 5) + 1,
    }
    for index in range(60)
]


@pytest.fixture
def site_with_data(project):
    """A private admin site holding one dataset, one model, one agent, one eval."""
    from mlango.agents import Agent, tool
    from mlango.evals import Eval, exact_match
    from mlango.training import Model

    class Reviews(Dataset):
        """Reviews for the admin tests."""

        id = fields.IntegerField()
        text = fields.TextField()
        label = fields.LabelField(["neg", "pos"])
        stars = fields.IntegerField(min_value=1, max_value=5)

        class Meta:
            source = InMemorySource(ROWS)
            primary_key = "id"

    class Sentiment(Model):
        """A model for the admin tests."""

        C = fields.FloatField(default=1.0, tunable=True)

        class Meta:
            dataset = Reviews
            trainer = "sklearn"
            task = "classification"
            features = ["text"]

        def build(self):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline

            return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=500))

    @tool
    def echo(text: str) -> str:
        """Echo the text.

        Args:
            text: What to echo.
        """
        return text

    class Helper(Agent):
        """An agent for the admin tests."""

        class Meta:
            tools = [echo]

    class Accuracy(Eval):
        """An eval for the admin tests."""

        class Meta:
            dataset = Reviews
            target = Sentiment
            input_field = "text"
            expected_field = "label"
            case_id_field = "id"
            scorers = {"correct": exact_match}
            max_cases = 10

    site = AdminSite(name="test-admin")
    site.autodiscover()
    return site, Reviews, Sentiment, Helper, Accuracy


@pytest.fixture
def client(site_with_data):
    from fastapi.testclient import TestClient

    from mlango.admin.app import build_admin_app

    site, *_ = site_with_data
    with TestClient(build_admin_app(site)) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


class TestRegistration:
    def test_everything_declared_appears_without_registration(self, site_with_data):
        site, reviews, sentiment, helper, accuracy = site_with_data
        labels = {entry.label for entry in site.all()}
        assert {
            reviews._meta.label,
            sentiment._meta.label,
            helper._meta.label,
            accuracy._meta.label,
        } <= labels

    def test_the_default_options_match_the_family(self, site_with_data):
        site, reviews, sentiment, helper, accuracy = site_with_data
        assert isinstance(site.get(reviews._meta.label), DatasetAdmin)
        assert isinstance(site.get(sentiment._meta.label), ModelAdmin)
        assert isinstance(site.get(helper._meta.label), AgentAdmin)
        assert isinstance(site.get(accuracy._meta.label), EvalAdmin)

    def test_explicit_registration_wins_over_autodiscovery(self, project):
        class Rows(Dataset):
            id = fields.IntegerField()
            text = fields.TextField()

            class Meta:
                source = InMemorySource([{"id": 1, "text": "a"}])

        site = AdminSite()

        @site.register(Rows)
        class RowsAdmin(ObjectAdmin):
            list_display = ("text",)

        site.autodiscover()
        assert site.get(Rows._meta.label).get_list_display() == ["text"]

    def test_registering_twice_is_an_error(self, project):
        class Rows(Dataset):
            id = fields.IntegerField()

            class Meta:
                source = InMemorySource([{"id": 1}])

        site = AdminSite()
        site.register(Rows, ObjectAdmin)
        with pytest.raises(ImproperlyConfigured, match="already registered"):
            site.register(Rows, ObjectAdmin)

    def test_registering_a_non_declarative_class_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="not an mlango declarative"):
            AdminSite().register(dict, ObjectAdmin)

    def test_unregister(self, project):
        class Rows(Dataset):
            id = fields.IntegerField()

            class Meta:
                source = InMemorySource([{"id": 1}])

        site = AdminSite()
        site.register(Rows, ObjectAdmin)
        assert Rows._meta.label in site
        site.unregister(Rows)
        assert Rows._meta.label not in site

    def test_an_unknown_label_lists_alternatives(self, site_with_data):
        site, *_ = site_with_data
        with pytest.raises(LookupError, match="Registered:"):
            site.get("nope.Nope")

    def test_app_list_groups_by_app_then_kind(self, site_with_data):
        site, *_ = site_with_data
        grouped = site.app_list()
        assert grouped

        order = ["dataset", "model", "agent", "eval"]
        for app in grouped:
            # The sidebar order is fixed within an app; apps themselves are
            # separate sections, so the sequence only has to hold inside one.
            kinds = [group["kind"] for group in app["kinds"]]
            assert kinds == sorted(kinds, key=order.index)

    def test_check_reports_a_field_that_does_not_exist(self, project):
        class Rows(Dataset):
            id = fields.IntegerField()

            class Meta:
                source = InMemorySource([{"id": 1}])

        site = AdminSite()

        @site.register(Rows)
        class RowsAdmin(ObjectAdmin):
            list_display = ("nope",)

        problems = site.check()
        assert problems
        assert "nope" in problems[0]

    def test_repr_reports_the_count(self, site_with_data):
        site, *_ = site_with_data
        assert "objects" in repr(site)
        assert len(site) >= 4


# --------------------------------------------------------------------------- #
# ObjectAdmin options
# --------------------------------------------------------------------------- #


class TestObjectAdminOptions:
    def test_default_columns_are_the_first_fields(self, site_with_data):
        site, reviews, *_ = site_with_data
        assert site.get(reviews._meta.label).get_list_display() == ["id", "text", "label", "stars"]

    def test_default_filters_are_bounded_fields(self, site_with_data):
        site, reviews, *_ = site_with_data
        assert "label" in site.get(reviews._meta.label).get_list_filter()

    def test_default_search_fields_are_text(self, site_with_data):
        site, reviews, *_ = site_with_data
        assert site.get(reviews._meta.label).get_search_fields() == ["text"]

    def test_filter_values_come_from_the_declaration(self, site_with_data):
        site, reviews, *_ = site_with_data
        assert site.get(reviews._meta.label).filter_values("label") == ["neg", "pos"]

    def test_filter_values_fall_back_to_scanning(self, site_with_data):
        site, reviews, *_ = site_with_data
        values = site.get(reviews._meta.label).filter_values("stars")
        assert set(values) == {1, 2, 3, 4, 5}

    def test_filtering_narrows_the_queryset(self, site_with_data):
        site, reviews, *_ = site_with_data
        entry = site.get(reviews._meta.label)
        assert entry.filtered(filters={"label": "pos"}).count() == 30

    def test_search_matches_case_insensitively(self, site_with_data):
        site, reviews, *_ = site_with_data
        entry = site.get(reviews._meta.label)
        assert entry.filtered(search="TERRIBLE").count() == 30

    def test_search_and_filter_combine(self, site_with_data):
        site, reviews, *_ = site_with_data
        entry = site.get(reviews._meta.label)
        assert entry.filtered(search="great", filters={"label": "pos"}).count() == 30

    def test_render_truncates_long_values(self, site_with_data):
        from mlango.data.query import Record

        site, reviews, *_ = site_with_data
        entry = site.get(reviews._meta.label)
        rendered = entry.render(Record({"text": "x" * 500}), "text")
        assert len(rendered) <= 160
        assert rendered.endswith("…")

    def test_render_shows_a_dash_for_none(self, site_with_data):
        from mlango.data.query import Record

        site, reviews, *_ = site_with_data
        assert site.get(reviews._meta.label).render(Record({"text": None}), "text") == "—"

    def test_a_custom_renderer_is_used(self, project):
        from mlango.data.query import Record

        class Rows(Dataset):
            id = fields.IntegerField()
            text = fields.TextField()

            class Meta:
                source = InMemorySource([{"id": 1, "text": "abc"}])

        site = AdminSite()

        @site.register(Rows)
        class RowsAdmin(ObjectAdmin):
            list_display = ("length",)

            def render_length(self, record):
                return f"{len(record['text'])} chars"

        assert site.get(Rows._meta.label).render(Record({"text": "abc"}), "length") == "3 chars"

    def test_ordering_is_applied(self, project):
        class Rows(Dataset):
            id = fields.IntegerField()

            class Meta:
                source = InMemorySource([{"id": 3}, {"id": 1}, {"id": 2}])

        site = AdminSite()

        @site.register(Rows)
        class RowsAdmin(ObjectAdmin):
            ordering = ("id",)

        assert [r["id"] for r in site.get(Rows._meta.label).get_queryset()] == [1, 2, 3]

    def test_an_object_without_a_manager_is_reported(self, site_with_data):
        from mlango.core.exceptions import FieldError

        site, _reviews, _sentiment, helper, _accuracy = site_with_data
        with pytest.raises(FieldError, match="no objects manager"):
            site.get(helper._meta.label).get_queryset()

    def test_summary_survives_an_incomplete_declaration(self, project):
        from mlango.training import Model

        class Incomplete(Model):
            """No dataset, on purpose."""

        site = AdminSite()
        site.register(Incomplete, ModelAdmin)
        summary = site.get(Incomplete._meta.label).summary()
        # A half-declared object must still render a page rather than 500.
        assert "label" in summary or "error" in summary


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


class TestPages:
    def test_the_overview_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Overview" in response.text
        assert "DATASETS" in response.text.upper()

    def test_a_dataset_page_previews_rows(self, client, site_with_data):
        _site, reviews, *_ = site_with_data
        response = client.get(f"/o/{reviews._meta.label}")
        assert response.status_code == 200
        assert "Data preview" in response.text
        assert "great movie 1" in response.text
        assert "Declaration" in response.text

    def test_a_dataset_page_honours_search(self, client, site_with_data):
        _site, reviews, *_ = site_with_data
        response = client.get(f"/o/{reviews._meta.label}", params={"q": "terrible"})
        assert response.status_code == 200
        assert "terrible movie" in response.text
        assert "great movie 1<" not in response.text

    def test_a_dataset_page_honours_a_filter(self, client, site_with_data):
        _site, reviews, *_ = site_with_data
        response = client.get(f"/o/{reviews._meta.label}", params={"f_label": "neg"})
        assert response.status_code == 200
        assert "terrible movie" in response.text

    def test_a_dataset_page_paginates(self, client, site_with_data):
        _site, reviews, *_ = site_with_data
        first = client.get(f"/o/{reviews._meta.label}", params={"page": 1})
        second = client.get(f"/o/{reviews._meta.label}", params={"page": 2})
        assert "next" in first.text
        assert second.status_code == 200
        assert "page 2" in second.text

    def test_a_model_page_renders_before_training(self, client, site_with_data):
        _site, _reviews, sentiment, *_ = site_with_data
        response = client.get(f"/o/{sentiment._meta.label}")
        assert response.status_code == 200
        assert "Not trained yet" in response.text

    def test_an_agent_page_renders(self, client, site_with_data):
        _site, _reviews, _sentiment, helper, _accuracy = site_with_data
        response = client.get(f"/o/{helper._meta.label}")
        assert response.status_code == 200
        assert "has not run yet" in response.text

    def test_an_eval_page_renders(self, client, site_with_data):
        *_, accuracy = site_with_data
        response = client.get(f"/o/{accuracy._meta.label}")
        assert response.status_code == 200
        assert "Runs" in response.text


class TestEvalDiffCard:
    """The eval page shows what the last run changed — no command needed.

    Cheap enough for a page load because nothing is loaded and nothing scored:
    ``evaluate`` already wrote a verdict per case.
    """

    def _run(self, label, cases, outputs=None):
        from mlango.metastore.models import EvalResult, Run, RunKind, RunStatus
        from mlango.metastore.session import session_scope

        outputs = outputs or {}
        with session_scope() as session:
            run = Run(kind=RunKind.EVAL, target=label, status=RunStatus.FINISHED)
            session.add(run)
            session.flush()
            for case_id, passed in cases.items():
                session.add(
                    EvalResult(
                        run_id=run.id,
                        eval_label=label,
                        case_id=case_id,
                        passed=passed,
                        output=outputs.get(case_id, "pass" if passed else "fail"),
                        expected="pass",
                    )
                )
            return run

    def test_one_run_shows_no_card(self, client, site_with_data):
        """With nothing to compare against, an empty card would be noise."""
        *_, accuracy = site_with_data
        self._run(accuracy._meta.label, {"a": True})

        assert "Since the previous run" not in client.get(f"/o/{accuracy._meta.label}").text

    def test_a_regression_is_visible_without_running_a_command(self, client, site_with_data):
        *_, accuracy = site_with_data
        label = accuracy._meta.label
        self._run(label, {"a": True, "b": True, "c": False})
        self._run(label, {"a": True, "b": False, "c": True})

        text = client.get(f"/o/{label}").text

        assert "Since the previous run" in text
        assert "3 shared cases" in text
        assert "case that had been passing" in text, "the broken one is spelled out"
        assert 'class="tag bad">1<' in text, "and marked, because it is the number that matters"

    def test_the_verdict_says_whether_it_is_a_coin(self, client, site_with_data):
        """Twenty-five rescued against twenty-five lost is a coin, and must read as one."""
        *_, accuracy = site_with_data
        label = accuracy._meta.label
        self._run(label, {f"c{i}": i >= 25 for i in range(50)})
        self._run(label, {f"c{i}": i < 25 for i in range(50)})

        text = client.get(f"/o/{label}").text
        assert "is a coin, not a change" in text

    def test_a_landslide_of_fixes_reads_as_a_real_improvement(self, client, site_with_data):
        *_, accuracy = site_with_data
        label = accuracy._meta.label
        self._run(label, {f"c{i}": False for i in range(30)})
        self._run(label, {f"c{i}": i > 0 for i in range(30)})

        assert "a real improvement" in client.get(f"/o/{label}").text

    def test_cases_only_in_one_run_are_named_on_the_page(self, client, site_with_data):
        """A suite that grew is a different suite, and the page has to say so."""
        *_, accuracy = site_with_data
        label = accuracy._meta.label
        self._run(label, {"shared": True})
        self._run(label, {"shared": True, "brand_new": True})

        text = client.get(f"/o/{label}").text
        assert "only in the newer run" in text
        assert "brand_new" in text

    def test_two_runs_sharing_no_case_render_nothing_rather_than_a_lie(
        self, client, site_with_data
    ):
        *_, accuracy = site_with_data
        label = accuracy._meta.label
        self._run(label, {"old_numbering": True})
        self._run(label, {"new_numbering": True})

        response = client.get(f"/o/{label}")
        assert response.status_code == 200
        assert "Since the previous run" not in response.text

    def test_a_model_page_does_not_grow_an_eval_card(self, client, site_with_data):
        _site, _reviews, sentiment, *_ = site_with_data
        assert "Since the previous run" not in client.get(f"/o/{sentiment._meta.label}").text

    def test_the_runs_list_renders_when_empty(self, client):
        response = client.get("/runs")
        assert response.status_code == 200
        assert "No runs match" in response.text

    def test_the_traces_list_renders_when_empty(self, client):
        response = client.get("/traces")
        assert response.status_code == 200
        assert "No traces yet" in response.text

    def test_the_versions_page_renders_when_empty(self, client):
        response = client.get("/versions")
        assert response.status_code == 200
        assert "No model versions registered" in response.text

    def test_compare_without_ids_asks_for_them(self, client):
        response = client.get("/compare")
        assert response.status_code == 200
        assert "Paste two or more run ids" in response.text

    def test_a_missing_run_renders_a_not_found_page(self, client):
        response = client.get("/runs/ffffffff")
        assert response.status_code == 404
        assert "Not found" in response.text

    def test_a_missing_trace_renders_a_not_found_page(self, client):
        response = client.get("/traces/ffffffff")
        assert response.status_code == 404
        assert "Not found" in response.text

    def test_an_unknown_object_is_a_404_page_not_a_500(self, client):
        response = client.get("/o/nope.Nope")
        assert response.status_code == 404
        assert "Not found" in response.text
        # The 404 still carries the sidebar, which lists every real label.
        assert "Reviews" in response.text

    def test_promoting_an_unknown_version_does_not_explode(self, client):
        response = client.post(
            "/versions/9999/promote", data={"stage": "production"}, follow_redirects=False
        )
        assert response.status_code == 303

    def test_promote_redirects_to_the_page_this_app_serves(self, client):
        """The redirect must not assume the default mount point.

        Building it from settings.ADMIN_URL sent anyone running the admin on a
        different prefix (or standalone) to a path that does not exist.
        """
        response = client.post(
            "/versions/9999/promote", data={"stage": "production"}, follow_redirects=False
        )
        assert response.headers["location"].endswith("/versions")

        followed = client.post("/versions/9999/promote", data={"stage": "production"})
        assert followed.status_code == 200
        assert "Model versions" in followed.text


class TestPagesWithData:
    @pytest.fixture
    def populated(self, client, site_with_data, sklearn_or_skip):
        """Train, evaluate and run the agent so the pages have content."""
        _site, reviews, sentiment, helper, accuracy = site_with_data

        model = sentiment()
        run = model.train()
        accuracy.evaluate()
        result = helper().run("hello")
        reviews.materialize(reviews.objects.get_queryset())
        return client, run, result, sentiment, reviews

    def test_the_overview_lists_what_happened(self, populated):
        client, *_ = populated
        text = client.get("/").text
        assert "Recent runs" in text
        assert "Recent agent traces" in text
        assert "Latest model versions" in text

    def test_a_run_page_shows_the_full_record(self, populated):
        client, run, *_ = populated
        text = client.get(f"/runs/{run.uuid}").text
        assert "Environment" in text
        assert "Parameters" in text
        assert "Metrics" in text
        assert "Artifacts" in text
        assert "_data_fingerprint" in text

    def test_a_run_page_charts_multi_point_metrics(self, populated):
        client, run, *_ = populated
        text = client.get(f"/runs/{run.uuid}").text
        # Charts are inline SVG, so no network fetch is needed to render them.
        assert "<svg" in text or "No metrics recorded" in text

    def test_an_eval_run_lists_its_cases(self, populated):
        client, *_ = populated
        from mlango.training import recent_runs

        eval_run = next(r for r in recent_runs(limit=10) if r.kind == "eval")
        text = client.get(f"/runs/{eval_run.uuid}").text
        assert "Evaluation cases" in text

    def test_the_runs_list_shows_the_run(self, populated):
        client, run, *_ = populated
        assert run.short_id in client.get("/runs").text

    def test_the_runs_list_filters(self, populated):
        client, *_ = populated
        assert client.get("/runs", params={"kind": "train"}).status_code == 200
        assert client.get("/runs", params={"status": "finished"}).status_code == 200
        assert client.get("/runs", params={"target": "nothing"}).status_code == 200

    def test_compare_lines_two_runs_up(self, populated):
        client, *_ = populated
        from mlango.training import recent_runs

        runs = recent_runs(limit=2)
        text = client.get("/compare", params={"ids": f"{runs[0].uuid},{runs[1].uuid}"}).text
        assert runs[0].short_id in text
        assert runs[1].short_id in text

    def test_compare_with_unknown_ids_says_so(self, populated):
        client, *_ = populated
        assert (
            "None of those run ids matched"
            in client.get("/compare", params={"ids": "zzzz,yyyy"}).text
        )

    def test_a_trace_page_shows_the_steps(self, populated):
        client, _run, result, *_ = populated
        text = client.get(f"/traces/{result.trace_uuid}").text
        assert "Conversation" in text
        assert "Steps" in text

    def test_the_traces_list_filters_by_agent(self, populated):
        client, *_ = populated
        assert client.get("/traces", params={"agent": "nope"}).status_code == 200

    def test_the_versions_page_lists_both_kinds(self, populated):
        client, *_ = populated
        text = client.get("/versions").text
        assert "Model versions" in text
        assert "Dataset versions" in text

    def test_a_model_page_lists_versions_and_runs(self, populated):
        client, _run, _result, sentiment, _reviews = populated
        text = client.get(f"/o/{sentiment._meta.label}").text
        assert "Registered versions" in text
        assert "v1" in text

    def test_no_drift_card_without_a_prediction_log(self, populated):
        """An empty drift table on every page teaches people to ignore it."""
        client, _run, _result, sentiment, _reviews = populated
        assert "Input drift" not in client.get(f"/o/{sentiment._meta.label}").text

    def test_the_drift_card_appears_once_traffic_is_logged(self, populated):
        from mlango.conf import settings

        client, _run, _result, sentiment, _reviews = populated
        before = settings.PREDICTION_LOG
        settings.PREDICTION_LOG = {"ENABLED": True, "SAMPLE": 1.0, "MAX_ROWS": 0}
        try:
            sentiment.load().predict(["ok"] * 30)
            text = client.get(f"/o/{sentiment._meta.label}").text
        finally:
            settings.PREDICTION_LOG = before

        assert "Input drift" in text
        assert "30 logged predictions from the last 7 days" in text
        assert 'class="tag significant"' in text
        assert "label (predicted)" in text

    def test_a_model_page_charts_feature_importance(self, populated):
        client, _run, _result, sentiment, _reviews = populated
        text = client.get(f"/o/{sentiment._meta.label}").text
        assert "Feature importance" in text
        assert "What v1 weighted most heavily" in text
        assert 'class="importances"' in text
        assert "width: " in text

    def test_bar_widths_are_relative_to_the_largest_weight(self, populated):
        from mlango.admin.app import _importance_bars

        class Version:
            version = 1
            importances = {"a": 2.0, "b": -1.0, "c": 0.0}

        bars = _importance_bars([Version()])
        assert bars["version"] == 1
        assert [row["name"] for row in bars["rows"]] == ["a", "b", "c"]
        assert [round(row["width"]) for row in bars["rows"]] == [100, 50, 0]
        assert bars["signed"], "a negative weight needs the legend"

    def test_positive_only_weights_need_no_legend(self):
        from mlango.admin.app import _importance_bars

        class Version:
            version = 3
            importances = {"a": 0.6, "b": 0.4}

        assert _importance_bars([Version()])["signed"] is False

    def test_a_model_with_no_weights_shows_no_chart(self, populated):
        from mlango.admin.app import _importance_bars

        class Version:
            version = 1
            importances = None

        assert _importance_bars([Version()]) is None

    def test_a_dataset_page_lists_materialised_versions(self, populated):
        client, _run, _result, _sentiment, reviews = populated
        text = client.get(f"/o/{reviews._meta.label}").text
        assert "Materialised versions" in text

    def test_promoting_from_the_admin_works(self, populated):
        client, _run, _result, sentiment, _reviews = populated

        response = client.post("/versions/1/promote", data={"stage": "production"})
        assert response.status_code in (200, 303)
        assert sentiment.versions()[0].stage == "production"


# --------------------------------------------------------------------------- #
# Django-shaped customisation
# --------------------------------------------------------------------------- #


class TestTemplateOverriding:
    def test_the_framework_templates_are_found_by_default(self, project):
        from mlango.admin.app import TEMPLATE_DIR, _template_dirs

        assert _template_dirs() == [TEMPLATE_DIR]

    def test_a_project_directory_is_searched_first(self, project):
        """Django's admin is customised by shadowing a template, not by forking."""
        from mlango.admin.app import TEMPLATE_DIR, _template_dirs

        override = project / "templates" / "admin"
        override.mkdir(parents=True)

        dirs = _template_dirs()
        assert dirs[0] == os.path.normpath(str(override))
        assert dirs[-1] == TEMPLATE_DIR

    def test_a_directory_that_does_not_exist_is_skipped(self, project):
        from mlango.admin.app import TEMPLATE_DIR, _template_dirs
        from mlango.conf import settings

        settings.ADMIN_TEMPLATE_DIRS = ["nope/at/all"]
        assert _template_dirs() == [TEMPLATE_DIR]

    def test_overriding_one_template_leaves_the_rest_alone(self, project, site_with_data):
        from fastapi.testclient import TestClient

        from mlango.admin.app import build_admin_app

        override = project / "templates" / "admin"
        override.mkdir(parents=True)
        (override / "missing.html").write_text(
            "{% extends 'base.html' %}{% block content %}<h1>Nothing here</h1>{% endblock %}",
            encoding="utf-8",
        )

        site, *_ = site_with_data
        with TestClient(build_admin_app(site)) as client:
            replaced = client.get("/runs/ffffffff")
            assert "Nothing here" in replaced.text
            # base.html was not overridden, so the shipped chrome still applies.
            assert "breadcrumbs" in replaced.text
            # And every other page still resolves from the framework's directory.
            assert client.get("/").status_code == 200

    def test_an_absolute_directory_is_accepted(self, project, tmp_path):
        from mlango.admin.app import _template_dirs
        from mlango.conf import settings

        elsewhere = tmp_path / "shared-admin"
        elsewhere.mkdir()
        settings.ADMIN_TEMPLATE_DIRS = [str(elsewhere)]
        assert _template_dirs()[0] == os.path.normpath(str(elsewhere))


class TestActions:
    @pytest.fixture
    def with_action(self, project):
        performed: list[list] = []

        class Rows(Dataset):
            id = fields.IntegerField()
            text = fields.TextField()

            class Meta:
                source = InMemorySource([{"id": i, "text": f"row {i}"} for i in range(5)])
                primary_key = "id"

        site = AdminSite()

        @site.register(Rows)
        class RowsAdmin(ObjectAdmin):
            def action_export(self, records):
                "Export the selected rows as JSONL"
                performed.append(records)
                return f"Exported {len(records)} row(s)."

            def action_touch(self, records):
                "Touch them"
                performed.append(records)

        return site, Rows, performed

    def test_actions_are_discovered_from_method_names(self, with_action):
        site, rows, _ = with_action
        assert site.get(rows._meta.label).get_actions() == {
            "export": "Export the selected rows as JSONL",
            "touch": "Touch them",
        }

    def test_the_label_comes_from_the_docstring(self, with_action):
        """So what the user reads and what the developer reads cannot drift."""
        site, rows, _ = with_action
        assert site.get(rows._meta.label).get_actions()["export"].startswith("Export")

    def test_an_explicit_order_is_honoured(self, project):
        class Rows(Dataset):
            id = fields.IntegerField()

            class Meta:
                source = InMemorySource([{"id": 1}])

        site = AdminSite()

        @site.register(Rows)
        class RowsAdmin(ObjectAdmin):
            actions = ("second", "first")

            def action_first(self, records):
                "One"

            def action_second(self, records):
                "Two"

        assert list(site.get(Rows._meta.label).get_actions()) == ["second", "first"]

    def test_an_admin_without_actions_offers_none(self, site_with_data):
        site, reviews, *_ = site_with_data
        assert site.get(reviews._meta.label).get_actions() == {}

    def test_running_one_returns_its_own_message(self, with_action):
        site, rows, performed = with_action
        entry = site.get(rows._meta.label)

        message = entry.run_action("export", list(rows.objects.take(3)))
        assert message == "Exported 3 row(s)."
        assert len(performed[0]) == 3

    def test_an_action_that_returns_nothing_gets_a_default_message(self, with_action):
        site, rows, _ = with_action
        entry = site.get(rows._meta.label)
        assert "2 row(s)" in entry.run_action("touch", list(rows.objects.take(2)))

    def test_an_unknown_action_lists_the_real_ones(self, with_action):
        from mlango.core.exceptions import FieldError

        site, rows, _ = with_action
        with pytest.raises(FieldError, match="export, touch"):
            site.get(rows._meta.label).run_action("nope", [])

    def test_the_checkboxes_and_action_bar_are_rendered(self, with_action):
        from fastapi.testclient import TestClient

        from mlango.admin.app import build_admin_app

        site, rows, _ = with_action
        with TestClient(build_admin_app(site)) as client:
            body = client.get(f"/o/{rows._meta.label}").text

        assert 'name="selected"' in body
        assert "Export the selected rows as JSONL" in body

    def test_posting_an_action_applies_it_to_the_ticked_rows(self, with_action):
        from fastapi.testclient import TestClient

        from mlango.admin.app import build_admin_app

        site, rows, performed = with_action
        with TestClient(build_admin_app(site)) as client:
            response = client.post(
                f"/o/{rows._meta.label}/action",
                data={"action": "export", "selected": ["1", "3"]},
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert [record["id"] for record in performed[-1]] == [1, 3]

    def test_a_failing_action_reports_instead_of_500ing(self, project):
        from fastapi.testclient import TestClient

        from mlango.admin.app import build_admin_app

        class Rows(Dataset):
            id = fields.IntegerField()

            class Meta:
                source = InMemorySource([{"id": 1}])
                primary_key = "id"

        site = AdminSite()

        @site.register(Rows)
        class RowsAdmin(ObjectAdmin):
            def action_explode(self, records):
                "Break on purpose"
                raise RuntimeError("boom")

        with TestClient(build_admin_app(site)) as client:
            response = client.post(
                f"/o/{Rows._meta.label}/action",
                data={"action": "explode", "selected": ["1"]},
                follow_redirects=False,
            )
            assert response.status_code == 303
            assert "Failed" in response.headers["location"]

    def test_an_action_on_an_unknown_object_redirects_home(self, with_action):
        from fastapi.testclient import TestClient

        from mlango.admin.app import build_admin_app

        site, *_ = with_action
        with TestClient(build_admin_app(site)) as client:
            response = client.post(
                "/o/nope.Nope/action", data={"action": "export"}, follow_redirects=False
            )
        assert response.status_code == 303


class TestDateHierarchy:
    @pytest.fixture
    def dated(self, project):
        class Events(Dataset):
            id = fields.IntegerField()
            happened = fields.DateTimeField()

            class Meta:
                source = InMemorySource(
                    [
                        {"id": 1, "happened": "2026-01-15T10:00:00"},
                        {"id": 2, "happened": "2026-01-20T10:00:00"},
                        {"id": 3, "happened": "2026-03-02T10:00:00"},
                    ]
                )
                primary_key = "id"

        site = AdminSite()

        @site.register(Events)
        class EventsAdmin(ObjectAdmin):
            date_hierarchy = "happened"

        return site, Events

    def test_the_periods_offered_are_the_months_present(self, dated):
        site, events = dated
        assert site.get(events._meta.label).date_periods() == ["2026-03", "2026-01"]

    def test_selecting_a_period_narrows_the_rows(self, dated):
        site, events = dated
        entry = site.get(events._meta.label)
        assert entry.filtered(period="2026-01").count() == 2
        assert entry.filtered(period="2026-03").count() == 1

    def test_no_period_shows_everything(self, dated):
        site, events = dated
        assert site.get(events._meta.label).filtered().count() == 3

    def test_an_admin_without_a_hierarchy_offers_no_periods(self, site_with_data):
        site, reviews, *_ = site_with_data
        assert site.get(reviews._meta.label).date_periods() == []

    def test_the_drill_down_is_rendered(self, dated):
        from fastapi.testclient import TestClient

        from mlango.admin.app import build_admin_app

        site, events = dated
        with TestClient(build_admin_app(site)) as client:
            body = client.get(f"/o/{events._meta.label}").text

        assert "date-hierarchy" in body
        assert "2026-01" in body
        assert "All dates" in body


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


class TestAuth:
    def test_off_by_default(self, project):
        from mlango.admin.auth import auth_configured, describe

        assert auth_configured() is False
        assert describe()["enabled"] is False

    def test_pages_are_open_when_no_password_is_set(self, client):
        assert client.get("/").status_code == 200

    def test_a_password_challenges(self, project, site_with_data):
        from fastapi.testclient import TestClient

        from mlango.admin.app import build_admin_app
        from mlango.conf import settings

        settings.ADMIN_PASSWORD = "s3cret"
        site, *_ = site_with_data

        with TestClient(build_admin_app(site)) as test_client:
            unauthenticated = test_client.get("/")
            assert unauthenticated.status_code == 401
            assert "Basic" in unauthenticated.headers["WWW-Authenticate"]

            assert test_client.get("/", auth=("admin", "s3cret")).status_code == 200
            assert test_client.get("/", auth=("admin", "wrong")).status_code == 401
            assert test_client.get("/", auth=("root", "s3cret")).status_code == 401

    def test_a_custom_username(self, project, site_with_data):
        from fastapi.testclient import TestClient

        from mlango.admin.app import build_admin_app
        from mlango.conf import settings

        settings.ADMIN_USERNAME = "denis"
        settings.ADMIN_PASSWORD = "s3cret"
        site, *_ = site_with_data

        with TestClient(build_admin_app(site)) as test_client:
            assert test_client.get("/", auth=("denis", "s3cret")).status_code == 200
            assert test_client.get("/", auth=("admin", "s3cret")).status_code == 401

    def test_a_malformed_header_is_rejected(self, project, site_with_data):
        from fastapi.testclient import TestClient

        from mlango.admin.app import build_admin_app
        from mlango.conf import settings

        settings.ADMIN_PASSWORD = "s3cret"
        site, *_ = site_with_data

        with TestClient(build_admin_app(site)) as test_client:
            for header in ("", "Basic", "Basic !!!notbase64", "Bearer token", "Basic "):
                response = test_client.get("/", headers={"Authorization": header})
                assert response.status_code == 401, header


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


class TestFormatting:
    def test_short_truncates(self):
        from mlango.admin.app import _short

        assert _short("abcdef", 4) == "abc…"
        assert _short("ab", 4) == "ab"
        assert _short(None) == ""

    def test_duration_scales(self):
        from mlango.admin.app import _duration

        assert _duration(None) == "—"
        assert _duration(0.05) == "50 ms"
        assert _duration(2.5) == "2.5 s"
        assert _duration(125) == "2m 5s"

    def test_number_formats(self):
        from mlango.admin.app import _number

        assert _number(None) == "—"
        assert _number(1234567) == "1,234,567"
        assert _number(1.5) == "1.5"
        assert _number(1.0) == "1"

    def test_the_sparkline_is_self_contained_svg(self):
        from mlango.admin.app import _sparkline

        assert _sparkline([(0, 1.0)]) == ""  # a single point is not a line

        svg = _sparkline([(0, 0.1), (1, 0.5), (2, 0.9)])
        assert svg.startswith("<svg")
        assert "polyline" in svg
        assert "http" not in svg  # nothing fetched from the network

    def test_the_sparkline_survives_a_flat_series(self):
        from mlango.admin.app import _sparkline

        # A constant metric would divide by a zero span if not guarded.
        assert "<svg" in _sparkline([(0, 1.0), (1, 1.0), (2, 1.0)])
