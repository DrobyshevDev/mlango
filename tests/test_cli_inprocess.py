"""The same commands, run in-process.

``test_cli.py`` drives ``manage.py`` in a subprocess, which is the honest way to
prove the CLI works as a user hits it — settings discovery, argv parsing, exit
codes. Coverage cannot see into those subprocesses, and a fast feedback loop
matters too, so the command bodies are also exercised here directly.

Both layers earn their keep: if only these existed, a broken ``manage.py`` or a
mis-wired settings module would go unnoticed.
"""

from __future__ import annotations

import json
import sys

import pytest

from mlango.management.manager import load_command

BUILTIN = "mlango.management.commands"


def run(name: str, *argv: str) -> int:
    """Run a built-in command in this process and return its exit code."""
    command = load_command(name, f"{BUILTIN}.{name}")
    return command.run_from_argv(list(argv))


@pytest.fixture(scope="module")
def live_project(tmp_path_factory):
    """A scaffolded project loaded into *this* interpreter.

    The registry and settings are process-global, so the fixture snapshots both,
    installs the project, and restores everything afterwards — otherwise the rest
    of the suite would run against a half-replaced registry.
    """
    import os

    from mlango.conf import ENVIRONMENT_VARIABLE, settings
    from mlango.core.registry import apps
    from mlango.metastore.session import dispose_all
    from mlango.storage import reset_default_storage
    from mlango.template import render_project

    root = tmp_path_factory.mktemp("inproc")
    project = root / "liveproject"
    render_project("liveproject", str(project), demo=True)

    # Snapshot the global state this fixture is about to replace.
    saved_objects = {kind: dict(entries) for kind, entries in apps._objects.items()}
    saved_configs = dict(apps.app_configs)
    saved_ready = apps.ready
    saved_env = os.environ.get(ENVIRONMENT_VARIABLE)
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)

    sys.path.insert(0, str(project))
    os.environ[ENVIRONMENT_VARIABLE] = "liveproject.settings"

    dispose_all()
    reset_default_storage()
    settings.reset()
    apps.clear()

    import mlango

    mlango.setup()

    yield project

    # Restore, so later modules see the registry they declared into.
    dispose_all()
    reset_default_storage()
    settings.reset()
    apps.clear()

    apps._objects.update(saved_objects)
    apps.app_configs.update(saved_configs)
    apps.ready = saved_ready

    for module in set(sys.modules) - saved_modules:
        if module.split(".")[0] in {"liveproject", "demo"}:
            sys.modules.pop(module, None)

    sys.path[:] = saved_path
    if saved_env is None:
        os.environ.pop(ENVIRONMENT_VARIABLE, None)
    else:
        os.environ[ENVIRONMENT_VARIABLE] = saved_env


@pytest.fixture(scope="module")
def live_migrated(live_project):
    run("migrate", "-v", "0")
    run("makemigrations", "-v", "0")
    run("migrate", "-v", "0")
    return live_project


@pytest.fixture(scope="module")
def live_trained(live_migrated):
    pytest.importorskip("sklearn")
    assert run("train", "demo.Sentiment", "-v", "0") == 0
    return live_migrated


class TestMigrate:
    def test_creates_the_metastore(self, live_project, capsys):
        assert run("migrate") == 0
        assert "metastore tables ready" in capsys.readouterr().out

    def test_makemigrations_then_migrate(self, live_project, capsys):
        assert run("makemigrations") == 0
        assert "0001_initial" in capsys.readouterr().out

        assert run("migrate") == 0
        assert "applied" in capsys.readouterr().out.lower()

    def test_showmigrations(self, live_migrated, capsys):
        assert run("showmigrations") == 0
        out = capsys.readouterr().out
        assert "demo" in out
        assert "0001_initial" in out

    def test_plan_applies_nothing(self, live_migrated, capsys):
        assert run("migrate", "--plan") == 0
        assert "No migrations to apply" in capsys.readouterr().out

    def test_an_unknown_app_is_reported(self, live_migrated, capsys):
        assert run("makemigrations", "nosuchapp") == 1
        assert "No installed app" in capsys.readouterr().err


class TestCheck:
    def test_reports_every_section(self, live_migrated, capsys):
        assert run("check") == 0
        out = capsys.readouterr().out
        for heading in ("Project", "Metastore", "Backends", "Wiring", "Migrations", "Admin"):
            assert heading in out

    def test_admin_auth_status_is_reported(self, live_migrated, capsys):
        assert run("check") == 0
        assert "auth" in capsys.readouterr().out

    def test_fail_level_warning(self, live_migrated, capsys):
        # The scaffold ships DEBUG = True, which is reported as a warning.
        assert run("check", "--fail-level", "warning") == 1
        assert "warning(s) found" in capsys.readouterr().err


class TestDataset:
    def test_list(self, live_migrated, capsys):
        assert run("dataset", "list") == 0
        assert "demo.Reviews" in capsys.readouterr().out

    def test_show(self, live_migrated, capsys):
        assert run("dataset", "show", "demo.Reviews") == 0
        out = capsys.readouterr().out
        assert "LabelField" in out
        assert "targets" in out

    def test_head(self, live_migrated, capsys):
        assert run("dataset", "head", "demo.Reviews", "-n", "2") == 0
        assert capsys.readouterr().out.count("\n") >= 4

    def test_validate(self, live_migrated, capsys):
        assert run("dataset", "validate", "demo.Reviews") == 0
        assert "validated against" in capsys.readouterr().out

    def test_materialize_and_versions(self, live_migrated, capsys):
        assert run("dataset", "materialize", "demo.Reviews", "--notes", "in-process") == 0
        assert "row(s)" in capsys.readouterr().out

        assert run("dataset", "versions", "demo.Reviews") == 0
        assert "v1" in capsys.readouterr().out

    def test_force_creates_another_version(self, live_migrated, capsys):
        run("dataset", "materialize", "demo.Reviews", "-v", "0")
        capsys.readouterr()
        assert run("dataset", "materialize", "demo.Reviews", "--force") == 0
        assert "v" in capsys.readouterr().out

    def test_a_missing_label_is_reported(self, live_migrated, capsys):
        assert run("dataset", "head") == 1
        assert "needs a dataset label" in capsys.readouterr().err


class TestTrain:
    def test_trains_and_reports(self, live_trained, capsys):
        assert run("train", "demo.Sentiment", "-p", "C=1.5", "--tag", "inproc") == 0
        out = capsys.readouterr().out
        assert "Registered" in out
        assert "accuracy" in out

    def test_notes_and_seed_are_accepted(self, live_trained, capsys):
        assert run("train", "demo.Sentiment", "--notes", "why", "--seed", "7", "-v", "0") == 0

    def test_materialize_first(self, live_trained, capsys):
        assert run("train", "demo.Sentiment", "--materialize", "-v", "0") == 0

    def test_an_unknown_model_is_reported(self, live_trained, capsys):
        assert run("train", "demo.Nope") == 1
        assert "Registered models" in capsys.readouterr().err

    def test_a_bad_parameter_is_reported(self, live_trained, capsys):
        assert run("train", "demo.Sentiment", "-p", "C=notanumber") == 1
        assert "--param C" in capsys.readouterr().err


class TestSweep:
    def test_grid(self, live_trained, capsys):
        assert (
            run(
                "sweep",
                "demo.Sentiment",
                "-p",
                "C=0.5,2.0",
                "--metric",
                "accuracy",
                "--mode",
                "max",
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "trial 1" in out
        assert "Best:" in out

    def test_default_space_from_tunable_fields(self, live_trained, capsys):
        assert run("sweep", "demo.Sentiment", "--trials", "2", "-v", "0") == 0

    def test_random_without_trials_is_reported(self, live_trained, capsys):
        assert run("sweep", "demo.Sentiment", "-p", "C=1,2", "--strategy", "random") == 1

    def test_an_empty_value_list_is_reported(self, live_trained, capsys):
        assert run("sweep", "demo.Sentiment", "-p", "C=") == 1
        assert "lists no values" in capsys.readouterr().err


class TestEvaluate:
    def test_runs(self, live_trained, capsys):
        assert run("evaluate", "demo.SentimentAccuracy") == 0
        out = capsys.readouterr().out
        assert "pass_rate" in out
        assert "cases passed" in out

    def test_show_failures(self, live_trained, capsys):
        assert run("evaluate", "demo.SentimentAccuracy", "--show-failures", "-v", "2") == 0

    def test_min_pass_rate_gate(self, live_trained, capsys):
        assert run("evaluate", "demo.SentimentAccuracy", "--min-pass-rate", "1.01") == 1
        assert "below the required" in capsys.readouterr().err

    def test_an_unknown_eval_is_reported(self, live_trained, capsys):
        assert run("evaluate", "demo.Nope") == 1


class TestAgent:
    def test_one_message(self, live_migrated, capsys):
        assert run("agent", "demo.Helper", "hello", "there") == 0
        out = capsys.readouterr().out
        assert "echo:" in out
        assert "trace" in out

    def test_show_steps(self, live_trained, capsys):
        assert (
            run(
                "agent",
                "demo.Helper",
                'use classify_review {"text": "great movie 2"}',
                "--show-steps",
            )
            == 0
        )
        captured = capsys.readouterr()
        # Step output goes to stderr so it does not pollute piped answers.
        assert "classify_review" in captured.out + captured.err

    def test_max_steps_override(self, live_migrated, capsys):
        assert run("agent", "demo.Helper", "hello", "--max-steps", "2", "-v", "0") == 0

    def test_session_is_accepted(self, live_migrated, capsys):
        assert run("agent", "demo.Helper", "hello", "--session", "s1", "-v", "0") == 0


class TestRuns:
    def test_list(self, live_trained, capsys):
        assert run("runs", "list") == 0
        assert "demo.Sentiment" in capsys.readouterr().out

    def test_filter_by_kind_and_status(self, live_trained, capsys):
        assert run("runs", "list", "--kind", "train", "--status", "finished") == 0

    def test_filter_by_target(self, live_trained, capsys):
        assert run("runs", "list", "--target", "demo.Sentiment") == 0
        assert "demo.Sentiment" in capsys.readouterr().out

    def test_show(self, live_trained, capsys):
        from mlango.training import recent_runs

        run_id = recent_runs(limit=1)[0].uuid
        assert run("runs", "show", run_id) == 0
        out = capsys.readouterr().out
        assert "Parameters" in out
        assert "Metrics" in out

    def test_compare(self, live_trained, capsys):
        from mlango.training import recent_runs

        runs = recent_runs(limit=2)
        assert run("runs", "compare", runs[0].uuid, runs[1].uuid) == 0
        assert "target" in capsys.readouterr().out

    def test_compare_needs_two(self, live_trained, capsys):
        assert run("runs", "compare", "abc") == 1

    def test_an_unknown_run(self, live_trained, capsys):
        assert run("runs", "show", "ffffffffff") == 1


class TestTraces:
    def test_list(self, live_migrated, capsys):
        run("agent", "demo.Helper", "hello", "-v", "0")
        capsys.readouterr()

        assert run("traces", "list") == 0
        assert "Helper" in capsys.readouterr().out

    def test_filter_by_agent(self, live_migrated, capsys):
        run("agent", "demo.Helper", "hello", "-v", "0")
        capsys.readouterr()

        assert run("traces", "list", "--agent", "demo.Helper") == 0

    def test_show(self, live_migrated, capsys):
        from mlango.agents.tracing import recent_traces

        run("agent", "demo.Helper", "hello", "-v", "0")
        capsys.readouterr()

        trace = recent_traces(limit=1)[0]
        assert run("traces", "show", trace.uuid, "-v", "2") == 0
        out = capsys.readouterr().out
        assert "Input" in out
        assert "Steps" in out

    def test_an_unknown_trace(self, live_migrated, capsys):
        assert run("traces", "show", "ffffffffff") == 1


class TestShell:
    def test_runs_code(self, live_migrated, capsys):
        assert run("shell", "-c", "print('rows', Reviews.objects.count())") == 0
        assert "rows 400" in capsys.readouterr().out

    def test_helpers_are_present(self, live_trained, capsys):
        assert run("shell", "-c", "print('n', len(recent_runs()))") == 0
        assert "n " in capsys.readouterr().out

    def test_the_banner_lists_declarations(self, live_migrated):
        command = load_command("shell", f"{BUILTIN}.shell")
        banner = command._banner()
        assert "datasets: Reviews" in banner
        assert "models: Sentiment" in banner
        assert "helpers:" in banner

    def test_the_namespace_holds_every_declaration(self, live_migrated):
        command = load_command("shell", f"{BUILTIN}.shell")
        namespace = command._namespace()
        assert {"Reviews", "Sentiment", "Helper", "SentimentAccuracy"} <= set(namespace)
        assert {"apps", "settings", "recent_runs", "get_trace"} <= set(namespace)


class TestStartApp:
    def test_creates_an_app(self, live_project, capsys):
        assert run("startapp", "inproc_app") == 0
        assert (live_project / "inproc_app" / "datasets.py").exists()
        assert "Next steps" in capsys.readouterr().out

    def test_a_reserved_name_is_refused(self, live_project, capsys):
        assert run("startapp", "admin") == 1
        assert "shadow an existing module" in capsys.readouterr().err

    def test_a_non_empty_target_is_refused(self, live_project, capsys):
        assert run("startapp", "demo") == 1
        assert "already exists and is not empty" in capsys.readouterr().err


class TestScaffoldedProjectCanRunItsOwnTests:
    """A project that ships a test file it cannot run teaches the wrong lesson.

    `startproject` writes `tests/test_demo.py` and `manage.py test` runs it, but
    that needs pytest, which is deliberately absent from requirements.txt
    because the generated Dockerfile installs that file into the production
    image. Following the printed steps exactly used to end at "pytest is not
    installed".
    """

    def rendered(self, tmp_path):
        from mlango.template import render_project

        target = tmp_path / "scaffolded"
        render_project("scaffolded", str(target), demo=True)
        return target

    def test_the_dev_requirements_bring_in_a_test_runner(self, tmp_path):
        dev = (self.rendered(tmp_path) / "requirements-dev.txt").read_text(encoding="utf-8")
        assert "pytest" in dev

    def test_the_dev_requirements_include_the_production_ones(self, tmp_path):
        # Otherwise installing them alone gives you a runner and no framework.
        dev = (self.rendered(tmp_path) / "requirements-dev.txt").read_text(encoding="utf-8")
        assert "-r requirements.txt" in dev

    def test_the_production_requirements_stay_free_of_a_test_runner(self, tmp_path):
        # This file is what the generated Dockerfile installs. pytest does not ship.
        prod = (self.rendered(tmp_path) / "requirements.txt").read_text(encoding="utf-8")
        assert "pytest" not in prod

    def test_a_test_file_is_actually_scaffolded(self, tmp_path):
        # If this ever stops being true, the rest of this class is pointless.
        assert (self.rendered(tmp_path) / "tests" / "test_demo.py").exists()

    def test_the_scaffold_says_how_to_run_the_tests_it_wrote(self, tmp_path, capsys):
        assert run("startproject", "spelled_out", str(tmp_path / "spelled_out")) == 0
        out = capsys.readouterr().out
        assert "requirements-dev.txt" in out
        assert "manage.py test" in out


class TestInspectData:
    @pytest.fixture(scope="class")
    def csv_file(self, live_project):
        path = live_project / "incoming.csv"
        lines = ["id,body,stars,country,verified,label"]
        for index in range(30):
            positive = index % 2 == 0
            lines.append(
                f"{index + 1},"
                f'"a review of some length written out here, number {index}",'
                f"{(index % 5) + 1},{'GB' if positive else 'US'},"
                f"{'true' if positive else 'false'},{'pos' if positive else 'neg'}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_it_prints_a_declaration(self, csv_file, capsys):
        assert run("inspectdata", "incoming.csv") == 0

        out = capsys.readouterr().out
        assert "class Incoming(Dataset):" in out
        assert 'label = LabelField(["neg", "pos"])' in out
        assert "verified = BooleanField()" in out

    def test_the_summary_names_the_key_and_target(self, csv_file, capsys):
        run("inspectdata", "incoming.csv")
        out = capsys.readouterr().out
        assert "primary_key  id" in out
        assert "target       label" in out

    def test_a_jsonl_file(self, live_project, capsys):
        path = live_project / "rows.jsonl"
        path.write_text(
            "\n".join(
                f'{{"id": {i}, "note": "some text {i}", "tier": "{"a" if i % 2 else "b"}"}}'
                for i in range(10)
            ),
            encoding="utf-8",
        )
        assert run("inspectdata", "rows.jsonl") == 0
        assert "JSONLSource" in capsys.readouterr().out

    def test_a_custom_name(self, csv_file, capsys):
        assert run("inspectdata", "incoming.csv", "--name", "Feedback") == 0
        assert "class Feedback(Dataset):" in capsys.readouterr().out

    def test_a_name_that_is_not_an_identifier(self, csv_file, capsys):
        assert run("inspectdata", "incoming.csv", "--name", "not a class") == 1
        assert "not a valid class name" in capsys.readouterr().err

    def test_the_sample_size(self, csv_file, capsys):
        assert run("inspectdata", "incoming.csv", "-n", "5") == 0
        out = capsys.readouterr().out
        assert "Read 5 rows." in out
        assert "min_value=1, max_value=5" in out

    def test_a_sample_of_zero_is_refused(self, csv_file, capsys):
        assert run("inspectdata", "incoming.csv", "-n", "0") == 1
        assert "at least 1" in capsys.readouterr().err

    def test_a_missing_file(self, live_project, capsys):
        assert run("inspectdata", "nope.csv") == 1
        assert "No such file" in capsys.readouterr().err

    def test_an_unsupported_extension(self, live_project, capsys):
        (live_project / "sheet.xlsx").write_text("nope", encoding="utf-8")
        assert run("inspectdata", "sheet.xlsx") == 1
        assert "Recognised extensions" in capsys.readouterr().err

    def test_an_empty_file(self, live_project, capsys):
        (live_project / "empty.jsonl").write_text("", encoding="utf-8")
        assert run("inspectdata", "empty.jsonl") == 1
        assert "no records" in capsys.readouterr().err

    def test_awkward_column_names_are_warned_about(self, live_project, capsys):
        path = live_project / "awkward.csv"
        path.write_text("Review Text,label\nhello there,a\nhi again,b\n", encoding="utf-8")

        assert run("inspectdata", "awkward.csv") == 0
        out = capsys.readouterr().out
        assert "review_text" in out
        assert "not a valid Python name" in out

    def test_write_needs_an_app(self, csv_file, capsys):
        assert run("inspectdata", "incoming.csv", "--write") == 1
        assert "--write needs --app" in capsys.readouterr().err

    def test_write_rejects_an_app_that_does_not_exist(self, csv_file, capsys):
        assert run("inspectdata", "incoming.csv", "--write", "--app", "ghost") == 1
        assert "No such app directory" in capsys.readouterr().err

    def test_write_refuses_to_clobber(self, csv_file, capsys):
        assert run("inspectdata", "incoming.csv", "--write", "--app", "demo") == 1
        err = capsys.readouterr().err
        assert "already declares something" in err
        assert "--force" in err

    def test_write_with_force(self, csv_file, live_project, capsys):
        run("startapp", "written_app", "-v", "0")
        capsys.readouterr()

        assert run("inspectdata", "incoming.csv", "--write", "--app", "written_app") == 0
        written = (live_project / "written_app" / "datasets.py").read_text(encoding="utf-8")
        assert "class Incoming(Dataset):" in written
        assert "Wrote" in capsys.readouterr().out

    def test_the_generated_file_is_importable(self, csv_file, live_project):
        """The point of the command is a file you can actually use."""
        import ast

        from mlango.data.inspect import profile_source, render_declaration, source_for

        source, expression = source_for("incoming.csv")
        declaration = render_declaration(
            profile_source(source), name="Incoming", source_expr=expression
        )
        tree = ast.parse(declaration)
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert classes == ["Incoming", "Meta"]


class TestPredict:
    def test_a_literal_input(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment", "an absolute delight") == 0
        out = capsys.readouterr().out
        assert "prediction" in out
        assert "1 prediction(s)." in out

    def test_probabilities(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment", "wonderful", "--proba") == 0
        assert "probabilities" in capsys.readouterr().out

    def test_the_declared_dataset(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment", "--dataset", "-n", "5") == 0
        assert "5 prediction(s)." in capsys.readouterr().out

    def test_a_filter(self, live_trained, capsys):
        assert (
            run("predict", "demo.Sentiment", "--dataset", "--filter", "label=pos", "-n", "3") == 0
        )
        assert "3 prediction(s)." in capsys.readouterr().out

    def test_a_numeric_filter_is_coerced(self, live_trained, capsys):
        """`--filter stars=5` has to compare as a number, not a string."""
        assert run("predict", "demo.Sentiment", "--dataset", "--filter", "id=1") == 0
        assert "1 prediction(s)." in capsys.readouterr().out

    def test_a_filter_matching_nothing(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment", "--dataset", "--filter", "label=zzz") == 1
        err = capsys.readouterr().err
        assert "Filters applied: label=zzz" in err
        assert "dataset head" in err

    def test_a_malformed_filter(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment", "--dataset", "--filter", "label") == 1
        assert "FIELD=VALUE" in capsys.readouterr().err

    def test_an_unknown_filter_field(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment", "--dataset", "--filter", "nope=1") == 1
        assert "has no field" in capsys.readouterr().err

    def test_a_file(self, live_trained, live_project, capsys):
        path = live_project / "score.jsonl"
        path.write_text(
            '{"id": 1, "text": "wonderful and warm"}\n{"id": 2, "text": "dull and dreadful"}\n',
            encoding="utf-8",
        )
        assert run("predict", "demo.Sentiment", "--file", "score.jsonl") == 0
        assert "2 prediction(s)." in capsys.readouterr().out

    def test_a_file_limited(self, live_trained, live_project, capsys):
        assert run("predict", "demo.Sentiment", "--file", "score.jsonl", "-n", "1") == 0
        assert "1 prediction(s)." in capsys.readouterr().out

    def test_a_file_missing_the_features(self, live_trained, live_project, capsys):
        path = live_project / "wrong.jsonl"
        path.write_text('{"id": 1, "body": "wonderful"}\n', encoding="utf-8")

        assert run("predict", "demo.Sentiment", "--file", "wrong.jsonl") == 1
        err = capsys.readouterr().err
        assert "needs text" in err
        assert "Columns found: body, id" in err

    def test_an_empty_file(self, live_trained, live_project, capsys):
        (live_project / "nothing.jsonl").write_text("", encoding="utf-8")
        assert run("predict", "demo.Sentiment", "--file", "nothing.jsonl") == 1
        assert "contained no records" in capsys.readouterr().err

    def test_a_missing_file(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment", "--file", "ghost.jsonl") == 1
        assert "No such file" in capsys.readouterr().err

    def test_an_unreadable_extension(self, live_trained, live_project, capsys):
        (live_project / "data.xlsx").write_text("nope", encoding="utf-8")
        assert run("predict", "demo.Sentiment", "--file", "data.xlsx") == 1
        assert "Recognised extensions" in capsys.readouterr().err

    def test_jsonl_output(self, live_trained, capsys):
        import json

        assert run("predict", "demo.Sentiment", "--dataset", "-n", "3", "--format", "jsonl") == 0
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
        rows = [json.loads(ln) for ln in lines]
        assert len(rows) == 3
        assert {"id", "input", "prediction"} <= set(rows[0])

    def test_csv_output(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment", "--dataset", "-n", "2", "--format", "csv") == 0
        assert "id,input,prediction" in capsys.readouterr().out

    def test_writing_to_a_file(self, live_trained, live_project, capsys):
        assert (
            run(
                "predict",
                "demo.Sentiment",
                "--dataset",
                "-n",
                "4",
                "--format",
                "jsonl",
                "--output",
                "out.jsonl",
            )
            == 0
        )
        assert (
            len((live_project / "out.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 4
        )
        assert "Wrote 4 prediction" in capsys.readouterr().out

    def test_a_file_target_writes_data_even_in_table_mode(self, live_trained, live_project):
        """A table is for a terminal; a file wants something parseable."""
        import json

        assert run("predict", "demo.Sentiment", "--dataset", "-n", "2", "--output", "t.jsonl") == 0
        lines = (live_project / "t.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(lines[0])["prediction"]

    def test_csv_to_a_file(self, live_trained, live_project):
        assert (
            run(
                "predict",
                "demo.Sentiment",
                "--dataset",
                "-n",
                "2",
                "--format",
                "csv",
                "--output",
                "out.csv",
            )
            == 0
        )
        body = (live_project / "out.csv").read_text(encoding="utf-8")
        assert body.splitlines()[0].startswith("id,input,prediction")

    def test_an_explicit_version(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment", "wonderful", "--version", "1") == 0
        assert "demo.Sentiment@v1" in capsys.readouterr().out

    def test_a_version_that_does_not_exist(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment", "wonderful", "--version", "999") == 1
        assert capsys.readouterr().err

    def test_an_unknown_model(self, live_trained, capsys):
        assert run("predict", "demo.Nope", "hello") == 1
        assert "No model named" in capsys.readouterr().err

    def test_no_input(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment") == 1
        assert "--dataset" in capsys.readouterr().err

    def test_literals_and_dataset_together(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment", "hello", "--dataset") == 1
        assert "not both" in capsys.readouterr().err

    def test_file_and_dataset_together(self, live_trained, capsys):
        assert run("predict", "demo.Sentiment", "--file", "score.jsonl", "--dataset") == 1
        assert "not both" in capsys.readouterr().err


class TestExplain:
    def test_the_default_is_the_newest_version(self, live_trained, capsys):
        assert run("explain", "demo.Sentiment") == 0
        out = capsys.readouterr().out
        assert "demo.Sentiment@v" in out
        assert "largest weight first" in out
        assert "█" in out, "the chart should draw bars"

    def test_it_names_words_from_the_vectoriser(self, live_trained, capsys):
        assert run("explain", "demo.Sentiment", "--json") == 0
        weights = json.loads(capsys.readouterr().out)
        assert weights
        assert all(isinstance(value, float) for value in weights.values())
        assert not any(name.startswith("feature_") for name in weights)

    def test_top_limits_the_rows(self, live_trained, capsys):
        assert run("explain", "demo.Sentiment", "--json", "-n", "3") == 0
        assert len(json.loads(capsys.readouterr().out)) == 3

    def test_it_reads_the_metastore_rather_than_the_artifact(self, live_trained, capsys):
        """No load means explaining stays cheap and works without the weights."""
        from mlango.training.trainer import get_trainer

        trainer = get_trainer("sklearn")

        def refuse(model, path):
            raise AssertionError("explain must not load the artifact by default")

        original, trainer.load = trainer.load, refuse
        try:
            assert run("explain", "demo.Sentiment") == 0
        finally:
            trainer.load = original

    def test_recompute_falls_back_to_the_artifact(self, live_trained, capsys):
        """A version registered before the column existed can still be filled in."""
        import sqlalchemy as sa

        from mlango.metastore.models import ModelVersion
        from mlango.metastore.session import session_scope

        with session_scope() as session:
            row = session.execute(
                sa.select(ModelVersion)
                .where(ModelVersion.label == "demo.Sentiment")
                .order_by(ModelVersion.version.desc())
                .limit(1)
            ).scalar_one()
            row.importances = None
            version = row.version

        assert run("explain", "demo.Sentiment") == 0
        assert "demo.Sentiment@v" in capsys.readouterr().out

        with session_scope() as session:
            restored = session.execute(
                sa.select(ModelVersion).where(
                    ModelVersion.label == "demo.Sentiment", ModelVersion.version == version
                )
            ).scalar_one()
            assert restored.importances, "the recomputed weights should be stored"

    def test_an_unknown_version_says_which_one(self, live_trained, capsys):
        assert run("explain", "demo.Sentiment", "--version", "999") == 1
        assert "v999" in capsys.readouterr().err

    def test_a_stage_with_no_version_points_at_train(self, live_trained, capsys):
        assert run("explain", "demo.Sentiment", "--stage", "nowhere") == 1
        err = capsys.readouterr().err
        assert "stage 'nowhere'" in err
        assert "manage.py train demo.Sentiment" in err


class TestDrift:
    @pytest.fixture
    def with_traffic(self, live_trained):
        """Serve some obviously different input, so there is drift to find."""
        from mlango.conf import settings
        from mlango.core.registry import apps

        before = settings.PREDICTION_LOG
        settings.PREDICTION_LOG = {"ENABLED": True, "SAMPLE": 1.0, "MAX_ROWS": 0}
        model = apps.get_model("demo.Sentiment").load()
        model.predict(["ok"] * 40)
        yield live_trained
        settings.PREDICTION_LOG = before

    def test_it_reports_a_verdict_per_column(self, with_traffic, capsys):
        assert run("drift", "demo.Sentiment") == 0
        out = capsys.readouterr().out
        assert "logged predictions over the last 7d" in out
        assert "PSI" in out and "Verdict" in out
        assert "significant" in out

    def test_the_predicted_label_is_compared_too(self, with_traffic, capsys):
        """Prediction drift is the half that needs no ground truth."""
        assert run("drift", "demo.Sentiment", "--json") == 0
        scores = json.loads(capsys.readouterr().out)
        assert "label (predicted)" in scores

    def test_the_training_data_itself_is_stable(self, with_traffic, capsys):
        assert run("drift", "demo.Sentiment", "--against", "demo.Reviews", "--json") == 0
        scores = json.loads(capsys.readouterr().out)
        assert scores["text"]["verdict"] == "stable"

    def test_fail_on_turns_drift_into_an_exit_code(self, with_traffic, capsys):
        assert run("drift", "demo.Sentiment", "--fail-on", "significant") == 1
        assert "Drift at or above significant" in capsys.readouterr().err

    def test_fail_on_stays_quiet_when_nothing_moved(self, with_traffic, capsys):
        assert (
            run("drift", "demo.Sentiment", "--against", "demo.Reviews", "--fail-on", "significant")
            == 0
        )

    def test_a_bad_window_says_what_it_wanted(self, with_traffic, capsys):
        assert run("drift", "demo.Sentiment", "--since", "soon") == 1
        assert "24h, 7d or 4w" in capsys.readouterr().err

    def test_an_empty_window_points_at_the_setting(self, with_traffic, capsys):
        assert run("drift", "demo.Sentiment", "--since", "1h", "-n", "0") == 1
        assert "PREDICTION_LOG" in capsys.readouterr().err

    def test_a_version_without_a_baseline_says_so(self, with_traffic, capsys):
        import sqlalchemy as sa

        from mlango.metastore.models import ModelVersion
        from mlango.metastore.session import session_scope

        with session_scope() as session:
            row = session.execute(
                sa.select(ModelVersion)
                .where(ModelVersion.label == "demo.Sentiment")
                .order_by(ModelVersion.version.desc())
                .limit(1)
            ).scalar_one()
            saved, row.baseline = row.baseline, None
            version = row.id

        try:
            assert run("drift", "demo.Sentiment") == 1
            assert "no training profile" in capsys.readouterr().err
        finally:
            with session_scope() as session:
                session.execute(
                    sa.update(ModelVersion).where(ModelVersion.id == version).values(baseline=saved)
                )


class TestDiff:
    @pytest.fixture(scope="module")
    def two_versions(self, live_trained):
        """A deliberately weak fit next to the good one already trained.

        The version numbers are read back rather than assumed: this module
        shares one metastore, so how many versions exist by now depends on
        which other tests have run.
        """
        good = _newest_version("demo.Sentiment")
        assert (
            run("train", "demo.Sentiment", "-p", "max_features=1", "-p", "C=0.01", "-v", "0") == 0
        )
        return good, _newest_version("demo.Sentiment")

    def test_it_reports_what_moved_and_what_broke(self, two_versions, capsys):
        good, weak = two_versions
        assert run("diff", "demo.Sentiment", str(good), str(weak), "-n", "120") == 0
        out = capsys.readouterr().out
        assert f"demo.Sentiment v{good} → v{weak}" in out
        assert "agreement" in out
        assert "fixed" in out and "broke" in out

    def test_json_carries_the_whole_report(self, two_versions, capsys):
        good, weak = two_versions
        assert run("diff", "demo.Sentiment", str(good), str(weak), "-n", "80", "--json") == 0
        report = json.loads(capsys.readouterr().out)
        assert report["rows"] == 80
        assert set(report) >= {"agreement", "changed", "transitions", "fixed", "broke", "metrics"}

    def test_show_changes_prints_the_rows(self, two_versions, capsys):
        good, weak = two_versions
        assert (
            run("diff", "demo.Sentiment", str(good), str(weak), "-n", "120", "--show-changes", "3")
            == 0
        )
        out = capsys.readouterr().out
        assert "Rows where they disagree" in out
        assert "expected" in out

    def test_fail_on_regression_is_an_exit_code(self, two_versions, capsys):
        """Good fit to weak fit is exactly what this is meant to catch."""
        good, weak = two_versions
        assert (
            run("diff", "demo.Sentiment", str(good), str(weak), "-n", "120", "--fail-on-regression")
            == 1
        )
        assert "got right" in capsys.readouterr().err

    def test_no_regression_passes(self, two_versions, capsys):
        good, weak = two_versions
        assert (
            run("diff", "demo.Sentiment", str(weak), str(good), "-n", "120", "--fail-on-regression")
            == 0
        )
        assert "No regression" in capsys.readouterr().out

    def test_the_bare_flag_still_means_what_it_meant(self, two_versions, capsys):
        # It grew an optional mode. The old spelling must not have changed.
        good, weak = two_versions
        assert (
            run("diff", "demo.Sentiment", str(good), str(weak), "-n", "120", "--fail-on-regression")
            == 1
        )
        assert "got right" in capsys.readouterr().err

    def test_significant_mode_blocks_a_real_regression(self, two_versions, capsys):
        good, weak = two_versions
        assert (
            run(
                "diff",
                "demo.Sentiment",
                str(good),
                str(weak),
                "-n",
                "120",
                "--fail-on-regression",
                "significant",
            )
            == 1
        )
        assert "more than chance" in capsys.readouterr().err

    def test_significant_mode_lets_an_improvement_through(self, two_versions, capsys):
        good, weak = two_versions
        assert (
            run(
                "diff",
                "demo.Sentiment",
                str(weak),
                str(good),
                "-n",
                "120",
                "--fail-on-regression",
                "significant",
            )
            == 0
        )

    def test_the_report_says_whether_the_balance_is_a_change_or_a_coin(self, two_versions, capsys):
        good, weak = two_versions
        assert run("diff", "demo.Sentiment", str(good), str(weak), "-n", "120") == 0
        assert "verdict" in capsys.readouterr().out

    def test_json_carries_the_significance(self, two_versions, capsys):
        good, weak = two_versions
        assert run("diff", "demo.Sentiment", str(good), str(weak), "-n", "80", "--json") == 0
        stats = json.loads(capsys.readouterr().out)["significance"]
        assert set(stats) == {"discordant", "p_value", "direction", "verdict"}
        assert 0.0 <= stats["p_value"] <= 1.0

    def test_one_version_number_is_refused(self, two_versions, capsys):
        assert run("diff", "demo.Sentiment", "1") == 1
        assert "two version numbers" in capsys.readouterr().err

    def test_with_no_versions_and_nothing_promoted_it_says_so(self, two_versions, capsys):
        assert run("diff", "demo.Sentiment") == 1
        err = capsys.readouterr().err
        assert "production" in err
        assert "manage.py diff demo.Sentiment" in err

    def test_with_a_promoted_version_it_picks_the_pair(self, two_versions, capsys):
        """No arguments answers the question people actually have."""
        from mlango.core.registry import apps
        from mlango.metastore.models import Stage

        good, weak = two_versions
        model_class = apps.get_model("demo.Sentiment")
        model_class.promote(good, Stage.PRODUCTION)
        try:
            assert run("diff", "demo.Sentiment", "-n", "60") == 0
            assert f"v{good} → v{weak}" in capsys.readouterr().out
        finally:
            model_class.promote(good, Stage.ARCHIVED)

    def test_an_unknown_version_is_reported(self, two_versions, capsys):
        assert run("diff", "demo.Sentiment", "1", "999") == 1
        assert "999" in capsys.readouterr().err


class TestDiffOnArtefactsMlangoDidNotTrain:
    """The door: two saved models and a declared dataset, no Model class.

    The comparison is the same comparison — only where the two sides came from
    differs — so this proves the seam, not the statistics.
    """

    @pytest.fixture(scope="module")
    def two_artefacts(self, live_trained, tmp_path_factory):
        import pickle

        directory = tmp_path_factory.mktemp("artefacts")

        class Constant:
            def __init__(self, answer):
                self.answer = answer

            def predict(self, inputs):
                return [self.answer for _ in inputs]

        # Module-level so pickle can find it again on load.
        globals()["_ConstantPredictor"] = Constant
        Constant.__module__ = __name__
        Constant.__qualname__ = "_ConstantPredictor"

        paths = []
        for answer in ("positive", "negative"):
            path = directory / f"{answer}.pkl"
            with path.open("wb") as handle:
                pickle.dump(Constant(answer), handle)
            paths.append(str(path))
        return paths

    def test_it_compares_two_files(self, two_artefacts, capsys):
        left, right = two_artefacts
        assert (
            run("diff", "--dataset", "demo.Reviews", "--left", left, "--right", right, "-n", "40")
            == 0
        )
        out = capsys.readouterr().out
        # Two constant predictors that never agree: nothing matches, everything moved.
        assert "agreement      0.0%" in out
        assert "demo.Reviews" in out
        assert left in out and right in out

    def test_a_uri_is_never_printed_as_a_version_number(self, two_artefacts, capsys):
        # Found by running it rather than by testing it: the report prefixed
        # every side with `v`, which reads as `vmodel.pkl`. Registered versions
        # still read as `v3`; a path reads as itself, everywhere it appears.
        left, right = two_artefacts
        assert run(
            "diff",
            "--dataset",
            "demo.Reviews",
            "--left",
            left,
            "--right",
            right,
            "-n",
            "40",
            "--fail-on-regression",
            "significant",
        ) in (0, 1)
        printed = capsys.readouterr()
        assert "v" + left not in printed.out + printed.err
        assert "v" + right not in printed.out + printed.err

    def test_the_report_carries_the_same_verdict(self, two_artefacts, capsys):
        left, right = two_artefacts
        assert (
            run(
                "diff",
                "--dataset",
                "demo.Reviews",
                "--left",
                left,
                "--right",
                right,
                "-n",
                "40",
                "--json",
            )
            == 0
        )
        report = json.loads(capsys.readouterr().out)
        assert report["labelled"] is True
        assert set(report["significance"]) == {"discordant", "p_value", "direction", "verdict"}

    def test_the_gate_works_on_artefacts_too(self, two_artefacts, capsys):
        left, right = two_artefacts
        code = run(
            "diff",
            "--dataset",
            "demo.Reviews",
            "--left",
            left,
            "--right",
            right,
            "-n",
            "40",
            "--fail-on-regression",
            "significant",
        )
        # Which side wins depends on the label balance; either way the gate ran
        # and said something about it rather than crashing.
        assert code in (0, 1)

    def test_one_side_alone_is_refused(self, live_trained, capsys):
        assert run("diff", "--left", "a.pkl") == 1
        assert "go together" in capsys.readouterr().err

    def test_without_a_dataset_it_says_where_to_get_one(self, live_trained, capsys):
        assert run("diff", "--left", "a.pkl", "--right", "b.pkl") == 1
        err = capsys.readouterr().err
        assert "--dataset" in err
        assert "inspectdata" in err

    def test_no_model_and_no_artefacts_explains_both_ways_in(self, live_trained, capsys):
        assert run("diff") == 1
        err = capsys.readouterr().err
        assert "Name a model" in err
        assert "--left" in err


def _newest_version(label: str) -> int:
    import sqlalchemy as sa

    from mlango.metastore.models import ModelVersion
    from mlango.metastore.session import session_scope

    with session_scope() as session:
        return int(
            session.execute(
                sa.select(sa.func.max(ModelVersion.version)).where(ModelVersion.label == label)
            ).scalar_one()
        )


class TestDiffEval:
    """``diff --eval`` — the same question, for a suite that has no version number."""

    @pytest.fixture
    def two_eval_runs(self, live_trained):
        """Two finished runs of the shipped suite, so there is a pair to diff."""
        assert run("evaluate", "demo.SentimentAccuracy", "-v", "0") == 0
        assert run("evaluate", "demo.SentimentAccuracy", "-v", "0") == 0
        return live_trained

    def test_it_compares_the_two_most_recent_runs(self, two_eval_runs, capsys):
        assert run("diff", "--eval", "demo.SentimentAccuracy") == 0
        out = capsys.readouterr().out
        assert "demo.SentimentAccuracy" in out
        assert "pass rate" in out
        assert "fixed" in out and "broke" in out

    def test_json_carries_the_whole_report(self, two_eval_runs, capsys):
        assert run("diff", "--eval", "demo.SentimentAccuracy", "--json") == 0
        report = json.loads(capsys.readouterr().out)
        assert report["kind"] == "eval"
        assert set(report) >= {
            "cases",
            "fixed",
            "broke",
            "pass_rate",
            "agreement",
            "significance",
            "only_left",
            "only_right",
        }

    def test_named_runs_are_used_when_given(self, two_eval_runs, capsys):
        from mlango.evals.comparison import recent_runs

        newer, older = recent_runs("demo.SentimentAccuracy", limit=2)
        assert (
            run("diff", "--eval", "demo.SentimentAccuracy", "--runs", older.uuid, newer.uuid) == 0
        )
        out = capsys.readouterr().out
        assert f"{older.short_id} \u2192 {newer.short_id}" in out

    def test_a_run_of_another_suite_is_refused(self, two_eval_runs, capsys):
        """Joining two different suites by case id would compare nothing real."""
        from mlango.evals.comparison import recent_runs
        from mlango.training.run import recent_runs as recent_any

        newer, older = recent_runs("demo.SentimentAccuracy", limit=2)
        training = [r for r in recent_any(limit=25) if r.kind == "train"][0]

        assert (
            run("diff", "--eval", "demo.SentimentAccuracy", "--runs", training.uuid, newer.uuid)
            == 1
        )
        assert "not" in capsys.readouterr().err

    def test_an_unknown_suite_names_what_is_registered(self, live_trained, capsys):
        assert run("diff", "--eval", "demo.Nope") == 1
        err = capsys.readouterr().err
        assert "demo.Nope" in err
        assert "demo.SentimentAccuracy" in err, "the error should list the real ones"

    def test_it_does_not_mix_with_the_artefact_mode(self, two_eval_runs, capsys):
        assert run("diff", "--eval", "demo.SentimentAccuracy", "--left", "a.joblib") == 1
        assert "--left" in capsys.readouterr().err

    def test_fail_on_regression_passes_when_nothing_moved(self, two_eval_runs, capsys):
        """The shipped model is deterministic, so two runs agree case for case."""
        assert run("diff", "--eval", "demo.SentimentAccuracy", "--fail-on-regression") == 0
        assert "No regression" in capsys.readouterr().out

    def test_the_gate_speaks_in_cases_not_rows(self, two_eval_runs, capsys):
        assert (
            run("diff", "--eval", "demo.SentimentAccuracy", "--fail-on-regression", "significant")
            == 0
        )
        out = capsys.readouterr().out
        assert "row(s)" not in out


class TestTestCommand:
    def test_a_scaffolded_project_is_green_before_it_is_edited(self, live_project, capsys):
        """startproject ships tests, and they must pass on a fresh checkout."""
        assert run("test") == 0

        out = capsys.readouterr().out
        assert "passed" in out
        assert "Tests passed." in out

    def test_no_tests_is_reported(self, live_project, capsys):
        """With the directory gone, the message has to say where to put one."""
        import shutil

        shipped = live_project / "tests"
        moved = live_project / "tests-moved"
        shutil.move(str(shipped), str(moved))
        try:
            assert run("test") == 1
            assert "No tests found" in capsys.readouterr().err
        finally:
            shutil.move(str(moved), str(shipped))

    def test_a_keyword_selects_a_subset(self, live_project, capsys):
        assert run("test", "-k", "dataset_loads") == 0
        assert "1 passed" in capsys.readouterr().out

    def test_an_installed_app_s_own_tests_are_collected(self, live_project, capsys):
        """startapp writes <app>/tests.py, so the default run has to find it.

        A scaffold that creates a file the test command then ignores teaches
        people the file does not matter.
        """
        import re

        def passed() -> int:
            """How many tests the last run reported.

            Counting is the behaviour; the file names pytest prints are output
            formatting, and asserting on those failed on a runner where the
            summary is quiet.
            """
            assert run("test") == 0
            match = re.search(r"(\d+) passed", capsys.readouterr().out)
            assert match, "no pytest summary to read"
            return int(match.group(1))

        before = passed()

        (live_project / "demo" / "tests.py").write_text(
            "def test_declared_in_the_app():\n    assert True\n", encoding="utf-8"
        )
        try:
            assert passed() == before + 1
        finally:
            (live_project / "demo" / "tests.py").unlink()

        assert passed() == before

    def test_a_project_with_only_app_tests_still_runs(self, live_project, capsys):
        import shutil

        shipped = live_project / "tests"
        moved = live_project / "tests-aside"
        shutil.move(str(shipped), str(moved))
        (live_project / "demo" / "tests.py").write_text(
            "def test_only_one():\n    assert True\n", encoding="utf-8"
        )
        try:
            assert run("test") == 0
            assert "1 passed" in capsys.readouterr().out
        finally:
            (live_project / "demo" / "tests.py").unlink()
            shutil.move(str(moved), str(shipped))

    def test_a_failed_run_leaves_settings_alone(self, live_project, capsys):
        """The sandbox must not outlive the command.

        The redirection happened before the "no tests" check, so that error path
        left BASE_DIR pointing at a temp directory the command then deleted —
        and every later command in the process read from it.
        """
        import shutil

        from mlango.conf import settings

        before = str(settings.BASE_DIR)
        shipped = live_project / "tests"
        moved = live_project / "tests-away"
        shutil.move(str(shipped), str(moved))
        try:
            assert run("test") == 1
            assert str(settings.BASE_DIR) == before
            assert "test-metastore" not in settings.METASTORE["URL"]
        finally:
            shutil.move(str(moved), str(shipped))
