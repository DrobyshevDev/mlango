"""mlango — a batteries-included framework for machine learning and LLM agents.

The design goal is simple: everything that made Django pleasant for web
development should be available for ML work. Declarative classes with a
``_meta``, an app registry, migrations, an auto-generated admin, management
commands and a settings module — but the nouns are datasets, models, runs,
agents and evaluations instead of tables and views.

Typical entry point::

    import mlango
    mlango.setup()
"""

__version__ = "0.3.0"

from typing import Any

from mlango.core.registry import apps

__all__ = ["__version__", "apps", "setup", "notebook", "get_version"]


def get_version() -> str:
    return __version__


def setup(settings_module: str | None = None, set_prefix: bool = True) -> None:
    """Bootstrap the framework.

    Loads the settings module, then populates the application registry, which
    in turn imports every declarative module of every installed app so that
    datasets, models, agents and evals register themselves.

    ``mlango.setup()`` is idempotent, so calling it from a script, a notebook
    or a test fixture is safe.
    """
    from mlango.conf import settings

    if settings_module is not None:
        settings.configure_from_module(settings_module)

    apps.populate(settings.INSTALLED_APPS)


def notebook(base_dir: str | None = None, **overrides: Any) -> None:
    """Configure mlango for a notebook or a shell, with no project on disk.

        import mlango
        mlango.notebook()

    Everything then works as it does inside a project: declare a Dataset and a
    Model in a cell, call ``train()``, and the run is recorded with its seed,
    metrics and artifacts. The metastore is SQLite beside the notebook and its
    tables are created on first use, so there is no ``migrate`` step.

    Agents default to the offline provider, so nothing here needs an API key.
    Point ``DEFAULT_PROVIDER`` at ``"anthropic"`` when you want a real model.

    Safe to call twice: a second call leaves existing settings alone, which is
    what re-running the first cell should do.
    """
    import os

    from mlango.conf import settings

    if not settings.configured:
        # Merged rather than passed alongside, so naming one of these in
        # **overrides replaces it instead of colliding with it.
        defaults: dict[str, Any] = {
            "BASE_DIR": base_dir or os.getcwd(),
            "METASTORE": {"URL": "sqlite:///mlango.db"},
            "STORAGE": {"BACKEND": "mlango.storage.local.LocalStorage", "ROOT": "artifacts"},
            "DEFAULT_PROVIDER": "echo",
            "INSTALLED_APPS": [],
        }
        settings.configure(**{**defaults, **overrides})
    setup()
