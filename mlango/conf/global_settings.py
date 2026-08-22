"""Default values for every mlango setting.

A project's ``settings.py`` only needs to override what differs. Anything not
named here is a typo, and the settings object will say so.
"""

from __future__ import annotations

import os

# --- Core -------------------------------------------------------------------

#: Absolute path to the project root. ``startproject`` fills this in.
BASE_DIR: str = os.getcwd()

#: Extra entropy for hashing and any future signing. Keep it out of git.
SECRET_KEY: str = ""

#: Verbose tracebacks, admin stack traces, autoreload.
DEBUG: bool = False

#: Apps whose ``datasets``/``models``/``agents``/``evals``/``admin`` modules are
#: imported at startup. Either a dotted path to a package or to an AppConfig.
INSTALLED_APPS: list[str] = []

#: Modules autodiscovered inside every installed app, in import order.
APP_MODULES: tuple[str, ...] = (
    "datasets",
    "models",
    "agents",
    "evals",
    "admin",
    "signals",
)

# --- Metastore --------------------------------------------------------------

#: Where runs, metrics, traces and versions live. SQLite by default so a fresh
#: project works with zero infrastructure; point it at Postgres for a team.
METASTORE: dict[str, object] = {
    "URL": "sqlite:///mlango.db",
    "ECHO": False,
    "POOL_PRE_PING": True,
}

# --- Artifact storage -------------------------------------------------------

#: Backend for checkpoints, materialised datasets and any file a run produces.
STORAGE: dict[str, object] = {
    "BACKEND": "mlango.storage.local.LocalStorage",
    "ROOT": "artifacts",
}

# --- Training ---------------------------------------------------------------

#: Trainer backends available to ``Model.Meta.trainer``.
TRAINERS: dict[str, str] = {
    "sklearn": "mlango.training.backends.sklearn_backend.SklearnTrainer",
    "torch": "mlango.training.backends.torch_backend.TorchTrainer",
    "transformers": "mlango.training.backends.transformers_backend.TransformersTrainer",
}

#: Default device for backends that care. "auto" picks cuda when available.
DEVICE: str = "auto"

#: Seed applied to python/numpy/torch at the start of every run.
SEED: int | None = 1337

#: Callbacks appended to every training run, as dotted paths. Metric recording
#: is built into the framework, so this list is purely additive — emptying it
#: never costs you run history.
DEFAULT_CALLBACKS: list[str] = []

#: What a served model records about the requests it answers, so drift has
#: something to measure. Off by default: a prediction log is a copy of user
#: input in a database, which is a decision a project makes rather than one it
#: wakes up with.
#:
#: ``SAMPLE`` is the fraction of predictions kept — 0.05 is plenty to see a
#: distribution move and costs almost nothing on a hot endpoint. ``MAX_ROWS``
#: trims the oldest rows so an always-on log cannot fill the disk.
PREDICTION_LOG: dict[str, object] = {
    "ENABLED": False,
    "SAMPLE": 1.0,
    "MAX_ROWS": 100_000,
}

#: Run a candidate version beside the one being served, on the same requests,
#: and log both. Off by default: it doubles the work an endpoint does, and a
#: framework that silently doubled your inference bill would deserve what it got.
#:
#: ``STAGE`` names where the candidate comes from — ``staging`` by convention,
#: so promoting to staging is what starts the shadow. ``SAMPLE`` bounds the
#: cost: 0.1 shadows a tenth of requests, which is usually enough to compare
#: two versions within a day.
SHADOW: dict[str, object] = {
    "ENABLED": False,
    "STAGE": "staging",
    "SAMPLE": 1.0,
}

# --- Agents -----------------------------------------------------------------

#: LLM providers available to ``Agent.provider``.
PROVIDERS: dict[str, str] = {
    "anthropic": "mlango.agents.providers.anthropic.AnthropicProvider",
    "echo": "mlango.agents.providers.echo.EchoProvider",
}

#: Provider used when an agent does not name one.
DEFAULT_PROVIDER: str = "anthropic"

#: Model id used when an agent does not name one.
DEFAULT_AGENT_MODEL: str = "claude-opus-5"

#: Thinking mode passed to providers that support it. "adaptive" lets the model
#: decide how much to think; None disables the parameter entirely.
DEFAULT_THINKING: str | None = "adaptive"

#: Hard stop on the tool-use loop, so a confused agent cannot spin forever.
AGENT_MAX_STEPS: int = 12

#: Record every LLM call and tool call as spans in the metastore.
TRACING: bool = True

# --- Admin & serving --------------------------------------------------------

ADMIN_ENABLED: bool = True
ADMIN_URL: str = "/admin"
ADMIN_SITE_HEADER: str = "mlango administration"
ADMIN_SITE_TITLE: str = "mlango"
ADMIN_PAGE_SIZE: int = 25

#: Directories searched for admin templates before the framework's own, so a
#: project can replace any of them by name. A file at
#: ``<BASE_DIR>/templates/admin/base.html`` overrides the shipped base.html and
#: nothing else, the way Django's admin has always worked. Relative paths
#: resolve against BASE_DIR.
ADMIN_TEMPLATE_DIRS: list[str] = ["templates/admin"]

#: Set a password to require HTTP Basic auth on the admin. Empty means the
#: admin is unauthenticated, which is fine locally and never fine in public.
#: For real deployments, put the admin behind your identity provider.
ADMIN_USERNAME: str = "admin"
ADMIN_PASSWORD: str = ""

#: Shared keys accepted on /api routes by ApiKeyMiddleware. Empty disables it.
SERVE_API_KEYS: list[str] = []

#: Terms that cause GuardrailMiddleware to reject a request body.
SERVE_BLOCKED_TERMS: list[str] = []

#: Dotted path to the module holding ``urlpatterns`` for the inference API.
ROOT_ROUTECONF: str | None = None

#: Middleware wrapped around every inference endpoint, outermost first.
SERVE_MIDDLEWARE: list[str] = [
    "mlango.serve.middleware.RequestLogMiddleware",
]

SERVE_HOST: str = "127.0.0.1"
SERVE_PORT: int = 8000

# --- Logging ----------------------------------------------------------------

LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
