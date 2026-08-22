# Settings

One module, resolved from the `MLANGO_SETTINGS_MODULE` environment variable —
`manage.py` sets it for you. Every default lives in
`mlango.conf.global_settings`; override only what differs.

An unknown setting is an error rather than a silent no-op:

```python
>>> settings.METASTOR
AttributeError: 'METASTOR' is not a known mlango setting. Settings must be
uppercase and declared in mlango.conf.global_settings.
```

## Core

| Setting | Default | Purpose |
|---|---|---|
| `BASE_DIR` | cwd | Everything relative resolves from here |
| `SECRET_KEY` | `""` | Extra entropy for hashing. Keep it out of git |
| `DEBUG` | `False` | Verbose tracebacks and autoreload |
| `INSTALLED_APPS` | `[]` | Apps whose declarations are loaded at startup |
| `APP_MODULES` | `("datasets", "models", "agents", "evals", "admin", "signals")` | Modules autodiscovered in each app |

## Metastore

```python
METASTORE = {
    "URL": "sqlite:///mlango.db",   # or postgresql://user@host/mlango
    "ECHO": False,                  # log every SQL statement
    "POOL_PRE_PING": True,          # non-SQLite only
}
```

A relative SQLite path resolves against `BASE_DIR`, so the database does not
follow your shell's working directory around. SQLite runs in WAL mode, so the
admin can read while a training loop writes.

Move to Postgres when more than one worker records runs.

## Storage

```python
STORAGE = {
    "BACKEND": "mlango.storage.local.LocalStorage",
    "ROOT": "artifacts",
}
```

Checkpoints, materialised datasets and run outputs go here. Artifacts are
recorded in the metastore by storage-relative name, so a run on one machine
resolves on another.

```python
STORAGE = {
    "BACKEND": "mlango.storage.s3.S3Storage",
    "ROOT": "s3://my-bucket/mlango",
    # Optional — anything S3-compatible: MinIO, R2, B2.
    "ENDPOINT_URL": os.environ.get("S3_ENDPOINT_URL"),
    "REGION": "eu-west-1",
}
```

```bash
pip install "mlango[s3]"
```

Credentials are boto3's, so the usual environment variables, instance roles and
profiles all work and mlango never handles a secret it does not have to. Point
`BACKEND` at your own `Storage` subclass for anything else — see
[Training somewhere else](serving.md#training-somewhere-else).

## Training

| Setting | Default | Purpose |
|---|---|---|
| `TRAINERS` | sklearn and torch | `{name: dotted path}` available to `Meta.trainer` |
| `DEVICE` | `"auto"` | `"auto"` picks CUDA when available |
| `SEED` | `1337` | Seeds python, numpy and torch at the start of every run |
| `DEFAULT_CALLBACKS` | `[]` | Callbacks appended to every run |
| `PREDICTION_LOG` | off | What a served model records about its requests |
| `SHADOW` | off | Run a candidate version beside the served one |
| `TELEMETRY` | off | Emit OpenTelemetry spans for runs and agent loops |

```python
TRAINERS = {"lightgbm": "myproject.trainers.LightGBMTrainer"}

DEFAULT_CALLBACKS = [
    "mlango.training.callbacks.ProgressBar",
]
```

!!! note
    `DEFAULT_CALLBACKS` is purely additive — metric recording is built into the
    framework, so emptying the list never costs you run history.

```python
PREDICTION_LOG = {
    "ENABLED": True,   # off by default
    "SAMPLE": 0.05,    # keep 5% of predictions
    "MAX_ROWS": 100_000,
}
```

The log is what `manage.py drift` measures against a version's training profile.
It is off by default because it is a copy of user input in a database — a
decision a project makes rather than one it wakes up with. See
[Monitoring](monitoring.md).

```python
SHADOW = {"ENABLED": True, "STAGE": "staging", "SAMPLE": 0.1}
```

Answers every request twice — production replies, the candidate runs on the
same input — so two versions can be compared on real traffic. See
[Shadow deployment](serving.md#shadow-deployment).

## Agents

| Setting | Default | Purpose |
|---|---|---|
| `PROVIDERS` | anthropic and echo | `{name: dotted path}` available to `Meta.provider` |
| `DEFAULT_PROVIDER` | `"anthropic"` | Provider when an agent does not name one |
| `DEFAULT_AGENT_MODEL` | `"claude-opus-5"` | Model when an agent does not name one |
| `DEFAULT_THINKING` | `"adaptive"` | Thinking mode; `None` omits the parameter |
| `AGENT_MAX_STEPS` | `12` | Hard stop on the tool-use loop |
| `TRACING` | `True` | Record spans for every model and tool call |

A scaffolded project starts on `"echo"` so it runs with no credentials. Switch to
`"anthropic"` and export `ANTHROPIC_API_KEY` when you want a real model.

## Admin

| Setting | Default | Purpose |
|---|---|---|
| `ADMIN_ENABLED` | `True` | Mount the admin at all |
| `ADMIN_URL` | `"/admin"` | Where to mount it |
| `ADMIN_SITE_HEADER` | `"mlango administration"` | Header text |
| `ADMIN_SITE_TITLE` | `"mlango"` | Browser title |
| `ADMIN_PAGE_SIZE` | `25` | Default rows per page |
| `ADMIN_USERNAME` | `"admin"` | Basic auth username |
| `ADMIN_PASSWORD` | `""` | Set it to require Basic auth |

## Serving

| Setting | Default | Purpose |
|---|---|---|
| `ROOT_ROUTECONF` | `None` | Module holding `urlpatterns` |
| `SERVE_MIDDLEWARE` | request logging | Middleware, outermost first |
| `SERVE_HOST` | `"127.0.0.1"` | Bind address |
| `SERVE_PORT` | `8000` | Port |
| `SERVE_API_KEYS` | `[]` | Keys accepted by `ApiKeyMiddleware` |
| `SERVE_BLOCKED_TERMS` | `[]` | Terms rejected by `GuardrailMiddleware` |

## Logging

| Setting | Default |
|---|---|
| `LOG_LEVEL` | `"INFO"` |
| `LOG_FORMAT` | `"%(asctime)s %(levelname)-7s %(name)s: %(message)s"` |

## Per-environment settings

The usual pattern is a base module plus overrides:

```python title="myproject/settings/base.py"
INSTALLED_APPS = ["reviews", "support"]
METASTORE = {"URL": "sqlite:///mlango.db"}
```

```python title="myproject/settings/production.py"
import os

from myproject.settings.base import *  # noqa: F403

DEBUG = False
SECRET_KEY = os.environ["MLANGO_SECRET_KEY"]
ADMIN_PASSWORD = os.environ["MLANGO_ADMIN_PASSWORD"]
METASTORE = {"URL": os.environ["DATABASE_URL"]}
STORAGE = {"BACKEND": "mlango.storage.s3.S3Storage", "ROOT": "s3://bucket/mlango"}
DEFAULT_PROVIDER = "anthropic"
SERVE_API_KEYS = os.environ["MLANGO_API_KEYS"].split(",")
```

```bash
python manage.py migrate --settings myproject.settings.production
```

Nested dict settings merge with their defaults, so overriding
`METASTORE["URL"]` does not mean restating `ECHO` and `POOL_PRE_PING`.

## Configuring without a module

Used by tests, scripts and anything without a `manage.py`. For a notebook,
[`mlango.notebook()`](notebooks.md) wraps this with defaults chosen for one.

```python
from mlango.conf import settings

settings.configure(
    BASE_DIR="/tmp/scratch",
    METASTORE={"URL": "sqlite:///scratch.db"},
    DEFAULT_PROVIDER="echo",
    INSTALLED_APPS=[],
)

import mlango
mlango.setup()
```

## Checking your configuration

```bash
python manage.py check
```

Reports the resolved settings module, installed apps, what is declared, the
metastore URL and whether its tables exist, which trainers and providers are
importable, whether every dotted path in `DEFAULT_CALLBACKS`,
`SERVE_MIDDLEWARE` and `STORAGE` resolves, pending migrations, and whether the
admin is authenticated.

Resolving those paths up front matters: a typo in `DEFAULT_CALLBACKS` would
otherwise surface as the same import error repeated once per sweep trial.
