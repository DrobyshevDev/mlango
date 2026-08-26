# mlango

**Know what your new model version broke — before you promote it.**

*Read this in [Русский](https://github.com/DrobyshevDev/mlango/blob/master/README.ru.md).*

[![CI](https://github.com/DrobyshevDev/mlango/actions/workflows/ci.yml/badge.svg)](https://github.com/DrobyshevDev/mlango/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mlango)](https://pypi.org/project/mlango/)
[![Python](https://img.shields.io/pypi/pyversions/mlango)](https://pypi.org/project/mlango/)
[![License](https://img.shields.io/pypi/l/mlango)](https://opensource.org/licenses/MIT)

```bash
pip install "mlango[sklearn]"
```

Your new model is two points more accurate. Ship it?

Aggregate metrics cannot tell you that it also broke forty rows that used to
work — and those are usually the ones somebody complained about last month.
mlango can:

```bash
$ python manage.py diff reviews.Sentiment 1 2

reviews.Sentiment v1 → v2 on 500 rows of reviews.Reviews

  agreement      92.0%
  changed        40 row(s)
    pos → neg                22
    neg → pos                18

Against the labels
  v1     accuracy     0.7700
  v2     accuracy     0.8060   +0.0360
  fixed          29 row(s) wrong in v1
  broke          11 row(s) right in v1
  verdict        a real improvement: 29 fixed against 11 broken (p=0.006)
```

Accuracy went up three and a half points. Eleven rows that used to work now do
not — and `broke` is the only place that number appears. Reproduce it with
[`examples/promotion/`](https://github.com/DrobyshevDev/mlango/tree/master/examples/promotion).

`--fail-on-regression` turns that into an exit code you can put in front of a
promotion. The same command compares **two runs of an agent's eval suite**
(prompts have no version numbers, so it diffs the runs and tells you what you
changed), **two model files mlango never trained**, and **what a candidate would
have answered on live traffic** if you run it as a shadow.

None of it needed extra infrastructure. It reads what the framework already
recorded.

## Where that comes from

ML projects become a pile of scripts: one to load data, one to train, a notebook
that produced the number in the slide deck, a `checkpoints/` directory nobody can
map back to a commit. Nothing knows what anything else did, so nothing can
compare two of them.

Web development had this problem, and Django's answer was not a better library
but a **framework**: a project layout, a settings module, declarative classes,
migrations, an auto-generated admin, and a `manage.py` that ties it together.

mlango applies that answer to ML. You declare datasets, models, agents and
evaluations; the framework runs them, versions them, records them, and — because
it recorded them — can tell you what changed.

```python
# reviews/datasets.py
from mlango.core import fields
from mlango.data import Dataset, JSONLSource

class Reviews(Dataset):
    """Customer product reviews."""

    id = fields.IntegerField()
    text = fields.TextField()
    label = fields.LabelField(["negative", "positive"])

    class Meta:
        source = JSONLSource("data/reviews.jsonl")
        primary_key = "id"
```

```python
# reviews/models.py
from mlango.core import fields
from mlango.training import Model
from reviews.datasets import Reviews

class Sentiment(Model):
    """TF-IDF into logistic regression."""

    max_features = fields.IntegerField(default=20_000, tunable=True)
    C = fields.FloatField(default=1.0, min_value=0.0, tunable=True)

    class Meta:
        dataset = Reviews
        trainer = "sklearn"
        task = "classification"
        features = ["text"]

    def build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        return make_pipeline(
            TfidfVectorizer(max_features=self.max_features),
            LogisticRegression(C=self.C),
        )
```

```bash
python manage.py train reviews.Sentiment -p C=2.0
```

That one command resolves your class, opens a tracked run, seeds every RNG,
splits the data deterministically, calls your `build()`, drives the training
loop, records metrics, captures the git commit, saves the artifact and registers
a promotable model version. You wrote `build()` and four field declarations.

## What you can ask afterwards

| Question | Command |
|---|---|
| What did the new version break? | `manage.py diff reviews.Sentiment 3 4` |
| Is that difference real, or a coin? | the `verdict` line — McNemar, computed exactly |
| Why did it change? | the config delta, printed beside the result |
| What is it paying attention to? | `manage.py explain reviews.Sentiment` |
| Has the world moved since training? | `manage.py drift reviews.Sentiment` |
| What would the candidate say to real traffic? | `SHADOW`, then `diff --from-log` |
| Did my prompt change break the agent? | `manage.py diff --eval support.Quality` |

---

## Install

```bash
pip install "mlango[sklearn]"
```

Extras: `sklearn`, `torch`, `anthropic`, `dev`, or `all`.

## Five minutes from nothing

```bash
mlango startproject myproject
cd myproject
python manage.py migrate
python manage.py train demo.Sentiment
python manage.py runserver

mlango startplugin mlango-lightgbm --kind trainer   # a package others can install
```

Open <http://127.0.0.1:8000/admin/>. Unlike a bare scaffold, a fresh mlango
project **already contains a working example**: a dataset, a trained model
with real metrics, an agent with a tool, and an eval suite. The admin has
something in it the first time you look.

No configuration is required to get there: the metastore is SQLite, artifacts go
to a local directory, and agents run on an offline provider that needs no API
key.

---

## What you get

### Declarative classes with a `_meta`

Four families, one system. Everything generic in the framework (the admin,
migrations, the CLI, the API) is written against `_meta`. That is why a single
admin renders all four.

| You declare | You get |
|---|---|
| `Dataset` | A lazy queryset, schema validation, deterministic splits, content-addressed versioning |
| `Model` | Hyperparameters as validated fields, tracked runs, callbacks, a model registry with stages |
| `Agent` | A tool-use loop, tools with schemas derived from type hints, memory, full step-by-step tracing |
| `Eval` | Per-case scoring persisted to the metastore, so a regression is a diff between two runs |

### A queryset for data

Lazy, composable, and recorded alongside the run that used it:

```python
train, val = Reviews.objects.filter(label="positive").shuffle(seed=0).split(train=0.8, val=0.2).values()

for batch in train.batch(32):
    ...
```

Lookups follow Django's spelling: `filter(stars__gte=4)`,
`exclude(text__icontains="spam")`, `filter(language__in=["en", "de"])`. Splits
are assigned by hashing each record's key, so **adding rows never moves existing
ones between train and test**. That property is what keeps a held-out set
trustworthy six months later.

### Migrations for schemas

```bash
python manage.py makemigrations
python manage.py migrate
```

Changing a dataset's fields generates a real, reviewable migration file. Data
migrations use `RunPython`, exactly as you would expect.

### An admin you did not build

Every declared object appears automatically, with no registration required.
Register only to change how it looks:

```python
@admin.register(Reviews)
class ReviewsAdmin(admin.ObjectAdmin):
    list_display = ("id", "text", "label")
    list_filter = ("label",)
    search_fields = ("text",)
```

The admin shows data previews with filters and search, run history with metric
charts, side-by-side run comparison, dataset and model versions with one-click
promotion, and a step-by-step trace viewer for every agent call. It is
server-rendered with no build step and no CDN.

![The mlango admin: everything a project declares, and everything it has run](https://raw.githubusercontent.com/DrobyshevDev/mlango/master/docs/assets/admin-overview.png)

Every run keeps its environment, parameters, metrics and artifacts, so a number
from six months ago still says where it came from:

![A run page: environment, parameters, metrics and artifacts](https://raw.githubusercontent.com/DrobyshevDev/mlango/master/docs/assets/admin-run.png)

### Agents as declarations

```python
from mlango.agents import Agent, BufferMemory, tool

@tool
def search_docs(query: str, limit: int = 5) -> list[str]:
    """Search the product documentation.

    Args:
        query: What to search for.
        limit: Maximum number of results.
    """
    return retrieve(query, limit)

class Support(Agent):
    """Answers product questions from the docs."""

    class Meta:
        model = "claude-opus-5"
        system = "You are a support engineer. Cite the docs you used."
        tools = [search_docs]
        memory = BufferMemory(k=20)
```

The JSON schema comes from your type hints and docstring, so a tool is described
in exactly one place. The framework owns the loop, retries, tool dispatch, usage
accounting and tracing.

### Serving from the same declaration

```python
# myproject/routes.py
from mlango.serve import path

urlpatterns = [
    path("predict/", Sentiment.as_endpoint(stage="production")),
    path("chat/", Support.as_endpoint()),
]
```

`manage.py runserver` serves the admin and a documented API together; OpenAPI
schemas are derived from the declarations, so `/api/docs` describes your model's
inputs without you writing a schema.

---

## The command line

```bash
python manage.py check                          # validate the whole project
python manage.py inspectdata data/reviews.csv    # declare a Dataset from a file
python manage.py dataset head reviews.Reviews   # peek at the data
python manage.py dataset materialize reviews.Reviews
python manage.py makemigrations && python manage.py migrate
python manage.py train reviews.Sentiment -p C=2.0 --tag baseline
python manage.py predict reviews.Sentiment "loved every minute"
python manage.py explain reviews.Sentiment                # what the fit relied on
python manage.py drift reviews.Sentiment --since 24h      # has the input moved?
python manage.py diff reviews.Sentiment 3 4               # what did v4 break?
python manage.py sweep reviews.Sentiment -p C=0.25,1,4 --promote-best production
python manage.py runs list
python manage.py runs compare 7c8f1020 c089b7e6
python manage.py evaluate reviews.Accuracy --min-pass-rate 0.9
python manage.py agent support.Support           # interactive session
python manage.py traces show a1b2c3d4            # replay an agent call
python manage.py shell                           # everything pre-imported
python manage.py test                            # against a throwaway metastore
python manage.py runserver
```

`inspectdata` is Django's `inspectdb` for data files: it samples a CSV, JSONL or
Parquet file and prints a `Dataset` with the field types, ranges, label classes
and primary key already filled in, so your first declaration is an edit rather
than a blank page.

Apps can ship their own commands in `<app>/management/commands/`, and they
appear in `manage.py help` automatically, including ones that override a built-in.

---

## Configuration

One settings module, every default documented in
`mlango.conf.global_settings`. Backends are swapped by setting, not by rewriting
code:

```python
METASTORE = {"URL": "postgresql://user@host/mlango"}   # SQLite by default
STORAGE = {"BACKEND": "myproject.storage.S3Storage"}
TRAINERS = {"lightgbm": "myproject.trainers.LightGBMTrainer"}
PROVIDERS = {"vllm": "myproject.providers.VLLMProvider"}
SERVE_MIDDLEWARE = ["mlango.serve.middleware.ApiKeyMiddleware", ...]
```

---

## Why a framework and not a library

A library is something you call. A framework calls you. That inversion is the
whole point, and it is what buys the conveniences above:

- **Project layout and settings**: `manage.py`, `MLANGO_SETTINGS_MODULE`
- **An app registry** that autodiscovers `datasets.py`, `models.py`, `agents.py`, `evals.py`, `admin.py`
- **Migrations**: generated, reviewable files for declared schemas
- **An admin generated from declarations**
- **A management command system** apps can extend and override
- **Signals**: `run_finished`, `epoch_finished`, `tool_called`, and more
- **Pluggable backends** behind settings

If mlango were a library you would still be writing the run loop, the tracking
schema, the admin and the CLI. Being a framework is what removes them.

---

## How it works

Everything follows from one idea: **your class body is compiled into metadata,
and every generic subsystem reads that metadata instead of knowing about your
class.**

```
          your class body                    what reads it
    ┌──────────────────────────┐
    │  class Sentiment(Model): │           ┌──────────────► Admin page
    │      C = FloatField(…)   │           │
    │                          │           ├──────────────► POST /api/predict/
    │      class Meta:         │  ────►  _meta               + OpenAPI schema
    │          dataset = …     │        (Options)  │
    │          trainer  = …    │           ├──────────────► Migration file
    │                          │           │
    │      def build(self): …  │           ├──────────────► manage.py train
    └──────────────────────────┘           │                manage.py sweep
                                           └──────────────► Eval + model registry
```

Nothing on the right imports `Model`, `Dataset`, `Agent` or `Eval`. They all read
`_meta`, which is why one admin renders four different families and why adding a
fifth would not mean touching the admin.

### The layers

```
  core            fields · metaclass · Options · registry · settings · signals
    │             (imports nothing else in mlango)
    ├── metastore   9 tables: runs, metrics, artifacts, versions, traces, spans…
    ├── storage     artifacts, behind one narrow interface
    │
    ├── data ─────┐
    ├── training ─┤  the four families. They never import each other.
    ├── agents ───┤
    ├── evals ────┘
    │
    └── admin · serve · management     read everything, but only through _meta
```

The rule that matters most is the middle one. The four families not importing
each other is what lets you use the agent half without the ML half, and why a
project declaring only datasets never loads a line of agent code.

### What `manage.py train` actually does

```
  train reviews.Sentiment -p C=2.0
        │
        ├─ load settings, autodiscover every app's declarations
        ├─ resolve the label in the registry
        ├─ build the instance — fields validate C=2.0
        ├─ open a run: seed, device, git commit, host, Python version
        ├─ split the data by hashing each record's key, record the fingerprint
        ├─ call your build(), drive the loop, log metrics each epoch
        ├─ save the artifact and register a promotable version
        └─ close the run
```

You wrote `build()` and four field declarations. Everything else happens whether
you remembered it or not, which is the whole argument for a framework.

**[Architecture](https://drobyshevdev.github.io/mlango/architecture/)** has the
full picture: sequence diagrams, the metastore schema, every extension point and
its contract. **[Philosophy](https://drobyshevdev.github.io/mlango/philosophy/)**
explains the decisions behind it.

---

## Documentation

Full docs, including a tutorial that builds a project end to end:
**<https://drobyshevdev.github.io/mlango/>**

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/DrobyshevDev/mlango/blob/master/CONTRIBUTING.md) for the
development setup, and [CODE_OF_CONDUCT.md](https://github.com/DrobyshevDev/mlango/blob/master/CODE_OF_CONDUCT.md) for community
expectations. Good first issues are labelled
[`good first issue`](https://github.com/DrobyshevDev/mlango/labels/good%20first%20issue).

## License

MIT. See [LICENSE](https://github.com/DrobyshevDev/mlango/blob/master/LICENSE).

mlango is not affiliated with or endorsed by the Django Software Foundation. It
borrows Django's design philosophy, gratefully.
