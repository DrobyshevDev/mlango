# Architecture

How the pieces fit, what may import what, and what actually happens when you run
a command. Every diagram here is text in the repository, so it can be reviewed
in a diff.

## The layers

```mermaid
graph TD
    subgraph foundation[" "]
        CORE["<b>core</b><br/><small>fields · metaclass · Options · registry<br/>settings · signals · exceptions</small>"]
    end

    subgraph persistence[" "]
        META["<b>metastore</b><br/><small>12 tables · sessions</small>"]
        MIG["<b>migrations</b><br/><small>autodetector · writer · executor</small>"]
        STORE["<b>storage</b><br/><small>artifacts</small>"]
    end

    subgraph families[" "]
        DATA["<b>data</b><br/><small>Dataset · QuerySet · Sources</small>"]
        TRAIN["<b>training</b><br/><small>Model · trainers · runs · sweeps</small>"]
        AGENTS["<b>agents</b><br/><small>Agent · tools · memory · providers</small>"]
        EVALS["<b>evals</b><br/><small>Eval · scorers</small>"]
    end

    subgraph surfaces[" "]
        ADMIN["<b>admin</b>"]
        SERVE["<b>serve</b>"]
        CLI["<b>management</b><br/><small>manage.py</small>"]
    end

    CORE --> META
    CORE --> STORE
    META --> MIG
    CORE --> DATA
    CORE --> TRAIN
    CORE --> AGENTS
    CORE --> EVALS
    META -.-> DATA
    META -.-> TRAIN
    META -.-> AGENTS
    META -.-> EVALS
    DATA --> ADMIN
    TRAIN --> ADMIN
    AGENTS --> ADMIN
    EVALS --> ADMIN
    DATA --> SERVE
    TRAIN --> SERVE
    AGENTS --> SERVE
    ADMIN --> CLI
    SERVE --> CLI
```

The rules, which `tests/` and code review both enforce:

| Layer | May import | Must not import |
|---|---|---|
| `core` | the standard library | anything else in mlango |
| `metastore`, `storage` | `core` | the four families |
| `data`, `training`, `agents`, `evals` | `core`, `metastore`, `storage` | **each other** |
| `admin`, `serve`, `management` | everything, through `_meta` | — |

The one that matters most is the third. The four families not importing each
other is what lets you use the agent half without the ML half, and it is why a
project that declares only datasets never loads a line of agent code.

A helper needed by two families belongs in `core`. `core/serialization.py`
exists for exactly that reason: `agents` and `evals` were both reaching into a
private function in `training`.

## How a declaration becomes metadata

```mermaid
sequenceDiagram
    participant P as Your class body
    participant M as DeclarativeMeta
    participant O as Options (_meta)
    participant R as Registry

    P->>M: class Urgency(Model)
    M->>M: collect Field instances, in declaration order
    M->>M: walk the MRO for inherited fields and Meta options
    M->>O: build Options(kind, label, app_label, fields, extras)
    O->>O: validate Meta keys against _meta_options
    M->>P: replace each Field with a FieldDescriptor
    M->>P: call _prepare() (Dataset gains .objects)
    M->>R: register unless Meta.abstract
    Note over R: apps.get_model("tickets.Urgency") now resolves
```

Two consequences worth knowing.

**Meta options inherit.** Python class bodies do not inherit on their own, so a
subclass writing its own `class Meta` would silently lose everything the parent
declared. `Options.inherit_extras()` fills in what the subclass did not name,
which is what makes a reusable base class like `TextClassifier` possible.
`abstract` is excluded, or every subclass of an abstract base would be abstract
too.

**An unknown `Meta` key is an error.** It lists the allowed ones, because a
silently ignored option is a bug you find weeks later.

## What `manage.py train` actually does

```mermaid
sequenceDiagram
    autonumber
    participant U as You
    participant C as manage.py
    participant R as Registry
    participant M as Model
    participant Q as QuerySet
    participant T as Trainer
    participant DB as Metastore
    participant S as Storage

    U->>C: train tickets.Urgency -p C=2.0
    C->>C: load settings, mlango.setup()
    C->>R: autodiscover datasets/models/agents/evals per app
    C->>R: get_model("tickets.Urgency")
    C->>M: Urgency(C=2.0) — fields validate the value
    M->>DB: RunContext.start() — seed, device, git commit, host
    M->>Q: dataset.objects → split by hash of the key
    M->>DB: record _data_fingerprint
    M->>T: fit(train, validation, run, callbacks)
    loop each epoch
        T->>DB: log_metrics()
        T->>C: callbacks.emit(on_epoch_end)
    end
    T->>S: save the fitted artifact
    M->>DB: register ModelVersion v1
    M->>DB: run finished
```

Everything after step 5 happens whether you asked for it or not. That is the
inversion: the parts that are easy to forget are the parts you do not write.

## The metastore

Eleven tables, SQLite by default and the same schema on Postgres.

```mermaid
erDiagram
    RUN ||--o{ METRIC : records
    RUN ||--o{ ARTIFACT : writes
    RUN ||--o{ MODEL_VERSION : registers
    RUN ||--o{ TRACE : produces
    RUN ||--o{ EVAL_RESULT : scores
    DATASET_VERSION ||--o{ RUN : "was read by"
    TRACE ||--o{ SPAN : "step by step"
    MIGRATION {
        string app
        string name
        datetime applied
    }
    RUN {
        string uuid
        string kind
        string target
        string status
        json params
        json summary
        string git_commit
    }
    MODEL_VERSION {
        int version
        string stage
        string path
    }
    MODEL_VERSION ||--o{ PREDICTION : "answered"
    PREDICTION {
        int version
        json inputs
        json output
        datetime created_at
    }
    DATASET_VERSION {
        int version
        string content_hash
        string fingerprint
    }
```

`RUN.kind` is one of `train`, `sweep`, `eval` or `agent`, which is why a sweep
and its trials, or an agent call and a training run, all live in one history and
one admin.

Two identities are deliberately separate on a dataset version:
`fingerprint` is the hash of the *declaration*, `content_hash` is the hash of the
*rows*. A schema change and a data change are different events, and telling them
apart six months later is the point of storing both.

## Extension points

Every one of these is a settings entry pointing at a dotted path, so replacing
one never means forking the framework.

| Point | Setting | Contract |
|---|---|---|
| Trainer | `TRAINERS` | `fit`, `predict`, `save`, `load` |
| LLM provider | `PROVIDERS` | one method: `complete()` |
| Artifact storage | `STORAGE["BACKEND"]` | `path`, `open`, `save_bytes`, `read_bytes`, `exists`, `delete`, `size`, `listdir` |
| Serving middleware | `SERVE_MIDDLEWARE` | ASGI middleware, outermost first |
| Training callbacks | `DEFAULT_CALLBACKS` | any subset of the `Callback` hooks |
| Data source | `Meta.source` | iterable of dicts, optionally `count()` |
| Commands | `<app>/management/commands/` | a `Command(BaseCommand)`, may override a built-in |

The narrow ones are narrow on purpose. A provider is one method because the
agent loop, tool dispatch, memory and tracing belong to the framework: swapping
providers must not be able to change how an agent behaves.

## Request paths

```mermaid
graph LR
    REQ([HTTP request]) --> MW["SERVE_MIDDLEWARE<br/><small>logging · api key · rate limit · guardrails</small>"]
    MW --> R{path}
    R -->|/admin| ADM["Admin<br/><small>Jinja2, inline SVG</small>"]
    R -->|/api/...| EP["Endpoint from a declaration"]
    R -->|/api/docs| DOC["OpenAPI from _meta"]
    EP --> LOAD["Registered version<br/><small>loaded once, cached</small>"]
    ADM --> DB[(Metastore)]
    LOAD --> ST[(Storage)]
```

The admin and the API are one ASGI application, so `manage.py runserver` is a
single process in development and the same object behind gunicorn in production.
