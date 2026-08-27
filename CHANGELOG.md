# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `RecordingProvider` and `ReplayProvider` record an agent's model calls to a
  file and play them back. An eval suite with an agent in it otherwise talks to
  a model: slow, paid for by the call, and different every time. A cassette is a
  `Provider`, so the agent, loop, tools and tracing are the real ones and only
  the model is absent.
- `Agent.run()` and `Agent.stream()` take `provider=` to override the declared
  one for that call. That is the seam recording hangs off, and being an argument
  rather than a setting is what stops a recording leaking into the next test.
- **`manage.py diff --format markdown`.** The report is a decision one person
  made as long as it only exists in a terminal, and promoting a model is rarely
  one person's decision. This renders the same comparison for a pull request:
  the broken count in the heading, because a comment is first read as a
  notification, and the disagreeing rows folded away, because one two hundred
  rows tall is one nobody scrolls. A pure function of the report dictionary, so
  it covers all four things `diff` compares — registered versions, evaluation
  runs, files mlango never trained, and shadow traffic — without knowing any of
  them exist. `--output PATH` writes it without touching the exit code, so a CI
  job can keep the report and still fail on it, and every report opens with a
  stable marker so a job can edit its own last comment instead of leaving a
  thread. `docs/ci.md` has the workflow.
- **`manage.py promote`.** The workflow this project leads with ended one step
  short: the framework could say what a new version broke and could promote one
  from Python or the admin, but not from a terminal. `--check` closes the loop
  the other way — it compares the candidate with whoever holds the stage and
  refuses the promotion if rows were lost, with `significant` allowing a loss the
  evidence cannot distinguish from a coin. One verb for models and agents.
- `manage.py sweep --workers N` runs trials concurrently. Threads rather than
  processes: settings and the registry are already shared, the metastore is
  built for overlapping access, and sklearn and torch release the GIL for the
  numeric work. The seed is process-global, so concurrent trials no longer each
  start from it — said plainly in the help and the docs, because a sweep is a
  search rather than a number to reproduce.
- `examples/promotion/` reproduces the comparison the README opens with, in one
  command and with no project to set up.

### Fixed

- **Two trainings of the same model racing for a version number lost one of
  them.** `SELECT max(version)` then `INSERT max+1` is a read-then-write with
  nothing holding the gap, so the loser hit the unique constraint and its run
  finished having registered nothing. Retried rather than locked, because the
  racers need not be threads: two `manage.py train` invocations collide the same
  way and an in-process lock cannot see them. Found by the first parallel sweep.
- **Two threads reaching an untouched metastore both created it**, and the loser
  got `table already exists`. A threaded server answering its first two requests
  at once is the ordinary way to meet this; schema creation now holds a lock.

### Changed

- The README leads with what mlango does that nothing else does — telling you
  what a new model version broke before you promote it — rather than with the
  category it belongs to. The Django framing stays, as the *how*.


## 0.3.0 — 2026-08-22

0.2.0 answered questions about a single version: why did it say that, has the
world moved. This release is about the moment you replace one thing with
another. What did the new version break, was the difference real or a coin, what
did I change to cause it, and what would the candidate have told the people who
actually wrote in — asked of models, of agents, and of artefacts mlango never
trained.

### Added

- `manage.py diff --eval LABEL` compares two runs of an evaluation suite, case by
  case. An agent has no version number, so the promotion question — did this
  change break anything that used to work — had no answer on that half of the
  framework, though the per-case results were already stored. Cases are joined
  on `case_id`; `fixed` and `broke` are counted separately, the McNemar verdict
  says whether the balance is a change or a coin, and `--fail-on-regression`
  gates a prompt change the way it already gates a promotion. Cases present in
  only one run are named rather than folded into the totals, because a suite
  that grew is a different suite.
- **OpenTelemetry** (`pip install "mlango[otel]"`). With `TELEMETRY` on,
  training runs, agent loops and tool calls are emitted as spans, namespaced so
  they are findable beside a dozen other libraries — a run started by an HTTP
  request shows up as a child of that request. mlango configures no exporter:
  the process does that, as every other instrumented library expects. Optional,
  silent by default, and incapable of failing the work it describes.
- **Shadow deployment.** With `SHADOW` on, every request is answered twice:
  production replies to the caller and the candidate at `staging` runs on the
  same input, both logged against one request id. `manage.py diff A B --from-log`
  then compares two versions on what they actually answered to real people,
  which is the question a promotion is about and the only one available before
  labels exist. The caller is never affected — a candidate that raises is a
  logged warning, not an outage — and `SAMPLE` bounds the cost. A candidate that
  resolves to the same version as the served one is skipped rather than compared
  with itself, which is what an endpoint serving `latest` would otherwise do
  right after a promotion to staging.
- `Prediction.request_id` pairs the rows one request produced. Matching on
  inputs instead would fuse two callers who happened to ask the same question.
- `mlango.training.model.current_request` is the ContextVar the serving layer
  sets so that `predict()`, which knows nothing about shadows, still logs rows
  that can be paired.
- **Agents have versions.** Models had a registry, stages and `promote()`; an
  agent had none of it, though its behaviour is entirely its declaration. One is
  recorded the first time an agent runs and again whenever the prompt, model or
  any other `Meta` option changes — idempotent by fingerprint, resolved once per
  process, so a served agent writes one row rather than one per request. Every
  trace records which version answered. `manage.py agent LABEL --versions` lists
  them and marks the one matching the code; `--promote N [--stage S]` moves it.
  A version pins configuration and never code: tools are callables, so their
  names are recorded and their implementations cannot be.
- `mlango_agent_versions` is the eleventh metastore table, added on connect to an
  existing database like any other additive column.
- Every evaluation run now records what it evaluated *was configured like* — an
  agent's prompt, model and step limit; a model's registered version and
  hyperparameters — so `diff --eval` puts a cause beside the effect instead of
  leaving the user to remember what they changed. A long value such as a system
  prompt is reported as changed rather than printed. When nothing about the
  target moved the report says so, which is informative in itself: the
  difference is then the target's own. Runs from before this existed are
  reported as unknown rather than as unchanged.
- `Options.recordable()` is the public name for the `Meta` options that survive
  a round trip through JSON — the same filter the fingerprint already used.
- The eval page in the admin shows the same comparison for the last two runs, so
  a broken case is visible without running a command. It renders for free where
  the model diff would not: nothing is loaded and nothing is scored, because
  `evaluate` already wrote a verdict per case. Two runs sharing no case id show
  nothing rather than a table comparing things that are not comparable.
- `reworded` counts cases that still pass and answer differently — nothing for a
  classifier, half the product for something whose output a person reads.
- `mlango.core.stats.significance` is where McNemar's test now lives, because
  evaluations ask it too and may not import `training`. It is still re-exported
  from `mlango.training.comparison`.
- `manage.py diff` now says whether the fixed-against-broken balance is a change
  or a coin. Only the rows two versions disagree about carry information, so the
  question is whether a coin that came up `fixed` heads in `fixed + broke`
  tosses was fair — McNemar's test, computed exactly rather than by
  approximation, because promotions are usually decided on a few hundred rows.
- `--fail-on-regression significant` gates on that instead of on any lost row.
  The strict rule is right for a curated suite where nothing may be lost; on a
  real dataset it blocks a version that fixed two hundred rows and lost three.
  The bare `--fail-on-regression` is unchanged and still means "lose nothing".
  `--alpha` sets the level, default 0.05.
- `manage.py diff --left URI --right URI --dataset LABEL` compares two models
  mlango did not train. The comparison never needed a `Model` class — two things
  that can `predict` and a dataset to score them on is the whole requirement —
  so this is a way in for a team with artefacts and a CSV who have not adopted
  anything yet. A plain path is loaded with joblib, falling back to pickle;
  other schemes come from packages registering under the `mlango.loaders`
  entry-point group, so a registry client stays in its own package rather than
  becoming a dependency here.
- `mlango.training.comparison.compare_predictors` is the seam that made the
  above possible, and is public: the comparison, given two predictors, with
  where they came from no longer part of the question.

### Fixed

- `startproject` now writes `requirements-dev.txt` and says how to use it. The
  scaffold ships `tests/test_demo.py` and `manage.py test` runs it, but
  following the printed steps ended at "pytest is not installed" — pytest cannot
  go in `requirements.txt`, which the generated Dockerfile installs into the
  production image.
- The quickstart CI job installs the built wheel rather than an editable
  checkout. The editable install meant a module missing from the wheel would
  still import, and the `dev` extra it pulled in meant `manage.py test` passed
  there for a reason no user has.

## 0.2.0 — 2026-07-31

Everything in this release answers a question a project has *after* the model
trains: why did it say that, has the world moved, what did the new version
break, and how does any of it leave the laptop it was fitted on.

### Added

- **Deployment is scaffolded, not improvised.** `startproject` now writes
  `asgi.py`, a `Dockerfile`, a `.dockerignore` and a `compose.yaml`, and
  `mlango.serve` exports `create_app`. The generated `settings.py` reads
  `MLANGO_SECRET_KEY`, `MLANGO_DEBUG` and `DATABASE_URL` from the environment, so
  a container changes what it must without editing a file.
- **Feature importance.** Registering a version records what the fit weighted, so
  `manage.py explain <model>` and the admin's model page can answer "why did it
  say that" without loading the artifact. Trainers opt in with one method,
  `importances()`; the sklearn backend names a pipeline's columns from its
  vectoriser, so a text model reports words rather than indices. Backends that
  cannot name a feature report nothing rather than inventing a list.
- **Drift detection.** Training records a profile of the split it fitted on;
  serving records what it was asked, when `PREDICTION_LOG` is turned on; and
  `manage.py drift` compares the two with a population stability index. Input
  drift and prediction drift are both reported — the second needs no ground
  truth, which is the point, because accuracy in production waits on labels that
  arrive late or never. `--fail-on` makes it usable from a scheduled job, and the
  admin's model page shows the same table. New docs page: Monitoring.
- **`mlango.storage.s3.S3Storage`** (`pip install "mlango[s3]"`), and with it a
  story for training somewhere that is not your laptop: a shared metastore plus
  shared artifacts means `manage.py train` on a GPU box and `Model.load()` on a
  laptop are the same project. Works against anything S3-compatible via
  `ENDPOINT_URL`; credentials stay boto3's. `Storage` gains `writable()`,
  `readable()` and `fetch()`, which is what lets a backend that is not a
  filesystem serve libraries that only know how to open files.
- **Extensions are found, not configured.** A package advertising itself through
  the `mlango.trainers` or `mlango.providers` entry point is merged into the
  registries at start-up, so `pip install mlango-lightgbm` is enough for a
  project to write `trainer = "lightgbm"`. Framework defaults, then installed
  packages, then the project's own settings — the project always wins, because
  an extension you cannot override is worse than the dotted path it replaced.
  Nothing is imported during discovery, and `manage.py check` says which
  backends arrived from a package.
- **`mlango startplugin NAME --kind trainer|provider|storage|source`** scaffolds
  the package to publish: pyproject with the entry point already declared, the
  contract with its interesting parts commented, a LICENSE, and tests including
  one that proves the entry point resolves. It needs no project — writing an
  extension should not require inventing somewhere to write it. New docs page:
  Extending mlango, plus `GOVERNANCE.md` on what belongs in the framework and
  what belongs in a package.
- **`manage.py diff`** — what two registered versions actually predict on the
  same data. Aggregate metrics answer "is the new one better" and hide the
  answer people are afraid of: a version two points more accurate overall can
  still have broken forty rows that used to work. The report counts `fixed` and
  `broke` separately, lists the rows where the two disagree, and
  `--fail-on-regression` turns "it lost a row that used to work" into an exit
  code you can put in front of a promotion. With no version numbers it compares
  production against the newest. Regression models are compared by distance
  rather than equality; unlabelled data still reports what changed without
  pretending to say what improved.

### Fixed

- **An artifact trained on one machine could not be read on another.** Runs
  recorded the absolute path they wrote to, so a shared metastore handed a
  laptop rows pointing at `/home/gpu/artifacts/...`. Artifacts are now recorded
  by storage-relative name. Versions registered earlier still carry an absolute
  path and still load where they were written.
- **Upgrading mlango no longer means deleting the metastore.** `create_all`
  creates missing tables and ignores missing columns, so a database written by an
  older release failed at the first query with `no such column`. Additive columns
  are now applied on connect, with their declared default so existing rows stay
  valid.
- **A UTF-8 file with a byte-order mark no longer fails to parse.** Excel,
  Notepad and PowerShell all write a BOM by default, so data exported on Windows
  routinely has one — and `JSONLSource`, `JSONSource` and `CSVSource` died on the
  first record with a message about byte 0xEF that said nothing about the cause.
  They read as `utf-8-sig`, which decodes a plain UTF-8 file identically.
- The README's links were relative, so on PyPI they resolved against the project
  page and 404ed, and the CI badge rendered as a broken image because it points
  at a workflow only a public repository exposes. Links are absolute and the
  badges read from PyPI. `tests/test_pipelines.py` now checks this, and that the
  English and Russian copies do not drift apart.

### Changed

- The tutorial starts from a file you already have: `inspectdata` writes the
  first `Dataset` declaration, and `predict` scores the trained model without
  starting a server. Both were added after the tutorial was written and it had
  never mentioned them.

## 0.1.0 — 2026-07-30

First release. mlango applies Django's design philosophy to machine learning,
analytics and LLM agents: declarative classes, migrations, an auto-generated
admin, and a `manage.py` that ties it together.

### Highlights

- **`manage.py inspectdata`** — Django's `inspectdb`, for data files. Point it at
  a CSV, TSV, JSONL, JSON or Parquet file and it samples the rows and prints a
  `Dataset` declaration: field types, numeric ranges, label classes, nullability,
  the primary key and the likely target. `--write --app myapp` puts it straight
  into `myapp/datasets.py`. It needs no declarations of its own, so it runs on a
  project you have only just created — which is the point, because bringing your
  own data was the one step that still meant writing a field per column by hand.

  Exactly one column becomes a target, since two would leave `Model.get_target()`
  unable to choose. A `max_length` is only imposed when every sampled value is
  short: a limit that turns out too small rejects valid data later, while
  `TextField` never rejects anything.
- **`manage.py predict`** — score data without starting a server, using the
  registered version the API would serve. Takes literal values, `--file`, or
  `--dataset` with repeatable `--filter FIELD=VALUE`; emits a table, JSONL or CSV,
  to stdout or `--output`. An `id`/`uuid`/`pk` on the input is carried through so
  a scored file can be joined back to its source. Data missing a feature the
  model needs is reported by column name, instead of failing inside a vectoriser.

- **Transformers trainer** (`mlango[transformers]`) — fine-tune a pretrained
  encoder for text classification or regression. The loop is mlango's own, so
  callbacks, early stopping, metric recording and run tracking behave the same
  as for any other backend; tokenisation, pretrained weights and heads come from
  Hugging Face. Two text fields become a sentence pair automatically.
- **Model presets** — `TextClassifier`, `TextRegressor`, `TabularClassifier`,
  `TabularRegressor`, `TransformerModel`. A complete declaration is now three
  lines, with every default overridable.
- **Meta options inherit.** A subclass writing its own `class Meta` keeps
  everything the parent declared and overrides only what it names. Python class
  bodies do not inherit on their own; without this, a reusable base class was
  impossible to write.
- **Agent streaming.** `Agent.stream()` yields typed events —  `Started`,
  `Thinking`, `TextChunk`, `ToolCalled`, `ToolFinished`, `StepFinished`,
  `Finished`, `Failed` — as they happen, and `Agent.as_stream_endpoint()` serves
  them as Server-Sent Events. `run()` and `stream()` share one loop
  implementation, so they cannot disagree about what the agent did.
- **Data sources**: `ParquetSource` (streamed in row-group batches, count from
  the file footer), `SQLSource` (server-side cursor, defaults to the metastore),
  `HuggingFaceSource`, and `DatasetVersionSource` for pinning a derived dataset
  to an exact upstream snapshot.
- **`Registry.unregister()`** and a documented registry-isolation pattern, so
  tests and notebooks can redeclare a class.
- `py.typed`, so downstream type checkers honour mlango's annotations.
- Release workflow using PyPI trusted publishing, with the tag checked against
  `__version__`, the changelog checked for a matching section, and the built
  wheel verified to contain the admin templates and `py.typed`.
- **`startproject` ships tests.** A new project comes with a working `tests/`
  directory — eight tests covering datasets, training, evaluation and agents —
  so it is green before anyone edits it and `manage.py test` works immediately.
  `startapp` writes a `tests.py` to match.
- **Documentation complete in English and Russian** — all 14 pages in both.
- **The type surface is now checked.** `mypy mlango` is clean and blocking in
  CI. Generic subsystems that took a bare `type` now name the family they mean
  (`mlango.core.typing`), and declared fields read as values inside your own
  `build()` rather than as `Field` objects, so user projects type-check too.

### Core

- Lazy settings resolved from `MLANGO_SETTINGS_MODULE`, with every default
  documented in `mlango.conf.global_settings`. An unknown setting is an error,
  not a silent no-op.
- App registry that autodiscovers `datasets.py`, `models.py`, `agents.py`,
  `evals.py`, `admin.py` and `signals.py` in every installed app.
- Declarative metaclass producing a `_meta` that the admin, migrations, CLI and
  API are all written against.
- 17 field types serving four jobs: dataset schemas, model hyperparameters,
  agent configuration and inference input validation.
- Signal dispatcher whose receivers cannot take a run down.

### Metastore and migrations

- Nine tables recording runs, metrics, artifacts, dataset versions, model
  versions, agent traces, spans, evaluation results and applied migrations.
- SQLite by default; the same schema runs on Postgres via one setting.
- Generated, reviewable migration files. The autodetector is deliberately
  conservative and never guesses a rename.

### Data

- `Dataset` with a lazy, composable queryset and Django-style `field__lookup`.
- Splits assigned by hashing each record's key, so adding rows never moves
  existing ones between train and test.
- Content-addressed materialisation that deduplicates identical snapshots, and
  distinguishes a schema change from a data change.

### Training

- `Model` with hyperparameters as validated fields, recorded on every run.
- Pluggable trainers behind one contract; scikit-learn and PyTorch included.
- Run tracking that captures seed, device, git commit, host, Python version and
  a data fingerprint.
- Metric recording built into the framework rather than a callback, so
  customising `DEFAULT_CALLBACKS` never costs run history.
- Callbacks for early stopping, checkpointing, progress and CI thresholds.
- Model registry with promotable stages.
- Hyperparameter sweeps (`manage.py sweep`) over grid or random search, with the
  winning version promotable in the same command.

### Agents

- Declarative `Agent` owning the tool-use loop, tool dispatch, retries and usage
  accounting.
- `@tool` deriving JSON Schema from type hints and Google-style docstrings.
- Four memory backends, including one that rebuilds history from traces.
- Anthropic provider and a deterministic offline provider, so the framework and
  its tests run with no credentials.
- Step-by-step tracing into ordered spans.

### Evaluation

- Declarative `Eval` with 13 scorers, including an LLM judge and a scorer that
  asserts on which tools an agent reached for.
- Per-case results persisted, so a regression is a diff between two runs.

### Admin and serving

- Server-rendered admin with no build step: data previews with filters and
  search, run history with inline SVG metric charts, side-by-side run
  comparison, version promotion and a trace viewer.
- Everything declared appears without registration; register to customise.
- Optional HTTP Basic auth, with `manage.py check` warning when the admin is
  open and `DEBUG` is off.
- Inference API deriving OpenAPI schemas from the declarations.

### Command line

- Sixteen commands: `startproject`, `startapp`, `check`, `makemigrations`,
  `migrate`, `showmigrations`, `train`, `sweep`, `evaluate`, `agent`, `runs`,
  `traces`, `dataset`, `shell`, `test`, `runserver`.
- Apps can ship their own commands, and override built-ins.
- `python -m mlango` works when the console script is not on `PATH`.

### Getting started

- `mlango startproject` scaffolds a project that already works — a dataset, a
  trained model, an agent and an eval — so the admin is populated the first time
  you open it. `--bare` skips the demo.
- Documentation in English and Russian, with a structure that accepts new
  languages one file at a time.
