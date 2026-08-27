# Command line

Every project gets a `manage.py`. The `mlango` script does the same job before a
project exists, and `python -m mlango` works when the script is not on `PATH`.

```bash
python manage.py help
python manage.py help train
```

## The commands

### Getting started

| Command | Does |
|---|---|
| `mlango startproject NAME [DIR]` | Scaffold a project that already works. `--bare` skips the demo app |
| `manage.py startapp NAME` | Scaffold an app: datasets, models, agents, evals, admin, migrations |
| `manage.py check` | Validate settings, backends, wiring, migrations and the admin |
| `mlango startplugin NAME --kind trainer` | Scaffold a publishable package that extends mlango |

`startplugin` needs no project: it writes a distributable package — pyproject
with the entry point already declared, the contract with its interesting parts
commented, a LICENSE and tests — so `pip install` is all a project needs to reach
it. `--kind` is `trainer`, `provider`, `storage` or `source`. See
[Extending mlango](extending.md).

### Bringing your own data

Django has `inspectdb` for an existing database. This is the same idea for a
data file: it samples the file and prints a `Dataset` you can paste into
`datasets.py`, so your first declaration is an edit rather than a blank page.

```bash
python manage.py inspectdata data/reviews.csv
python manage.py inspectdata data/reviews.csv --name Feedback -n 5000
python manage.py inspectdata data/reviews.csv --write --app reviews
```

Reads `.csv`, `.tsv`, `.jsonl`, `.ndjson`, `.json` and `.parquet`. It needs no
declarations of its own, so it works on a project you have only just created.

```python
class Reviews(Dataset):
    """40 rows, 6 columns."""

    id = IntegerField(min_value=1, max_value=40)
    body = TextField()
    stars = IntegerField(min_value=1, max_value=5)
    country = CharField(max_length=16, choices=["GB", "US"])
    verified = BooleanField()
    label = LabelField(["neg", "pos"])

    class Meta:
        source = CSVSource("data/reviews.csv")
        primary_key = "id"
```

What it decides, and why:

| Signal | Becomes |
|---|---|
| All values parse as whole numbers | `IntegerField` with the observed range |
| Any value has a decimal point | `FloatField` with the observed range |
| `true`/`yes`/`t`/`on` and their opposites | `BooleanField` |
| A dict, a list, or a string parsing as either | `JSONField` |
| ISO timestamps | `DateTimeField` |
| Few distinct values, and they repeat | `CharField(choices=…)` |
| Any value longer than 32 characters | `TextField` |
| A column named `label`, `target`, `y`, `class`… | `LabelField` or `TargetField` |
| A unique column named `id`, `uuid` or `*_id` | `Meta.primary_key` |
| Some values missing | `null=True, required=False` |

Two rules worth knowing. **Exactly one column becomes a target** — declaring two
would leave `Model.get_target()` unable to choose, so other categorical columns
stay `CharField` with `choices`. And a column is only given a `max_length` when
every sampled value is short, because a limit that turns out to be too small
rejects valid data later, while `TextField` never rejects anything.

It is a starting point, not an oracle. Anything it guessed at carries a comment
saying so, and a column name that cannot be a Python attribute is reported
rather than silently mangled.

### Data

```bash
python manage.py dataset list
python manage.py dataset show reviews.Reviews
python manage.py dataset head reviews.Reviews -n 20
python manage.py dataset validate reviews.Reviews
python manage.py dataset materialize reviews.Reviews --notes "nightly snapshot"
python manage.py dataset versions reviews.Reviews
```

### Migrations

```bash
python manage.py makemigrations [app] [-n NAME] [--dry-run] [--empty]
python manage.py migrate [app] [--plan] [--fake]
python manage.py showmigrations [app]
```

### Training

```bash
python manage.py train reviews.Sentiment -p C=2.0 -p max_features=5000 \
    --tag baseline --notes "first attempt" --materialize

python manage.py sweep reviews.Sentiment -p C=0.25,1,4 \
    --strategy grid --metric accuracy --mode max --promote-best production
```

| Flag | Effect |
|---|---|
| `-p NAME=VALUE` | Override a hyperparameter. Repeatable |
| `--dataset LABEL` | Train on a different dataset |
| `--tag TAG` | Tag the run. Repeatable |
| `--seed N` | Override the seed |
| `--materialize` | Freeze the training view into a dataset version first |
| `--no-register` | Train without adding to the registry |

Trials run one after another by default. `--workers` runs them together:

```bash
python manage.py sweep reviews.Sentiment -p C=0.25,1,4 --workers 4
```

Threads, not processes — settings and the registry are already shared, the
metastore is SQLite in WAL mode built for overlapping readers and writers, and
the numeric work in sklearn and torch releases the GIL.

One honest cost: **the RNG seed is process-global**, so concurrent trials no
longer each begin from the same state. A sweep is a search rather than a number
to reproduce, which is why the option exists — re-run the winning point on its
own if you need its exact score back.

### Prediction

Scoring without starting a server. The model comes from the registry, so this
runs the same artefact the API would serve.

```bash
python manage.py predict reviews.Sentiment "loved every minute of it"
python manage.py predict reviews.Sentiment "great" "awful" --proba

python manage.py predict reviews.Sentiment --dataset -n 100
python manage.py predict reviews.Sentiment --dataset --filter label=pos

python manage.py predict reviews.Sentiment --file incoming.jsonl \
    --format jsonl --output scored.jsonl
```

| Flag | Effect |
|---|---|
| `--dataset` | Score the model's own declared dataset |
| `--filter FIELD=VALUE` | Narrow the dataset. Repeatable |
| `--file PATH` | Score a csv/tsv/jsonl/json/parquet file |
| `-n N` | Stop after N records |
| `--version N` / `--stage NAME` | Which registered version to load |
| `--proba` | Include class probabilities |
| `--format table\|jsonl\|csv` | How to print it |
| `--output PATH` | Write to a file instead of stdout |

An `id`, `uuid` or `pk` on the input is carried through to the output, so a
scored file can be joined back to where it came from. If the data is missing a
feature the model needs, the command says which column is absent and what the
data does have — rather than letting the trainer fail somewhere deep inside a
vectoriser.

### Explaining a version

Which features a trained version actually leaned on. The weights are recorded on
the version row when it is registered, so this reads the metastore and never
loads the artifact:

```bash
python manage.py explain reviews.Sentiment
python manage.py explain reviews.Sentiment --stage production -n 10
python manage.py explain reviews.Sentiment --json
```

```
reviews.Sentiment@v4
top 10 of 40, largest weight first

delightful   ████████████████████████████████  2.4439
dull         ████████████████████████████···· -2.1614
brilliant    ███████████████████████████·····  2.0495
boring       ███████████████████████████····· -2.0407
badly        ██████████████████████████······ -1.9794
beautifully  ██████████████████████████······  1.9708
awful        █████████████████████████······· -1.8902
excellent    █████████████████████████·······  1.8844
waste        ███████████████████············· -1.4853
every        ███████████████████·············  1.4288
```

A pipeline's vectoriser names its own columns, which is what turns 40,000
numbered slots into the words above. The sign is the direction of the effect —
kept for binary and regression fits, where it means something, and dropped for
multiclass, where a feature arguing for one class argues against another.

| Flag | Effect |
|---|---|
| `--version N` / `--stage NAME` | Which version to explain (default: newest) |
| `-n N` | How many features to show |
| `--json` | Emit the weights instead of a chart |
| `--recompute` | Load the artifact, re-derive the weights and store them |

`--recompute` is the escape hatch for a version registered before mlango knew
how to explain it. Backends that cannot name a feature — the neural ones —
report nothing rather than inventing a plausible list.

### Diffing two versions

Aggregate metrics answer "is the new one better" and hide the answer you are
afraid of: a version two points more accurate overall can still have broken
forty rows that used to work. This scores both on the same data and diffs the
answers.

```bash
python manage.py diff reviews.Sentiment 3 4
python manage.py diff reviews.Sentiment                      # production vs newest
python manage.py diff reviews.Sentiment 3 4 --show-changes 20
python manage.py diff reviews.Sentiment 3 4 --fail-on-regression
```

```
reviews.Sentiment v3 → v4 on 500 rows of reviews.Reviews

  agreement      94.2%
  changed        29 row(s)
    neg → pos                18
    pos → neg                11

Against the labels
  v3 accuracy      0.8840
  v4 accuracy      0.9020   +0.0180
  fixed          22 row(s) wrong in v3
  broke           4 row(s) right in v3
  verdict        a real improvement: 22 fixed against 4 broken (p=0.001)
```

**`broke`** is the number nobody reports and everybody wants. A promotion that
improves the average while losing rows that used to work is the kind that gets
reverted a week later, and `--fail-on-regression` turns it into an exit code you
can put in front of a promotion.

**`verdict`** answers the question those two counts invite. Rows both versions
get right say nothing about which is better, and neither do rows both get wrong;
only the disagreements carry information. So the question is whether a coin that
came up 22 heads in 26 tosses was fair, which is
[McNemar's test](https://en.wikipedia.org/wiki/McNemar%27s_test), computed
exactly rather than by approximation because promotions are usually decided on a
few hundred rows.

The distinction it draws is the one that matters before a promotion: 200 fixed
against 3 broken is an improvement, 38 fixed against 40 broken is a coin, and a
rule that counts broken rows calls both of them a regression.

`--from-log` compares the two versions on requests they *already* answered,
rather than scoring a dataset now. That needs a
[shadow deployment](serving.md#shadow-deployment) — both versions answering the
same traffic — and the report then carries no `fixed`/`broke`, because
production traffic has no labels:

```bash
python manage.py diff reviews.Sentiment 4 5 --from-log --since 24h
```

With no version numbers it compares what is in production against the newest —
which is the question you have when you are about to promote something.

| Flag | Effect |
|---|---|
| `--dataset LABEL` | Score a different dataset, e.g. a held-out set |
| `-n N` | Stop after N rows |
| `--show-changes N` | Print up to N rows where the two disagree |
| `--json` | Emit the whole report |
| `--fail-on-regression` | Exit non-zero if the newer one lost a row the older one got right |
| `--fail-on-regression significant` | Exit non-zero only when the losses beat the gains by more than chance |
| `--alpha P` | Significance level for the mode above. Default `0.05` |

```bash
# A curated regression suite: nothing may be lost.
python manage.py diff reviews.Sentiment --fail-on-regression

# A real dataset before a promotion: noise may pass, a real loss may not.
python manage.py diff reviews.Sentiment --fail-on-regression significant
```

Three renderings, one report. `--format markdown` produces something meant to be
posted in a pull request rather than read in a terminal, and `--output` writes it
to a file without touching the exit code — so a CI job can keep the report and
still go red:

```bash
python manage.py diff reviews.Sentiment     --format markdown --show-changes 20     --output diff.md --fail-on-regression significant
```

`--json` is the older spelling of `--format json` and still means what it meant.
The workflow around this is in [Continuous integration](ci.md).

### Promoting a version

The other half of the diff. `promote` moves a model or agent version to a
stage, and `--check` compares it with whoever holds that stage first — refusing
the promotion if the candidate lost rows.

```bash
python manage.py promote reviews.Sentiment 4                     # to production
python manage.py promote reviews.Sentiment                       # the newest version
python manage.py promote reviews.Sentiment 4 --stage staging
python manage.py promote reviews.Sentiment 4 --check             # lose nothing
python manage.py promote reviews.Sentiment 4 --check significant # lose nothing that matters
```

```
$ python manage.py promote reviews.Sentiment 2 --check

v1 → v2 on 500 rows of reviews.Reviews
  accuracy       0.7700 → 0.8060   +0.0360
  fixed          29 row(s)
  broke          11 row(s)

error: Refusing to promote: v2 is wrong on 11 row(s) that v1 got right.
Inspect them with: manage.py diff reviews.Sentiment 1 2 --show-changes 11
```

Note that v2 is **more accurate** and the strict check still refuses it. That
rule is for a curated suite where nothing may be lost. On a real dataset use
`--check significant`, which allows a loss the evidence cannot distinguish from
a coin and refuses one it can:

```
  verdict        a real improvement: 29 fixed against 11 broken (p=0.006)
reviews.Sentiment@v2 is now at stage 'production'.
```

| Flag | Effect |
|---|---|
| `--stage NAME` | Which stage. Default `production` |
| `--check [any\|significant]` | Compare with the incumbent first, and refuse a regression |
| `--dataset LABEL` | Score `--check` against this dataset |
| `-n N` | Cap the rows `--check` scores |
| `--notes TEXT` | Why, recorded with the move |
| `--history` | List what has been promoted instead of promoting |

One verb covers models and agents — an agent version is the same idea, so
`promote support.Support 3` works too. `--check` needs a model, because it
compares predictions; for an agent, compare two runs of its evaluation suite
with `diff --eval`.

Every move is recorded. The `stage` column is mutable — promoting v3 overwrites
what v2 was — so on its own a registry can say what is live and nothing about
how it got there:

```bash
python manage.py promote reviews.Sentiment --history   # one model
python manage.py promote --history                     # everything
```

```
reviews.Sentiment — 3 move(s), newest first

when              version  move                   who      on the strength of
----------------  -------  ---------------------  -------  -------------------------------------
2026-08-27 11:26  v2       none → production      denis    29 fixed / 11 broke, accuracy +0.0360
2026-08-27 11:26  v1       production → archived  denis    superseded by v2
2026-08-20 09:03  v1       none → production      denis    first one live
```

Three things are deliberate there. The demotion is logged too, so the history
reads as a history rather than as a list of winners. `--check` writes its
verdict into the row, because a promotion made on a comparison and one made on a
hunch look identical a month later unless the comparison was written down — and
a move nobody checked says **not checked** rather than showing a blank, which is
the most useful thing a promotion log can tell you. And the actor is the local
user, `git`-style; set `MLANGO_ACTOR` to override it, which is what a CI job
should do, since the runner's account is nobody.

From Python the same log is `mlango.metastore.history`:

```python
from mlango.metastore.history import history, stage_at

history("reviews.Sentiment")                       # moves, newest first
stage_at("reviews.Sentiment", when=last_tuesday)   # what was live then
```

`stage_at` replays the log rather than reading the version rows, because the
version rows only know about now — which is exactly the wrong thing to ask when
something broke last Tuesday.

### Models mlango did not train

The comparison does not care where the two models came from — it needs two
things that can `predict` and a dataset to score them on. So you can point it at
artefacts you already have, without a `Model` class and without adopting
anything:

```bash
python manage.py diff --dataset reviews.Reviews \
  --left  models/sentiment-v3.joblib \
  --right models/sentiment-v4.joblib
```

The dataset is required, because a saved model carries neither the rows to score
it on nor the column that holds the answer. If you do not have one declared yet,
`manage.py inspectdata data/rows.csv` writes it from a file.

| Flag | Effect |
|---|---|
| `--left URI`, `--right URI` | The two models. A path, or `scheme:reference` |
| `--task` | `classification` (default) or `regression` |
| `--target` | Column to score against. Defaults to the dataset's declared target |
| `--features` | Comma-separated inputs. Defaults to every field but the target and primary key |

A plain path is loaded with joblib, falling back to pickle. Other schemes come
from packages registering under the `mlango.loaders` entry-point group:

```toml
[project.entry-points."mlango.loaders"]
mlflow = "my_package.loaders:load_mlflow_model"
```

The function takes the part after the scheme — `models:/Sentiment/3` for
`mlflow:models:/Sentiment/3` — and returns anything with a `predict` method.
Registry clients live in those packages rather than here, because a framework
that installs somebody else's SDK to read one file is not one you want.

Regression models are compared by distance rather than equality — two float
predictions are never equal — so the report gives mean and largest delta, and
counts rows that got closer to the truth against rows that got further away.

Unlabelled data still works: the report then says what changed, and does not
pretend to say what improved.

This one is not in the admin, and that is deliberate: it loads two models and
scores a dataset, which belongs behind a command you chose to run rather than a
page that loads when you click a link.

### Diffing two evaluation runs

An agent has no version number. You change a prompt, a tool description or a
model, re-run the suite, and the only thing that moves is a pass rate — which
hides exactly what an accuracy hides: some of the cases that used to pass now do
not, and they are usually the ones somebody complained about.

The per-case results are already stored, so this joins two runs on `case_id`:

```bash
python manage.py diff --eval support.AnswerQuality
python manage.py diff --eval support.AnswerQuality --runs 7c8f1020 c089b7e6
python manage.py diff --eval support.AnswerQuality --show-changes 20
python manage.py diff --eval support.AnswerQuality --fail-on-regression significant
```

```
support.AnswerQuality 7c8f1020 → c089b7e6 on 120 shared case(s)

  7c8f1020 pass rate    0.8250
  c089b7e6 pass rate    0.8667   +0.0417
  fixed          7 case(s) failing in 7c8f1020
  broke          2 case(s) passing in 7c8f1020
  verdict        7 fixed against 2 broken is not distinguishable from noise (p=0.180)
  reworded       11 case(s) answered differently and still passed
```

With no `--runs`, the two most recent finished runs of that suite are compared.

The eval page in the admin shows the same comparison for the last two runs. Unlike the model diff it costs nothing to render — nothing is loaded and nothing is scored, because `evaluate` already wrote a verdict per case.

The report also says **what changed about the thing being evaluated**. Each run
records the target's configuration — an agent's prompt, model and step limit;
a model's registered version and hyperparameters — so the diff can put a cause
beside the effect:

```
  verdict        a real regression: 50 broken against 0 fixed (p=0.000)

What changed about it
  version        21 → 22
  C              8.0 → 0.01
  max_features   5000 → 1
```

A long value such as a system prompt is reported as changed with both lengths
rather than printed; a page of text in a terminal report helps nobody. When
nothing about the target moved, the report says so — and that is informative in
its own right, because the difference is then the target's own: sampling, a
temperature, a tool that answered differently.

Runs recorded before this existed carry no configuration, and are reported as
unknown rather than as unchanged.


**`reworded`** is the line that only matters for an agent: cases that still pass
but answer differently. For a classifier that is nothing; for something whose
output a person reads, half the product just changed without failing a test.

Cases present in only one of the runs are **named, never absorbed**. A suite
that grew between the two runs is a different suite, and quietly folding the new
cases into the totals is how a pass rate improves by adding easy questions.

| Flag | Effect |
|---|---|
| `--runs OLDER NEWER` | Which two runs. Defaults to the two most recent |
| `--show-changes N` | Print up to N cases, those whose verdict moved first |
| `--json` | Emit the whole report |
| `--fail-on-regression [any\|significant]` | Exit non-zero. Same rule as for models |
| `--alpha P` | Significance level. Default 0.05 |

The verdict line is the same McNemar test a model diff uses, for the same
reason: seven fixed against two broken on a suite of 120 is not evidence, and a
gate that treats it as evidence will be turned off within a month.

### Watching for drift

Whether the input has moved away from what a version was trained on. Reads the
prediction log, which is off until you turn it on — see [Monitoring](monitoring.md).

```bash
python manage.py drift reviews.Sentiment
python manage.py drift reviews.Sentiment --stage production --since 24h
python manage.py drift reviews.Sentiment --against reviews.Incoming
python manage.py drift reviews.Sentiment --since 24h --fail-on significant
```

```
reviews.Sentiment@v4 vs 2841 logged predictions over the last 7d

Column             Kind         PSI     Verdict
-----------------  -----------  ------  -----------
text               text         0.4132  significant
label (predicted)  categorical  0.1801  moderate
```

`--fail-on` exits non-zero, which is what makes this usable from a scheduled job
rather than only from a terminal.

### Evaluation

```bash
python manage.py evaluate support.AnswerQuality
python manage.py evaluate support.AnswerQuality --show-failures
python manage.py evaluate support.AnswerQuality --min-pass-rate 0.9
```

### Agents

```bash
python manage.py agent support.Support                         # interactive
python manage.py agent support.Support "how do I ...?"          # one shot
python manage.py agent support.Support "..." --show-steps       # print tool calls
python manage.py agent support.Support "..." --session user-42  # with memory
```


### Agent versions

A model version is an artifact; an agent's behaviour *is* its declaration, so a
version is the declaration. One is recorded the first time an agent runs, and
again whenever the prompt, the model or any other `Meta` option changes:

```bash
python manage.py agent support.Support --versions
python manage.py agent support.Support --promote 3
python manage.py agent support.Support --promote 3 --stage staging
```

```
Version  Stage       Fingerprint   Tools            Recorded          Current
-------  ----------  ------------  ---------------  ----------------  -------
v3       none        16c5d2295ade  search_docs      2026-08-22 11:09  ←
v2       production  58fa45bab53f  search_docs      2026-08-19 09:22
v1       archived    a1b2c3d4e5f6  search_docs      2026-08-14 17:40
```

The `←` marks the version matching the declaration in front of you. When nothing
is marked, the code has been edited since anything was recorded — what is
written down and what would run have parted company, and the command says so.

Registration is idempotent by fingerprint and resolved once per process, so a
served agent answering a thousand requests writes one row and runs one query.
Every trace records which version answered, so a trace read next month is not
interpreted against today's prompt.

!!! warning "A version pins configuration, not code"
    Tools are callables and live in your source. A recorded version keeps their
    *names*, so a removed tool is visible, but it cannot restore an
    implementation. A registry that claimed otherwise would be lying.

Reverting a prompt records a new version with the earlier fingerprint rather
than reusing the old row: the history is a log of what the declaration was and
when, and "it changed back on Tuesday" is part of that.

### Inspecting what happened

```bash
python manage.py runs list --kind train --status finished -n 20
python manage.py runs show 7c8f1020
python manage.py runs compare 7c8f1020 c089b7e6

python manage.py traces list --agent support.Support
python manage.py traces show a1b2c3d4 -v 2
```

### Development

```bash
python manage.py runserver              # 127.0.0.1:8000
python manage.py runserver 8080
python manage.py runserver 0.0.0.0:8080 --reload
python manage.py runserver --no-admin

python manage.py shell                  # IPython when available
python manage.py shell -c "print(Reviews.objects.count())"

python manage.py test                   # pytest, against a throwaway metastore
python manage.py test -k splits -x
python manage.py test --coverage
```

`manage.py test` points the metastore and artifact store at a temporary
directory for the duration of the run, so a test can never touch real data —
the same idea as Django creating a test database.

## Common flags

Available on every command:

| Flag | Effect |
|---|---|
| `--settings MODULE` | Use a different settings module for this run |
| `-v 0..3` | Quiet, normal, verbose, very verbose |
| `--traceback` | Show the full traceback instead of a message |

## The shell

`manage.py shell` pre-imports every declared object plus a few helpers:

```python
>>> Reviews.objects.filter(label="positive").count()
1284
>>> Sentiment.versions()
[<ModelVersion reviews.Sentiment@v2 stage=production>, ...]
>>> recent_runs(limit=3)
>>> get_trace("a1b2c3d4").spans
>>> apps.summary()
```

## Your own commands

Drop a module in `<app>/management/commands/` and it appears in
`manage.py help` — including one that **overrides a built-in**, which is how a
project customises `train` without forking the framework.

```python title="reviews/management/commands/import_reviews.py"
from mlango.management import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Import reviews from the warehouse."

    def add_arguments(self, parser):
        parser.add_argument("since", help="ISO date to import from.")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, **options):
        rows = fetch_since(options["since"])
        if not rows:
            raise CommandError(f"Nothing to import since {options['since']}.")

        self.table(
            ["id", "subject"],
            [[r["id"], r["subject"]] for r in rows[:10]],
        )
        if options["dry_run"]:
            self.warn("Dry run: nothing written.")
            return
        write(rows)
        self.ok(f"Imported {len(rows)} review(s).")
```

Helpers available on `self`:

| Helper | Prints |
|---|---|
| `self.write(msg, level=1)` | A line, respecting `-v` |
| `self.ok(msg)` / `self.warn(msg)` | Green / yellow |
| `self.stderr(msg)` | To stderr |
| `self.table(headers, rows)` | An aligned table |
| `self.style.bold(...)` etc. | Colour, disabled when output is redirected |

Raise `CommandError` for anything the user should see as a message rather than a
traceback. Set `requires_apps = False` for a command that must run before apps
load, and `requires_settings = False` for one that runs without a project.
