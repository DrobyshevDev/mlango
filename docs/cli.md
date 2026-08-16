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
