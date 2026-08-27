# Continuous integration

A diff read in a terminal is a decision one person made. Promoting a model is
usually not one person's decision, and the place the others are is a pull
request — so the comparison has to be able to go there, on a machine with no
terminal and nobody watching.

Two flags do that. `--fail-on-regression` turns the comparison into an exit
code, and `--format markdown` turns it into something a person can read where
they already are.

## The report, as a comment

```bash
python manage.py diff reviews.Sentiment --format markdown --show-changes 20
```

```markdown
<!-- mlango:diff:model:reviews.Sentiment -->

### ⚠️ reviews.Sentiment v1 → v2

**11 rows broken**, 29 fixed, over 500 rows of `reviews.Reviews`.

| | v1 → v2 |
|---|---:|
| `accuracy` | 0.7700 → **0.8060** (+0.0360) |
| agreement | 92.0% |
| changed | 40 rows |
| fixed | 29 rows |
| broke | **11 rows** |

Movement: `pos → neg` 22 · `neg → pos` 18

> a real improvement: 29 fixed against 11 broken (p=0.006)

<details>
<summary>20 of 40 rows where they disagree</summary>
…
</details>
```

The broken count is in the heading and the first sentence, because a pull
request comment is usually first read as a notification and never opened. The
rows are folded away for the opposite reason: whoever *has* opened it wants the
evidence, and a comment two hundred rows tall is one nobody scrolls past.

It renders the same four comparisons the command already makes — registered
versions, evaluation runs, two files mlango never trained, and live shadow
traffic — because it is a function of the report rather than of the command.

| Flag | Effect |
|---|---|
| `--format text\|markdown\|json` | How to render it. `text` is the default |
| `--output PATH` | Write it to a file. The exit code is unaffected |
| `--show-changes N` | Include up to N disagreeing rows |

`--json` is the older spelling of `--format json` and still means what it meant.

## A GitHub Actions workflow

```yaml
name: model

on: pull_request

permissions:
  contents: read
  pull-requests: write

jobs:
  diff:
    runs-on: ubuntu-latest
    env:
      DATABASE_URL: ${{ secrets.DATABASE_URL }}
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - run: pip install -r requirements.txt

      - name: Train the candidate
        run: python manage.py train reviews.Sentiment -v 0

      - name: Compare it with what is live
        id: diff
        continue-on-error: true
        run: |
          python manage.py diff reviews.Sentiment \
            --format markdown --show-changes 20 \
            --output diff.md \
            --fail-on-regression significant

      - name: Say so in the pull request
        if: always()
        env:
          GH_TOKEN: ${{ github.token }}
          NUMBER: ${{ github.event.number }}
        run: gh pr comment "$NUMBER" --body-file diff.md --edit-last --create-if-none

      - name: Fail if it regressed
        if: steps.diff.outcome == 'failure'
        run: exit 1
```

Three things in there are deliberate.

**`continue-on-error` then an explicit fail.** The comparison has to be posted
whether or not it passed — a job that goes red without saying why is the thing
this is trying to replace. So the diff is allowed to fail, the comment is
posted, and the failure is re-raised afterwards.

**`--edit-last --create-if-none`** replaces the previous comment instead of
adding one, so a branch with nine pushes has one report rather than a thread
nobody reads.

**`significant`, not the bare flag.** On real data a strictly-better version is
a fiction: something always regresses. The bare flag fails on a single lost row,
which is the right rule for a curated suite and the wrong one here.
`significant` fails only when the losses outweigh the gains by more than chance
— see [Diffing two versions](cli.md#diffing-two-versions).

### More than one model in a job

`--edit-last` finds the last comment the job's own account wrote, which is the
wrong one as soon as a second model reports. Every rendered report opens with a
marker naming what it compared:

```
<!-- mlango:diff:model:reviews.Sentiment -->
```

It is stable across runs and unique per model, so a job comparing several can
find each report's own comment and update that one.

## What CI has to be able to see

The comparison reads the registry. A fresh checkout has no registry in it, so a
job that trains and diffs against `sqlite:///mlango.db` is comparing the
candidate with nothing.

Point the runner at the metastore where promotions actually happen:

```python
METASTORE = {"URL": os.environ.get("DATABASE_URL", "sqlite:///mlango.db")}
```

Postgres is the ordinary answer. If your registry is a SQLite file that lives
somewhere, fetching it in a step before the diff works too — read-only is
enough, because the diff registers nothing.

## GitLab CI

The same commands, and the same argument for splitting the failure from the
report:

```yaml
model-diff:
  image: python:3.12
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  script:
    - pip install -r requirements.txt
    - python manage.py train reviews.Sentiment -v 0
    - |
      python manage.py diff reviews.Sentiment \
        --format markdown --show-changes 20 \
        --output diff.md --fail-on-regression significant
  artifacts:
    when: always
    paths:
      - diff.md
```

`when: always` is the same idea as `continue-on-error`: the report is worth
keeping precisely when the job failed.

## Evaluations, too

An agent has no version number, so the pair to compare is two runs of its
evaluation suite. Everything above applies unchanged:

```bash
python manage.py evaluate support.AnswerQuality
python manage.py diff --eval support.AnswerQuality \
    --format markdown --output diff.md --fail-on-regression significant
```

That is a prompt change gated the way a promotion is, and the report names what
you changed — the prompt, the model, the step limit — beside what it did.
