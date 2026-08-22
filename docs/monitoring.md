# Monitoring

A model does not fail loudly when the world moves. It keeps returning confident
answers about data it has never seen, and the metric that would catch it —
accuracy against labels — does not exist in production, because the labels are
the thing you are waiting for.

What is available on the first request is the input. Its distribution moving is
the earliest honest signal that a model is being asked questions it was not
trained for.

## The two halves

Training already records what it fitted on. Every registered version carries a
profile of its training split:

```python
Sentiment.load()._version.baseline
# {'text': {'kind': 'text', 'mean': 37.5, 'edges': [...], 'counts': [...]},
#  'label': {'kind': 'categorical', 'values': {'neg': 162, 'pos': 155}}}
```

Serving records the other half, when you turn it on:

```python title="myproject/settings.py"
PREDICTION_LOG = {
    "ENABLED": True,
    "SAMPLE": 0.05,        # 5% is plenty to see a distribution move
    "MAX_ROWS": 100_000,   # oldest rows are trimmed past this
}
```

Off by default on purpose. A prediction log is a copy of user input in a
database, which is a decision a project makes rather than one it wakes up with.
Sampling is per row, not per request, so a batch endpoint records a fraction of
every batch rather than all of some and none of others.

## Measuring the gap

```bash
python manage.py drift reviews.Sentiment
python manage.py drift reviews.Sentiment --stage production --since 24h
python manage.py drift reviews.Sentiment --against reviews.Incoming
```

```
reviews.Sentiment@v4 vs 2841 logged predictions over the last 7d

Column             Kind         PSI     Verdict
-----------------  -----------  ------  -----------
text               text         0.4132  significant
label (predicted)  categorical  0.1801  moderate

PSI below 0.1 is stable, 0.1–0.25 moderate, above 0.25 significant.
A column has moved significantly. Retraining is likely overdue.
```

Two things are compared, and the second matters most when the first cannot say
much:

- **Input drift** — each feature against the same feature at training time.
- **Prediction drift** — what the model is saying against the labels it was
  trained on. A model whose output used to split evenly and now returns 90% one
  class is worth looking at, and this needs no ground truth at all.

| Flag | Effect |
|---|---|
| `--version N` / `--stage NAME` | Which version's profile to compare against |
| `--since 24h\|7d\|4w` | Window of logged predictions. Default `7d` |
| `--against LABEL` | Compare a dataset instead of the log — for a batch you are about to score |
| `-n N` | Cap the rows read. Default 10,000 |
| `--json` | Emit the scores |
| `--fail-on moderate\|significant` | Exit non-zero. For a scheduled job |

The admin's model page shows the same table for the last seven days, and says
nothing at all when the log is off — an empty drift table on every page teaches
people to ignore the drift table.

## What PSI is

The population stability index, standard in credit risk for exactly this job:

```
PSI = Σ (actual_i − expected_i) · ln(actual_i / expected_i)
```

over buckets `i`, as proportions of their totals. One number per column,
comparable across columns of different types, with
thresholds nobody has to invent: below 0.1 stable, 0.1–0.25 moderate, above 0.25
significant.

How a column is bucketed depends on what it holds, decided once at training time:

| Kind | Bucketing | Notes |
|---|---|---|
| Numeric | Deciles of the training split | Quantiles, not equal widths — equal widths put a skewed column in one bucket and detect nothing |
| Categorical | One bucket per value | Values unseen at training get their own |
| Text | Deciles of the *length* | A crude proxy, and an honest one: free-text values never repeat, so their frequencies say nothing |

Observed data is always read as the kind the baseline decided on. Inferring
twice looks harmless and is not: production text that has collapsed to a handful
of repeated values would be read as categorical, and the comparison would return
nothing at exactly the moment worth catching.

## In a scheduled job

`--fail-on` makes drift a check rather than a report:

```yaml
- name: Check for drift
  run: python manage.py drift reviews.Sentiment --since 24h --fail-on significant
```

## OpenTelemetry

mlango keeps its own record of every run, trace and span, and the admin reads
that. This is the bridge for organisations that already have somewhere to look:

```bash
pip install "mlango[otel]"
```

```python title="myproject/settings.py"
TELEMETRY = {"ENABLED": True, "SERVICE_NAME": "reviews-api"}
```

Training runs, agent loops and every tool call are then emitted as spans, with
mlango's own attributes namespaced (`mlango.target`, `mlango.run`,
`mlango.status`, `mlango.step`) so they are findable in a view holding spans
from a dozen libraries. A training run started by an HTTP request appears as a
child of that request, which is the whole point.

**mlango configures no exporter.** No endpoint, no sampler, no headers. The
process does that, the way every other OpenTelemetry-instrumented library
expects — usually a few lines at start-up or the `opentelemetry-instrument`
launcher. Owning that configuration would mean owning a second, worse copy of
the SDK's own settings.

**It cannot fail your work.** A collector that is down, an exporter that raises,
a tracer that explodes: all of it is logged at debug and the run continues.
Telemetry describes the work; it does not get to fail it.

Nothing is emitted with the setting off, and the dependency is optional — with
`opentelemetry-api` absent the setting warns once and every span is a no-op.

## What this is not

It is not a monitoring product, and it does not try to be. There are no alerts,
no dashboards over time, no anomaly detection. It answers one question — has the
input moved away from what this version was fitted on — from data the framework
already had, with no extra infrastructure. When you outgrow it, the prediction
log is an ordinary table you can ship anywhere.
