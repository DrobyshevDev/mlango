# mlango and the alternatives

You almost certainly already have tools for this. This page is here to help you
decide whether mlango replaces any of them, sits alongside them, or is the wrong
choice for what you are doing.

## The one-sentence difference

Most ML tools solve **one** of these: tracking, orchestration, data versioning,
serving, or agents. mlango's bet is that the problem is the **seam between
them** — and that the way to close a seam is a shared declaration.

```python
class Sentiment(Model):
    C = fields.FloatField(default=1.0, tunable=True)

    class Meta:
        dataset = Reviews
        trainer = "sklearn"
```

That class body is simultaneously: a page in the admin, a `POST /api/predict/`
endpoint with an OpenAPI schema, a row in a migration, a target for
`manage.py train`, a search space for `manage.py sweep`, and a versioned entry in
a model registry. You did not wire any of it up. That is the Django move —
declare once, and everything generic reads the declaration — applied to ML.

The second difference: **agents are a first-class family beside models.** An
LLM agent and a gradient-boosted tree are both declared classes with a `_meta`,
both tracked in the same metastore, both visible in the same admin, both scored
by the same `Eval`. Almost every tool in this space is either an ML tool or an
LLM tool. Most teams in 2026 are doing both.

## Where each tool actually sits

| Tool | What it is | Overlap with mlango |
|---|---|---|
| **MLflow** | Experiment tracking, a model registry, and packaging | Real overlap on tracking and the registry. No project structure, no admin over *your data*, no declarative classes, no agents |
| **Weights & Biases** | Hosted tracking with an excellent UI | Overlaps on tracking. Hosted, not a framework; it does not tell you how to lay out a project |
| **Kedro** | Project structure and pipelines for data science | The closest in spirit. Node-and-pipeline shaped rather than declarative-class shaped; no admin, no serving, no agents |
| **ZenML / Metaflow / Flyte** | Pipeline orchestration, usually cloud-first | Little overlap. They run steps across infrastructure; mlango declares objects and runs locally by default |
| **Airflow / Dagster / Prefect** | General orchestration and scheduling | No overlap. They schedule work; mlango is work that gets scheduled |
| **DVC** | Data and model versioning on top of git | Complementary. mlango versions data by content hash into its own store; DVC does it in git's model |
| **LangChain / LlamaIndex** | Building LLM applications | Overlaps on the agent loop and tools. They have no training, no dataset versioning, no model registry, no admin |
| **FastAPI** | The web framework | None — mlango *uses* it. Your endpoints are generated from declarations rather than written |

## Honest answers to the obvious questions

### "Why not just MLflow?"

If tracking is the only thing you are missing, MLflow is less to adopt and has a
far larger ecosystem. Use it.

mlango is worth looking at when the tracking gap is not the real problem — when
the real problem is that every project on the team is laid out differently,
nobody can tell which dataset version produced a number, serving is a separate
hand-written service, and the LLM work lives in a repository that shares nothing
with the ML work. Those are structural problems, and a tracking library does not
solve structural problems.

You can also run both: nothing stops you logging to MLflow from a callback.

### "Why not Kedro?"

Kedro is the most similar thing here, and if you like pipelines-of-nodes it is a
good answer. The difference is what the unit of declaration is. In Kedro you
declare a pipeline; in mlango you declare an *object* — a dataset, a model, an
agent, an eval — and the framework derives the pipeline, the admin page, the API
and the migration from it.

The practical consequence: mlango gives you a working admin and a documented API
without writing either, and Kedro does not aim to. In return, Kedro's pipeline
model is more expressive for complex multi-step data engineering, which mlango
deliberately does not try to be.

### "Why not just scikit-learn and some scripts?"

For one model that one person maintains, scripts are fine and a framework is
overhead. The cost of scripts shows up on the second model, the second person,
and the first time somebody asks which data produced the number in the deck.

### "Do I have to use the agent half?"

No. The four families are independent. A project that declares only datasets and
models never imports the agent code, and the agent extras are not installed
unless you ask for them.

### "Do I have to use the ML half?"

Also no. mlango is a reasonable way to build an agent application on its own:
you get tool schemas from type hints, tracing into the admin, memory backends,
declarative evals for agent quality, and an SSE endpoint. What you additionally
get, if you ever train a model, is that it lands in the same system.

## When mlango is the wrong choice

Written plainly, because a framework that claims to fit everything fits nothing:

- **You need distributed training across many machines.** mlango runs a training
  loop; it is not a scheduler and has no cluster story. Use Ray, or your
  cloud's training service, and let mlango track the result if you like.
- **Your problem is a large data-engineering DAG.** That is Dagster or Airflow
  territory. mlango's queryset is for data a model reads, not for a warehouse
  pipeline.
- **You need a mature plugin ecosystem today.** mlango is 0.3.0. MLflow has a
  decade and hundreds of integrations. That gap is real and will take time.
- **You only need a hosted dashboard for a small team.** W&B will make you
  happier faster.
- **You are already productive.** If your current setup is not hurting, a
  framework is a cost with no benefit yet.

## What mlango takes from Django on purpose

Worth naming, because it explains several decisions that would otherwise look
arbitrary:

| Django | mlango | Why |
|---|---|---|
| `models.py` | `datasets.py`, `models.py`, `agents.py`, `evals.py` | Autodiscovered per app, so declaring is enough |
| `ModelAdmin` | `ObjectAdmin` | Everything declared appears without registration; register to customise |
| Migrations | The same, for dataset schemas | A schema change six months ago should be readable today |
| `manage.py` | The same, extensible by apps | Your project's commands sit beside the framework's |
| `settings.py` | The same, with unknown settings as errors | A typo should fail, not silently do nothing |
| Third-party apps | `mlango-*` packages | The ecosystem is the point; see [Contributing](contributing.md) |

And what it deliberately does not take: there is no ORM. A `Dataset` describes
records that live in files, a warehouse or the Hugging Face hub — mlango does
not want to own your data, only to know its shape.
