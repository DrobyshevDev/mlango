"""The metastore schema.

These tables are mlango's equivalent of Django's built-in ``auth`` and
``contenttypes``: framework-owned bookkeeping the user never declares but the
whole system depends on. Everything reproducible about a project lives here —
which data version trained which model, with which hyperparameters, producing
which metrics, and what an agent actually said on the way to an answer.

SQLite by default so a new project needs no infrastructure; the same schema
runs on Postgres by changing one setting.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> dt.datetime:
    """Naive UTC — consistent across SQLite and Postgres without adapter games."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def new_uuid() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class RunStatus:
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    KILLED = "killed"

    ALL = (PENDING, RUNNING, FINISHED, FAILED, KILLED)
    TERMINAL = (FINISHED, FAILED, KILLED)


class RunKind:
    TRAIN = "train"
    EVAL = "eval"
    AGENT = "agent"
    SWEEP = "sweep"

    ALL = (TRAIN, EVAL, AGENT, SWEEP)


class Stage:
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

    ALL = (NONE, STAGING, PRODUCTION, ARCHIVED)


# --------------------------------------------------------------------------- #
# Migrations
# --------------------------------------------------------------------------- #


class MigrationRecord(Base):
    __tablename__ = "mlango_migrations"
    __table_args__ = (UniqueConstraint("app", "name", name="uq_migration_app_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    app: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(255))
    applied_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    def __repr__(self) -> str:
        return f"<Migration {self.app}.{self.name}>"


# --------------------------------------------------------------------------- #
# Data & model versions
# --------------------------------------------------------------------------- #


class DatasetVersion(Base):
    """A materialised, content-addressed snapshot of a dataset."""

    __tablename__ = "mlango_dataset_versions"
    __table_args__ = (
        UniqueConstraint("label", "version", name="uq_dataset_label_version"),
        Index("ix_dataset_fingerprint", "fingerprint"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(255), index=True)
    version: Mapped[int] = mapped_column(Integer)
    #: Hash of the declared schema — changes when fields change.
    fingerprint: Mapped[str] = mapped_column(String(64))
    #: Hash of the materialised rows — changes when the data changes.
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    schema: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    pipeline: Mapped[list[Any]] = mapped_column(JSON, default=list)
    path: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)

    runs: Mapped[list[Run]] = relationship(back_populates="dataset_version")

    @property
    def ref(self) -> str:
        return f"{self.label}@v{self.version}"

    def __repr__(self) -> str:
        return f"<DatasetVersion {self.ref} rows={self.row_count}>"


class ModelVersion(Base):
    """A trained artifact, promotable through stages like a Django deployment."""

    __tablename__ = "mlango_model_versions"
    __table_args__ = (
        UniqueConstraint("label", "version", name="uq_model_label_version"),
        Index("ix_model_stage", "label", "stage"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(255), index=True)
    version: Mapped[int] = mapped_column(Integer)
    fingerprint: Mapped[str] = mapped_column(String(64))
    run_id: Mapped[int | None] = mapped_column(ForeignKey("mlango_runs.id"), nullable=True)
    path: Mapped[str | None] = mapped_column(Text, nullable=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: Feature weights, largest first, as the backend reported them at fit time.
    #: Kept on the row rather than recomputed, so explaining a version never
    #: means loading its weights — and an artifact that has been deleted can
    #: still say what it was paying attention to.
    importances: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    #: A summary of the data this version was fitted on, one entry per feature
    #: plus the target. Without it "has the input drifted" has no answer: you
    #: can see what production looks like today and nothing to compare it to,
    #: and by the time anyone asks, the training split is usually gone.
    baseline: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    stage: Mapped[str] = mapped_column(String(32), default=Stage.NONE)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)

    run: Mapped[Run | None] = relationship(back_populates="model_versions")

    @property
    def ref(self) -> str:
        return f"{self.label}@v{self.version}"

    def __repr__(self) -> str:
        return f"<ModelVersion {self.ref} stage={self.stage}>"


class AgentVersion(Base):
    """A recorded state of an agent's declaration, promotable like a model.

    The difference from :class:`ModelVersion` is that there is no artifact. An
    agent's behaviour is its declaration — the prompt, the model, the step
    limit — so the row *is* the version, and there is nothing to save or load
    from storage.

    That also bounds what a version can restore: tools are callables and live
    in code, so a recorded version pins the configuration and never the
    implementation. Saying so is better than a registry that pretends to
    reproduce something it cannot.
    """

    __tablename__ = "mlango_agent_versions"
    __table_args__ = (
        UniqueConstraint("label", "version", name="uq_agent_label_version"),
        Index("ix_agent_stage", "label", "stage"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(255), index=True)
    version: Mapped[int] = mapped_column(Integer)
    #: Hash of the declaration. A new version exists exactly when this changes.
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    #: The Meta options that survive being written down.
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: Tool names at the time, recorded so a missing tool is visible later even
    #: though the code behind it is not.
    tools: Mapped[list[str]] = mapped_column(JSON, default=list)
    stage: Mapped[str] = mapped_column(String(32), default=Stage.NONE)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)

    @property
    def ref(self) -> str:
        return f"{self.label}@v{self.version}"

    def __repr__(self) -> str:
        return f"<AgentVersion {self.ref} stage={self.stage}>"


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


class Run(Base):
    """One execution of training, evaluation or an agent."""

    __tablename__ = "mlango_runs"
    __table_args__ = (
        Index("ix_run_kind_status", "kind", "status"),
        Index("ix_run_target", "target"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uuid: Mapped[str] = mapped_column(String(32), default=new_uuid, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), default="")
    kind: Mapped[str] = mapped_column(String(32), default=RunKind.TRAIN, index=True)
    #: Label of the model / agent / eval this run exercised.
    target: Mapped[str] = mapped_column(String(255), default="")
    status: Mapped[str] = mapped_column(String(32), default=RunStatus.PENDING, index=True)

    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    dataset_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("mlango_dataset_versions.id"), nullable=True
    )

    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    device: Mapped[str] = mapped_column(String(32), default="")
    git_commit: Mapped[str] = mapped_column(String(64), default="")
    git_dirty: Mapped[bool] = mapped_column(default=False)
    host: Mapped[str] = mapped_column(String(255), default="")
    python_version: Mapped[str] = mapped_column(String(32), default="")

    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    notes: Mapped[str] = mapped_column(Text, default="")

    dataset_version: Mapped[DatasetVersion | None] = relationship(back_populates="runs")
    metrics: Mapped[list[Metric]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    model_versions: Mapped[list[ModelVersion]] = relationship(back_populates="run")
    traces: Mapped[list[Trace]] = relationship(back_populates="run")
    eval_results: Mapped[list[EvalResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def short_id(self) -> str:
        return self.uuid[:8]

    @property
    def is_terminal(self) -> bool:
        return self.status in RunStatus.TERMINAL

    def __repr__(self) -> str:
        return f"<Run {self.short_id} {self.kind}:{self.target} {self.status}>"


class Metric(Base):
    """One scalar observation. Many rows per run — indexed for chart queries."""

    __tablename__ = "mlango_metrics"
    __table_args__ = (Index("ix_metric_run_key_step", "run_id", "key", "step"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("mlango_runs.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(128), index=True)
    value: Mapped[float] = mapped_column(Float)
    step: Mapped[int] = mapped_column(Integer, default=0)
    epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    split: Mapped[str] = mapped_column(String(32), default="")
    recorded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    run: Mapped[Run] = relationship(back_populates="metrics")

    def __repr__(self) -> str:
        return f"<Metric {self.key}={self.value} step={self.step}>"


class Artifact(Base):
    """A file produced by a run: checkpoint, plot, report, prediction dump."""

    __tablename__ = "mlango_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("mlango_runs.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(64), default="file")
    path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), default="")
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    run: Mapped[Run] = relationship(back_populates="artifacts")

    def __repr__(self) -> str:
        return f"<Artifact {self.name} ({self.kind})>"


# --------------------------------------------------------------------------- #
# Agent traces
# --------------------------------------------------------------------------- #


class Trace(Base):
    """One agent invocation, start to finish."""

    __tablename__ = "mlango_traces"
    __table_args__ = (Index("ix_trace_agent_started", "agent", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uuid: Mapped[str] = mapped_column(String(32), default=new_uuid, unique=True, index=True)
    agent: Mapped[str] = mapped_column(String(255), index=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("mlango_runs.id"), nullable=True)
    session_id: Mapped[str] = mapped_column(String(64), default="", index=True)

    input: Mapped[str] = mapped_column(Text, default="")
    output: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default=RunStatus.RUNNING, index=True)
    steps: Mapped[int] = mapped_column(Integer, default=0)

    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")

    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped[Run | None] = relationship(back_populates="traces")
    spans: Mapped[list[Span]] = relationship(
        back_populates="trace",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Span.ordering",
    )

    @property
    def short_id(self) -> str:
        return self.uuid[:8]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __repr__(self) -> str:
        return f"<Trace {self.short_id} {self.agent} {self.status}>"


class Span(Base):
    """A single step inside a trace: an LLM call, a tool call, a memory read."""

    __tablename__ = "mlango_spans"
    __table_args__ = (Index("ix_span_trace_order", "trace_id", "ordering"),)

    LLM = "llm"
    TOOL = "tool"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trace_id: Mapped[int] = mapped_column(ForeignKey("mlango_traces.id", ondelete="CASCADE"))
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("mlango_spans.id", ondelete="CASCADE"), nullable=True
    )
    ordering: Mapped[int] = mapped_column(Integer, default=0)

    name: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(32), default=LLM, index=True)
    input: Mapped[Any] = mapped_column(JSON, default=dict)
    output: Mapped[Any] = mapped_column(JSON, default=dict)
    usage: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")

    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    trace: Mapped[Trace] = relationship(back_populates="spans")
    children: Mapped[list[Span]] = relationship(
        back_populates="parent", cascade="all, delete-orphan", passive_deletes=True
    )
    parent: Mapped[Span | None] = relationship(back_populates="children", remote_side="Span.id")

    def __repr__(self) -> str:
        return f"<Span {self.kind}:{self.name}>"


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


class EvalResult(Base):
    """The score for one case in an evaluation run."""

    __tablename__ = "mlango_eval_results"
    __table_args__ = (Index("ix_eval_run_case", "run_id", "case_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("mlango_runs.id", ondelete="CASCADE"))
    eval_label: Mapped[str] = mapped_column(String(255), index=True)
    case_id: Mapped[str] = mapped_column(String(255), default="")

    passed: Mapped[bool | None] = mapped_column(nullable=True)
    scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    inputs: Mapped[Any] = mapped_column(JSON, default=dict)
    output: Mapped[Any] = mapped_column(JSON, default=dict)
    expected: Mapped[Any] = mapped_column(JSON, default=dict)
    trace_uuid: Mapped[str] = mapped_column(String(32), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    run: Mapped[Run] = relationship(back_populates="eval_results")

    def __repr__(self) -> str:
        return f"<EvalResult {self.eval_label}:{self.case_id} passed={self.passed}>"


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #


class StageTransition(Base):
    """One promotion, recorded because ``stage`` only remembers the present.

    The stage on a version row is a mutable column: promoting v3 overwrites
    what v2 was, so a registry that has been in use for a year can say what is
    live now and nothing at all about how it got there. The question a
    post-mortem opens with — what was in production on the fourteenth, and what
    was it promoted on the strength of — had no answer.

    A row per move, demotions included, so the log reads as a history rather
    than as a list of winners. The evidence is kept beside the move because a
    promotion made on a comparison and one made on a hunch are indistinguishable
    a month later unless the comparison was written down.
    """

    __tablename__ = "mlango_stage_transitions"
    __table_args__ = (Index("ix_transition_target", "kind", "label", "at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    #: ``model`` or ``agent``. Both families promote, and both are logged here
    #: rather than in two tables, because "what changed last week" is one
    #: question and answering it should not need a union.
    kind: Mapped[str] = mapped_column(String(16), index=True)
    label: Mapped[str] = mapped_column(String(255), index=True)
    version: Mapped[int] = mapped_column(Integer)
    from_stage: Mapped[str] = mapped_column(String(32))
    to_stage: Mapped[str] = mapped_column(String(32), index=True)
    #: What the comparison said, when one was run: the counts and the verdict.
    #: Null means nobody checked, which is itself worth being able to see.
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    #: Who moved it, when that can be known. An audit trail without an actor
    #: answers half the question, so this follows git's precedent and records
    #: the local user, overridable through ``MLANGO_ACTOR`` and empty when
    #: neither is available.
    actor: Mapped[str] = mapped_column(String(255), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)

    @property
    def ref(self) -> str:
        return f"{self.label}@v{self.version}"

    def __repr__(self) -> str:
        return f"<StageTransition {self.ref} {self.from_stage}→{self.to_stage}>"


class Prediction(Base):
    """One request a deployed version answered.

    Off by default. A prediction log is the only record of what a model was
    asked in production — training data says what it was *expected* to be asked
    — but it is also a copy of user input in a database, so turning it on is a
    decision the project makes rather than one it wakes up with.
    """

    __tablename__ = "mlango_predictions"
    __table_args__ = (
        Index("ix_prediction_label_at", "label", "created_at"),
        Index("ix_prediction_version", "label", "version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(255), index=True)
    version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    inputs: Mapped[Any] = mapped_column(JSON, default=dict)
    output: Mapped[Any] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Shared by every version that answered one request, so a shadow row and
    #: the row it shadowed can be paired. Matching on inputs instead would fuse
    #: two callers who happened to ask the same question.
    request_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)

    @property
    def ref(self) -> str:
        return f"{self.label}@v{self.version}" if self.version else self.label

    def __repr__(self) -> str:
        return f"<Prediction {self.ref} at={self.created_at}>"


ALL_TABLES = (
    MigrationRecord,
    DatasetVersion,
    ModelVersion,
    AgentVersion,
    Run,
    Metric,
    Artifact,
    Trace,
    Span,
    EvalResult,
    Prediction,
    StageTransition,
)

__all__ = [
    "Base",
    "utcnow",
    "new_uuid",
    "RunStatus",
    "RunKind",
    "Stage",
    "MigrationRecord",
    "DatasetVersion",
    "ModelVersion",
    "AgentVersion",
    "Run",
    "Metric",
    "Artifact",
    "Trace",
    "Span",
    "EvalResult",
    "Prediction",
    "StageTransition",
    "ALL_TABLES",
]
