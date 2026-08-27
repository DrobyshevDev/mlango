"""Reading and writing the log of promotions.

``stage`` on a version row is a mutable column, so it answers what is live and
nothing about how it got there. This is the other half: one row per move, kept
next to the evidence the move was made on.

It lives in the metastore rather than beside either family because both of them
promote and neither may import the other. The same reason the table is one
table: "what changed last week" is a single question, and answering it should
not need a union.
"""

from __future__ import annotations

from typing import Any

__all__ = ["actor_name", "history", "record_transition"]


def actor_name() -> str:
    """Who is promoting, when that can be known.

    An audit trail without an actor answers half the question, so this follows
    git's precedent and records the local user. ``MLANGO_ACTOR`` overrides it,
    which is what a CI job should set — the runner's user account is nobody.
    """
    import getpass
    import os

    named = os.environ.get("MLANGO_ACTOR", "").strip()
    if named:
        return named[:255]
    try:
        return getpass.getuser()[:255]
    except Exception:
        # No passwd entry and no environment to fall back on. Ordinary in a
        # container, and not worth failing a promotion over.
        return ""


def record_transition(
    session: Any,
    *,
    kind: str,
    label: str,
    version: int,
    from_stage: str,
    to_stage: str,
    evidence: dict[str, Any] | None = None,
    notes: str = "",
    actor: str | None = None,
) -> Any:
    """Log one move, in the caller's transaction.

    Taking the session rather than opening one is deliberate: the log entry and
    the stage change have to land together or the history is fiction.
    """
    from mlango.metastore.models import StageTransition

    row = StageTransition(
        kind=kind,
        label=label,
        version=version,
        from_stage=from_stage,
        to_stage=to_stage,
        evidence=evidence,
        actor=actor if actor is not None else actor_name(),
        notes=notes,
    )
    session.add(row)
    return row


def history(
    label: str | None = None,
    *,
    kind: str | None = None,
    stage: str | None = None,
    limit: int = 50,
) -> list[Any]:
    """Moves, newest first.

    With no label this is the whole registry's history, which is the shape the
    question usually arrives in: nobody asks what happened to one model until
    after they have found out that something happened at all.
    """
    from sqlalchemy import select

    from mlango.metastore.models import StageTransition
    from mlango.metastore.session import session_scope

    query = select(StageTransition).order_by(StageTransition.at.desc(), StageTransition.id.desc())
    if label:
        query = query.where(StageTransition.label == label)
    if kind:
        query = query.where(StageTransition.kind == kind)
    if stage:
        query = query.where(StageTransition.to_stage == stage)

    with session_scope() as session:
        return list(session.execute(query.limit(limit)).scalars())


def stage_at(label: str, when: Any, *, stage: str = "production") -> int | None:
    """Which version held a stage at a given moment, or None if nothing did.

    The question a post-mortem opens with. It is answered by replaying the log
    rather than by reading the version rows, because the version rows only know
    about now.
    """
    from sqlalchemy import select

    from mlango.metastore.models import StageTransition
    from mlango.metastore.session import session_scope

    with session_scope() as session:
        rows = list(
            session.execute(
                select(StageTransition)
                .where(StageTransition.label == label, StageTransition.at <= when)
                .order_by(StageTransition.at.asc(), StageTransition.id.asc())
            ).scalars()
        )

    holder: int | None = None
    for row in rows:
        if row.to_stage == stage:
            holder = row.version
        elif row.version == holder and row.from_stage == stage:
            # Moved out of the stage without anything replacing it, which is
            # what a demotion or an archive looks like on its own.
            holder = None
    return holder
