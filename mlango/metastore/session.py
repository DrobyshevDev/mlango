"""Engine and session management for the metastore.

The engine is created lazily from ``settings.METASTORE`` and cached per URL, so
importing this module has no side effects and tests can swap the database by
reconfiguring settings.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from mlango.metastore.models import Base

logger = logging.getLogger("mlango.metastore")

#: Guards the create-then-record below. Held only on the first touch of a
#: URL, so it costs one uncontended acquisition per process afterwards.
_schema_lock = threading.Lock()

_engines: dict[str, Engine] = {}
_sessionmakers: dict[str, sessionmaker[Session]] = {}
_ensured: set[str] = set()


def metastore_url() -> str:
    from mlango.conf import settings

    url = str(settings.METASTORE.get("URL", "sqlite:///mlango.db"))
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        # Resolve relative SQLite paths against BASE_DIR so the database does
        # not follow the shell's working directory around.
        rel = url[len("sqlite:///") :]
        if rel and rel != ":memory:" and not os.path.isabs(rel):
            url = "sqlite:///" + os.path.join(str(settings.BASE_DIR), rel).replace("\\", "/")
    return url


def get_engine(url: str | None = None) -> Engine:
    from mlango.conf import settings

    url = url or metastore_url()
    if url in _engines:
        return _engines[url]

    options: dict[str, Any] = {
        "echo": bool(settings.METASTORE.get("ECHO", False)),
        "future": True,
    }
    if url.startswith("sqlite"):
        # A training loop writing metrics while the admin reads them is the
        # normal case, so allow cross-thread use and turn on WAL below.
        options["connect_args"] = {"check_same_thread": False}
    else:
        options["pool_pre_ping"] = bool(settings.METASTORE.get("POOL_PRE_PING", True))

    engine = create_engine(url, **options)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    _engines[url] = engine
    _sessionmakers[url] = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return engine


def get_sessionmaker(url: str | None = None) -> sessionmaker[Session]:
    url = url or metastore_url()
    if url not in _sessionmakers:
        get_engine(url)
    return _sessionmakers[url]


def ensure_schema(url: str | None = None) -> None:
    """Create the metastore tables once per process.

    These tables are framework-owned and never change shape at a user's
    request, so there is nothing to be gained by making people run ``migrate``
    before their first ``materialize()``. Declarative migrations are a separate
    concern and still explicit.

    They do change shape when mlango itself is upgraded, which is why
    :func:`align_schema` runs here too.
    """
    url = url or metastore_url()
    if url in _ensured:
        return

    # Checked, created, recorded — with nothing holding the gap, two threads
    # reaching an untouched metastore together both decide to create it and the
    # loser gets "table already exists". A threaded server answering its first
    # two requests at once is the ordinary way to meet this.
    with _schema_lock:
        if url in _ensured:
            return
        create_all(url)
        align_schema(url)
        _ensured.add(url)


def new_session(url: str | None = None) -> Session:
    """A session the caller owns and must close."""
    ensure_schema(url)
    return get_sessionmaker(url)()


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error."""
    session = new_session(url)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all(url: str | None = None) -> None:
    """Create every metastore table that does not exist yet."""
    Base.metadata.create_all(get_engine(url))


def align_schema(url: str | None = None) -> list[str]:
    """Add columns a newer mlango declares that an existing database lacks.

    ``create_all`` creates missing *tables* and ignores missing *columns*, so
    upgrading mlango against a metastore written by an older version failed at
    the first query with ``no such column``. The choice then was to delete the
    database — throwing away every run, metric and registered version — for a
    column that was only ever appended.

    Only additive changes are handled. Anything narrower — a rename, a type
    change, a dropped column — is a real migration and stays out of an implicit
    startup path. Returns the ``table.column`` names that were added, so
    ``check`` can report them.
    """
    from sqlalchemy import inspect, text

    engine = get_engine(url)
    inspector = inspect(engine)
    present = set(inspector.get_table_names())
    added: list[str] = []

    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            if table.name not in present:
                continue
            known = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in known:
                    continue
                clause = _add_column_clause(column, engine.dialect)
                if clause is None:
                    logger.warning(
                        "Metastore column %s.%s is missing and cannot be added "
                        "automatically: it is NOT NULL and its default is computed "
                        "per row, so existing rows have no value to take. Migrate "
                        "it by hand, or recreate the metastore.",
                        table.name,
                        column.name,
                    )
                    continue
                connection.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {clause}"))
                added.append(f"{table.name}.{column.name}")

    if added:
        logger.info("Metastore schema aligned: added %s", ", ".join(added))
    return added


def _add_column_clause(column: Any, dialect: Any) -> str | None:
    """The ``ADD COLUMN`` body for a new column, or None if it cannot be added.

    A NOT NULL column needs a value for the rows that already exist. A constant
    default supplies one — that is what a hand-written migration would do — and
    both SQLite and Postgres accept it in a single statement. A default that is
    computed per row (``utcnow``, ``dict``) has no single value to backfill
    with, so it is refused rather than guessed at.
    """
    from sqlalchemy import literal

    body = f'"{column.name}" {column.type.compile(dialect)}'
    if column.nullable:
        return body

    default = column.server_default
    if default is not None:
        return f"{body} NOT NULL DEFAULT {default.arg.text if hasattr(default.arg, 'text') else default.arg}"

    default = column.default
    if default is None or default.is_callable or default.is_clause_element:
        return None

    rendered = literal(default.arg, column.type).compile(
        dialect=dialect, compile_kwargs={"literal_binds": True}
    )
    return f"{body} NOT NULL DEFAULT {rendered}"


def drop_all(url: str | None = None) -> None:
    """Drop every metastore table. Destructive; used by ``flush`` and tests."""
    Base.metadata.drop_all(get_engine(url))


def dispose_all() -> None:
    """Close every pooled connection — call before deleting a SQLite file."""
    for engine in _engines.values():
        engine.dispose()
    _engines.clear()
    _sessionmakers.clear()
    _ensured.clear()


def table_names(url: str | None = None) -> list[str]:
    from sqlalchemy import inspect

    return sorted(inspect(get_engine(url)).get_table_names())


def metastore_ready(url: str | None = None) -> bool:
    """True when the core tables exist — i.e. ``migrate`` has been run."""
    try:
        return "mlango_runs" in table_names(url)
    except Exception:
        return False
