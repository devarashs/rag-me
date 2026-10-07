"""Postgres connections for the app and for Alembic migrations."""

import psycopg
from pgvector.psycopg import register_vector

CONNECT_TIMEOUT_SECONDS = 15


def connect(database_url: str) -> psycopg.Connection:
    """Open a Postgres connection with pgvector's `vector` type registered.

    The connection is in autocommit mode: callers group statements with
    `connection.transaction()`.

    `prepare_threshold=None` disables server-side prepared statements. Neon's
    pooled endpoint runs PgBouncer in transaction mode, where a statement
    prepared on one server connection may not exist on the next.

    The connect timeout is generous because a Neon free-tier compute that has
    scaled to zero takes a moment to wake on the first connection.

    Raises:
        psycopg.OperationalError: If the database is unreachable.
        psycopg.ProgrammingError: If the `vector` extension is not installed yet
            (run `uv run alembic upgrade head`).
    """
    # autocommit=True so every `with connection.transaction():` block is a real,
    # explicitly bounded transaction. Without it psycopg opens an implicit
    # transaction on the first query (register_vector's type lookup), and later
    # transaction blocks silently become savepoints inside it.
    connection = psycopg.connect(
        database_url,
        autocommit=True,
        prepare_threshold=None,
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
    )
    try:
        register_vector(connection)
    except Exception:
        connection.close()
        raise
    return connection


def sqlalchemy_url(database_url: str) -> str:
    """Rewrite a `postgresql://` URL to select SQLAlchemy's psycopg 3 driver.

    Alembic runs on SQLAlchemy, whose bare `postgresql://` means psycopg2, which
    this project does not install.
    """
    for scheme in ("postgresql://", "postgres://"):
        if database_url.startswith(scheme):
            return "postgresql+psycopg://" + database_url.removeprefix(scheme)
    raise ValueError("database URL must start with postgresql:// or postgres://")
