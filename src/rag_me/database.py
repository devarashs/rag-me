"""Postgres connections for the app and for Alembic migrations."""

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

CONNECT_TIMEOUT_SECONDS = 15

# Settings shared by single connections and pooled ones; see `connect` for why.
_CONNECTION_OPTIONS = {
    "autocommit": True,
    "prepare_threshold": None,
    "connect_timeout": CONNECT_TIMEOUT_SECONDS,
}


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
    connection = psycopg.connect(database_url, **_CONNECTION_OPTIONS)
    try:
        register_vector(connection)
    except Exception:
        connection.close()
        raise
    return connection


def create_connection_pool(database_url: str, max_size: int = 4) -> ConnectionPool:
    """Create a closed connection pool for the HTTP API; call `.open()` to start it.

    `min_size=0` keeps no idle connections open, so Neon's free-tier compute can
    still scale to zero between visitors. Connections are checked before use
    because Neon drops idle ones when it suspends, and a pooled connection
    from before the suspension would otherwise fail on first query.
    """
    return ConnectionPool(
        database_url,
        min_size=0,
        max_size=max_size,
        kwargs=_CONNECTION_OPTIONS,
        configure=register_vector,
        check=ConnectionPool.check_connection,
        timeout=CONNECT_TIMEOUT_SECONDS,
        open=False,
    )


def sqlalchemy_url(database_url: str) -> str:
    """Rewrite a `postgresql://` URL to select SQLAlchemy's psycopg 3 driver.

    Alembic runs on SQLAlchemy, whose bare `postgresql://` means psycopg2, which
    this project does not install.
    """
    for scheme in ("postgresql://", "postgres://"):
        if database_url.startswith(scheme):
            return "postgresql+psycopg://" + database_url.removeprefix(scheme)
    raise ValueError("database URL must start with postgresql:// or postgres://")
