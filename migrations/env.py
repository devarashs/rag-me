"""Alembic environment: where migrations get their database connection.

Migrations are hand-written SQL (`op.execute`), because the app uses psycopg
directly and has no SQLAlchemy models for Alembic to diff against.

Two ways in:
- From the command line (`uv run alembic upgrade head`): connects to
  DATABASE_URL from the environment or `.env`.
- From tests: a caller passes an open SQLAlchemy connection in
  `config.attributes["connection"]`, so migrations run inside a throwaway schema.
"""

from alembic import context
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import Connection

from rag_me.config import load_settings
from rag_me.database import sqlalchemy_url


def run_migrations(connection: Connection) -> None:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    provided_connection = context.config.attributes.get("connection")
    if provided_connection is not None:
        run_migrations(provided_connection)
        return

    database_url = load_settings().database_url.get_secret_value()
    # NullPool: a migration run is one short-lived connection, not a server.
    engine = create_engine(
        sqlalchemy_url(database_url),
        poolclass=pool.NullPool,
        # Same reason as rag_me.database.connect: Neon's pooled endpoint is
        # PgBouncer in transaction mode, which breaks prepared statements.
        connect_args={"prepare_threshold": None},
    )
    with engine.connect() as connection:
        run_migrations(connection)


if context.is_offline_mode():
    raise SystemExit("Offline (--sql) migrations are not supported; run against a database.")
run_migrations_online()
