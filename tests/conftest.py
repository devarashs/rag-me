"""Shared fixtures. Integration tests run against the real DATABASE_URL, inside a
throwaway schema created per test and dropped afterwards, so they never touch the
app's own `public.knowledge_chunks` table."""

import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from dotenv import dotenv_values
from pgvector.psycopg import register_vector
from sqlalchemy import create_engine, pool, text

from rag_me.database import sqlalchemy_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def direct_database_url() -> str:
    """DATABASE_URL from the environment or .env, pointed at Neon's direct endpoint.

    Tests set a per-session `search_path`. Neon's `-pooler` host is PgBouncer in
    transaction mode, which does not keep session settings between transactions,
    so tests connect to the direct host instead (same name without `-pooler`).
    A non-Neon URL is used unchanged.
    """
    import os

    database_url = os.environ.get("DATABASE_URL") or dotenv_values(PROJECT_ROOT / ".env").get(
        "DATABASE_URL"
    )
    if not database_url:
        pytest.skip("integration test needs DATABASE_URL (environment or .env)")
    return database_url.replace("-pooler.", ".", 1)


@pytest.fixture
def migrated_schema_connection() -> Iterator[psycopg.Connection]:
    """An autocommit connection whose search_path is a fresh, fully migrated schema."""
    database_url = direct_database_url()
    schema = f"test_{uuid.uuid4().hex[:12]}"

    engine = create_engine(sqlalchemy_url(database_url), poolclass=pool.NullPool)
    with engine.begin() as setup_connection:
        setup_connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        # `public` stays on the path so the `vector` type, installed there by the
        # extension, still resolves.
        setup_connection.execute(text(f'SET search_path TO "{schema}", public'))
        alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
        alembic_config.attributes["connection"] = setup_connection
        alembic_config.attributes["version_table_schema"] = schema
        command.upgrade(alembic_config, "head")

    connection = psycopg.connect(database_url, autocommit=True)
    try:
        connection.execute(f'SET search_path TO "{schema}", public')
        assert_table_resolves_to_schema(connection, "knowledge_chunks", schema)
        register_vector(connection)
        yield connection
    finally:
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        connection.close()
        engine.dispose()


def assert_table_resolves_to_schema(
    connection: psycopg.Connection, table_name: str, schema: str
) -> None:
    """Stop the test unless the unqualified table name resolves into `schema`.

    Tests delete rows. If the throwaway schema lacked the table, the name would
    fall through the search_path to `public` and the test would delete real data,
    which is exactly what happened once (2026-10-07) before the version-table fix.
    """
    row = connection.execute(
        """
        SELECT n.nspname
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.oid = to_regclass(%s)
        """,
        [table_name],
    ).fetchone()
    resolved_schema = row[0] if row else None
    if resolved_schema != schema:
        pytest.fail(
            f"Refusing to run: {table_name} resolves to schema {resolved_schema!r}, "
            f"not the throwaway schema {schema!r}"
        )
