"""Add a full-text search column to knowledge_chunks, for hybrid search.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-09
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A generated column: Postgres computes it from the row's text on every
    # insert and update, so ingest needs no change and it can never go stale.
    # 'english' stems words ("used" -> "use") and drops stopwords ("has").
    op.execute(
        """
        ALTER TABLE knowledge_chunks
        ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (
            to_tsvector('english', document_title || ' ' || section_heading || ' ' || body)
        ) STORED
        """
    )
    # Serves the `@@` pre-filter in keyword search, so only sections containing
    # at least one question term are scored once the table grows.
    op.execute(
        "CREATE INDEX knowledge_chunks_search_vector_idx "
        "ON knowledge_chunks USING gin (search_vector)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX knowledge_chunks_search_vector_idx")
    op.execute("ALTER TABLE knowledge_chunks DROP COLUMN search_vector")
