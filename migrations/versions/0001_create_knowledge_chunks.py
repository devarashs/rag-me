"""Create knowledge_chunks: one row per knowledge base section, with its embedding.

Revision ID: 0001
Revises:
Create Date: 2026-10-07
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # embedding_input_hash fingerprints the exact text sent to the embedding model
    # plus the model name, so ingest re-embeds a row only when either changes.
    #
    # vector(768) must match rag_me.embeddings.EMBEDDING_DIMENSIONS.
    #
    # No approximate-nearest-neighbour index (HNSW or IVFFlat) on purpose: at
    # ~100 rows an exact sequential scan takes well under a millisecond and is
    # exact. Add HNSW with vector_cosine_ops once the table reaches thousands of rows.
    op.execute(
        """
        CREATE TABLE knowledge_chunks (
            chunk_id             text PRIMARY KEY,
            source_path          text NOT NULL,
            document_title       text NOT NULL,
            section_heading      text NOT NULL,
            body                 text NOT NULL,
            embedding_model      text NOT NULL,
            embedding_input_hash text NOT NULL,
            embedding            vector(768) NOT NULL,
            created_at           timestamptz NOT NULL DEFAULT now(),
            updated_at           timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    # The vector extension is left installed: other schemas may use it, and
    # dropping an extension is not this migration's call.
    op.execute("DROP TABLE knowledge_chunks")
