"""Create rate_limit_counters: fixed-window request counts per visitor and globally.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-07
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # bucket_key is either "global" or "visitor:<HMAC of the client IP>". Raw IPs
    # are never stored; see rag_me.rate_limit.
    #
    # One row per bucket per window. Rows are deleted once their window is long
    # past, so the table stays at roughly one row per recent visitor.
    op.execute(
        """
        CREATE TABLE rate_limit_counters (
            bucket_key    text NOT NULL,
            window_start  timestamptz NOT NULL,
            request_count integer NOT NULL CHECK (request_count > 0),
            PRIMARY KEY (bucket_key, window_start)
        )
        """
    )
    # Serves the expired-window cleanup, which filters on window_start alone.
    op.execute(
        "CREATE INDEX rate_limit_counters_window_start_idx ON rate_limit_counters (window_start)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE rate_limit_counters")
