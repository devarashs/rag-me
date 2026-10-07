"""Read and write embedded chunks in Postgres (`knowledge_chunks`, see migrations/)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import psycopg
from pgvector import Vector

from rag_me.chunking import Chunk


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A chunk together with its embedding and the fingerprint of what was embedded."""

    chunk: Chunk
    embedding_model: str
    embedding_input_hash: str
    embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class SyncResult:
    """What `ChunkStore.sync` changed."""

    upserted_ids: list[str]
    deleted_ids: list[str]


class ChunkStore(Protocol):
    """What ingest needs from storage. `PostgresChunkStore` is the real one."""

    def fetch_embedding_input_hashes(self) -> dict[str, str]:
        """Return `{chunk_id: embedding_input_hash}` for every stored chunk."""
        ...

    def sync(self, upserts: Sequence[EmbeddedChunk], keep_ids: Sequence[str]) -> SyncResult:
        """Insert or update `upserts`, delete every chunk not in `keep_ids`, atomically."""
        ...


class PostgresChunkStore:
    """`ChunkStore` on a psycopg connection with pgvector registered.

    Args:
        connection: From `rag_me.database.connect`. The caller owns and closes it.
    """

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    def fetch_embedding_input_hashes(self) -> dict[str, str]:
        # Unbounded by design: the result is one short row per knowledge base
        # section, and ingest needs all of them to diff against the files.
        with self._connection.transaction():
            rows = self._connection.execute(
                "SELECT chunk_id, embedding_input_hash FROM knowledge_chunks"
            ).fetchall()
        return {chunk_id: input_hash for chunk_id, input_hash in rows}

    def sync(self, upserts: Sequence[EmbeddedChunk], keep_ids: Sequence[str]) -> SyncResult:
        """Write all changes in one transaction, so readers never see a half-ingest.

        Deletion is computed here, against the table as it is now, rather than
        from an earlier read: rows inserted since that read are still judged
        against `keep_ids`.

        Args:
            upserts: Chunks to insert or overwrite, matched by `chunk_id`.
            keep_ids: Every chunk ID that should exist afterwards. Must not be
                empty, which would delete the whole table.

        Raises:
            ValueError: If `keep_ids` is empty, or an upsert is not in `keep_ids`.
        """
        if not keep_ids:
            raise ValueError("keep_ids is empty; refusing to delete every stored chunk")
        keep_id_set = set(keep_ids)
        stray_ids = [
            item.chunk.chunk_id for item in upserts if item.chunk.chunk_id not in keep_id_set
        ]
        if stray_ids:
            raise ValueError(f"upserts not in keep_ids would be deleted again: {stray_ids}")

        # Writing rows in primary-key order keeps lock acquisition order the same
        # for every writer, so two concurrent ingests cannot deadlock each other.
        ordered_upserts = sorted(upserts, key=lambda item: item.chunk.chunk_id)
        with self._connection.transaction(), self._connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO knowledge_chunks (
                    chunk_id, source_path, document_title, section_heading, body,
                    embedding_model, embedding_input_hash, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    source_path = EXCLUDED.source_path,
                    document_title = EXCLUDED.document_title,
                    section_heading = EXCLUDED.section_heading,
                    body = EXCLUDED.body,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_input_hash = EXCLUDED.embedding_input_hash,
                    embedding = EXCLUDED.embedding,
                    updated_at = now()
                """,
                [
                    (
                        item.chunk.chunk_id,
                        item.chunk.source_path,
                        item.chunk.document_title,
                        item.chunk.section_heading,
                        item.chunk.body,
                        item.embedding_model,
                        item.embedding_input_hash,
                        Vector(list(item.embedding)),
                    )
                    for item in ordered_upserts
                ],
            )
            deleted_rows = cursor.execute(
                """
                DELETE FROM knowledge_chunks
                WHERE NOT (chunk_id = ANY(%s))
                RETURNING chunk_id
                """,
                [list(keep_ids)],
            ).fetchall()
        return SyncResult(
            upserted_ids=[item.chunk.chunk_id for item in ordered_upserts],
            deleted_ids=sorted(chunk_id for (chunk_id,) in deleted_rows),
        )
