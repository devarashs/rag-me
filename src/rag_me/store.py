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
class RetrievedChunk:
    """A stored chunk returned by a similarity search.

    Attributes:
        chunk: The stored section.
        similarity: Cosine similarity to the query, from -1 to 1; higher is closer.
    """

    chunk: Chunk
    similarity: float


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


class ChunkSearcher(Protocol):
    """What answering needs from storage. `PostgresChunkStore` is the real one."""

    def search(
        self, question: str, query_embedding: Sequence[float], limit: int
    ) -> list[RetrievedChunk]:
        """Return up to `limit` chunks, best first, each with its cosine similarity."""
        ...


MAX_SEARCH_LIMIT = 50

# Hybrid search tuning. Standard values, not fitted to this corpus:
# - BM25 k1 and b are the defaults used by Lucene and Elasticsearch.
# - RRF k = 60 is the constant from the original reciprocal rank fusion paper
#   (Cormack, Clarke and Buettcher, 2009); it keeps one list's top rank from
#   drowning out the other list.
# - Each search contributes its best 20 candidates to the fusion.
# - Question terms found in more than half of all sections are ignored: they say
#   nothing about which section is relevant ("Arash" is in nearly every one).
_BM25_K1 = 1.2
_BM25_B = 0.75
_RRF_K = 60
_CANDIDATES_PER_SEARCH = 20
_MAX_TERM_DOCUMENT_FRACTION = 0.5

# One round trip: vector top-N and BM25 keyword top-N, fused by reciprocal rank,
# returned with every result's real cosine similarity (keyword-only results
# included) so the relevance gate keeps working on vector scores.
#
# Postgres full-text search has no built-in BM25 (ts_rank ignores how common a
# word is), so BM25 is computed here from the tsvector's own term positions.
_HYBRID_SEARCH_SQL = """
WITH question_terms AS (
    SELECT DISTINCT term.lexeme
    FROM unnest(to_tsvector('english', %(question)s)) AS term(lexeme, positions, weights)
),
question_query AS (
    -- Lexemes are already stemmed, so the 'simple' config must not stem again.
    SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')) AS tsquery
    FROM question_terms
),
corpus AS (
    SELECT count(*) AS document_count, avg(length(search_vector)) AS average_length
    FROM knowledge_chunks
),
postings AS (
    SELECT chunk.chunk_id,
           term.lexeme,
           coalesce(array_length(term.positions, 1), 1) AS term_frequency,
           length(chunk.search_vector) AS document_length
    FROM knowledge_chunks AS chunk
    CROSS JOIN question_query
    CROSS JOIN LATERAL unnest(chunk.search_vector) AS term(lexeme, positions, weights)
    WHERE chunk.search_vector @@ question_query.tsquery
      AND term.lexeme IN (SELECT lexeme FROM question_terms)
),
document_frequency AS (
    SELECT lexeme, count(*) AS documents FROM postings GROUP BY lexeme
),
keyword_scores AS (
    SELECT postings.chunk_id,
           sum(
               ln(1 + (corpus.document_count - document_frequency.documents + 0.5)
                      / (document_frequency.documents + 0.5))
               * postings.term_frequency * (%(k1)s + 1)
               / (postings.term_frequency
                  + %(k1)s * (1 - %(b)s + %(b)s * postings.document_length
                                              / corpus.average_length))
           ) AS bm25
    FROM postings
    JOIN document_frequency USING (lexeme)
    CROSS JOIN corpus
    WHERE document_frequency.documents <= corpus.document_count * %(max_term_fraction)s
    GROUP BY postings.chunk_id
),
keyword_ranked AS (
    SELECT chunk_id, row_number() OVER (ORDER BY bm25 DESC, chunk_id) AS keyword_rank
    FROM keyword_scores
    ORDER BY bm25 DESC, chunk_id
    LIMIT %(candidates)s
),
vector_ranked AS (
    SELECT chunk_id,
           row_number() OVER (ORDER BY embedding <=> %(query)s, chunk_id) AS vector_rank
    FROM knowledge_chunks
    ORDER BY embedding <=> %(query)s, chunk_id
    LIMIT %(candidates)s
),
fused AS (
    SELECT chunk_id,
           coalesce(1.0 / (%(rrf_k)s + vector_ranked.vector_rank), 0)
           + coalesce(1.0 / (%(rrf_k)s + keyword_ranked.keyword_rank), 0) AS fused_score
    FROM vector_ranked
    FULL OUTER JOIN keyword_ranked USING (chunk_id)
)
SELECT chunk.chunk_id, chunk.source_path, chunk.document_title, chunk.section_heading,
       chunk.body, 1 - (chunk.embedding <=> %(query)s) AS similarity
FROM fused
JOIN knowledge_chunks AS chunk USING (chunk_id)
ORDER BY fused.fused_score DESC, similarity DESC, chunk.chunk_id
LIMIT %(limit)s
"""

_VECTOR_SEARCH_SQL = """
SELECT chunk_id, source_path, document_title, section_heading, body,
       1 - (embedding <=> %(query)s) AS similarity
FROM knowledge_chunks
ORDER BY embedding <=> %(query)s, chunk_id
LIMIT %(limit)s
"""


class PostgresChunkStore:
    """`ChunkStore` and `ChunkSearcher` on a psycopg connection with pgvector registered.

    Args:
        connection: From `rag_me.database.connect`. The caller owns and closes it.
        hybrid_search: Combine keyword (BM25) and vector search; False is
            vector only, kept so the two can be compared on the eval suite.
    """

    def __init__(self, connection: psycopg.Connection, *, hybrid_search: bool = True) -> None:
        self._connection = connection
        self._hybrid_search = hybrid_search

    def fetch_embedding_input_hashes(self) -> dict[str, str]:
        # Unbounded by design: the result is one short row per knowledge base
        # section, and ingest needs all of them to diff against the files.
        with self._connection.transaction():
            rows = self._connection.execute(
                "SELECT chunk_id, embedding_input_hash FROM knowledge_chunks"
            ).fetchall()
        return {chunk_id: input_hash for chunk_id, input_hash in rows}

    def search(
        self, question: str, query_embedding: Sequence[float], limit: int
    ) -> list[RetrievedChunk]:
        """Return the `limit` best chunks for a question.

        Hybrid (default): vector and BM25 keyword rankings fused by reciprocal
        rank, so an exact term mentioned once (a tool name, say) still finds its
        section even when the section's embedding is dominated by other topics.
        Vector only: cosine distance alone.

        Every result carries its cosine similarity to `query_embedding`;
        `<=>` is pgvector's cosine distance, so similarity is `1 - distance`.
        Results are in ranking order, which in hybrid mode is not similarity order.

        Raises:
            ValueError: If `limit` is outside 1..MAX_SEARCH_LIMIT.
        """
        if not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_SEARCH_LIMIT}, got {limit}")
        parameters: dict[str, object] = {"query": Vector(list(query_embedding)), "limit": limit}
        if self._hybrid_search:
            sql = _HYBRID_SEARCH_SQL
            parameters |= {
                "question": question,
                "k1": _BM25_K1,
                "b": _BM25_B,
                "rrf_k": _RRF_K,
                "candidates": _CANDIDATES_PER_SEARCH,
                "max_term_fraction": _MAX_TERM_DOCUMENT_FRACTION,
            }
        else:
            sql = _VECTOR_SEARCH_SQL
        rows = self._connection.execute(sql, parameters).fetchall()
        return [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id=chunk_id,
                    source_path=source_path,
                    document_title=document_title,
                    section_heading=section_heading,
                    body=body,
                ),
                similarity=float(similarity),
            )
            for chunk_id, source_path, document_title, section_heading, body, similarity in rows
        ]

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
