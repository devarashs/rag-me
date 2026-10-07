"""In-memory stand-ins for the embedding model and the chunk store."""

from collections.abc import Sequence

from rag_me.embeddings import EMBEDDING_DIMENSIONS
from rag_me.store import EmbeddedChunk, SyncResult


class FakeEmbedder:
    """Deterministic `Embedder`: each vector encodes its input's length, so tests
    can check which vector was stored for which chunk."""

    def __init__(self, model_name: str = "fake-embedding-model") -> None:
        self.model_name = model_name
        self.embedded_batches: list[list[str]] = []

    def format_document(self, title: str, text: str) -> str:
        return f"{title}::{text}"

    def embed_documents(self, formatted_documents: Sequence[str]) -> list[list[float]]:
        self.embedded_batches.append(list(formatted_documents))
        return [vector_for(text) for text in formatted_documents]

    def embed_query(self, question: str) -> list[float]:
        return vector_for(question)


def vector_for(text: str) -> list[float]:
    return [float(len(text))] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


class FakeChunkStore:
    """`ChunkStore` backed by a dict of `{chunk_id: EmbeddedChunk}`."""

    def __init__(self, rows: dict[str, EmbeddedChunk] | None = None) -> None:
        self.rows: dict[str, EmbeddedChunk] = dict(rows or {})
        self.sync_calls = 0

    def fetch_embedding_input_hashes(self) -> dict[str, str]:
        return {chunk_id: row.embedding_input_hash for chunk_id, row in self.rows.items()}

    def sync(self, upserts: Sequence[EmbeddedChunk], keep_ids: Sequence[str]) -> SyncResult:
        self.sync_calls += 1
        for item in upserts:
            self.rows[item.chunk.chunk_id] = item
        keep = set(keep_ids)
        deleted = sorted(chunk_id for chunk_id in self.rows if chunk_id not in keep)
        for chunk_id in deleted:
            del self.rows[chunk_id]
        return SyncResult(
            upserted_ids=sorted(item.chunk.chunk_id for item in upserts), deleted_ids=deleted
        )
