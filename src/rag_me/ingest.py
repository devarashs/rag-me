"""Bring the vector store in line with the knowledge base, embedding only what changed.

The flow is plan, then embed, then write:

1. `plan_ingest` (pure) compares each chunk's fingerprint with the stored one.
2. Only new or changed chunks are sent to the embedding API.
3. One transaction upserts them and deletes chunks whose sections are gone.

No database transaction is open while the embedding API is called: those calls
can take seconds and retry on rate limits, and holding a transaction across them
would keep locks for no benefit.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from rag_me.chunking import Chunk
from rag_me.embeddings import Embedder, EmbeddingError
from rag_me.store import ChunkStore, EmbeddedChunk


@dataclass(frozen=True, slots=True)
class PlannedChunk:
    """A chunk with the exact embedding input it produces and that input's fingerprint."""

    chunk: Chunk
    embedding_input: str
    embedding_input_hash: str


@dataclass(frozen=True, slots=True)
class IngestPlan:
    """What an ingest would do, before any embedding or writing happens."""

    new: list[PlannedChunk]
    changed: list[PlannedChunk]
    unchanged: list[PlannedChunk]
    deleted_ids: list[str]

    @property
    def to_embed(self) -> list[PlannedChunk]:
        return self.new + self.changed

    @property
    def keep_ids(self) -> list[str]:
        return [planned.chunk.chunk_id for planned in self.new + self.changed + self.unchanged]


@dataclass(frozen=True, slots=True)
class IngestReport:
    """Outcome of an ingest. `written` is False for a dry run."""

    plan: IngestPlan
    written: bool


def fingerprint_embedding_input(embedding_model: str, embedding_input: str) -> str:
    """SHA-256 over the model name and the exact text it embeds.

    Both are included because changing either one changes the vector: a new
    model must re-embed everything even when no text changed.
    """
    # The NUL separator cannot occur in a model name, so ("a", "bc") and
    # ("ab", "c") can never produce the same digest.
    return hashlib.sha256(f"{embedding_model}\0{embedding_input}".encode()).hexdigest()


def plan_ingest(
    chunks: Sequence[Chunk],
    stored_hashes: dict[str, str],
    embedder: Embedder,
) -> IngestPlan:
    """Sort chunks into new, changed and unchanged, and find stored chunks to delete.

    Args:
        chunks: The full current knowledge base.
        stored_hashes: `{chunk_id: embedding_input_hash}` from the store.
        embedder: Supplies the model name and document formatting. Not called
            over the network.

    Raises:
        ValueError: If `chunks` is empty (that would delete everything) or
            contains a duplicate ID.
    """
    if not chunks:
        raise ValueError("No chunks to ingest; refusing to plan deleting every stored chunk")
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        duplicates = sorted({chunk_id for chunk_id in chunk_ids if chunk_ids.count(chunk_id) > 1})
        raise ValueError(f"Duplicate chunk IDs: {duplicates}")

    new: list[PlannedChunk] = []
    changed: list[PlannedChunk] = []
    unchanged: list[PlannedChunk] = []
    for chunk in chunks:
        embedding_input = embedder.format_document(chunk.title, chunk.body)
        planned = PlannedChunk(
            chunk=chunk,
            embedding_input=embedding_input,
            embedding_input_hash=fingerprint_embedding_input(embedder.model_name, embedding_input),
        )
        stored_hash = stored_hashes.get(chunk.chunk_id)
        if stored_hash is None:
            new.append(planned)
        elif stored_hash != planned.embedding_input_hash:
            changed.append(planned)
        else:
            unchanged.append(planned)

    current_ids = set(chunk_ids)
    deleted_ids = sorted(chunk_id for chunk_id in stored_hashes if chunk_id not in current_ids)
    return IngestPlan(new=new, changed=changed, unchanged=unchanged, deleted_ids=deleted_ids)


def run_ingest(
    chunks: Sequence[Chunk],
    store: ChunkStore,
    embedder: Embedder,
    *,
    dry_run: bool = False,
) -> IngestReport:
    """Plan the ingest, then (unless `dry_run`) embed what changed and write it.

    If embedding fails part-way, nothing is written: the store keeps its previous
    consistent state and the next run retries the same chunks.

    Raises:
        EmbeddingError: If the embedding API returns an unexpected result.
        ValueError: See `plan_ingest`.
    """
    plan = plan_ingest(chunks, store.fetch_embedding_input_hashes(), embedder)
    if dry_run:
        return IngestReport(plan=plan, written=False)

    to_embed = plan.to_embed
    vectors = embedder.embed_documents([planned.embedding_input for planned in to_embed])
    if len(vectors) != len(to_embed):
        raise EmbeddingError(f"Got {len(vectors)} vectors for {len(to_embed)} chunks")

    store.sync(
        upserts=[
            EmbeddedChunk(
                chunk=planned.chunk,
                embedding_model=embedder.model_name,
                embedding_input_hash=planned.embedding_input_hash,
                embedding=vector,
            )
            for planned, vector in zip(to_embed, vectors, strict=True)
        ],
        keep_ids=plan.keep_ids,
    )
    return IngestReport(plan=plan, written=True)
