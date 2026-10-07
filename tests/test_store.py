import psycopg
import pytest

from rag_me.chunking import Chunk
from rag_me.embeddings import EMBEDDING_DIMENSIONS
from rag_me.ingest import run_ingest
from rag_me.store import EmbeddedChunk, PostgresChunkStore
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


def make_embedded(chunk_id: str, body: str = "Body.", first_value: float = 1.0) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=Chunk(chunk_id, chunk_id.split("#")[0], "Doc", "Heading", body),
        embedding_model="model",
        embedding_input_hash=f"hash-of-{body}",
        embedding=[first_value] + [0.0] * (EMBEDDING_DIMENSIONS - 1),
    )


def fetch_rows(connection: psycopg.Connection) -> list[tuple]:
    return connection.execute(
        "SELECT chunk_id, body, embedding_input_hash, created_at, updated_at "
        "FROM knowledge_chunks ORDER BY chunk_id"
    ).fetchall()


def test_empty_table_has_no_hashes(migrated_schema_connection) -> None:
    assert PostgresChunkStore(migrated_schema_connection).fetch_embedding_input_hashes() == {}


def test_sync_inserts_rows_with_their_vectors(migrated_schema_connection) -> None:
    store = PostgresChunkStore(migrated_schema_connection)

    result = store.sync(
        [make_embedded("b.md#x"), make_embedded("a.md#y", first_value=2.5)],
        keep_ids=["a.md#y", "b.md#x"],
    )

    assert result.upserted_ids == ["a.md#y", "b.md#x"]
    assert result.deleted_ids == []
    assert store.fetch_embedding_input_hashes() == {
        "a.md#y": "hash-of-Body.",
        "b.md#x": "hash-of-Body.",
    }
    stored = (
        migrated_schema_connection.execute(
            "SELECT embedding FROM knowledge_chunks WHERE chunk_id = 'a.md#y'"
        )
        .fetchone()[0]
        .to_list()
    )
    assert len(stored) == EMBEDDING_DIMENSIONS
    assert stored[0] == pytest.approx(2.5)


def test_sync_updates_existing_rows_and_bumps_updated_at(migrated_schema_connection) -> None:
    store = PostgresChunkStore(migrated_schema_connection)
    store.sync([make_embedded("a.md#x", body="Old.")], keep_ids=["a.md#x"])
    [(_, _, _, created_before, updated_before)] = fetch_rows(migrated_schema_connection)

    store.sync([make_embedded("a.md#x", body="New.")], keep_ids=["a.md#x"])

    [(chunk_id, body, input_hash, created_after, updated_after)] = fetch_rows(
        migrated_schema_connection
    )
    assert (chunk_id, body, input_hash) == ("a.md#x", "New.", "hash-of-New.")
    assert created_after == created_before
    assert updated_after > updated_before


def test_sync_deletes_rows_not_kept_and_reports_them(migrated_schema_connection) -> None:
    store = PostgresChunkStore(migrated_schema_connection)
    store.sync(
        [make_embedded("a.md#keep"), make_embedded("a.md#drop")],
        keep_ids=["a.md#keep", "a.md#drop"],
    )

    result = store.sync([], keep_ids=["a.md#keep"])

    assert result.deleted_ids == ["a.md#drop"]
    assert list(store.fetch_embedding_input_hashes()) == ["a.md#keep"]


def test_sync_refuses_empty_keep_ids(migrated_schema_connection) -> None:
    store = PostgresChunkStore(migrated_schema_connection)
    store.sync([make_embedded("a.md#x")], keep_ids=["a.md#x"])

    with pytest.raises(ValueError, match="refusing"):
        store.sync([], keep_ids=[])

    assert list(store.fetch_embedding_input_hashes()) == ["a.md#x"]


def test_sync_refuses_upsert_that_would_be_deleted(migrated_schema_connection) -> None:
    with pytest.raises(ValueError, match="not in keep_ids"):
        PostgresChunkStore(migrated_schema_connection).sync(
            [make_embedded("a.md#x")], keep_ids=["a.md#other"]
        )


def test_failed_sync_rolls_back_everything(migrated_schema_connection) -> None:
    store = PostgresChunkStore(migrated_schema_connection)
    store.sync([make_embedded("a.md#old")], keep_ids=["a.md#old"])
    wrong_size = EmbeddedChunk(
        chunk=Chunk("a.md#bad", "a.md", "Doc", "Heading", "Body."),
        embedding_model="model",
        embedding_input_hash="h",
        embedding=[1.0, 2.0],  # the column is vector(768)
    )

    with pytest.raises(psycopg.errors.DataException):
        store.sync([make_embedded("a.md#new"), wrong_size], keep_ids=["a.md#new", "a.md#bad"])

    # Neither the valid insert nor the delete of a.md#old survived.
    assert list(store.fetch_embedding_input_hashes()) == ["a.md#old"]


def test_full_ingest_round_trip_against_postgres(migrated_schema_connection) -> None:
    store = PostgresChunkStore(migrated_schema_connection)
    chunks = [Chunk(f"a.md#{name}", "a.md", "Doc", name, f"About {name}.") for name in "xyz"]

    first = run_ingest(chunks, store, FakeEmbedder())
    second_embedder = FakeEmbedder()
    second = run_ingest(chunks[:2], store, second_embedder)

    assert len(first.plan.new) == 3
    assert len(second.plan.unchanged) == 2
    assert second.plan.deleted_ids == ["a.md#z"]
    assert second_embedder.embedded_batches == [[]]
    assert sorted(store.fetch_embedding_input_hashes()) == ["a.md#x", "a.md#y"]
