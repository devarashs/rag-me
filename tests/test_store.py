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


# --- search ---------------------------------------------------------------------


def direction(x: float, y: float) -> list[float]:
    """A vector in the first two dimensions, zero elsewhere."""
    return [x, y] + [0.0] * (EMBEDDING_DIMENSIONS - 2)


def embedded_with_vector(chunk_id: str, vector: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=Chunk(chunk_id, chunk_id.split("#")[0], "Doc", "Heading", f"Body of {chunk_id}."),
        embedding_model="model",
        embedding_input_hash="h",
        embedding=vector,
    )


@pytest.fixture
def store_with_three_directions(migrated_schema_connection) -> PostgresChunkStore:
    store = PostgresChunkStore(migrated_schema_connection)
    rows = {
        "a.md#same": direction(1.0, 0.0),
        "a.md#diagonal": direction(1.0, 1.0),
        "a.md#opposite": direction(-1.0, 0.0),
    }
    store.sync([embedded_with_vector(i, v) for i, v in rows.items()], keep_ids=list(rows))
    return store


def test_search_orders_by_cosine_similarity(store_with_three_directions) -> None:
    results = store_with_three_directions.search("", direction(1.0, 0.0), limit=3)

    assert [result.chunk.chunk_id for result in results] == [
        "a.md#same",
        "a.md#diagonal",
        "a.md#opposite",
    ]
    assert [result.similarity for result in results] == pytest.approx([1.0, 0.7071, -1.0], abs=1e-3)


def test_search_returns_full_chunk_content(store_with_three_directions) -> None:
    [top] = store_with_three_directions.search("", direction(1.0, 0.0), limit=1)

    assert top.chunk == Chunk("a.md#same", "a.md", "Doc", "Heading", "Body of a.md#same.")


def test_search_similarity_ignores_vector_length(store_with_three_directions) -> None:
    [top] = store_with_three_directions.search("", direction(5.0, 0.0), limit=1)

    assert top.similarity == pytest.approx(1.0, abs=1e-6)


def test_search_limit_caps_results(store_with_three_directions) -> None:
    assert len(store_with_three_directions.search("", direction(1.0, 0.0), limit=2)) == 2
    assert len(store_with_three_directions.search("", direction(1.0, 0.0), limit=50)) == 3


@pytest.mark.parametrize("limit", [0, -1, 51])
def test_search_rejects_out_of_range_limit(migrated_schema_connection, limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be between 1 and 50"):
        PostgresChunkStore(migrated_schema_connection).search("", direction(1.0, 0.0), limit=limit)


def test_search_on_empty_table_returns_nothing(migrated_schema_connection) -> None:
    assert (
        PostgresChunkStore(migrated_schema_connection).search("", direction(1.0, 0.0), limit=5)
        == []
    )


# --- hybrid search ----------------------------------------------------------------


def chunk_with(chunk_id: str, body: str, vector: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=Chunk(chunk_id, chunk_id.split("#")[0], "Doc", chunk_id.split("#")[1], body),
        embedding_model="model",
        embedding_input_hash="h",
        embedding=vector,
    )


@pytest.fixture
def corpus_with_one_rare_term(migrated_schema_connection) -> psycopg.Connection:
    """Six sections that all mention Arash. Only the launcher one names Sentry, and
    its vector points away from the query, as a multi-topic section's would."""
    rows = [
        chunk_with("a.md#faq-start", "Arash can start immediately.", direction(1.0, 0.0)),
        chunk_with("a.md#faq-go", "Arash writes Go for networking tools.", direction(0.98, 0.2)),
        chunk_with("a.md#faq-ai", "Arash built AI products with agents.", direction(0.95, 0.3)),
        chunk_with("a.md#faq-tests", "Arash wrote thousands of tests.", direction(0.9, 0.4)),
        chunk_with("a.md#faq-mcp", "Arash built an MCP server for agents.", direction(0.85, 0.5)),
        chunk_with(
            "b.md#launcher",
            "Arash built the Electron launcher with auto-updates, code signing, delta "
            "patching and content-addressed uploads, and added Sentry for crash reporting.",
            direction(0.0, 1.0),
        ),
    ]
    PostgresChunkStore(migrated_schema_connection).sync(
        rows, keep_ids=[r.chunk.chunk_id for r in rows]
    )
    return migrated_schema_connection


def test_hybrid_search_finds_a_term_mentioned_once(corpus_with_one_rare_term) -> None:
    store = PostgresChunkStore(corpus_with_one_rare_term)

    results = store.search("Has Arash used Sentry?", direction(1.0, 0.0), limit=3)

    assert "b.md#launcher" in [r.chunk.chunk_id for r in results]


def test_vector_only_search_misses_it(corpus_with_one_rare_term) -> None:
    store = PostgresChunkStore(corpus_with_one_rare_term, hybrid_search=False)

    results = store.search("Has Arash used Sentry?", direction(1.0, 0.0), limit=3)

    assert "b.md#launcher" not in [r.chunk.chunk_id for r in results]


def test_keyword_only_result_reports_its_true_cosine_similarity(corpus_with_one_rare_term) -> None:
    results = PostgresChunkStore(corpus_with_one_rare_term).search(
        "Has Arash used Sentry?", direction(1.0, 0.0), limit=3
    )

    launcher = next(r for r in results if r.chunk.chunk_id == "b.md#launcher")
    assert launcher.similarity == pytest.approx(0.0, abs=1e-6)  # orthogonal to the query


@pytest.mark.parametrize(
    "question",
    [
        "Arash",  # in every section: ignored as a keyword
        "",  # no terms
        "Has the?",  # stopwords only
        "What's O'Reilly's & Arash's | take?",  # quotes and tsquery operators
    ],
)
def test_questions_without_useful_terms_fall_back_to_vector_order(
    corpus_with_one_rare_term, question: str
) -> None:
    hybrid = PostgresChunkStore(corpus_with_one_rare_term).search(
        question, direction(1.0, 0.0), limit=6
    )
    vector = PostgresChunkStore(corpus_with_one_rare_term, hybrid_search=False).search(
        question, direction(1.0, 0.0), limit=6
    )

    assert [r.chunk.chunk_id for r in hybrid] == [r.chunk.chunk_id for r in vector]


def test_hybrid_results_are_unique_and_capped(corpus_with_one_rare_term) -> None:
    results = PostgresChunkStore(corpus_with_one_rare_term).search(
        "Arash Sentry launcher agents tests", direction(1.0, 0.0), limit=4
    )

    chunk_ids = [r.chunk.chunk_id for r in results]
    assert len(chunk_ids) == 4
    assert len(set(chunk_ids)) == 4
