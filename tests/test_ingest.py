import pytest

from rag_me.chunking import Chunk
from rag_me.embeddings import EmbeddingError
from rag_me.ingest import fingerprint_embedding_input, plan_ingest, run_ingest
from tests.fakes import FakeChunkStore, FakeEmbedder, vector_for


def make_chunk(chunk_id: str, body: str = "Body.", heading: str = "Heading") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_path=chunk_id.split("#")[0],
        document_title="Doc",
        section_heading=heading,
        body=body,
    )


def ingest_into(store: FakeChunkStore, chunks: list[Chunk], embedder: FakeEmbedder | None = None):
    return run_ingest(chunks, store, embedder or FakeEmbedder())


# --- fingerprint_embedding_input ------------------------------------------------


def test_fingerprint_is_stable_for_same_model_and_input() -> None:
    assert fingerprint_embedding_input("m", "text") == fingerprint_embedding_input("m", "text")


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("model-a", "text"), ("model-b", "text")),
        (("model", "text"), ("model", "text changed")),
        # Shifting characters across the boundary must not collide.
        (("ab", "c"), ("a", "bc")),
    ],
)
def test_fingerprint_differs_when_model_or_input_differs(first, second) -> None:
    assert fingerprint_embedding_input(*first) != fingerprint_embedding_input(*second)


# --- plan_ingest ----------------------------------------------------------------


def test_everything_is_new_against_an_empty_store() -> None:
    chunks = [make_chunk("a.md#one"), make_chunk("a.md#two")]

    plan = plan_ingest(chunks, {}, FakeEmbedder())

    assert [p.chunk.chunk_id for p in plan.new] == ["a.md#one", "a.md#two"]
    assert plan.changed == plan.unchanged == plan.deleted_ids == []


def test_plan_classifies_new_changed_unchanged_and_deleted() -> None:
    embedder = FakeEmbedder()
    store = FakeChunkStore()
    ingest_into(store, [make_chunk("a.md#same"), make_chunk("a.md#edit"), make_chunk("a.md#gone")])

    plan = plan_ingest(
        [make_chunk("a.md#same"), make_chunk("a.md#edit", body="Edited."), make_chunk("b.md#new")],
        store.fetch_embedding_input_hashes(),
        embedder,
    )

    assert [p.chunk.chunk_id for p in plan.new] == ["b.md#new"]
    assert [p.chunk.chunk_id for p in plan.changed] == ["a.md#edit"]
    assert [p.chunk.chunk_id for p in plan.unchanged] == ["a.md#same"]
    assert plan.deleted_ids == ["a.md#gone"]
    assert sorted(plan.keep_ids) == ["a.md#edit", "a.md#same", "b.md#new"]


def test_heading_change_alone_marks_chunk_changed() -> None:
    store = FakeChunkStore()
    ingest_into(store, [make_chunk("a.md#x", heading="Old heading")])

    plan = plan_ingest(
        [make_chunk("a.md#x", heading="New heading")],
        store.fetch_embedding_input_hashes(),
        FakeEmbedder(),
    )

    assert [p.chunk.chunk_id for p in plan.changed] == ["a.md#x"]


def test_switching_embedding_model_marks_every_chunk_changed() -> None:
    chunks = [make_chunk("a.md#one"), make_chunk("a.md#two")]
    store = FakeChunkStore()
    ingest_into(store, chunks, FakeEmbedder("model-v1"))

    plan = plan_ingest(chunks, store.fetch_embedding_input_hashes(), FakeEmbedder("model-v2"))

    assert len(plan.changed) == 2
    assert plan.unchanged == []


def test_planned_input_uses_the_embedders_document_format() -> None:
    plan = plan_ingest([make_chunk("a.md#x", body="Text.")], {}, FakeEmbedder())

    assert plan.new[0].embedding_input == "Doc - Heading::Text."


def test_empty_chunk_list_is_refused_rather_than_deleting_everything() -> None:
    with pytest.raises(ValueError, match="refusing"):
        plan_ingest([], {"a.md#x": "hash"}, FakeEmbedder())


def test_duplicate_chunk_ids_are_refused() -> None:
    with pytest.raises(ValueError, match=r"Duplicate chunk IDs: \['a.md#x'\]"):
        plan_ingest([make_chunk("a.md#x"), make_chunk("a.md#x")], {}, FakeEmbedder())


# --- run_ingest -----------------------------------------------------------------


def test_first_ingest_embeds_and_stores_every_chunk_with_its_own_vector() -> None:
    store = FakeChunkStore()
    embedder = FakeEmbedder()
    chunks = [make_chunk("a.md#short", body="x"), make_chunk("a.md#long", body="much longer")]

    report = ingest_into(store, chunks, embedder)

    assert report.written
    assert sorted(store.rows) == ["a.md#long", "a.md#short"]
    for chunk in chunks:
        row = store.rows[chunk.chunk_id]
        expected_input = embedder.format_document(chunk.title, chunk.body)
        assert list(row.embedding) == vector_for(expected_input)
        assert row.embedding_model == "fake-embedding-model"
        assert row.embedding_input_hash == fingerprint_embedding_input(
            "fake-embedding-model", expected_input
        )


def test_rerun_without_changes_embeds_nothing() -> None:
    store = FakeChunkStore()
    chunks = [make_chunk("a.md#one"), make_chunk("a.md#two")]
    ingest_into(store, chunks)
    embedder = FakeEmbedder()

    report = ingest_into(store, chunks, embedder)

    assert embedder.embedded_batches == [[]]
    assert len(report.plan.unchanged) == 2


def test_rerun_embeds_only_changed_chunks_and_deletes_removed_ones() -> None:
    store = FakeChunkStore()
    ingest_into(store, [make_chunk("a.md#keep"), make_chunk("a.md#edit"), make_chunk("a.md#drop")])
    embedder = FakeEmbedder()

    ingest_into(store, [make_chunk("a.md#keep"), make_chunk("a.md#edit", body="New.")], embedder)

    assert embedder.embedded_batches == [["Doc - Heading::New."]]
    assert sorted(store.rows) == ["a.md#edit", "a.md#keep"]
    assert store.rows["a.md#edit"].chunk.body == "New."


def test_dry_run_reports_the_plan_without_embedding_or_writing() -> None:
    store = FakeChunkStore()
    embedder = FakeEmbedder()

    report = run_ingest([make_chunk("a.md#x")], store, embedder, dry_run=True)

    assert not report.written
    assert [p.chunk.chunk_id for p in report.plan.new] == ["a.md#x"]
    assert embedder.embedded_batches == []
    assert store.sync_calls == 0
    assert store.rows == {}


class FailingEmbedder(FakeEmbedder):
    def embed_documents(self, formatted_documents):
        raise EmbeddingError("simulated failure")


class ShortChangingEmbedder(FakeEmbedder):
    def embed_documents(self, formatted_documents):
        return super().embed_documents(formatted_documents)[:-1]


@pytest.mark.parametrize("embedder", [FailingEmbedder(), ShortChangingEmbedder()])
def test_embedding_failure_leaves_the_store_untouched(embedder: FakeEmbedder) -> None:
    store = FakeChunkStore()
    ingest_into(store, [make_chunk("a.md#old")])
    rows_before = dict(store.rows)

    with pytest.raises(EmbeddingError):
        run_ingest(
            [make_chunk("a.md#old", body="Changed."), make_chunk("a.md#new")], store, embedder
        )

    assert store.rows == rows_before
    assert store.sync_calls == 1  # only the setup ingest
