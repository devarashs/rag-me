from types import SimpleNamespace

import pytest

from rag_me.embeddings import EMBEDDING_DIMENSIONS, EmbeddingError, GeminiEmbedder


class FakeModels:
    """Records embed_content calls and answers with one vector per Content."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS, drop_one: bool = False) -> None:
        self.calls: list[dict] = []
        self._dimensions = dimensions
        self._drop_one = drop_one

    def embed_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        texts = [content.parts[0].text for content in contents]
        embeddings = [
            SimpleNamespace(values=[float(len(text))] + [0.0] * (self._dimensions - 1))
            for text in texts
        ]
        if self._drop_one:
            embeddings = embeddings[:-1]
        return SimpleNamespace(embeddings=embeddings)


def make_embedder(models: FakeModels, batch_size: int = 50) -> GeminiEmbedder:
    return GeminiEmbedder(SimpleNamespace(models=models), "gemini-embedding-2", batch_size)


def sent_texts(call: dict) -> list[str]:
    return [content.parts[0].text for content in call["contents"]]


def test_document_format_is_the_retrieval_prefix_gemini_recommends() -> None:
    embedder = make_embedder(FakeModels())

    assert embedder.format_document("Skills - Go", "Arash writes Go.") == (
        "title: Skills - Go | text: Arash writes Go."
    )


@pytest.mark.parametrize(("title", "expected_title"), [("A | B", "A / B"), ("   ", "none")])
def test_document_title_cannot_break_the_prefix(title: str, expected_title: str) -> None:
    embedder = make_embedder(FakeModels())

    assert embedder.format_document(title, "x") == f"title: {expected_title} | text: x"


def test_each_input_is_sent_as_its_own_content_so_vectors_are_not_merged() -> None:
    models = FakeModels()

    vectors = make_embedder(models).embed_documents(["a", "bb", "ccc"])

    assert len(models.calls) == 1
    assert sent_texts(models.calls[0]) == ["a", "bb", "ccc"]
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0]


def test_requests_the_stored_dimension_from_the_configured_model() -> None:
    models = FakeModels()

    make_embedder(models).embed_documents(["a"])

    assert models.calls[0]["model"] == "gemini-embedding-2"
    assert models.calls[0]["config"].output_dimensionality == EMBEDDING_DIMENSIONS


@pytest.mark.parametrize(
    ("input_count", "batch_size", "expected_batch_sizes"),
    [(0, 2, []), (1, 2, [1]), (2, 2, [2]), (5, 2, [2, 2, 1]), (3, 1, [1, 1, 1])],
)
def test_inputs_are_split_into_batches_preserving_order(
    input_count: int, batch_size: int, expected_batch_sizes: list[int]
) -> None:
    models = FakeModels()
    texts = ["x" * (index + 1) for index in range(input_count)]

    vectors = make_embedder(models, batch_size).embed_documents(texts)

    assert [len(call["contents"]) for call in models.calls] == expected_batch_sizes
    assert [vector[0] for vector in vectors] == [float(len(text)) for text in texts]


def test_query_uses_question_answering_prefix() -> None:
    models = FakeModels()

    make_embedder(models).embed_query("Does Arash know Go?")

    assert sent_texts(models.calls[0]) == ["task: question answering | query: Does Arash know Go?"]


def test_missing_embedding_in_response_is_an_error() -> None:
    with pytest.raises(EmbeddingError, match="returned 1 embeddings for 2 inputs"):
        make_embedder(FakeModels(drop_one=True)).embed_documents(["a", "b"])


def test_wrong_vector_size_is_an_error() -> None:
    with pytest.raises(EmbeddingError, match="3072-dimension vector"):
        make_embedder(FakeModels(dimensions=3072)).embed_documents(["a"])


def test_batch_size_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        make_embedder(FakeModels(), batch_size=0)
