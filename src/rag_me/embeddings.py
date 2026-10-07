"""Turn text into embedding vectors with Gemini.

Two Gemini-specific behaviours shape this module, both verified against the live
API and both silent when gotten wrong:

1. **`gemini-embedding-2` merges a plain list of strings into ONE embedding.**
   Each input must be wrapped in its own `Content` to get one vector per input.
   Passing a bare list would give every chunk the same vector, and retrieval
   would quietly return garbage.
2. **The model has no `task_type` setting.** Retrieval quality instead depends on
   a text prefix: documents as `title: ... | text: ...`, queries as
   `task: question answering | query: ...`. Formatting therefore belongs to the
   embedder, not the chunker, and changing it changes every stored vector, which
   is why ingest fingerprints the formatted input rather than the raw chunk.
"""

from collections.abc import Sequence
from typing import Protocol

from google import genai
from google.genai import types

# Stored vectors are vector(768) in the database (see migrations/). 768 is one of
# the dimensions Google recommends for gemini-embedding-2, which returns them
# already normalised. Changing this needs a migration and a full re-ingest.
EMBEDDING_DIMENSIONS = 768

# Attempts per request including the first. The SDK retries 429 and 5xx with
# exponential backoff, which is what the free tier's per-minute limit needs.
_REQUEST_ATTEMPTS = 5


class EmbeddingError(Exception):
    """Raised when the embedding API returns something other than one vector of
    the expected size per input."""


class Embedder(Protocol):
    """What ingest and retrieval need from an embedding model."""

    model_name: str

    def format_document(self, title: str, text: str) -> str:
        """Return the exact text to embed for one document."""
        ...

    def embed_documents(self, formatted_documents: Sequence[str]) -> list[list[float]]:
        """Embed already-formatted documents, returning one vector per input, in order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Format and embed one search query."""
        ...


class GeminiEmbedder:
    """`Embedder` backed by the Gemini API.

    Args:
        client: A configured `google.genai.Client`. Injected so tests can pass a
            fake; build a real one with `create_gemini_client`.
        model_name: Embedding model ID, e.g. `gemini-embedding-2`.
        batch_size: Maximum documents per API request.

    Example:
        >>> embedder = GeminiEmbedder(create_gemini_client(key), "gemini-embedding-2", 50)
        >>> vectors = embedder.embed_documents([embedder.format_document("Skills", "Go")])
    """

    def __init__(self, client: genai.Client, model_name: str, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self._client = client
        self.model_name = model_name
        self._batch_size = batch_size

    def format_document(self, title: str, text: str) -> str:
        # `|` separates the prefix fields, so a `|` inside the title would blur
        # where the title ends. Titles come from headings, so replace it there.
        safe_title = title.replace("|", "/").strip() or "none"
        return f"title: {safe_title} | text: {text}"

    def embed_documents(self, formatted_documents: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(formatted_documents), self._batch_size):
            batch = formatted_documents[start : start + self._batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        return self._embed_batch([f"task: question answering | query: {question}"])[0]

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self.model_name,
            # One Content per input; a bare list of strings would be merged into
            # a single embedding (see the module docstring).
            contents=[types.Content(parts=[types.Part(text=text)]) for text in texts],
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
        )
        embeddings = response.embeddings or []
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"{self.model_name} returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )
        vectors = [list(embedding.values or []) for embedding in embeddings]
        for vector in vectors:
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise EmbeddingError(
                    f"{self.model_name} returned a {len(vector)}-dimension vector; "
                    f"expected {EMBEDDING_DIMENSIONS}"
                )
        return vectors


def create_gemini_client(api_key: str) -> genai.Client:
    """Build a Gemini client that retries rate-limit and server errors."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=_REQUEST_ATTEMPTS),
        ),
    )
