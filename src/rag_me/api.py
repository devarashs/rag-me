"""Public HTTP API: `POST /api/ask` streams an answer, `GET /api/health` checks readiness.

Response formats:

- `/api/ask` success is a Server-Sent Events stream (`text/event-stream`), so the
  first words reach the visitor while the rest is generated. Events, in order:

      event: sources   data: {"sources": [{"number", "chunk_id", "title",
                                 "document_title", "section_heading", "similarity"}]}
      event: delta     data: {"text": "..."}            (repeated)
      event: done      data: {"grounded": bool, "cited": [numbers]}

  If generation fails after streaming has begun, the HTTP status is already
  200, so an `error` event (`data: {"detail": "..."}`) replaces `done`.
- Every error before streaming is RFC 9457 Problem Details
  (`application/problem+json`) with a matching HTTP status: 422 for a bad
  question, 429 with `Retry-After` when rate limited, 503 when a dependency is
  down.

Database connections are borrowed only for the rate-limit check and the vector
search, never while waiting on Gemini.
"""

import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from google.genai import errors as genai_errors
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field

from rag_me.answering import (
    AnswerStream,
    InvalidQuestionError,
    answer_question,
    extract_cited_source_numbers,
    normalize_question,
)
from rag_me.config import ApiSettings
from rag_me.database import create_connection_pool
from rag_me.embeddings import Embedder, EmbeddingError, GeminiEmbedder, create_gemini_client
from rag_me.generation import GeminiGenerator, Generator
from rag_me.rate_limit import PostgresRateLimiter, RateLimitDecision, RateLimiter, RateLimitPolicy
from rag_me.store import ChunkSearcher, PostgresChunkStore, RetrievedChunk

logger = logging.getLogger("rag_me.api")

# Caps the raw request body field before whitespace normalization; the real
# question limit (answering.MAX_QUESTION_LENGTH) is applied after it.
MAX_RAW_QUESTION_LENGTH = 2_000
SECONDS_PER_DAY = 86_400


# --- request and response models --------------------------------------------------


class AskRequest(BaseModel):
    """Body of `POST /api/ask`."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_RAW_QUESTION_LENGTH)


@dataclass(frozen=True, slots=True)
class AnsweringOptions:
    """The answering settings the API passes through to `answer_question`."""

    top_k: int
    min_similarity: float
    contact_email: str


@dataclass(frozen=True, slots=True)
class ApiDependencies:
    """Everything the API calls out to, injected so tests can substitute fakes.

    Attributes:
        embedder: Embeds questions.
        generator: Writes answers.
        searcher: Vector search. Must borrow a database connection per call.
        rate_limiter: Must borrow a database connection per call.
        check_database: Raises if the database is unreachable.
        answering: Retrieval and prompt options.
        client_ip_header: Trusted proxy header with the client IP, or None.
        clock: Current time, timezone-aware.
    """

    embedder: Embedder
    generator: Generator
    searcher: ChunkSearcher
    rate_limiter: RateLimiter
    check_database: Callable[[], None]
    answering: AnsweringOptions
    client_ip_header: str | None
    clock: Callable[[], datetime]


# --- application -------------------------------------------------------------------


def create_app(dependencies: ApiDependencies, lifespan: Any = None) -> FastAPI:
    """Build the FastAPI app around `dependencies`.

    Args:
        dependencies: See `ApiDependencies`.
        lifespan: Optional FastAPI lifespan, used in production to open and
            close the connection pool.
    """
    app = FastAPI(
        title="rag-me",
        summary="Ask questions about Arash Salehkhah, answered from his own documents.",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
            for issue in error.errors()
        )
        return problem_response(422, "Invalid request", detail)

    @app.post(
        "/api/ask",
        response_class=StreamingResponse,
        responses={
            200: {"content": {"text/event-stream": {}}, "description": "Answer stream"},
            422: {"description": "Invalid question (Problem Details)"},
            429: {"description": "Rate limited (Problem Details, Retry-After)"},
            503: {"description": "A dependency is unavailable (Problem Details)"},
        },
    )
    def ask(body: AskRequest, request: Request):
        started_at = time.monotonic()
        try:
            question = normalize_question(body.question)
        except InvalidQuestionError as error:
            return problem_response(422, "Invalid question", str(error))

        try:
            decision = dependencies.rate_limiter.check_and_count(
                resolve_client_ip(request, dependencies.client_ip_header), dependencies.clock()
            )
            if not decision.allowed:
                log_request(
                    request,
                    status=429,
                    started_at=started_at,
                    rate_limited_scope=decision.exceeded_scope,
                )
                return rate_limited_response(decision)

            answer = answer_question(
                question,
                embedder=dependencies.embedder,
                searcher=dependencies.searcher,
                generator=dependencies.generator,
                top_k=dependencies.answering.top_k,
                min_similarity=dependencies.answering.min_similarity,
                contact_email=dependencies.answering.contact_email,
            )
        except (psycopg.Error, genai_errors.APIError, EmbeddingError):
            logger.exception("ask failed before streaming")
            return problem_response(
                503, "Temporarily unavailable", "The service could not answer right now."
            )

        return StreamingResponse(
            stream_answer_events(answer, on_finish=lambda: log_answer(request, answer, started_at)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/health")
    def health():
        try:
            dependencies.check_database()
        except psycopg.Error:
            logger.exception("health check failed")
            return problem_response(503, "Unhealthy", "The database is unreachable.")
        return {"status": "ok"}

    return app


def create_production_app() -> FastAPI:
    """Build the app from environment settings with real Gemini and Postgres.

    Raises:
        ConfigurationError: If settings are missing or invalid (at import time
            of the entrypoint, so a misconfigured deployment fails loudly).
    """
    from rag_me.config import load_settings

    settings = load_settings(ApiSettings)
    pool = create_connection_pool(settings.database_url.get_secret_value())
    client = create_gemini_client(settings.google_ai_api_key.get_secret_value())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        pool.open()
        try:
            yield
        finally:
            pool.close()

    dependencies = ApiDependencies(
        embedder=GeminiEmbedder(client, settings.embedding_model, settings.embedding_batch_size),
        generator=GeminiGenerator(client, settings.generation_model),
        searcher=PooledChunkSearcher(pool),
        rate_limiter=PooledRateLimiter(
            pool,
            visitor_policy=RateLimitPolicy(
                settings.visitor_rate_limit_requests, settings.visitor_rate_limit_window_seconds
            ),
            global_policy=RateLimitPolicy(settings.daily_question_cap, SECONDS_PER_DAY),
            hash_key=settings.rate_limit_hash_key.get_secret_value().encode("utf-8"),
        ),
        check_database=lambda: check_pool(pool),
        answering=AnsweringOptions(
            top_k=settings.retrieval_top_k,
            min_similarity=settings.min_similarity,
            contact_email=settings.contact_email,
        ),
        client_ip_header=settings.client_ip_header,
        clock=lambda: datetime.now(UTC),
    )
    return create_app(dependencies, lifespan=lifespan)


# --- pooled adapters -------------------------------------------------------------


class PooledChunkSearcher:
    """`ChunkSearcher` that borrows a pooled connection only for the query."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def search(self, query_embedding, limit: int) -> list[RetrievedChunk]:
        with self._pool.connection() as connection:
            return PostgresChunkStore(connection).search(query_embedding, limit)


class PooledRateLimiter:
    """`RateLimiter` that borrows a pooled connection only for the check."""

    def __init__(self, pool: ConnectionPool, **limiter_options: Any) -> None:
        self._pool = pool
        self._limiter_options = limiter_options

    def check_and_count(self, client_ip: str, now: datetime) -> RateLimitDecision:
        with self._pool.connection() as connection:
            return PostgresRateLimiter(connection, **self._limiter_options).check_and_count(
                client_ip, now
            )


def check_pool(pool: ConnectionPool) -> None:
    with pool.connection() as connection:
        connection.execute("SELECT 1")


# --- helpers ---------------------------------------------------------------------


def resolve_client_ip(request: Request, client_ip_header: str | None) -> str:
    """The client IP: from the trusted proxy header if configured, else the socket peer.

    A proxy header may hold a comma-separated chain; the first entry is the
    original client. Falls back to "unknown" (one shared bucket) rather than
    skipping the limit when no address is available.
    """
    if client_ip_header:
        header_value = request.headers.get(client_ip_header, "")
        first_address = header_value.split(",")[0].strip()
        if first_address:
            return first_address
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def stream_answer_events(answer: AnswerStream, on_finish: Callable[[], None]) -> Iterator[str]:
    """Render an `AnswerStream` as Server-Sent Events (see the module docstring)."""
    yield format_sse("sources", {"sources": [describe_source(n, s) for n, s in numbered(answer)]})
    answer_text = ""
    try:
        for text_piece in answer.text_pieces:
            answer_text += text_piece
            yield format_sse("delta", {"text": text_piece})
    except Exception:
        # Headers (200) are already sent, so the failure must travel in-band.
        logger.exception("answer stream failed mid-generation")
        yield format_sse(
            "error", {"detail": "The answer could not be completed. Please try again."}
        )
        return
    finally:
        on_finish()
    yield format_sse(
        "done",
        {
            "grounded": answer.is_grounded,
            "cited": extract_cited_source_numbers(answer_text, len(answer.sources)),
        },
    )


def numbered(answer: AnswerStream) -> list[tuple[int, RetrievedChunk]]:
    return list(enumerate(answer.sources, start=1))


def describe_source(number: int, source: RetrievedChunk) -> dict[str, Any]:
    """Public view of a source: deliberately no body text, just enough to cite it."""
    return {
        "number": number,
        "chunk_id": source.chunk.chunk_id,
        "title": source.chunk.title,
        "document_title": source.chunk.document_title,
        "section_heading": source.chunk.section_heading,
        "similarity": round(source.similarity, 4),
    }


def format_sse(event: str, data: dict[str, Any]) -> str:
    # json.dumps escapes newlines, so `data` is always a single SSE line.
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def problem_response(
    status: int, title: str, detail: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    """An RFC 9457 Problem Details response."""
    return JSONResponse(
        status_code=status,
        content={"type": "about:blank", "title": title, "status": status, "detail": detail},
        media_type="application/problem+json",
        headers=headers,
    )


def rate_limited_response(decision: RateLimitDecision) -> JSONResponse:
    if decision.exceeded_scope == "global":
        detail = "The bot has reached its daily question limit. Please try again tomorrow."
    else:
        detail = "Too many questions. Please wait a few minutes and try again."
    return problem_response(
        429, "Too many requests", detail, {"Retry-After": str(decision.retry_after_seconds)}
    )


def log_request(request: Request, *, status: int, started_at: float, **fields: Any) -> None:
    """One structured line per request. Never logs the question or the client IP."""
    record = {
        "path": request.url.path,
        "status": status,
        "duration_ms": round((time.monotonic() - started_at) * 1000),
        "request_id": request.headers.get("x-vercel-id"),
    }
    record.update(fields)
    logger.info(json.dumps(record))


def log_answer(request: Request, answer: AnswerStream, started_at: float) -> None:
    log_request(
        request,
        status=200,
        started_at=started_at,
        grounded=answer.is_grounded,
        top_similarity=round(answer.sources[0].similarity, 4) if answer.sources else None,
    )
