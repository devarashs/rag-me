import json
from collections.abc import Iterator
from datetime import UTC, datetime

import psycopg
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from rag_me.answering import MAX_QUESTION_LENGTH, no_answer_message
from rag_me.api import (
    MAX_RAW_QUESTION_LENGTH,
    AnsweringOptions,
    ApiDependencies,
    create_app,
)
from rag_me.rate_limit import ALLOWED, RateLimitDecision
from tests.fakes import FakeEmbedder
from tests.test_answering import FakeGenerator, FakeSearcher, retrieved

CONTACT_EMAIL = "me@example.test"
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


class FakeRateLimiter:
    def __init__(self, decision: RateLimitDecision = ALLOWED) -> None:
        self.decision = decision
        self.calls: list[tuple[str, datetime]] = []

    def check_and_count(self, client_ip: str, now: datetime) -> RateLimitDecision:
        self.calls.append((client_ip, now))
        return self.decision


class FailingGenerator(FakeGenerator):
    def stream(self, system_instruction: str, prompt: str) -> Iterator[str]:
        yield "Partial "
        raise genai_errors.ServerError(503, {"error": {"message": "overloaded"}})


class FailingEmbedder(FakeEmbedder):
    def embed_query(self, question: str) -> list[float]:
        raise genai_errors.ServerError(503, {"error": {"message": "overloaded"}})


def make_client(
    *,
    searcher=None,
    generator=None,
    embedder=None,
    rate_limiter=None,
    check_database=lambda: None,
    client_ip_header=None,
) -> TestClient:
    dependencies = ApiDependencies(
        embedder=embedder or FakeEmbedder(),
        generator=generator or FakeGenerator(),
        searcher=searcher or FakeSearcher([retrieved("a.md#go", 0.9), retrieved("b.md#ts", 0.8)]),
        rate_limiter=rate_limiter or FakeRateLimiter(),
        check_database=check_database,
        answering=AnsweringOptions(top_k=5, min_similarity=0.62, contact_email=CONTACT_EMAIL),
        client_ip_header=client_ip_header,
        clock=lambda: NOW,
    )
    return TestClient(create_app(dependencies), client=("10.0.0.1", 50000))


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def ask(client: TestClient, question: str = "Does Arash know Go?", **kwargs):
    return client.post("/api/ask", json={"question": question}, **kwargs)


def assert_problem(response, status: int, title: str) -> dict:
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["status"] == status
    assert body["title"] == title
    assert body["type"] == "about:blank"
    return body


# --- happy path -------------------------------------------------------------------


def test_ask_streams_sources_then_deltas_then_done() -> None:
    response = ask(make_client(generator=FakeGenerator(["Arash ", "knows Go [1]."])))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    assert parse_sse(response.text) == [
        (
            "sources",
            {
                "sources": [
                    {
                        "number": 1,
                        "chunk_id": "a.md#go",
                        "title": "Doc - go",
                        "document_title": "Doc",
                        "section_heading": "go",
                        "similarity": 0.9,
                    },
                    {
                        "number": 2,
                        "chunk_id": "b.md#ts",
                        "title": "Doc - ts",
                        "document_title": "Doc",
                        "section_heading": "ts",
                        "similarity": 0.8,
                    },
                ]
            },
        ),
        ("delta", {"text": "Arash "}),
        ("delta", {"text": "knows Go [1]."}),
        ("done", {"grounded": True, "cited": [1]}),
    ]


def test_sources_never_include_section_bodies() -> None:
    response = ask(make_client(searcher=FakeSearcher([retrieved("a.md#x", 0.9, "SECRET BODY")])))

    assert "SECRET BODY" not in response.text


def test_off_topic_question_streams_the_fixed_refusal() -> None:
    generator = FakeGenerator()

    response = ask(
        make_client(searcher=FakeSearcher([retrieved("a.md#x", 0.3)]), generator=generator)
    )

    assert parse_sse(response.text) == [
        ("sources", {"sources": []}),
        ("delta", {"text": no_answer_message(CONTACT_EMAIL)}),
        ("done", {"grounded": False, "cited": []}),
    ]
    assert generator.calls == []


UNICODE_TEXT = "2023 – 2025 ✓"  # noqa: RUF001 - en dash on purpose, as in data/


def test_unicode_in_answers_is_sent_unescaped() -> None:
    response = ask(make_client(generator=FakeGenerator([f"{UNICODE_TEXT} [1]"])))

    assert UNICODE_TEXT in response.text


# --- validation -------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": 42},
        {"question": "ok", "extra": "field"},
        {"question": "x" * (MAX_RAW_QUESTION_LENGTH + 1)},
    ],
)
def test_malformed_body_is_a_422_problem(body) -> None:
    rate_limiter = FakeRateLimiter()

    response = make_client(rate_limiter=rate_limiter).post("/api/ask", json=body)

    assert_problem(response, 422, "Invalid request")
    assert rate_limiter.calls == []


def test_non_json_body_is_a_422_problem() -> None:
    response = make_client().post(
        "/api/ask", content=b"not json", headers={"content-type": "application/json"}
    )

    assert_problem(response, 422, "Invalid request")


@pytest.mark.parametrize("question", ["   \n\t ", "x" * (MAX_QUESTION_LENGTH + 1)])
def test_blank_or_too_long_question_is_rejected_before_rate_limiting(question: str) -> None:
    rate_limiter = FakeRateLimiter()

    response = ask(make_client(rate_limiter=rate_limiter), question)

    assert_problem(response, 422, "Invalid question")
    assert rate_limiter.calls == []


def test_get_is_not_allowed_on_ask() -> None:
    assert make_client().get("/api/ask").status_code == 405


# --- rate limiting ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scope", "detail_fragment"),
    [("visitor", "wait a few minutes"), ("global", "daily question limit")],
)
def test_rate_limited_request_is_a_429_with_retry_after(scope: str, detail_fragment: str) -> None:
    generator = FakeGenerator()
    limiter = FakeRateLimiter(RateLimitDecision(False, scope, 170))  # type: ignore[arg-type]

    response = ask(make_client(rate_limiter=limiter, generator=generator))

    body = assert_problem(response, 429, "Too many requests")
    assert detail_fragment in body["detail"]
    assert response.headers["retry-after"] == "170"
    assert generator.calls == []


def test_rate_limiter_receives_socket_ip_and_clock_by_default() -> None:
    limiter = FakeRateLimiter()

    ask(make_client(rate_limiter=limiter), headers={"x-vercel-forwarded-for": "203.0.113.7"})

    # Header ignored: it is only trusted when configured.
    assert limiter.calls == [("10.0.0.1", NOW)]


@pytest.mark.parametrize(
    ("header_value", "expected_ip"),
    [
        ("203.0.113.7", "203.0.113.7"),
        ("203.0.113.7, 10.1.1.1", "203.0.113.7"),
        ("  203.0.113.7  ", "203.0.113.7"),
        ("", "10.0.0.1"),
    ],
)
def test_configured_proxy_header_supplies_the_client_ip(
    header_value: str, expected_ip: str
) -> None:
    limiter = FakeRateLimiter()
    client = make_client(rate_limiter=limiter, client_ip_header="x-vercel-forwarded-for")

    ask(client, headers={"x-vercel-forwarded-for": header_value})

    assert limiter.calls[0][0] == expected_ip


# --- dependency failures ----------------------------------------------------------


def test_embedding_failure_before_streaming_is_a_503_problem() -> None:
    response = ask(make_client(embedder=FailingEmbedder()))

    body = assert_problem(response, 503, "Temporarily unavailable")
    assert "overloaded" not in body["detail"]


def test_database_failure_in_rate_limiter_is_a_503_problem() -> None:
    class BrokenLimiter(FakeRateLimiter):
        def check_and_count(self, client_ip, now):
            raise psycopg.OperationalError("connection refused to secret-host")

    response = ask(make_client(rate_limiter=BrokenLimiter()))

    body = assert_problem(response, 503, "Temporarily unavailable")
    assert "secret-host" not in body["detail"]


def test_generation_failure_mid_stream_ends_with_an_error_event() -> None:
    response = ask(make_client(generator=FailingGenerator()))

    events = parse_sse(response.text)
    assert response.status_code == 200
    assert [name for name, _ in events] == ["sources", "delta", "error"]
    assert events[-1][1] == {"detail": "The answer could not be completed. Please try again."}


# --- health -----------------------------------------------------------------------


def test_health_is_ok_when_database_answers() -> None:
    response = make_client().get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_is_503_when_database_is_down() -> None:
    def broken_check() -> None:
        raise psycopg.OperationalError("down")

    assert_problem(make_client(check_database=broken_check).get("/api/health"), 503, "Unhealthy")
