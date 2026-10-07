from datetime import UTC, datetime, timedelta

import pytest

from rag_me.rate_limit import (
    GLOBAL_BUCKET_KEY,
    PostgresRateLimiter,
    RateLimitPolicy,
    seconds_until_window_ends,
    visitor_bucket_key,
    window_start,
)

HASH_KEY = b"test-hash-key-at-least-32-bytes-long!!"
NOON = datetime(2026, 10, 7, 12, 0, 0, tzinfo=UTC)


# --- pure helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "window_seconds", "expected_start"),
    [
        (NOON, 600, NOON),
        (NOON + timedelta(seconds=599), 600, NOON),
        (NOON + timedelta(seconds=600), 600, NOON + timedelta(seconds=600)),
        (NOON + timedelta(hours=5, minutes=3), 86_400, datetime(2026, 10, 7, tzinfo=UTC)),
    ],
)
def test_window_start_aligns_to_fixed_windows(now, window_seconds, expected_start) -> None:
    assert window_start(now, window_seconds) == expected_start


def test_window_start_converts_other_timezones_to_utc() -> None:
    dubai = datetime(2026, 10, 7, 2, 30, tzinfo=UTC).astimezone(
        __import__("zoneinfo").ZoneInfo("Asia/Dubai")
    )

    assert window_start(dubai, 86_400) == datetime(2026, 10, 7, tzinfo=UTC)


def test_window_start_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        window_start(datetime(2026, 10, 7, 12, 0), 600)


@pytest.mark.parametrize(("offset_seconds", "expected"), [(0, 600), (1, 599), (599, 1), (599.5, 1)])
def test_seconds_until_window_ends(offset_seconds: float, expected: int) -> None:
    assert seconds_until_window_ends(NOON + timedelta(seconds=offset_seconds), 600) == expected


def test_visitor_bucket_key_is_a_keyed_hash_not_the_ip() -> None:
    key = visitor_bucket_key("203.0.113.7", HASH_KEY)

    assert key.startswith("visitor:")
    assert "203.0.113.7" not in key
    assert key == visitor_bucket_key("203.0.113.7", HASH_KEY)
    assert key != visitor_bucket_key("203.0.113.8", HASH_KEY)
    assert key != visitor_bucket_key("203.0.113.7", b"a-different-secret-key-of-32-bytes!")


@pytest.mark.parametrize(("max_requests", "window_seconds"), [(0, 60), (5, 0), (-1, -1)])
def test_policy_rejects_non_positive_values(max_requests: int, window_seconds: int) -> None:
    with pytest.raises(ValueError):
        RateLimitPolicy(max_requests, window_seconds)


# --- PostgresRateLimiter (integration) ------------------------------------------


def make_limiter(connection, visitor_max=3, global_max=100) -> PostgresRateLimiter:
    return PostgresRateLimiter(
        connection,
        visitor_policy=RateLimitPolicy(visitor_max, 600),
        global_policy=RateLimitPolicy(global_max, 86_400),
        hash_key=HASH_KEY,
    )


def counter(connection, bucket_key: str) -> int | None:
    row = connection.execute(
        "SELECT sum(request_count) FROM rate_limit_counters WHERE bucket_key = %s", [bucket_key]
    ).fetchone()
    return row[0]


@pytest.mark.integration
def test_visitor_is_allowed_up_to_the_limit_then_refused(migrated_schema_connection) -> None:
    limiter = make_limiter(migrated_schema_connection, visitor_max=3)

    decisions = [limiter.check_and_count("203.0.113.7", NOON) for _ in range(4)]

    assert [decision.allowed for decision in decisions] == [True, True, True, False]
    assert decisions[-1].exceeded_scope == "visitor"
    assert decisions[-1].retry_after_seconds == 600


@pytest.mark.integration
def test_visitors_are_counted_separately(migrated_schema_connection) -> None:
    limiter = make_limiter(migrated_schema_connection, visitor_max=1)

    assert limiter.check_and_count("203.0.113.7", NOON).allowed
    assert not limiter.check_and_count("203.0.113.7", NOON).allowed
    assert limiter.check_and_count("198.51.100.4", NOON).allowed


@pytest.mark.integration
def test_new_window_resets_the_visitor_limit(migrated_schema_connection) -> None:
    limiter = make_limiter(migrated_schema_connection, visitor_max=1)
    limiter.check_and_count("203.0.113.7", NOON)

    assert not limiter.check_and_count("203.0.113.7", NOON + timedelta(seconds=599)).allowed
    assert limiter.check_and_count("203.0.113.7", NOON + timedelta(seconds=600)).allowed


@pytest.mark.integration
def test_global_cap_refuses_across_visitors_until_next_utc_day(migrated_schema_connection) -> None:
    limiter = make_limiter(migrated_schema_connection, visitor_max=10, global_max=2)

    assert limiter.check_and_count("203.0.113.1", NOON).allowed
    assert limiter.check_and_count("203.0.113.2", NOON).allowed
    refused = limiter.check_and_count("203.0.113.3", NOON)

    assert not refused.allowed
    assert refused.exceeded_scope == "global"
    assert refused.retry_after_seconds == 12 * 3600
    assert limiter.check_and_count("203.0.113.3", NOON + timedelta(hours=12)).allowed


@pytest.mark.integration
def test_visitor_refusal_does_not_spend_the_global_allowance(migrated_schema_connection) -> None:
    limiter = make_limiter(migrated_schema_connection, visitor_max=1, global_max=100)

    for _ in range(5):
        limiter.check_and_count("203.0.113.7", NOON)

    assert counter(migrated_schema_connection, GLOBAL_BUCKET_KEY) == 1
    assert counter(migrated_schema_connection, visitor_bucket_key("203.0.113.7", HASH_KEY)) == 5


@pytest.mark.integration
def test_raw_ip_is_never_stored(migrated_schema_connection) -> None:
    make_limiter(migrated_schema_connection).check_and_count("203.0.113.7", NOON)

    stored_keys = [
        key
        for (key,) in migrated_schema_connection.execute(
            "SELECT bucket_key FROM rate_limit_counters"
        ).fetchall()
    ]
    assert stored_keys and all("203.0.113.7" not in key for key in stored_keys)


@pytest.mark.integration
def test_counters_older_than_retention_are_deleted(migrated_schema_connection) -> None:
    limiter = make_limiter(migrated_schema_connection)
    limiter.check_and_count("203.0.113.7", NOON - timedelta(days=3))

    limiter.check_and_count("198.51.100.4", NOON)

    oldest = migrated_schema_connection.execute(
        "SELECT min(window_start) FROM rate_limit_counters"
    ).fetchone()[0]
    assert oldest >= NOON - timedelta(days=2)


def test_empty_hash_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="hash_key"):
        PostgresRateLimiter(
            None,
            RateLimitPolicy(1, 1),
            RateLimitPolicy(1, 1),
            hash_key=b"",  # type: ignore[arg-type]
        )
