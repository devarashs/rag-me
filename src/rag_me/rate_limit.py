"""Fixed-window rate limiting in Postgres: per visitor, and a global daily cap.

Why limit at all: every question spends Gemini free-tier quota (one embedding
and one generation call). Without a cap, one script could exhaust the day's
quota and take the bot down for everyone.

Why Postgres: Vercel functions keep no shared memory between instances, so the
counters need an external store, and the app already has one. An atomic
`INSERT ... ON CONFLICT DO UPDATE ... RETURNING` increments and reads a counter
in one statement, so concurrent requests cannot both slip under the limit.

Why fixed windows: simplest to reason about. The known weakness, up to twice the
limit across a window boundary, is acceptable for protecting a free quota.

Privacy: visitor buckets are keyed by an HMAC of the IP address, never the IP
itself. With a secret key, the stored value cannot be reversed by hashing every
possible IPv4 address, which a plain hash would allow.
"""

import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

import psycopg

GLOBAL_BUCKET_KEY = "global"
_COUNTER_RETENTION = timedelta(days=2)


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """At most `max_requests` per window of `window_seconds`."""

    max_requests: int
    window_seconds: int

    def __post_init__(self) -> None:
        if self.max_requests < 1 or self.window_seconds < 1:
            raise ValueError(f"Rate limit values must be positive, got {self}")


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Whether a request may proceed.

    Attributes:
        allowed: True if the request is within every limit.
        exceeded_scope: Which limit refused it, or None when allowed.
        retry_after_seconds: Seconds until the refusing window resets, or 0.
    """

    allowed: bool
    exceeded_scope: Literal["visitor", "global"] | None
    retry_after_seconds: int


ALLOWED = RateLimitDecision(allowed=True, exceeded_scope=None, retry_after_seconds=0)


class RateLimiter(Protocol):
    """What the API needs: count this request and say whether it may proceed."""

    def check_and_count(self, client_ip: str, now: datetime) -> RateLimitDecision: ...


def window_start(now: datetime, window_seconds: int) -> datetime:
    """Start of the fixed window containing `now`, aligned to the Unix epoch.

    Raises:
        ValueError: If `now` is naive; window boundaries must be in UTC.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    epoch_seconds = int(now.timestamp())
    return datetime.fromtimestamp(epoch_seconds - epoch_seconds % window_seconds, tz=UTC)


def seconds_until_window_ends(now: datetime, window_seconds: int) -> int:
    """Whole seconds until the next window starts, at least 1."""
    window_end = window_start(now, window_seconds) + timedelta(seconds=window_seconds)
    return max(1, int((window_end - now).total_seconds() + 0.999))


def visitor_bucket_key(client_ip: str, hash_key: bytes) -> str:
    """Bucket key for one visitor: an HMAC-SHA256 of the IP, never the IP itself."""
    digest = hmac.new(hash_key, client_ip.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"visitor:{digest}"


class PostgresRateLimiter:
    """`RateLimiter` storing counters in `rate_limit_counters`.

    Args:
        connection: Autocommit psycopg connection; borrowed for one call only.
        visitor_policy: Limit per visitor IP.
        global_policy: Limit across all visitors (normally a daily cap).
        hash_key: Secret used to HMAC visitor IPs.
    """

    def __init__(
        self,
        connection: psycopg.Connection,
        visitor_policy: RateLimitPolicy,
        global_policy: RateLimitPolicy,
        hash_key: bytes,
    ) -> None:
        if not hash_key:
            raise ValueError("hash_key must not be empty")
        self._connection = connection
        self._visitor_policy = visitor_policy
        self._global_policy = global_policy
        self._hash_key = hash_key

    def check_and_count(self, client_ip: str, now: datetime) -> RateLimitDecision:
        """Count this request against the visitor limit, then the global one.

        A request refused by the visitor limit is not counted globally, so one
        visitor hammering the API cannot use up everyone's daily allowance.
        Refused requests do still count against their own visitor window, which
        keeps a client that ignores 429s refused.
        """
        with self._connection.transaction():
            self._delete_expired_counters(now)
            visitor_count = self._increment(
                visitor_bucket_key(client_ip, self._hash_key),
                window_start(now, self._visitor_policy.window_seconds),
            )
            if visitor_count > self._visitor_policy.max_requests:
                return RateLimitDecision(
                    allowed=False,
                    exceeded_scope="visitor",
                    retry_after_seconds=seconds_until_window_ends(
                        now, self._visitor_policy.window_seconds
                    ),
                )
            global_count = self._increment(
                GLOBAL_BUCKET_KEY, window_start(now, self._global_policy.window_seconds)
            )
            if global_count > self._global_policy.max_requests:
                return RateLimitDecision(
                    allowed=False,
                    exceeded_scope="global",
                    retry_after_seconds=seconds_until_window_ends(
                        now, self._global_policy.window_seconds
                    ),
                )
        return ALLOWED

    def _increment(self, bucket_key: str, bucket_window_start: datetime) -> int:
        row = self._connection.execute(
            """
            INSERT INTO rate_limit_counters (bucket_key, window_start, request_count)
            VALUES (%s, %s, 1)
            ON CONFLICT (bucket_key, window_start)
            DO UPDATE SET request_count = rate_limit_counters.request_count + 1
            RETURNING request_count
            """,
            [bucket_key, bucket_window_start],
        ).fetchone()
        assert row is not None  # RETURNING on an upsert always yields one row
        return int(row[0])

    def _delete_expired_counters(self, now: datetime) -> None:
        # Cheap with the window_start index, and keeps the table small without a
        # scheduled job. Retention outlasts the longest window (one day).
        self._connection.execute(
            "DELETE FROM rate_limit_counters WHERE window_start < %s",
            [now - _COUNTER_RETENTION],
        )
