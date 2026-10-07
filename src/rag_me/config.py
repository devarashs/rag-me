"""Runtime configuration, read from environment variables and the project's `.env` file.

Configuration is validated once, at startup, so a missing or blank value fails
immediately with a message naming the variable, instead of surfacing later as an
authentication error from a remote API.
"""

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid.

    The message names the offending environment variables and never includes
    their values, because those values are secrets.
    """


class Settings(BaseSettings):
    """Validated application settings.

    Field names map to environment variables case-insensitively, so
    `google_ai_api_key` is read from `GOOGLE_AI_API_KEY`. Real environment
    variables take precedence over `.env`, which is what a hosted deployment
    relies on. Unrelated variables in `.env` are ignored.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_ai_api_key: SecretStr = Field(
        description="Google AI Studio API key, used for both generation and embeddings.",
    )
    database_url: SecretStr = Field(
        description="Postgres connection string (Neon), `postgresql://...`. A secret: "
        "it embeds the database password.",
    )
    embedding_model: str = Field(
        default="gemini-embedding-2",
        min_length=1,
        description="Gemini embedding model. Changing it re-embeds every chunk on next ingest.",
    )
    embedding_batch_size: int = Field(
        default=50,
        ge=1,
        le=100,
        # Upper bound is a conservative guard: batches above 100 are untested.
        description="Chunks per embedding request.",
    )
    generation_model: str = Field(
        # Chosen for latency: first words in ~1s, versus ~6.6s for
        # gemini-3.8-flash even at low thinking (measured 2026-10-07).
        default="gemini-3.1-flash-lite",
        min_length=1,
        description="Gemini model that writes answers.",
    )
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Sections retrieved and shown to the model per question.",
    )
    min_similarity: float = Field(
        # Sits in the gap measured on 18 sample questions: off-topic topped out at
        # 0.59, on-topic started at 0.65. Re-tune against the evaluation set.
        default=0.62,
        ge=0.0,
        le=1.0,
        description="Below this best-match similarity, answer 'I don't know' without "
        "calling the model.",
    )
    contact_email: str = Field(
        default="me@devarash.icu",
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        description="Address offered when a question cannot be answered.",
    )

    @field_validator("google_ai_api_key", "database_url")
    @classmethod
    def reject_blank_secret(cls, value: SecretStr) -> SecretStr:
        # An empty `NAME=` line in .env parses as a present-but-empty string;
        # treat it the same as a missing variable.
        if not value.get_secret_value().strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("database_url")
    @classmethod
    def require_postgres_scheme(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().startswith(("postgresql://", "postgres://")):
            # Name the expected shape only; the value itself is a secret.
            raise ValueError("must start with postgresql:// or postgres://")
        return value


class ApiSettings(Settings):
    """Settings the public HTTP API needs on top of the shared ones.

    Kept separate so the local CLI does not demand a rate-limit secret it never
    uses.
    """

    rate_limit_hash_key: SecretStr = Field(
        min_length=32,
        description="Secret for HMAC-ing visitor IPs in rate-limit counters. Generate "
        'with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.',
    )
    visitor_rate_limit_requests: int = Field(
        default=10, ge=1, description="Questions one visitor may ask per window."
    )
    visitor_rate_limit_window_seconds: int = Field(
        default=600, ge=1, le=86_400, description="Length of the per-visitor window."
    )
    daily_question_cap: int = Field(
        # Each question spends one embedding and one generation request. Keep this
        # under the Gemini free-tier daily limits shown in AI Studio.
        default=200,
        ge=1,
        description="Questions answered per UTC day across all visitors.",
    )
    client_ip_header: str | None = Field(
        default=None,
        description="Header holding the real client IP, set by a trusted proxy. On "
        "Vercel: x-vercel-forwarded-for. Unset: use the socket peer address. Never "
        "set this to a header clients can forge.",
    )

    @field_validator("client_ip_header")
    @classmethod
    def normalize_header_name(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return value.strip().lower()


def load_settings[SettingsT: Settings](
    settings_class: type[SettingsT] = Settings,  # type: ignore[assignment]
    **overrides: object,
) -> SettingsT:
    """Load and validate settings from the environment and `.env`.

    Args:
        settings_class: `Settings` for the CLI, `ApiSettings` for the HTTP API.
        overrides: Keyword arguments passed straight to the class, for tests
            (for example `_env_file=None` to ignore the real `.env`).

    Returns:
        The validated settings.

    Raises:
        ConfigurationError: If any required value is missing or invalid. The
            message lists each problem by environment variable name.
    """
    try:
        return settings_class(**overrides)  # type: ignore[arg-type]
    except ValidationError as error:
        # Pydantic's own message echoes the rejected input, which here would be
        # a secret. Rebuild the message from field locations and reasons only.
        problems = [
            f"{'.'.join(str(part) for part in issue['loc']).upper()}: {issue['msg']}"
            for issue in error.errors()
        ]
        raise ConfigurationError(
            "Invalid configuration (set these in the environment or in .env):\n  "
            + "\n  ".join(problems)
        ) from None
