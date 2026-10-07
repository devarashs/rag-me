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


def load_settings(**overrides: object) -> Settings:
    """Load and validate settings from the environment and `.env`.

    Args:
        overrides: Keyword arguments passed straight to `Settings`, for tests
            (for example `_env_file=None` to ignore the real `.env`).

    Returns:
        The validated settings.

    Raises:
        ConfigurationError: If any required value is missing or invalid. The
            message lists each problem by environment variable name.
    """
    try:
        return Settings(**overrides)  # type: ignore[arg-type]
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
