import pytest

from rag_me.config import ApiSettings, ConfigurationError, load_settings

FAKE_API_KEY = "test-key-not-a-real-secret"
FAKE_DATABASE_URL = "postgresql://user:not-a-real-password@db.example.test/rag"

CONFIG_VARIABLES = [
    "GOOGLE_AI_API_KEY",
    "DATABASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_BATCH_SIZE",
    "GENERATION_MODEL",
    "RETRIEVAL_TOP_K",
    "MIN_SIMILARITY",
    "CONTACT_EMAIL",
    "RATE_LIMIT_HASH_KEY",
    "VISITOR_RATE_LIMIT_REQUESTS",
    "VISITOR_RATE_LIMIT_WINDOW_SECONDS",
    "DAILY_QUESTION_CAP",
    "CLIENT_IP_HEADER",
]


@pytest.fixture(autouse=True)
def isolate_from_real_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with config variables unset, so the developer's shell never leaks in."""
    for name in CONFIG_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def valid_environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setenv("GOOGLE_AI_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("DATABASE_URL", FAKE_DATABASE_URL)
    return monkeypatch


# --- loading --------------------------------------------------------------------


def test_reads_required_values_and_applies_defaults(valid_environment) -> None:
    settings = load_settings(_env_file=None)

    assert settings.google_ai_api_key.get_secret_value() == FAKE_API_KEY
    assert settings.database_url.get_secret_value() == FAKE_DATABASE_URL
    assert settings.embedding_model == "gemini-embedding-2"
    assert settings.embedding_batch_size == 50
    assert settings.generation_model == "gemini-3.1-flash-lite"
    assert settings.retrieval_top_k == 5
    assert settings.min_similarity == 0.62
    assert settings.contact_email == "me@devarash.icu"


def test_reads_env_file_and_ignores_unrelated_variables(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"GOOGLE_AI_API_KEY={FAKE_API_KEY}\n"
        f'DATABASE_URL="{FAKE_DATABASE_URL}"\n'
        "GOOGLE_AI_API_PROJECT_ID=unrelated\n",
        encoding="utf-8",
    )

    settings = load_settings(_env_file=env_file)

    assert settings.google_ai_api_key.get_secret_value() == FAKE_API_KEY
    assert settings.database_url.get_secret_value() == FAKE_DATABASE_URL


def test_environment_variable_overrides_env_file(tmp_path, valid_environment) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_AI_API_KEY=from-file\n", encoding="utf-8")
    valid_environment.setenv("GOOGLE_AI_API_KEY", "from-environment")

    settings = load_settings(_env_file=env_file)

    assert settings.google_ai_api_key.get_secret_value() == "from-environment"


def test_optional_values_can_be_overridden(valid_environment) -> None:
    valid_environment.setenv("EMBEDDING_MODEL", "gemini-embedding-001")
    valid_environment.setenv("EMBEDDING_BATCH_SIZE", "100")

    settings = load_settings(_env_file=None)

    assert settings.embedding_model == "gemini-embedding-001"
    assert settings.embedding_batch_size == 100


def test_postgres_scheme_alias_is_accepted(valid_environment) -> None:
    valid_environment.setenv("DATABASE_URL", "postgres://user:pw@host/db")

    assert load_settings(_env_file=None).database_url.get_secret_value().startswith("postgres://")


@pytest.mark.parametrize(
    ("variable", "value", "field", "expected"),
    [
        ("GENERATION_MODEL", "gemini-3.8-flash", "generation_model", "gemini-3.8-flash"),
        ("RETRIEVAL_TOP_K", "20", "retrieval_top_k", 20),
        ("MIN_SIMILARITY", "0", "min_similarity", 0.0),
        ("CONTACT_EMAIL", "hello@example.test", "contact_email", "hello@example.test"),
    ],
)
def test_answering_settings_can_be_overridden(
    valid_environment, variable: str, value: str, field: str, expected
) -> None:
    valid_environment.setenv(variable, value)

    assert getattr(load_settings(_env_file=None), field) == expected


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("RETRIEVAL_TOP_K", "0"),
        ("RETRIEVAL_TOP_K", "21"),
        ("MIN_SIMILARITY", "-0.1"),
        ("MIN_SIMILARITY", "1.5"),
        ("CONTACT_EMAIL", "not-an-email"),
        ("CONTACT_EMAIL", "two@@example.test"),
        ("GENERATION_MODEL", ""),
    ],
)
def test_invalid_answering_settings_are_rejected(
    valid_environment, variable: str, value: str
) -> None:
    valid_environment.setenv(variable, value)

    with pytest.raises(ConfigurationError, match=variable):
        load_settings(_env_file=None)


# --- rejection ------------------------------------------------------------------


def test_every_missing_required_variable_is_named_at_once() -> None:
    with pytest.raises(ConfigurationError) as raised:
        load_settings(_env_file=None)

    assert "GOOGLE_AI_API_KEY: Field required" in str(raised.value)
    assert "DATABASE_URL: Field required" in str(raised.value)


@pytest.mark.parametrize("variable", ["GOOGLE_AI_API_KEY", "DATABASE_URL"])
@pytest.mark.parametrize("blank_value", ["", "   "])
def test_blank_secret_is_rejected(valid_environment, variable: str, blank_value: str) -> None:
    valid_environment.setenv(variable, blank_value)

    with pytest.raises(ConfigurationError, match=rf"{variable}: .*must not be blank"):
        load_settings(_env_file=None)


@pytest.mark.parametrize(
    "database_url", ["mysql://user:pw@host/db", "host=db user=x", "https://db.example.test"]
)
def test_non_postgres_database_url_is_rejected(valid_environment, database_url: str) -> None:
    valid_environment.setenv("DATABASE_URL", database_url)

    with pytest.raises(ConfigurationError, match=r"DATABASE_URL: .*must start with postgresql://"):
        load_settings(_env_file=None)


@pytest.mark.parametrize("batch_size", ["0", "-1", "101", "ten"])
def test_out_of_range_batch_size_is_rejected(valid_environment, batch_size: str) -> None:
    valid_environment.setenv("EMBEDDING_BATCH_SIZE", batch_size)

    with pytest.raises(ConfigurationError, match="EMBEDDING_BATCH_SIZE"):
        load_settings(_env_file=None)


def test_blank_embedding_model_is_rejected(valid_environment) -> None:
    valid_environment.setenv("EMBEDDING_MODEL", "")

    with pytest.raises(ConfigurationError, match="EMBEDDING_MODEL"):
        load_settings(_env_file=None)


# --- secrecy --------------------------------------------------------------------


def test_error_never_echoes_a_rejected_secret(valid_environment) -> None:
    # Pydantic's own ValidationError text includes "input_value=..."; neither that
    # text nor the original exception may reach the user, since inputs are secrets.
    leaked_password = "s3cret-password-in-url"
    valid_environment.setenv("DATABASE_URL", f"mysql://user:{leaked_password}@host/db")

    with pytest.raises(ConfigurationError) as raised:
        load_settings(_env_file=None)

    assert leaked_password not in str(raised.value)
    assert "input_value" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


def test_settings_repr_masks_secrets(valid_environment) -> None:
    settings = load_settings(_env_file=None)

    for text in (repr(settings), str(settings)):
        assert FAKE_API_KEY not in text
        assert "not-a-real-password" not in text


# --- ApiSettings ------------------------------------------------------------------

FAKE_HASH_KEY = "k" * 32


def test_cli_settings_do_not_require_the_rate_limit_key(valid_environment) -> None:
    load_settings(_env_file=None)


def test_api_settings_require_the_rate_limit_key(valid_environment) -> None:
    valid_environment.delenv("RATE_LIMIT_HASH_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="RATE_LIMIT_HASH_KEY: Field required"):
        load_settings(ApiSettings, _env_file=None)


def test_api_settings_defaults(valid_environment) -> None:
    valid_environment.setenv("RATE_LIMIT_HASH_KEY", FAKE_HASH_KEY)

    settings = load_settings(ApiSettings, _env_file=None)

    assert settings.visitor_rate_limit_requests == 10
    assert settings.visitor_rate_limit_window_seconds == 600
    assert settings.daily_question_cap == 200
    assert settings.client_ip_header is None
    assert FAKE_HASH_KEY not in repr(settings)


def test_short_rate_limit_key_is_rejected(valid_environment) -> None:
    valid_environment.setenv("RATE_LIMIT_HASH_KEY", "k" * 31)

    with pytest.raises(ConfigurationError, match="RATE_LIMIT_HASH_KEY"):
        load_settings(ApiSettings, _env_file=None)


@pytest.mark.parametrize(
    ("raw_header", "expected"),
    [("X-Vercel-Forwarded-For", "x-vercel-forwarded-for"), ("  ", None), ("", None)],
)
def test_client_ip_header_is_normalized(valid_environment, raw_header: str, expected) -> None:
    valid_environment.setenv("RATE_LIMIT_HASH_KEY", FAKE_HASH_KEY)
    valid_environment.setenv("CLIENT_IP_HEADER", raw_header)

    assert load_settings(ApiSettings, _env_file=None).client_ip_header == expected


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("VISITOR_RATE_LIMIT_REQUESTS", "0"),
        ("VISITOR_RATE_LIMIT_WINDOW_SECONDS", "0"),
        ("VISITOR_RATE_LIMIT_WINDOW_SECONDS", "86401"),
        ("DAILY_QUESTION_CAP", "0"),
    ],
)
def test_invalid_rate_limits_are_rejected(valid_environment, variable: str, value: str) -> None:
    valid_environment.setenv("RATE_LIMIT_HASH_KEY", FAKE_HASH_KEY)
    valid_environment.setenv(variable, value)

    with pytest.raises(ConfigurationError, match=variable):
        load_settings(ApiSettings, _env_file=None)
