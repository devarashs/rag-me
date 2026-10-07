import pytest

from rag_me.config import ConfigurationError, load_settings

FAKE_API_KEY = "test-key-not-a-real-secret"
FAKE_DATABASE_URL = "postgresql://user:not-a-real-password@db.example.test/rag"

CONFIG_VARIABLES = [
    "GOOGLE_AI_API_KEY",
    "DATABASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_BATCH_SIZE",
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
