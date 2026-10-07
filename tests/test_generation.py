from types import SimpleNamespace

from rag_me.generation import GeminiGenerator


class FakeModels:
    def __init__(self, pieces: list[str | None]) -> None:
        self._pieces = pieces
        self.calls: list[dict] = []

    def generate_content_stream(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return iter(SimpleNamespace(text=piece) for piece in self._pieces)


def test_streams_text_pieces_in_order_skipping_metadata_only_pieces() -> None:
    models = FakeModels(["Arash ", None, "", "knows Go."])
    generator = GeminiGenerator(SimpleNamespace(models=models), "gemini-test")

    assert list(generator.stream("rules", "prompt")) == ["Arash ", "knows Go."]


def test_sends_model_prompt_and_system_instruction() -> None:
    models = FakeModels(["ok"])
    generator = GeminiGenerator(SimpleNamespace(models=models), "gemini-test")

    list(generator.stream("the rules", "the prompt"))

    [call] = models.calls
    assert call["model"] == "gemini-test"
    assert call["contents"] == "the prompt"
    assert call["config"].system_instruction == "the rules"
    assert call["config"].automatic_function_calling.disable is True


def test_nothing_is_requested_until_the_stream_is_consumed() -> None:
    models = FakeModels(["ok"])

    GeminiGenerator(SimpleNamespace(models=models), "gemini-test").stream("rules", "prompt")

    assert models.calls == []
