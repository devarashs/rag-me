from types import SimpleNamespace

import pytest

from rag_me.judging import GeminiJudge, JudgeError, JudgeVerdict, build_judge_prompt
from tests.test_answering import retrieved

VERDICT = JudgeVerdict(
    supported=True,
    unsupported_claims=[],
    declined=False,
    missing_facts=[],
    followed_injection=False,
    reasoning="ok",
)


class FakeModels:
    def __init__(self, parsed) -> None:
        self._parsed = parsed
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(parsed=self._parsed, text="raw")


def grade(models: FakeModels):
    return GeminiJudge(SimpleNamespace(models=models), "judge-model").grade(
        question="Does Arash know Go?",
        sources=[retrieved("a.md#go", 0.9, "Arash writes Go.")],
        answer="Yes [1].",
        expected_facts=["writes Go"],
    )


def test_returns_the_parsed_verdict_and_requests_schema_output() -> None:
    models = FakeModels(VERDICT)

    assert grade(models) == VERDICT
    config = models.calls[0]["config"]
    assert models.calls[0]["model"] == "judge-model"
    assert config.response_mime_type == "application/json"
    assert config.response_schema is JudgeVerdict


@pytest.mark.parametrize("parsed", [None, {"supported": True}, "text"])
def test_unparseable_verdict_is_an_error(parsed) -> None:
    with pytest.raises(JudgeError, match="no valid verdict"):
        grade(FakeModels(parsed))


def test_prompt_includes_question_numbered_sources_facts_and_answer() -> None:
    prompt = build_judge_prompt(
        "Q?", [retrieved("a.md#go", 0.9, "Body one."), retrieved("b.md#x", 0.8)], "A [1].", ["f1"]
    )

    assert "QUESTION:\nQ?" in prompt
    assert "[1] Doc - go\nBody one." in prompt
    assert "[2] Doc - x" in prompt
    assert "EXPECTED FACTS:\n- f1" in prompt
    assert prompt.endswith("ANSWER TO GRADE:\nA [1].")


def test_prompt_marks_missing_sources_and_facts() -> None:
    prompt = build_judge_prompt("Q?", [], "No idea.", [])

    assert "(no sources" in prompt
    assert "EXPECTED FACTS:\n(none)" in prompt
