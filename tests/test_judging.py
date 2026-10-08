from types import SimpleNamespace

import pytest

from rag_me.judging import (
    BatchVerdicts,
    CaseVerdict,
    GeminiJudge,
    GradingItem,
    JudgeError,
    JudgeVerdict,
    build_batch_prompt,
    match_verdicts_to_items,
)
from tests.test_answering import retrieved


def case_verdict(case_id: str, **overrides) -> CaseVerdict:
    fields = {
        "case_id": case_id,
        "supported": True,
        "unsupported_claims": [],
        "declined": False,
        "missing_facts": [],
        "followed_injection": False,
        "reasoning": f"graded {case_id}",
    } | overrides
    return CaseVerdict(**fields)


def item(case_id: str, answer: str = "Yes [1].") -> GradingItem:
    return GradingItem(
        case_id=case_id,
        question=f"Question for {case_id}?",
        sources=[retrieved(f"{case_id}.md#src", 0.9, f"Body for {case_id}.")],
        answer=answer,
        expected_facts=[f"fact for {case_id}"],
    )


class FakeModels:
    def __init__(self, parsed) -> None:
        self._parsed = parsed
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(parsed=self._parsed, text="raw")


def judge_with(parsed) -> tuple[GeminiJudge, FakeModels]:
    models = FakeModels(parsed)
    return GeminiJudge(SimpleNamespace(models=models), "judge-model"), models


# --- GeminiJudge.grade_batch ------------------------------------------------------


def test_grades_a_batch_in_one_request_keyed_by_case_id() -> None:
    judge, models = judge_with(
        BatchVerdicts(verdicts=[case_verdict("b", declined=True), case_verdict("a")])
    )

    verdicts = judge.grade_batch([item("a"), item("b")])

    assert len(models.calls) == 1
    assert set(verdicts) == {"a", "b"}
    assert verdicts["a"].declined is False
    assert verdicts["b"].declined is True
    assert verdicts["b"].reasoning == "graded b"
    assert all(type(verdict) is JudgeVerdict for verdict in verdicts.values())


def test_requests_schema_constrained_json_from_the_judge_model() -> None:
    judge, models = judge_with(BatchVerdicts(verdicts=[case_verdict("a")]))

    judge.grade_batch([item("a")])

    call = models.calls[0]
    assert call["model"] == "judge-model"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is BatchVerdicts
    assert call["config"].automatic_function_calling.disable is True


@pytest.mark.parametrize("parsed", [None, {"verdicts": []}, "text"])
def test_unparseable_response_is_an_error(parsed) -> None:
    judge, _ = judge_with(parsed)

    with pytest.raises(JudgeError, match="no valid verdicts"):
        judge.grade_batch([item("a")])


def test_empty_batch_is_rejected_without_a_request() -> None:
    judge, models = judge_with(BatchVerdicts(verdicts=[]))

    with pytest.raises(ValueError, match="No items"):
        judge.grade_batch([])

    assert models.calls == []


def test_duplicate_case_ids_in_a_request_are_rejected() -> None:
    judge, models = judge_with(BatchVerdicts(verdicts=[]))

    with pytest.raises(ValueError, match="Duplicate case ids"):
        judge.grade_batch([item("a"), item("a")])

    assert models.calls == []


# --- match_verdicts_to_items --------------------------------------------------------


@pytest.mark.parametrize(
    ("returned_ids", "message"),
    [
        (["a"], r"missing \['b'\]"),
        (["a", "b", "c"], r"unexpected \['c'\]"),
        (["a", "a", "b"], r"duplicated \['a'\]"),
        (["a", "x"], r"missing \['b'\], unexpected \['x'\]"),
        ([], r"missing \['a', 'b'\]"),
    ],
)
def test_any_id_mismatch_rejects_the_whole_batch(returned_ids, message) -> None:
    with pytest.raises(JudgeError, match=message):
        match_verdicts_to_items([case_verdict(case_id) for case_id in returned_ids], ["a", "b"])


def test_verdicts_returned_in_a_different_order_still_match() -> None:
    verdicts = match_verdicts_to_items(
        [case_verdict("b", supported=False), case_verdict("a")], ["a", "b"]
    )

    assert verdicts["a"].supported is True
    assert verdicts["b"].supported is False


# --- build_batch_prompt -------------------------------------------------------------


def test_each_case_is_delimited_with_only_its_own_sources_and_facts() -> None:
    prompt = build_batch_prompt([item("alpha", "Answer A [1]."), item("beta", "Answer B.")])

    alpha_block, beta_block = prompt.split("\n\n=== CASE beta ===")
    assert alpha_block.startswith("=== CASE alpha ===")
    assert alpha_block.endswith("=== END CASE alpha ===")
    assert "Body for alpha." in alpha_block and "Body for beta." not in alpha_block
    assert "- fact for alpha" in alpha_block
    assert "ANSWER TO GRADE:\nAnswer A [1]." in alpha_block
    assert "Body for beta." in beta_block and "Body for alpha." not in beta_block
    assert beta_block.endswith("=== END CASE beta ===")


def test_case_without_sources_or_facts_is_marked() -> None:
    prompt = build_batch_prompt(
        [GradingItem("x", "Q?", sources=[], answer="No idea.", expected_facts=[])]
    )

    assert "(no sources" in prompt
    assert "EXPECTED FACTS:\n(none)" in prompt
