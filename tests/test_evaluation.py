from pathlib import Path

import pytest

from rag_me.evaluation import (
    CaseResult,
    EvalCase,
    EvalCaseError,
    EvalOptions,
    failure_reasons,
    first_relevant_rank,
    format_report,
    load_eval_cases,
    result_record,
    run_case,
    summarize,
)
from rag_me.judging import JudgeVerdict
from tests.fakes import FakeEmbedder
from tests.test_answering import FakeGenerator, FakeSearcher, retrieved

OPTIONS = EvalOptions(top_k=5, min_similarity=0.62, contact_email="me@example.test")
KNOWN_IDS = ["a.md#go", "b.md#ts", "c.md#other"]

ANSWER_CASE = EvalCase(
    id="knows-go",
    category="fact",
    question="Does Arash know Go?",
    expect="answer",
    relevant_chunks=("a.md#go",),
    expected_facts=("writes Go",),
)
DECLINE_CASE = EvalCase(id="salary", category="not_covered", question="Salary?", expect="decline")


def verdict(**overrides) -> JudgeVerdict:
    fields = {
        "supported": True,
        "unsupported_claims": [],
        "declined": False,
        "missing_facts": [],
        "followed_injection": False,
        "reasoning": "fine",
    } | overrides
    return JudgeVerdict(**fields)


class FakeJudge:
    model_name = "fake-judge"

    def __init__(self, result: JudgeVerdict | Exception | None = None) -> None:
        self._result = result if result is not None else verdict()
        self.calls: list[dict] = []

    def grade(self, *, question, sources, answer, expected_facts) -> JudgeVerdict:
        self.calls.append(
            {"question": question, "sources": sources, "answer": answer, "facts": expected_facts}
        )
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def run(case: EvalCase, *, results=None, judge=None, generator=None) -> CaseResult:
    return run_case(
        case,
        embedder=FakeEmbedder(),
        searcher=FakeSearcher(
            results
            if results is not None
            else [retrieved("b.md#ts", 0.8), retrieved("a.md#go", 0.7)]
        ),
        generator=generator or FakeGenerator(["Arash writes Go [2]."]),
        judge=judge or FakeJudge(),
        options=OPTIONS,
    )


# --- load_eval_cases ------------------------------------------------------------------


def write_cases(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cases.toml"
    path.write_text(body, encoding="utf-8")
    return path


VALID_CASES = """
[[case]]
id = "knows-go"
category = "fact"
question = "Does Arash know Go?"
expect = "answer"
relevant_chunks = ["a.md#go"]
expected_facts = ["writes Go"]

[[case]]
id = "salary"
category = "not_covered"
question = "Salary?"
expect = "decline"
"""


def test_loads_valid_cases(tmp_path: Path) -> None:
    cases = load_eval_cases(write_cases(tmp_path, VALID_CASES), KNOWN_IDS)

    assert cases == [ANSWER_CASE, DECLINE_CASE]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("", "no \\[\\[case\\]\\] entries"),
        (
            '[[case]]\nid = "x"\ncategory = "fact"\nexpect = "decline"',
            "id and question are required",
        ),
        (VALID_CASES + VALID_CASES, "duplicate id"),
        (VALID_CASES.replace('"fact"', '"trivia"'), "category must be one of"),
        (VALID_CASES.replace('"decline"', '"maybe"'), "expect must be"),
        (VALID_CASES.replace('expected_facts = ["writes Go"]', ""), "need relevant_chunks"),
        (VALID_CASES + 'relevant_chunks = ["a.md#go"]\n', "decline cases take no"),
        (VALID_CASES.replace("a.md#go", "a.md#renamed"), "unknown chunk ids \\['a.md#renamed'\\]"),
    ],
)
def test_malformed_cases_are_rejected_with_the_case_named(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(EvalCaseError, match=message):
        load_eval_cases(write_cases(tmp_path, body), KNOWN_IDS)


def test_project_cases_file_is_valid() -> None:
    from rag_me.chunking import load_knowledge_base_chunks

    root = Path(__file__).resolve().parents[1]
    known = [chunk.chunk_id for chunk in load_knowledge_base_chunks(root / "data")]

    cases = load_eval_cases(root / "evals" / "cases.toml", known)

    assert len(cases) >= 30


# --- first_relevant_rank -------------------------------------------------------------


@pytest.mark.parametrize(
    ("retrieved_ids", "relevant", "expected"),
    [
        (["a", "b", "c"], ["a"], 1),
        (["a", "b", "c"], ["c", "b"], 2),
        (["a", "b"], ["z"], None),
        ([], ["a"], None),
    ],
)
def test_first_relevant_rank(retrieved_ids, relevant, expected) -> None:
    assert first_relevant_rank(retrieved_ids, relevant) == expected


# --- run_case --------------------------------------------------------------------


def test_answer_case_records_retrieval_answer_citations_and_verdict() -> None:
    judge = FakeJudge()

    result = run(ANSWER_CASE, judge=judge)

    assert result.passed
    assert result.retrieved == [("b.md#ts", 0.8), ("a.md#go", 0.7)]
    assert result.first_relevant_rank == 2
    assert result.answer == "Arash writes Go [2]."
    assert result.cited == [2]
    assert judge.calls[0]["facts"] == ("writes Go",)
    assert [s.chunk.chunk_id for s in judge.calls[0]["sources"]] == ["b.md#ts", "a.md#go"]


def test_gate_refusal_is_recorded_without_calling_judge_but_keeps_retrieval() -> None:
    judge = FakeJudge()

    result = run(ANSWER_CASE, results=[retrieved("a.md#go", 0.3)], judge=judge)

    assert not result.gate_passed
    assert result.retrieved == [("a.md#go", 0.3)]
    assert judge.calls == []
    assert not result.passed
    assert failure_reasons(result) == ["refused by the relevance gate"]


def test_gate_refusal_passes_a_decline_case() -> None:
    result = run(DECLINE_CASE, results=[retrieved("c.md#other", 0.3)])

    assert result.passed


def test_exception_is_recorded_not_raised() -> None:
    result = run(ANSWER_CASE, judge=FakeJudge(RuntimeError("judge overloaded")))

    assert result.error == "RuntimeError: judge overloaded"
    assert result.retrieved  # retrieval had already completed
    assert not result.passed


# --- pass rules -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict_overrides", "answer", "reason"),
    [
        ({"supported": False, "unsupported_claims": ["x"]}, "Go [1].", "unsupported claims"),
        ({"declined": True}, "Go [1].", "model declined"),
        ({"missing_facts": ["writes Go"]}, "Go [1].", "missing facts"),
        ({}, "Arash writes Go.", "no citations"),
    ],
)
def test_answer_case_fails_for_each_requirement(verdict_overrides, answer, reason) -> None:
    result = run(
        ANSWER_CASE,
        judge=FakeJudge(verdict(**verdict_overrides)),
        generator=FakeGenerator([answer]),
    )

    assert not result.passed
    assert any(reason in text for text in failure_reasons(result))


@pytest.mark.parametrize(
    ("verdict_overrides", "passes"),
    [
        ({"declined": True}, True),
        ({"declined": False}, False),
        ({"declined": True, "supported": False, "unsupported_claims": ["guess"]}, False),
        ({"declined": True, "followed_injection": True}, False),
    ],
)
def test_decline_case_pass_rule(verdict_overrides, passes) -> None:
    result = run(DECLINE_CASE, judge=FakeJudge(verdict(**verdict_overrides)))

    assert result.passed is passes


# --- summarize and report ---------------------------------------------------------


def test_summary_metrics() -> None:
    results = [
        run(ANSWER_CASE, results=[retrieved("a.md#go", 0.9)]),  # rank 1
        run(ANSWER_CASE, results=[retrieved("b.md#ts", 0.8), retrieved("a.md#go", 0.7)]),  # rank 2
        run(ANSWER_CASE, results=[retrieved("c.md#other", 0.64)]),  # miss
        run(
            EvalCase("france", "off_topic", "Capital of France?", "decline"),
            results=[retrieved("c.md#other", 0.55)],
        ),
    ]

    summary = summarize(results, top_k=5)

    assert summary["cases"] == 4
    assert summary["retrieval"]["hit_at_1"] == pytest.approx(0.333, abs=1e-3)
    assert summary["retrieval"]["hit_at_k"] == pytest.approx(0.667, abs=1e-3)
    assert summary["retrieval"]["mrr"] == pytest.approx(0.5)  # (1 + 0.5 + 0) / 3
    assert summary["gate"]["off_topic_caught"] == 1.0
    assert summary["gate"]["lowest_answerable_similarity"] == 0.64
    assert summary["gate"]["highest_off_topic_or_injection_similarity"] == 0.55
    assert summary["by_category"]["off_topic"] == {"passed": 1, "total": 1}


def test_summary_of_empty_run_has_no_ratios() -> None:
    summary = summarize([], top_k=5)

    assert summary["cases"] == 0
    assert summary["retrieval"]["hit_at_1"] is None
    assert summary["latency_seconds"]["p50"] is None


def test_report_and_record_render_every_case() -> None:
    results = [run(ANSWER_CASE), run(DECLINE_CASE, judge=FakeJudge(verdict(declined=False)))]

    report = format_report(summarize(results, top_k=5), results)
    record = result_record(results[1])

    assert "PASS  fact" in report
    assert "FAIL  not_covered" in report
    assert "answered instead of declining" in report
    assert record["id"] == "salary"
    assert record["verdict"]["declined"] is False
