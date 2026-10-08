"""Measure the bot against a fixed set of questions (evals/cases.toml).

Each case runs through the same `answer_question` pipeline the API uses, against
the real vector store and models, then is scored on three independent axes:

1. **Retrieval** (answer cases): did a relevant section come back, and how high?
   hit@1, hit@k and mean reciprocal rank (MRR: 1/rank of the first relevant
   section, 0 if none).
2. **Relevance gate**: answerable questions wrongly refused, off-topic ones
   caught. The report also prints the similarity margin the gate sits in.
3. **Answer quality**, graded by `rag_me.judging`: grounded in the sources,
   states the expected facts, cites, and declines when it should.

A case passes only if everything its `expect` requires holds. Retrieval and gate
numbers are reported separately so a failure can be traced to the stage that
caused it.

Answering and grading are separate steps (`answer_case`, then `grade_results`)
so that grading can batch several answers per judge request, and so a failed
grading batch can be retried without regenerating answers that already exist.
"""

import statistics
import time
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rag_me.answering import answer_question, extract_cited_source_numbers
from rag_me.embeddings import Embedder
from rag_me.generation import Generator
from rag_me.judging import GradingItem, Judge, JudgeVerdict
from rag_me.store import ChunkSearcher, RetrievedChunk

CATEGORIES = ("fact", "paraphrase", "not_covered", "off_topic", "injection")
Expectation = Literal["answer", "decline"]


class EvalCaseError(ValueError):
    """Raised when the cases file is malformed. The message names the case."""


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    category: str
    question: str
    expect: Expectation
    relevant_chunks: tuple[str, ...] = ()
    expected_facts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalOptions:
    """The answering settings under test, recorded with every run."""

    top_k: int
    min_similarity: float
    contact_email: str


@dataclass(slots=True)
class CaseResult:
    """Everything observed for one case.

    `error` is set if answering crashed; `grading_error` if the answer exists
    but its grading batch failed, so a retry only needs to grade again.
    `sources` keeps the chunks shown to the model, for the judge.
    """

    case: EvalCase
    retrieved: list[tuple[str, float]] = field(default_factory=list)
    sources: list[RetrievedChunk] = field(default_factory=list)
    gate_passed: bool = False
    answer: str = ""
    cited: list[int] = field(default_factory=list)
    verdict: JudgeVerdict | None = None
    answer_seconds: float = 0.0
    error: str | None = None
    grading_error: str | None = None

    @property
    def failure(self) -> str | None:
        """The first error that stopped this case, if any."""
        return self.error or self.grading_error

    @property
    def needs_grading(self) -> bool:
        return self.error is None and self.gate_passed and self.verdict is None

    @property
    def top_similarity(self) -> float | None:
        return self.retrieved[0][1] if self.retrieved else None

    @property
    def first_relevant_rank(self) -> int | None:
        return first_relevant_rank(
            [chunk_id for chunk_id, _ in self.retrieved], self.case.relevant_chunks
        )

    @property
    def passed(self) -> bool:
        if self.failure is not None or self.verdict is None:
            return False
        verdict = self.verdict
        if self.case.expect == "answer":
            return (
                self.gate_passed
                and verdict.supported
                and not verdict.declined
                and not verdict.missing_facts
                and bool(self.cited)
            )
        return verdict.declined and verdict.supported and not verdict.followed_injection


# --- loading ---------------------------------------------------------------------


def load_eval_cases(path: Path, known_chunk_ids: Iterable[str]) -> list[EvalCase]:
    """Read and validate the cases file.

    Raises:
        EvalCaseError: On a missing or duplicate id, unknown category or
            expectation, an answer case without relevant chunks or facts, a
            decline case with them, or a chunk ID that does not exist in the
            knowledge base (usually a renamed heading).
    """
    raw_cases = tomllib.loads(path.read_text(encoding="utf-8")).get("case", [])
    if not raw_cases:
        raise EvalCaseError(f"{path}: no [[case]] entries")

    known_ids = set(known_chunk_ids)
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_cases, start=1):
        case_id = raw.get("id") or f"#{index}"
        if not raw.get("id") or not raw.get("question", "").strip():
            raise EvalCaseError(f"case {case_id}: id and question are required")
        if case_id in seen_ids:
            raise EvalCaseError(f"case {case_id}: duplicate id")
        seen_ids.add(case_id)
        if raw.get("category") not in CATEGORIES:
            raise EvalCaseError(f"case {case_id}: category must be one of {CATEGORIES}")
        if raw.get("expect") not in ("answer", "decline"):
            raise EvalCaseError(f"case {case_id}: expect must be 'answer' or 'decline'")

        relevant = tuple(raw.get("relevant_chunks", ()))
        facts = tuple(raw.get("expected_facts", ()))
        if raw["expect"] == "answer" and not (relevant and facts):
            raise EvalCaseError(
                f"case {case_id}: answer cases need relevant_chunks and expected_facts"
            )
        if raw["expect"] == "decline" and (relevant or facts):
            raise EvalCaseError(f"case {case_id}: decline cases take no relevant_chunks or facts")
        unknown = [chunk_id for chunk_id in relevant if chunk_id not in known_ids]
        if unknown:
            raise EvalCaseError(f"case {case_id}: unknown chunk ids {unknown}")

        cases.append(
            EvalCase(
                id=case_id,
                category=raw["category"],
                question=raw["question"],
                expect=raw["expect"],
                relevant_chunks=relevant,
                expected_facts=facts,
            )
        )
    return cases


# --- running ---------------------------------------------------------------------


class _RecordingSearcher:
    """Wraps a searcher to keep what it returned, including when the gate then
    refuses and `answer_question` discards the results."""

    def __init__(self, inner: ChunkSearcher) -> None:
        self._inner = inner
        self.last_results: list[RetrievedChunk] = []

    def search(self, query_embedding: Sequence[float], limit: int) -> list[RetrievedChunk]:
        self.last_results = self._inner.search(query_embedding, limit)
        return self.last_results


def answer_case(
    case: EvalCase,
    *,
    embedder: Embedder,
    searcher: ChunkSearcher,
    generator: Generator,
    options: EvalOptions,
) -> CaseResult:
    """Ask one question through the real pipeline. Grading happens later.

    A gate refusal gets its verdict here: the fixed refusal makes no claims and
    calls no model, so sending it to the judge would only spend quota to confirm
    a constant.

    Never raises: a failure is recorded on the result so one bad case does not
    abort a run that has already spent quota on the others.
    """
    result = CaseResult(case=case)
    recording_searcher = _RecordingSearcher(searcher)
    try:
        started_at = time.monotonic()
        answer = answer_question(
            case.question,
            embedder=embedder,
            searcher=recording_searcher,
            generator=generator,
            top_k=options.top_k,
            min_similarity=options.min_similarity,
            contact_email=options.contact_email,
        )
        result.answer = "".join(answer.text_pieces)
        result.answer_seconds = time.monotonic() - started_at
        result.sources = list(answer.sources)
        result.gate_passed = answer.is_grounded
        result.cited = extract_cited_source_numbers(result.answer, len(answer.sources))
        if not answer.is_grounded:
            result.verdict = JudgeVerdict(
                supported=True,
                unsupported_claims=[],
                declined=True,
                missing_facts=list(case.expected_facts),
                followed_injection=False,
                reasoning="Refused by the relevance gate; the model was not called.",
            )
    except Exception as error:  # recorded per case, see docstring
        result.error = f"{type(error).__name__}: {error}"
    # Recorded even when generation failed: retrieval had already finished, and
    # the summary scores retrieval for every case that got that far.
    result.retrieved = [
        (source.chunk.chunk_id, source.similarity) for source in recording_searcher.last_results
    ]
    return result


def grade_results(results: Sequence[CaseResult], judge: Judge, batch_size: int) -> int:
    """Grade every result that needs it, `batch_size` answers per judge request.

    A failed batch marks only its own cases with `grading_error` (clearing any
    earlier one on success), so calling this again retries exactly those.

    Returns:
        The number of judge requests made.

    Raises:
        ValueError: If `batch_size` is below 1.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    pending = [result for result in results if result.needs_grading]
    requests = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        requests += 1
        try:
            verdicts = judge.grade_batch(
                [
                    GradingItem(
                        case_id=result.case.id,
                        question=result.case.question,
                        sources=result.sources,
                        answer=result.answer,
                        expected_facts=result.case.expected_facts,
                    )
                    for result in batch
                ]
            )
        except Exception as error:  # recorded per case, see docstring
            for result in batch:
                result.grading_error = f"{type(error).__name__}: {error}"
            continue
        for result in batch:
            result.verdict = verdicts[result.case.id]
            result.grading_error = None
    return requests


# --- scoring ---------------------------------------------------------------------


def first_relevant_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> int | None:
    """1-based rank of the first relevant chunk in `retrieved_ids`, or None."""
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return rank
    return None


def summarize(results: Sequence[CaseResult], top_k: int) -> dict[str, Any]:
    """Aggregate metrics for a run. See the module docstring for definitions."""
    # A case that failed later (say, the judge was overloaded) still has valid
    # retrieval data, so it counts for retrieval and the gate if it got that far.
    answer_results = [r for r in results if r.case.expect == "answer" and r.retrieved]
    ranks = [r.first_relevant_rank for r in answer_results]
    generated = [
        r for r in results if r.gate_passed and r.verdict is not None and r.failure is None
    ]
    gate_target = [r for r in results if r.case.category == "off_topic" and r.error is None]
    off_or_attack = [
        r.top_similarity
        for r in results
        if r.case.category in ("off_topic", "injection") and r.top_similarity is not None
    ]
    answer_similarities = [r.top_similarity for r in answer_results if r.top_similarity is not None]
    latencies = sorted(r.answer_seconds for r in generated)

    by_category: dict[str, dict[str, int]] = {}
    for category in CATEGORIES:
        in_category = [r for r in results if r.case.category == category]
        if in_category:
            by_category[category] = {
                "passed": sum(r.passed for r in in_category),
                "total": len(in_category),
            }

    return {
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "errors": [r.case.id for r in results if r.failure is not None],
        "by_category": by_category,
        "retrieval": {
            "k": top_k,
            "hit_at_1": _ratio(sum(rank == 1 for rank in ranks), len(ranks)),
            "hit_at_k": _ratio(sum(rank is not None for rank in ranks), len(ranks)),
            "mrr": round(statistics.fmean(1 / rank if rank else 0.0 for rank in ranks), 3)
            if ranks
            else None,
            "misses": [r.case.id for r in answer_results if r.first_relevant_rank is None],
        },
        "gate": {
            "answerable_refused": [r.case.id for r in answer_results if not r.gate_passed],
            "off_topic_caught": _ratio(
                sum(not r.gate_passed for r in gate_target), len(gate_target)
            ),
            "lowest_answerable_similarity": round(min(answer_similarities), 4)
            if answer_similarities
            else None,
            "highest_off_topic_or_injection_similarity": round(max(off_or_attack), 4)
            if off_or_attack
            else None,
        },
        "answers": {
            "generated": len(generated),
            "grounded": _ratio(sum(r.verdict.supported for r in generated), len(generated)),
            "answer_cases_with_citations": _ratio(
                sum(bool(r.cited) for r in answer_results), len(answer_results)
            ),
            "injections_followed": [
                r.case.id for r in results if r.verdict is not None and r.verdict.followed_injection
            ],
        },
        "latency_seconds": {
            "p50": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(latencies[max(0, round(0.95 * len(latencies)) - 1)], 2)
            if latencies
            else None,
        },
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


# --- reporting -------------------------------------------------------------------


def result_record(result: CaseResult) -> dict[str, Any]:
    """JSON-serialisable view of one case result."""
    return {
        "id": result.case.id,
        "category": result.case.category,
        "expect": result.case.expect,
        "question": result.case.question,
        "passed": result.passed,
        "gate_passed": result.gate_passed,
        "first_relevant_rank": result.first_relevant_rank,
        "retrieved": [{"chunk_id": c, "similarity": round(s, 4)} for c, s in result.retrieved],
        "answer": result.answer,
        "cited": result.cited,
        "verdict": result.verdict.model_dump() if result.verdict else None,
        "answer_seconds": round(result.answer_seconds, 2),
        "error": result.error,
        "grading_error": result.grading_error,
    }


def format_report(summary: dict[str, Any], results: Sequence[CaseResult]) -> str:
    """Human-readable run report: one line per case, then the metrics."""
    lines = ["Cases"]
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        rank = result.first_relevant_rank
        similarity = result.top_similarity
        details = [
            f"rank {rank}" if result.case.expect == "answer" else "",
            f"top {similarity:.3f}" if similarity is not None else "",
            "gate refused" if not result.gate_passed and result.error is None else "",
        ]
        lines.append(
            f"  {mark}  {result.case.category:<11} {result.case.id:<24} "
            + "  ".join(part for part in details if part)
        )
        if not result.passed:
            lines.extend(f"        - {reason}" for reason in failure_reasons(result))

    retrieval, gate, answers = summary["retrieval"], summary["gate"], summary["answers"]
    latency = summary["latency_seconds"]
    lines += [
        "",
        f"Passed {summary['passed']}/{summary['cases']}  "
        + "  ".join(
            f"{name} {c['passed']}/{c['total']}" for name, c in summary["by_category"].items()
        ),
        f"Retrieval  hit@1 {retrieval['hit_at_1']}  hit@{retrieval['k']} {retrieval['hit_at_k']}  "
        f"MRR {retrieval['mrr']}  misses {retrieval['misses'] or 'none'}",
        f"Gate       answerable refused {gate['answerable_refused'] or 'none'}  "
        f"off-topic caught {gate['off_topic_caught']}",
        f"           lowest answerable top similarity {gate['lowest_answerable_similarity']}, "
        f"highest off-topic/injection {gate['highest_off_topic_or_injection_similarity']}",
        f"Answers    grounded {answers['grounded']} of {answers['generated']} generated  "
        f"citations {answers['answer_cases_with_citations']}  "
        f"injections followed {answers['injections_followed'] or 'none'}",
        f"Latency    p50 {latency['p50']}s  p95 {latency['p95']}s "
        "(question to full answer, generated answers only)",
    ]
    if summary["errors"]:
        lines.append(f"Errors     {summary['errors']}")
    return "\n".join(lines)


def failure_reasons(result: CaseResult) -> list[str]:
    """Why a case failed, in the order the pipeline would have caused it."""
    if result.failure is not None:
        return [f"error: {result.failure}"]
    verdict = result.verdict
    if verdict is None:
        return ["no verdict"]
    reasons: list[str] = []
    if result.case.expect == "answer":
        if not result.gate_passed:
            reasons.append("refused by the relevance gate")
        if verdict.declined and result.gate_passed:
            reasons.append("model declined an answerable question")
        if verdict.missing_facts and result.gate_passed:
            reasons.append(f"missing facts: {verdict.missing_facts}")
        if not result.cited and result.gate_passed:
            reasons.append("no citations")
    elif not verdict.declined:
        reasons.append("answered instead of declining")
    if not verdict.supported:
        reasons.append(f"unsupported claims: {verdict.unsupported_claims}")
    if verdict.followed_injection:
        reasons.append("followed an injected instruction")
    return reasons
