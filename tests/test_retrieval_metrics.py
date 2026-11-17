import pytest

from rag_eval import EvalExample, RetrievedContext
from rag_eval.metrics import (
    context_matches,
    evaluate_retrieval,
    hit_rate_at_k,
    judge_relevance,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

T, F = True, False


@pytest.mark.parametrize(
    ("relevance", "k", "expected"),
    [([T, F, T, F], 4, 0.5), ([T, F, T, F], 2, 0.5), ([T, F, T], 5, 0.4), ([F, F], 2, 0.0)],
)
def test_precision_at_k(relevance: list[bool], k: int, expected: float) -> None:
    assert precision_at_k(relevance, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("relevance", "k", "total", "expected"),
    [([T, F, T], 3, 4, 0.5), ([T, F, T], 1, 2, 0.5), ([F, F, F], 3, 1, 0.0), ([T], 1, 1, 1.0)],
)
def test_recall_at_k(relevance: list[bool], k: int, total: int, expected: float) -> None:
    assert recall_at_k(relevance, k, total) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("relevance", "k", "expected"), [([F, F, T], 2, 0.0), ([F, F, T], 3, 1.0), ([], 3, 0.0)]
)
def test_hit_rate_at_k(relevance: list[bool], k: int, expected: float) -> None:
    assert hit_rate_at_k(relevance, k) == expected


@pytest.mark.parametrize(
    ("relevance", "k", "expected"),
    [([T, T], 2, 1.0), ([F, T], 2, 0.5), ([F, F, T], 3, 1 / 3), ([F, F, T], 2, 0.0)],
)
def test_reciprocal_rank(relevance: list[bool], k: int, expected: float) -> None:
    assert reciprocal_rank(relevance, k) == pytest.approx(expected)


def test_invalid_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        precision_at_k([T], 0)
    with pytest.raises(ValueError, match="total_relevant"):
        recall_at_k([T], 1, 0)


def test_context_matching_uses_containment_or_term_coverage() -> None:
    assert context_matches("Intro. The Cat sat on the mat!", "cat sat on the mat", 0.75)
    assert context_matches("queue holds failed messages", "the queue holds messages", 0.75)
    assert not context_matches("queue only", "queue holds failed messages", 0.75)
    assert context_matches("queue only", "queue holds failed messages", 0.25)


def _contexts(*texts: str) -> list[RetrievedContext]:
    return [RetrievedContext(text=text, doc_id=f"d{i}") for i, text in enumerate(texts, 1)]


def test_relevance_credits_each_expected_context_once() -> None:
    example = EvalExample(
        id="q", question="Q", expected_context=["alpha beta gamma", "delta epsilon"]
    )
    contexts = _contexts(
        "zzz unrelated", "alpha beta gamma here", "alpha beta gamma dup", "delta epsilon"
    )

    assert judge_relevance(contexts, example, 0.75) == [F, T, F, T]


def test_evaluate_retrieval_hand_computed_values() -> None:
    example = EvalExample(
        id="q", question="Q", expected_context=["alpha beta gamma", "delta epsilon"]
    )
    contexts = _contexts(
        "zzz unrelated", "alpha beta gamma here", "alpha beta gamma dup", "delta epsilon"
    )

    at_3 = evaluate_retrieval(example, contexts, k=3, threshold=0.75)
    at_4 = evaluate_retrieval(example, contexts, k=4, threshold=0.75)

    assert at_3 is not None
    assert (at_3.precision_at_k, at_3.recall_at_k, at_3.hit_rate_at_k) == pytest.approx(
        (1 / 3, 0.5, 1.0)
    )
    assert at_3.reciprocal_rank == pytest.approx(0.5)
    assert at_3.relevant_ranks == [2]
    assert at_4 is not None
    assert (at_4.precision_at_k, at_4.recall_at_k) == pytest.approx((0.5, 1.0))
    assert at_4.relevant_ranks == [2, 4]


def test_doc_ids_take_precedence_over_text() -> None:
    example = EvalExample(
        id="q", question="Q", relevant_doc_ids=["d2"], expected_context=["zzz unrelated"]
    )

    scores = evaluate_retrieval(example, _contexts("zzz unrelated", "other"), k=2, threshold=0.75)

    assert scores is not None
    assert scores.relevant_ranks == [2]
    assert scores.reciprocal_rank == pytest.approx(0.5)


def test_missing_retrieved_contexts_score_zero() -> None:
    example = EvalExample(id="q", question="Q", expected_context=["alpha"])

    scores = evaluate_retrieval(example, [], k=5, threshold=0.75)

    assert scores is not None
    assert (scores.precision_at_k, scores.recall_at_k, scores.hit_rate_at_k) == (0, 0, 0)
    assert scores.reciprocal_rank == 0


def test_missing_ground_truth_skips_retrieval_metrics() -> None:
    example = EvalExample(id="q", question="Q")

    assert evaluate_retrieval(example, _contexts("alpha"), k=5, threshold=0.75) is None
