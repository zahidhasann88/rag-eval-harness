from collections.abc import Sequence

from rag_eval.models import EvalExample, RetrievedContext
from rag_eval.results import RetrievalScores
from rag_eval.text import content_terms, normalize_answer


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    """Relevant items in the top k divided by k."""
    _check_k(k)
    return sum(relevance[:k]) / k


def recall_at_k(relevance: Sequence[bool], k: int, total_relevant: int) -> float:
    """Relevant items in the top k divided by the number of ground-truth items."""
    _check_k(k)
    if total_relevant < 1:
        raise ValueError("total_relevant must be >= 1")
    return min(sum(relevance[:k]), total_relevant) / total_relevant


def hit_rate_at_k(relevance: Sequence[bool], k: int) -> float:
    _check_k(k)
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool], k: int) -> float:
    """1 / rank of the first relevant item within the top k, or 0 if none."""
    _check_k(k)
    for rank, relevant in enumerate(relevance[:k], start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def context_matches(retrieved: str, expected: str, threshold: float) -> bool:
    """A retrieved passage matches when it contains the expected passage, or covers at least
    ``threshold`` of the expected passage's content terms."""
    normalized_expected = normalize_answer(expected)
    if not normalized_expected:
        return False
    if normalized_expected in normalize_answer(retrieved):
        return True
    expected_terms = set(content_terms(expected))
    if not expected_terms:
        return False
    covered = expected_terms & set(content_terms(retrieved))
    return len(covered) / len(expected_terms) >= threshold


def judge_relevance(
    contexts: Sequence[RetrievedContext], example: EvalExample, threshold: float
) -> list[bool]:
    """Mark each retrieved context relevant or not, crediting each ground-truth item at most once
    in rank order. Document ids take precedence over expected context text when both exist."""
    if example.relevant_doc_ids:
        remaining_ids = set(example.relevant_doc_ids)
        relevance = []
        for context in contexts:
            hit = context.doc_id is not None and context.doc_id in remaining_ids
            if hit:
                remaining_ids.discard(context.doc_id)
            relevance.append(hit)
        return relevance

    remaining = list(example.expected_context)
    relevance = []
    for context in contexts:
        match = next(
            (
                index
                for index, exp in enumerate(remaining)
                if context_matches(context.text, exp, threshold)
            ),
            None,
        )
        if match is not None:
            remaining.pop(match)
        relevance.append(match is not None)
    return relevance


def evaluate_retrieval(
    example: EvalExample, contexts: Sequence[RetrievedContext], k: int, threshold: float
) -> RetrievalScores | None:
    """Score the top-k retrieved contexts, or return None when the example has no ground truth."""
    _check_k(k)
    if not example.has_retrieval_ground_truth:
        return None
    expected_count = len(set(example.relevant_doc_ids)) or len(example.expected_context)
    relevance = judge_relevance(contexts[:k], example, threshold)
    return RetrievalScores(
        precision_at_k=precision_at_k(relevance, k),
        recall_at_k=recall_at_k(relevance, k, expected_count),
        hit_rate_at_k=hit_rate_at_k(relevance, k),
        reciprocal_rank=reciprocal_rank(relevance, k),
        relevant_ranks=[rank for rank, relevant in enumerate(relevance, start=1) if relevant],
        expected_count=expected_count,
    )
