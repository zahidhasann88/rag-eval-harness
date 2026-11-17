import math
from collections import Counter
from collections.abc import Sequence

from rag_eval.models import EvalExample
from rag_eval.results import AnswerScores
from rag_eval.text import content_terms, normalize_answer

_NGRAM = 3


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style F1 over normalized token multisets."""
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return 1.0 if predicted == expected else 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _char_ngrams(text: str) -> Counter[str]:
    padded = f" {text} "
    if len(padded) < _NGRAM:
        return Counter([padded])
    return Counter(padded[i : i + _NGRAM] for i in range(len(padded) - _NGRAM + 1))


def answer_similarity(prediction: str, reference: str) -> float:
    """Heuristic: cosine similarity of character trigram counts of the normalized strings."""
    left, right = normalize_answer(prediction), normalize_answer(reference)
    if not left or not right:
        return 1.0 if left == right else 0.0
    a, b = _char_ngrams(left), _char_ngrams(right)
    dot = sum(count * b[gram] for gram, count in a.items())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm


def groundedness(answer: str, contexts: Sequence[str]) -> tuple[float | None, list[str]]:
    """Heuristic: share of the answer's distinct content terms that appear in the contexts.

    Returns the score (None when the answer has no content terms) and the unsupported terms.
    """
    answer_terms = list(dict.fromkeys(content_terms(answer)))
    if not answer_terms:
        return None, []
    context_terms = {term for context in contexts for term in content_terms(context)}
    unsupported = [term for term in answer_terms if term not in context_terms]
    return 1 - len(unsupported) / len(answer_terms), unsupported


def evaluate_answer(example: EvalExample, answer: str, contexts: Sequence[str]) -> AnswerScores:
    grounded, unsupported = groundedness(answer, contexts)
    if not example.has_answer_ground_truth or example.expected_answer is None:
        return AnswerScores(groundedness=grounded, unsupported_terms=unsupported)
    reference = example.expected_answer
    return AnswerScores(
        exact_match=exact_match(answer, reference),
        token_f1=token_f1(answer, reference),
        answer_similarity=answer_similarity(answer, reference),
        groundedness=grounded,
        unsupported_terms=unsupported,
    )
