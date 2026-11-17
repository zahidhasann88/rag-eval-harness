import pytest

from rag_eval import EvalExample
from rag_eval.metrics import answer_similarity, evaluate_answer, exact_match, groundedness, token_f1


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("The Cat sat.", "cat sat", 1.0),
        ("  a   dog  ", "Dog!", 1.0),
        ("cat sat", "cat sat down", 0.0),
        ("", "", 1.0),
    ],
)
def test_exact_match_normalizes(prediction: str, reference: str, expected: float) -> None:
    assert exact_match(prediction, reference) == expected


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("the cat sat on the mat", "a cat sat", 2 / 3),
        ("cat cat cat", "cat", 0.5),
        ("red apple", "green pear", 0.0),
        ("Same answer.", "same answer", 1.0),
        ("", "something", 0.0),
        ("", "", 1.0),
    ],
)
def test_token_f1(prediction: str, reference: str, expected: float) -> None:
    assert token_f1(prediction, reference) == pytest.approx(expected)


def test_answer_similarity_bounds() -> None:
    assert answer_similarity("Dead letter queue", "dead letter queue") == pytest.approx(1.0)
    assert answer_similarity("xyz", "qqq") == 0.0
    assert answer_similarity("", "text") == 0.0
    partial = answer_similarity("stores failed messages", "stores messages that failed")
    assert 0.0 < partial < 1.0


def test_groundedness_hand_computed() -> None:
    score, unsupported = groundedness("Queues store failed messages", ["A queue holds messages"])

    assert score == pytest.approx(0.5)
    assert unsupported == ["store", "failed"]


def test_groundedness_without_contexts_is_zero() -> None:
    score, unsupported = groundedness("Queues store messages", [])

    assert score == 0.0
    assert unsupported == ["queue", "store", "message"]


def test_groundedness_is_undefined_for_answers_without_content_terms() -> None:
    assert groundedness("", ["anything"]) == (None, [])
    assert groundedness("It is what it is.", ["anything"]) == (None, [])


def test_evaluate_answer_skips_reference_metrics_without_expected_answer() -> None:
    scores = evaluate_answer(EvalExample(id="q", question="Q"), "alpha", ["alpha"])

    assert scores.exact_match is None
    assert scores.token_f1 is None
    assert scores.answer_similarity is None
    assert scores.groundedness == 1.0


def test_evaluate_answer_with_reference() -> None:
    example = EvalExample(id="q", question="Q", expected_answer="red apple")

    scores = evaluate_answer(example, "red apple", ["a red apple fact"])

    assert (scores.exact_match, scores.token_f1, scores.groundedness) == (1.0, 1.0, 1.0)
    assert scores.answer_similarity == pytest.approx(1.0)
