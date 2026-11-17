from pathlib import Path

import pytest

from rag_eval import DatasetError, load_corpus, load_dataset, pooled_corpus
from rag_eval.dataset import find_unknown_doc_ids, validate_dataset
from tests.conftest import EXAMPLES_DIR, WriteJsonl


def test_parses_example_dataset() -> None:
    dataset = load_dataset(EXAMPLES_DIR / "dataset.jsonl")

    assert len(dataset.examples) > 0
    assert dataset.source is not None
    assert len(dataset.source.sha256) == 64
    assert all(example.has_retrieval_ground_truth for example in dataset.examples)


def test_assigns_line_ids_and_coerces_single_context(write_jsonl: WriteJsonl) -> None:
    path = write_jsonl(
        "data.jsonl",
        [
            {"question": "  What is X?  ", "expected_answer": "X", "expected_context": "X is X."},
            "",
            {"id": 7, "question": "What is Y?", "relevant_doc_ids": "doc-y"},
        ],
    )

    first, second = load_dataset(path).examples

    assert first.id == "line-1"
    assert first.question == "What is X?"
    assert first.expected_context == ["X is X."]
    assert second.id == "7"
    assert second.relevant_doc_ids == ["doc-y"]
    assert not second.has_answer_ground_truth


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("{not json", "invalid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ({"expected_answer": "no question"}, "question"),
        ({"question": "   "}, "must not be blank"),
        ({"question": "Q", "expected_contexts": ["typo"]}, "expected_contexts"),
        ({"question": "Q", "expected_context": ["ok", " "]}, "entries must not be blank"),
    ],
)
def test_reports_invalid_records_with_line_numbers(
    write_jsonl: WriteJsonl, row: object, message: str
) -> None:
    path = write_jsonl("data.jsonl", [{"question": "valid"}, row])  # type: ignore[list-item]

    validation = validate_dataset(path)

    assert not validation.ok
    assert validation.errors[0].line == 2
    assert message in validation.errors[0].message


def test_rejects_duplicate_ids(write_jsonl: WriteJsonl) -> None:
    path = write_jsonl("data.jsonl", [{"id": "a", "question": "Q1"}, {"id": "a", "question": "Q2"}])

    with pytest.raises(DatasetError, match="duplicate id 'a'"):
        load_dataset(path)


def test_empty_dataset_is_an_error(write_jsonl: WriteJsonl) -> None:
    path = write_jsonl("empty.jsonl", ["", "   "])

    with pytest.raises(DatasetError, match="no records"):
        load_dataset(path)


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="cannot read file"):
        load_dataset(tmp_path / "missing.jsonl")


def test_tolerates_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b'\xef\xbb\xbf{"question": "Q"}\n')

    assert load_dataset(path).examples[0].question == "Q"


def test_warns_about_missing_ground_truth(write_jsonl: WriteJsonl) -> None:
    path = write_jsonl("data.jsonl", [{"question": "Q"}])

    validation = validate_dataset(path)

    assert validation.ok
    messages = [warning.message for warning in validation.warnings]
    assert any("retrieval metrics skipped" in message for message in messages)
    assert any("no expected_answer" in message for message in messages)


def test_loads_corpus_and_finds_unknown_doc_ids(write_jsonl: WriteJsonl) -> None:
    corpus = load_corpus(write_jsonl("corpus.jsonl", [{"id": "d1", "text": "alpha"}]))
    dataset = load_dataset(
        write_jsonl("data.jsonl", [{"id": "q", "question": "Q", "relevant_doc_ids": ["d1", "d9"]}])
    )

    issues = find_unknown_doc_ids(dataset.examples, corpus)

    assert [issue.message for issue in issues] == ["example 'q' references unknown doc id 'd9'"]


def test_pooled_corpus_deduplicates_contexts(write_jsonl: WriteJsonl) -> None:
    dataset = load_dataset(
        write_jsonl(
            "data.jsonl",
            [
                {"question": "Q1", "expected_context": ["A", "B"]},
                {"question": "Q2", "expected_context": ["B", "C"]},
            ],
        )
    )

    corpus = pooled_corpus(dataset)

    assert [(doc.id, doc.text) for doc in corpus.documents] == [
        ("ctx-1", "A"),
        ("ctx-2", "B"),
        ("ctx-3", "C"),
    ]
    assert corpus.source is not None
    assert corpus.source.path is None
