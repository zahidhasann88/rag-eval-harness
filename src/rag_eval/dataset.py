import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from rag_eval.models import Corpus, Dataset, Document, EvalExample, SourceInfo

_Record = TypeVar("_Record", EvalExample, Document)


class DatasetIssue(BaseModel):
    line: int | None = None
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}" if self.line is not None else self.message


class DatasetValidation(BaseModel):
    path: str
    examples: list[EvalExample]
    errors: list[DatasetIssue]
    warnings: list[DatasetIssue]
    sha256: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


class DatasetError(Exception):
    def __init__(self, path: Path | str, issues: Iterable[DatasetIssue]) -> None:
        self.path = str(path)
        self.issues = list(issues)
        details = "; ".join(str(issue) for issue in self.issues[:5])
        super().__init__(f"{self.path}: {details}")


def _format_validation_error(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "record"
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts)


def _parse_jsonl(
    path: Path, model: type[_Record], id_prefix: str
) -> tuple[list[tuple[int, _Record]], list[DatasetIssue], str | None]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [], [DatasetIssue(message=f"cannot read file: {exc.strerror or exc}")], None

    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return [], [DatasetIssue(message=f"file is not valid UTF-8: {exc}")], None

    records: list[tuple[int, _Record]] = []
    errors: list[DatasetIssue] = []
    seen_ids: dict[str, int] = {}

    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(DatasetIssue(line=line_number, message=f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(payload, dict):
            errors.append(DatasetIssue(line=line_number, message="record must be a JSON object"))
            continue
        payload.setdefault("id", f"{id_prefix}-{line_number}")
        try:
            record = model.model_validate(payload)
        except ValidationError as exc:
            errors.append(DatasetIssue(line=line_number, message=_format_validation_error(exc)))
            continue
        if record.id in seen_ids:
            first = seen_ids[record.id]
            message = f"duplicate id {record.id!r} (first seen on line {first})"
            errors.append(DatasetIssue(line=line_number, message=message))
            continue
        seen_ids[record.id] = line_number
        records.append((line_number, record))

    if not records and not errors:
        errors.append(DatasetIssue(message="file contains no records"))

    return records, errors, hashlib.sha256(raw).hexdigest()


def validate_dataset(path: Path | str) -> DatasetValidation:
    path = Path(path)
    records, errors, digest = _parse_jsonl(path, EvalExample, "line")
    warnings: list[DatasetIssue] = []
    for line_number, example in records:
        if not example.has_retrieval_ground_truth:
            warnings.append(
                DatasetIssue(
                    line=line_number,
                    message="no expected_context or relevant_doc_ids; retrieval metrics skipped",
                )
            )
        if not example.has_answer_ground_truth:
            warnings.append(
                DatasetIssue(
                    line=line_number,
                    message="no expected_answer; exact match, token F1 and similarity skipped",
                )
            )
    return DatasetValidation(
        path=str(path),
        examples=[example for _, example in records],
        errors=errors,
        warnings=warnings,
        sha256=digest,
    )


def load_dataset(path: Path | str) -> Dataset:
    validation = validate_dataset(path)
    if not validation.ok or validation.sha256 is None:
        raise DatasetError(validation.path, validation.errors)
    return Dataset(
        examples=validation.examples,
        source=SourceInfo(path=validation.path, sha256=validation.sha256),
    )


def load_corpus(path: Path | str) -> Corpus:
    path = Path(path)
    records, errors, digest = _parse_jsonl(path, Document, "doc")
    if errors or digest is None:
        raise DatasetError(path, errors)
    return Corpus(
        documents=[document for _, document in records],
        source=SourceInfo(path=str(path), sha256=digest),
    )


def pooled_corpus(dataset: Dataset) -> Corpus:
    """Build a corpus from the unique expected contexts of a dataset, in first-seen order."""
    texts = list(dict.fromkeys(text for ex in dataset.examples for text in ex.expected_context))
    documents = [Document(id=f"ctx-{index}", text=text) for index, text in enumerate(texts, 1)]
    digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
    return Corpus(documents=documents, source=SourceInfo(path=None, sha256=digest))


def find_unknown_doc_ids(examples: Iterable[EvalExample], corpus: Corpus) -> list[DatasetIssue]:
    known = {document.id for document in corpus.documents}
    return [
        DatasetIssue(message=f"example {example.id!r} references unknown doc id {doc_id!r}")
        for example in examples
        for doc_id in example.relevant_doc_ids
        if doc_id not in known
    ]
