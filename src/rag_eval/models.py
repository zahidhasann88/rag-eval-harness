from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _as_list(value: Any) -> Any:
    return [value] if isinstance(value, str) else value


class RetrievedContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    doc_id: str | None = None
    score: float | None = None


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class PipelineOutput(BaseModel):
    """What a RAG pipeline returns for a single question."""

    model_config = ConfigDict(frozen=True)

    answer: str
    contexts: list[RetrievedContext] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0)
    token_usage: TokenUsage | None = None
    model: str | None = None

    @field_validator("contexts", mode="before")
    @classmethod
    def _coerce_contexts(cls, value: Any) -> Any:
        if isinstance(value, list | tuple):
            return [{"text": item} if isinstance(item, str) else item for item in value]
        return value


class EvalExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, coerce_numbers_to_str=True)

    id: str = Field(min_length=1)
    question: str
    expected_answer: str | None = None
    expected_context: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("expected_context", "relevant_doc_ids", mode="before")
    @classmethod
    def _wrap_single_value(cls, value: Any) -> Any:
        return _as_list(value)

    @field_validator("expected_context", "relevant_doc_ids")
    @classmethod
    def _entries_not_blank(cls, value: list[str]) -> list[str]:
        if any(not entry.strip() for entry in value):
            raise ValueError("entries must not be blank")
        return value

    @property
    def has_retrieval_ground_truth(self) -> bool:
        return bool(self.expected_context or self.relevant_doc_ids)

    @property
    def has_answer_ground_truth(self) -> bool:
        return self.expected_answer is not None and bool(self.expected_answer.strip())


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, coerce_numbers_to_str=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str | None = None
    sha256: str


class Dataset(BaseModel):
    examples: list[EvalExample]
    source: SourceInfo | None = None

    def head(self, limit: int) -> "Dataset":
        return self.model_copy(update={"examples": self.examples[:limit]})


class Corpus(BaseModel):
    documents: list[Document]
    source: SourceInfo | None = None


class EvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    top_k: int = Field(default=5, ge=1)
    batch_size: int = Field(default=16, ge=1)
    context_match_threshold: float = Field(default=0.75, gt=0, le=1)
    min_recall: float = Field(default=0.5, ge=0, le=1)
    min_answer_f1: float = Field(default=0.5, ge=0, le=1)
    min_groundedness: float = Field(default=0.5, ge=0, le=1)
