from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from rag_eval.models import EvalConfig, RetrievedContext, SourceInfo, TokenUsage

SCHEMA_VERSION = 1

RETRIEVAL_METRICS = ("precision_at_k", "recall_at_k", "hit_rate_at_k", "mrr")
ANSWER_METRICS = ("exact_match", "token_f1", "answer_similarity", "groundedness")
HEURISTIC_METRICS = frozenset({"answer_similarity", "groundedness"})


def metric_label(name: str, top_k: int) -> str:
    labels = {
        "precision_at_k": f"Precision@{top_k}",
        "recall_at_k": f"Recall@{top_k}",
        "hit_rate_at_k": f"Hit Rate@{top_k}",
        "mrr": f"MRR@{top_k}",
        "exact_match": "Exact Match",
        "token_f1": "Token F1",
        "answer_similarity": "Answer Similarity",
        "groundedness": "Groundedness",
        "pass_rate": "Pass Rate",
    }
    return labels.get(name, name)


class RetrievalScores(BaseModel):
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    reciprocal_rank: float
    relevant_ranks: list[int] = Field(default_factory=list)
    expected_count: int


class AnswerScores(BaseModel):
    exact_match: float | None = None
    token_f1: float | None = None
    answer_similarity: float | None = None
    groundedness: float | None = None
    unsupported_terms: list[str] = Field(default_factory=list)


class FailureStage(StrEnum):
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    PIPELINE_ERROR = "pipeline_error"


class Failure(BaseModel):
    stage: FailureStage
    reasons: list[str]


class ExampleResult(BaseModel):
    id: str
    question: str
    expected_answer: str | None = None
    expected_context: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    answer: str | None = None
    contexts: list[RetrievedContext] = Field(default_factory=list)
    latency_ms: float | None = None
    token_usage: TokenUsage | None = None
    model: str | None = None
    retrieval: RetrievalScores | None = None
    answer_scores: AnswerScores | None = None
    judge_scores: dict[str, float] = Field(default_factory=dict)
    judge_errors: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    failure: Failure | None = None

    @property
    def passed(self) -> bool:
        return self.failure is None


class MetricSummary(BaseModel):
    mean: float | None
    count: int


class LatencySummary(BaseModel):
    count: int = 0
    mean_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    max_ms: float | None = None


class TokenSummary(BaseModel):
    examples_with_usage: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RunSummary(BaseModel):
    num_examples: int
    num_passed: int
    num_failed: int
    failures_by_stage: dict[FailureStage, int]


class RunMetadata(BaseModel):
    fingerprint: str
    tool_version: str
    created_at: datetime
    pipeline: str
    models: list[str] = Field(default_factory=list)
    dataset: SourceInfo
    corpus: SourceInfo | None = None
    config: EvalConfig
    judges: list[str] = Field(default_factory=list)
    python_version: str


class EvaluationReport(BaseModel):
    schema_version: int = SCHEMA_VERSION
    metadata: RunMetadata
    summary: RunSummary
    metrics: dict[str, MetricSummary]
    latency: LatencySummary
    tokens: TokenSummary
    results: list[ExampleResult]

    @property
    def failures(self) -> list[ExampleResult]:
        return [result for result in self.results if not result.passed]
