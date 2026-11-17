from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from rag_eval.models import PipelineOutput


@runtime_checkable
class RAGPipeline(Protocol):
    """Adapter contract: answer one question and report the contexts used to answer it."""

    name: str

    def run(self, question: str, *, top_k: int) -> PipelineOutput: ...


@runtime_checkable
class BatchRAGPipeline(RAGPipeline, Protocol):
    """Optional extension for pipelines that answer many questions in one call."""

    def run_batch(self, questions: Sequence[str], *, top_k: int) -> Sequence[PipelineOutput]: ...


PipelineFunction = Callable[[str, int], PipelineOutput | Mapping[str, Any]]


class FunctionPipeline:
    """Wrap a plain ``fn(question, top_k)`` callable that returns a PipelineOutput or a dict."""

    def __init__(self, fn: PipelineFunction, name: str | None = None) -> None:
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "function")

    def run(self, question: str, *, top_k: int) -> PipelineOutput:
        result = self._fn(question, top_k)
        if isinstance(result, PipelineOutput):
            return result
        return PipelineOutput.model_validate(result)
