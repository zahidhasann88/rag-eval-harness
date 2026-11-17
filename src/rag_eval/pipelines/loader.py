from collections.abc import Callable, Sequence

from rag_eval.models import Document
from rag_eval.pipelines.base import RAGPipeline
from rag_eval.pipelines.mock import MockPipeline
from rag_eval.pipelines.tfidf import TfidfPipeline
from rag_eval.plugins import PluginLoadError, import_object, materialize

BUILTIN_PIPELINES: dict[str, Callable[[Sequence[Document]], RAGPipeline]] = {
    "mock": MockPipeline,
    "tfidf": TfidfPipeline,
}


class PipelineLoadError(Exception):
    pass


def load_pipeline(spec: str, documents: Sequence[Document] = ()) -> RAGPipeline:
    """Build a pipeline from a built-in name or a ``module:attribute`` import path.

    Built-ins index ``documents``. An import path may point to a pipeline instance, a class, or a
    zero-argument factory.
    """
    if spec in BUILTIN_PIPELINES:
        if not documents:
            raise PipelineLoadError(f"pipeline {spec!r} requires a non-empty corpus")
        return BUILTIN_PIPELINES[spec](documents)
    try:
        pipeline = materialize(import_object(spec))
    except PluginLoadError as exc:
        known = ", ".join(sorted(BUILTIN_PIPELINES))
        raise PipelineLoadError(f"{exc} (built-in pipelines: {known})") from exc
    if not isinstance(pipeline, RAGPipeline):
        raise PipelineLoadError(
            f"{spec!r} does not satisfy RAGPipeline: needs a 'name' attribute and "
            "a run(question, *, top_k) method"
        )
    return pipeline
