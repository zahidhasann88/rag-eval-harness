from collections.abc import Sequence

from rag_eval.models import Document, PipelineOutput, RetrievedContext
from rag_eval.text import content_terms


class MockPipeline:
    """Deterministic demo pipeline: Jaccard keyword retrieval, answers with the top passage."""

    name = "mock"
    model_name = "mock-extractive"

    def __init__(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)
        self._terms = [frozenset(content_terms(document.text)) for document in self._documents]

    def run(self, question: str, *, top_k: int) -> PipelineOutput:
        query = frozenset(content_terms(question))
        scored = []
        for index, terms in enumerate(self._terms):
            union = query | terms
            score = len(query & terms) / len(union) if union else 0.0
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        contexts = [
            RetrievedContext(
                text=self._documents[index].text, doc_id=self._documents[index].id, score=score
            )
            for score, index in scored[:top_k]
        ]
        answer = contexts[0].text if contexts else ""
        return PipelineOutput(answer=answer, contexts=contexts, model=self.model_name)
