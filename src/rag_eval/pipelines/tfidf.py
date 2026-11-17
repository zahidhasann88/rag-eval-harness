import math
from collections import Counter
from collections.abc import Sequence

from rag_eval.models import Document, PipelineOutput, RetrievedContext
from rag_eval.text import content_terms, split_sentences

Vector = dict[str, float]


class TfidfIndex:
    """In-memory TF-IDF index with smoothed IDF, ``ln((1 + N) / (1 + df)) + 1``, and L2-normalized
    vectors, so cosine similarity is a dot product. Terms unseen at fit time are ignored."""

    def __init__(self, texts: Sequence[str]) -> None:
        tokenized = [content_terms(text) for text in texts]
        document_frequency = Counter(term for terms in tokenized for term in set(terms))
        total = len(texts)
        self._idf = {
            term: math.log((1 + total) / (1 + count)) + 1
            for term, count in document_frequency.items()
        }
        self._vectors = [self._weigh(terms) for terms in tokenized]

    def _weigh(self, terms: Sequence[str]) -> Vector:
        weights = {
            term: count * self._idf[term]
            for term, count in Counter(terms).items()
            if term in self._idf
        }
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        return {term: weight / norm for term, weight in weights.items()} if norm else {}

    def vectorize(self, text: str) -> Vector:
        return self._weigh(content_terms(text))

    @staticmethod
    def similarity(left: Vector, right: Vector) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(weight * right.get(term, 0.0) for term, weight in left.items())

    def search(self, text: str, k: int) -> list[tuple[int, float]]:
        query = self.vectorize(text)
        scored = [
            (index, score)
            for index, vector in enumerate(self._vectors)
            if (score := self.similarity(query, vector)) > 0
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:k]


class TfidfPipeline:
    """Local baseline: TF-IDF retrieval plus an extractive answer, the sentence from the top
    passages most similar to the question. No language model is involved."""

    name = "tfidf"
    model_name = "tfidf-extractive"

    def __init__(self, documents: Sequence[Document], answer_passages: int = 3) -> None:
        if answer_passages < 1:
            raise ValueError("answer_passages must be >= 1")
        self._documents = list(documents)
        self._index = TfidfIndex([document.text for document in self._documents])
        self._answer_passages = answer_passages

    def run(self, question: str, *, top_k: int) -> PipelineOutput:
        hits = self._index.search(question, top_k)
        contexts = [
            RetrievedContext(
                text=self._documents[index].text, doc_id=self._documents[index].id, score=score
            )
            for index, score in hits
        ]
        return PipelineOutput(
            answer=self._extract_answer(question, contexts),
            contexts=contexts,
            model=self.model_name,
        )

    def _extract_answer(self, question: str, contexts: Sequence[RetrievedContext]) -> str:
        if not contexts:
            return ""
        query = self._index.vectorize(question)
        best_sentence, best_score = "", 0.0
        for context in contexts[: self._answer_passages]:
            for sentence in split_sentences(context.text):
                score = self._index.similarity(query, self._index.vectorize(sentence))
                if score > best_score:
                    best_sentence, best_score = sentence, score
        return best_sentence or split_sentences(contexts[0].text)[0]
