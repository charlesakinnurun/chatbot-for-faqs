"""Similarity-based retrieval over the FAQ knowledge base.

``FAQRetriever`` converts every FAQ question into a TF-IDF vector once
(indexing step), then scores an incoming user question with cosine
similarity and returns the top-K matches.  Because retrieval is decoupled
from the chatbot logic, swapping TF-IDF for embeddings later only requires
a different ``TextVectorizer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.data_loading import load_faq_dataset
from src.preprocessing import TextPreprocessor
from src.vectorizer import TextVectorizer, TfidfTextVectorizer


@dataclass(frozen=True)
class RetrievalHit:
    """A single ranked candidate from the knowledge base."""

    rank: int
    faq_id: str
    category: str
    question: str
    answer: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked retrieval output for one user question."""

    query: str
    hits: tuple[RetrievalHit, ...]

    @property
    def best_hit(self) -> RetrievalHit | None:
        """The highest scoring hit, if any."""
        return self.hits[0] if self.hits else None

    @property
    def best_score(self) -> float:
        """Cosine similarity of the best hit (``0.0`` if no hits)."""
        hit = self.best_hit
        return hit.score if hit is not None else 0.0

    def faq_ids(self) -> list[str]:
        """Top-K matched FAQ ids, in rank order."""
        return [hit.faq_id for hit in self.hits]


class FAQRetriever:
    """Cosine-similarity retriever over a set of FAQ questions.

    Parameters
    ----------
    faqs:
        DataFrame with at least ``id``, ``category``, ``question`` and
        ``answer`` columns.
    vectorizer:
        A fitted-or-unfitted ``TextVectorizer``.  ``fit`` is applied to the
        FAQ questions on construction.
    max_results:
        Default number of ranked hits returned by ``retrieve``.
    """

    def __init__(
        self,
        faqs: pd.DataFrame,
        vectorizer: TextVectorizer | None = None,
        max_results: int = 3,
    ) -> None:
        required = {"id", "category", "question", "answer"}
        missing = required - set(faqs.columns)
        if missing:
            raise ValueError(f"faqs must contain columns {sorted(required)}; missing {sorted(missing)}")

        self.faqs = faqs.reset_index(drop=True)
        self.max_results = max(1, max_results)

        if vectorizer is not None:
            self.vectorizer = vectorizer
        else:
            preprocessor = TextPreprocessor()
            self.vectorizer = TfidfTextVectorizer(tokenizer=preprocessor.transform)

        self._indexed = False
        self._index_vectors: object | None = None  # set by _build_index

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        preprocessor: TextPreprocessor | None = None,
        max_results: int = 3,
        **vectorizer_kwargs: object,
    ) -> "FAQRetriever":
        """Build a retriever directly from a FAQ CSV file.

        Parameters
        ----------
        csv_path:
            Path to ``faq_dataset.csv``.
        preprocessor:
            Optional preprocessor to share with the vectorizer.
        max_results:
            Default number of hits.
        **vectorizer_kwargs:
            Forwarded to :class:`TfidfTextVectorizer` (e.g. ``ngram_range``).

        Returns
        -------
        FAQRetriever
        """
        faqs = load_faq_dataset(csv_path)
        vec = TfidfTextVectorizer(
            tokenizer=(preprocessor or TextPreprocessor()).transform,
            **vectorizer_kwargs,  # type: ignore[arg-type]
        )
        return cls(faqs=faqs, vectorizer=vec, max_results=max_results)

    def _build_index(self) -> None:
        """Fit the vectorizer on FAQ questions and cache the term-document matrix."""
        self._index_vectors = self.vectorizer.fit_transform(list(self.faqs["question"])).tocsc()
        self._indexed = True

    def _ensure_index(self) -> None:
        if not self._indexed:
            self._build_index()

    def _score_all(self, query: str) -> np.ndarray:
        """Return a cosine-similarity score for every FAQ entry."""
        self._ensure_index()
        query_vector = self.vectorizer.transform(query)
        # FAQ rows and the query are both L2-normalized by TF-IDF, so the dot
        # product equals cosine similarity.
        scores = self._index_vectors.dot(query_vector.T).toarray().ravel()  # type: ignore[union-attr]
        return np.asarray(scores, dtype=float)

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        """Rank FAQ entries by similarity to ``query``.

        Parameters
        ----------
        query:
            Raw user question (preprocessing happens inside the vectorizer).
        k:
            Number of ranked hits to return; defaults to ``max_results``.

        Returns
        -------
        RetrievalResult
            An empty result (no hits) if the query produces no features.
        """
        k = int(k) if k is not None else self.max_results
        k = max(1, k)

        if not query or not query.strip():
            return RetrievalResult(query=query, hits=())

        scores = self._score_all(query)
        if not np.any(scores > 0.0):
            # No shared vocabulary -> nothing to rank; avoid leaking garbage.
            return RetrievalResult(query=query, hits=())

        top_indices = np.argsort(-scores)[:k]
        selected = self.faqs.iloc[top_indices]
        hits = tuple(
            RetrievalHit(
                rank=rank,
                faq_id=str(row.id),
                category=str(row.category),
                question=str(row.question),
                answer=str(row.answer),
                score=float(scores[position]),
            )
            for rank, (position, row) in enumerate(
                zip(top_indices, selected.itertuples(index=False, name="FAQRow")), start=1
            )
            if float(scores[position]) > 0.0
        )
        return RetrievalResult(query=query, hits=hits)

    def category_coverage(self) -> pd.Series:
        """Number of FAQ entries per category (useful for reporting)."""
        return self.faqs["category"].value_counts().sort_index()