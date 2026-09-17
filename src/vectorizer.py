"""Vectorization layer: text -> numerical feature vectors.

``TextVectorizer`` defines a small interface that the retrieval layer
depends on.  The shipped implementation wraps scikit-learn's TF-IDF, but a
sentence-transformer or embedding-based vectorizer can slot in later with
zero changes to ``FAQRetriever`` -- this is the extension point that keeps
the project future-proof.
"""

from __future__ import annotations

from typing import Callable, Sequence

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    """Minimal interface for any document-to-vector transform.

    Implementations must be able to fit on a corpus and then transform
    individual strings into sparse vectors whose columns (features) are
    aligned with the fitted corpus.
    """

    def fit(self, documents: Sequence[str]) -> "TextVectorizer":
        """Fit the vectorizer on a corpus of documents."""
        raise NotImplementedError

    def transform(self, text: str) -> sparse.csr_matrix:
        """Convert a single document into a feature vector (1 x D)."""
        raise NotImplementedError

    def fit_transform(self, documents: Sequence[str]) -> sparse.csr_matrix:
        """Fit on the corpus and return the full document-term matrix."""
        self.fit(documents)
        return sparse.vstack([self.transform(doc) for doc in documents])


class TfidfTextVectorizer(TextVectorizer):
    """TF-IDF vectorizer configured for short FAQ questions.

    Parameters
    ----------
    tokenizer:
        A callable mapping a raw string to a list of tokens.  Pass the
        application's ``TextPreprocessor.transform`` to reuse the exact same
        preprocessing pipeline during retrieval.
    ngram_range:
        Lower/upper n-gram range; bigrams help match phrasal FAQ questions
        such as "reset password" or "return policy".
    sublinear_tf:
        Use 1 + log(tf), dampening the influence of very frequent terms.
    max_features:
        Optional cap on vocabulary size.
    """

    def __init__(
        self,
        tokenizer: Callable[[str], list[str]] | None = None,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        max_features: int | None = None,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            lowercase=False,  # the tokenizer already lowercases
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            max_features=max_features,
            token_pattern=None,  # silence warning: a callable tokenizer is given
        )
        self._fitted = False

    def fit(self, documents: Sequence[str]) -> "TfidfTextVectorizer":
        self._vectorizer.fit(documents)
        self._fitted = True
        return self

    def transform(self, text: str) -> sparse.csr_matrix:
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fitted on a corpus before transform().")
        return self._vectorizer.transform([text])

    def fit_transform(self, documents: Sequence[str]) -> sparse.csr_matrix:
        self.fit(documents)
        return self._vectorizer.transform(list(documents))

    @property
    def vocabulary_size(self) -> int:
        """Number of learned features (vocabulary size)."""
        if not self._fitted:
            return 0
        return len(self._vectorizer.vocabulary_)