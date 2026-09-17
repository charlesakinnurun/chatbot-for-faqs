"""Text preprocessing: raw user input -> clean, normalized tokens.

The pipeline is deliberately conservative.  FAQ questions are short and
query-like; aggressive cleaning (removing numbers, heavy stemming, dropping
every function word) destroys the very lexical clues that TF-IDF retrieval
relies on.  Every stage is a small, separately testable unit so the
pipeline can be tuned or extended without touching the rest of the system.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from src.config import EXTRA_STOPWORDS, NEGATION_WHITELIST

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

#: Matches words; contractions have been expanded before tokenization so only
#: plain alphanumeric runs survive.
_WORD_RE = re.compile(r"[a-z0-9]+")

#: Map of common English contractions to their expanded form.  Applied
#: *before* tokenization so that "can't" becomes ["can", "not"] and the
#: negation is preserved for matching.  Patterns intentionally avoid word
#: boundaries -- the apostrophe itself is the boundary and the suffix is
#: attached to a preceding word character.  Irregular contractions that the
#: generic "n't" rule would mangle are listed first.
_CONTRACTION_REPLACEMENTS = (
    (re.compile(r"can't"), "can not"),
    (re.compile(r"won't"), "will not"),
    (re.compile(r"shan't"), "shall not"),
    (re.compile(r"ain't"), "is not"),
    (re.compile(r"n't"), " not"),
    (re.compile(r"'ere"), " were"),
    (re.compile(r"'cause"), " because"),
    (re.compile(r"'em"), " them"),
    (re.compile(r"'ya"), " you"),
    (re.compile(r"'ll"), " will"),
    (re.compile(r"'re"), " are"),
    (re.compile(r"'ve"), " have"),
    (re.compile(r"'m"), " am"),
    (re.compile(r"'d"), " would"),
    (re.compile(r"'s"), ""),
)

#: Typographic apostrophes/quotes that should be treated as ASCII.
_UNICODE_QUOTES: dict[str, str] = {
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
}


def strip_curly_quotes(text: str) -> str:
    """Replace curly/unicode punctuation with ASCII equivalents."""
    for source, target in _UNICODE_QUOTES.items():
        text = text.replace(source, target)
    return text


def expand_contractions(text: str) -> str:
    """Expand common contractions into standalone tokens.

    Examples
    --------
    >>> expand_contractions("I can't reset my password")
    'I can not reset my password'
    """
    for pattern, replacement in _CONTRACTION_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def tokenize(text: str) -> list[str]:
    """Lowercase, expand contractions and split text into word tokens.

    Numbers are kept unless configured otherwise; hyphens split words so
    that "sign-in" and "sign in" produce the same tokens.
    """
    clean = expand_contractions(strip_curly_quotes(text.lower()))
    return _WORD_RE.findall(clean)


# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------

#: Curated, deliberately *conservative* stop-word list.  FAQ queries are short
#: and terse; aggressively removing every function word destroys the lexical
#: overlap that lexical retrieval depends on (e.g. "get my money back").
_DEFAULT_STOPWORDS: tuple[str, ...] = (
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "so",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "up",
    "about",
    "into",
    "over",
    "after",
    "before",
    "between",
    "i",
    "me",
    "my",
    "we",
    "our",
    "us",
    "you",
    "your",
    "he",
    "she",
    "it",
    "its",
    "they",
    "them",
    "their",
    "this",
    "that",
    "these",
    "those",
    "who",
    "whom",
    "whose",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "can",
    "could",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "there",
    "here",
    "where",
    "when",
    "why",
    "any",
    "each",
    "other",
    "some",
    "such",
    "only",
    "own",
    "same",
    "too",
    "very",
    "just",
    "now",
    "also",
    "him",
    "her",
    "than",
    "upon",
    "via",
    "few",
    "more",
    "most",
    "etc",
    "etc.",
)  # type: ignore[reportAssignment]


def _stopword_set(extra: Iterable[str] = ()) -> set[str]:
    """Merge the curated stop words with domain extras + negation whitelist.

    Negation words are explicitly whitelisted so that "can not log in" is not
    conflated with "can log in".
    """
    stopwords = set(_DEFAULT_STOPWORDS)
    stopwords.update(extra)
    stopwords.difference_update(NEGATION_WHITELIST)
    return stopwords


# ---------------------------------------------------------------------------
# Normalizers (stemming / lemmatization)
# ---------------------------------------------------------------------------

def _load_stemmer():
    """Return an offline Snowball stemmer (no external resources needed)."""
    from nltk.stem.snowball import EnglishStemmer  # noqa: PLC0415

    return EnglishStemmer(ignore_stopwords=True)


def _load_lemmatizer():
    """Return a WordNet lemmatizer, raising a helpful error if data is missing."""
    try:
        from nltk.stem import WordNetLemmatizer  # noqa: PLC0415
        from nltk.corpus import wordnet  # noqa: F401  (forces the proxy check)
        wordnet.ensure_loaded()  # type: ignore[attr-defined]
    except LookupError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "WordNet data is missing. Run `python -m nltk.downloader wordnet omw-1.4` "
            "and try again, or set normalization='stem'."
        ) from exc
    return WordNetLemmatizer()  # type: ignore[no-any-return]


class TextPreprocessor:
    """Configurable text -> token pipeline.

    Parameters
    ----------
    normalization:
        ``"stem"`` applies offline Snowball stemming (default), ``"lemma"``
        applies WordNet lemmatization, ``"none"`` keeps tokens verbatim.
    extra_stopwords:
        Additional stop words beyond the built-in English list.
    remove_numbers:
        If ``True``, pure-digit tokens are dropped.
    """

    def __init__(
        self,
        normalization: str = "stem",
        extra_stopwords: Iterable[str] = EXTRA_STOPWORDS,
        remove_numbers: bool = True,
    ) -> None:
        if normalization not in {"stem", "lemma", "none"}:
            raise ValueError(
                f"Unknown normalization {normalization!r}; " 'expected "stem", "lemma" or "none".'
            )
        self.normalization = normalization
        self.remove_numbers = remove_numbers
        self._stopwords = _stopword_set(extra_stopwords)

        self._stemmer = _load_stemmer() if normalization == "stem" else None
        self._lemmatizer = _load_lemmatizer() if normalization == "lemma" else None

    def transform(self, text: str) -> list[str]:
        """Run the full preprocessing pipeline on a single string.

        Returns
        -------
        list[str]
            Clean, lowercased, normalized tokens.
        """
        tokens = tokenize(text)
        tokens = self._filter_tokens(tokens)
        tokens = self._normalize(tokens)
        tokens = self._drop_stopwords(tokens)
        return tokens

    def _filter_tokens(self, tokens: list[str]) -> list[str]:
        """Remove tokens that carry no lexical signal (e.g. bare numbers)."""
        if not self.remove_numbers:
            return tokens
        return [tok for tok in tokens if not tok.isdigit()]

    def _drop_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stop words while preserving negations (whitelisted)."""
        return [tok for tok in tokens if tok not in self._stopwords]

    def _normalize(self, tokens: list[str]) -> list[str]:
        """Stem or lemmatize each token depending on the configured mode."""
        if self.normalization == "none":
            return tokens
        if self._stemmer is not None:
            return [self._stemmer.stem(tok) for tok in tokens]
        if self._lemmatizer is not None:
            # Prefer a verb reading (questions are action-oriented); fall back
            # to the default noun reading if the verb reading is unchanged.
            return [self._lemmatize_token(tok) for tok in tokens]
        return tokens

    def _lemmatize_token(self, token: str) -> str:
        """Lemmatize a token, checking the verb reading first."""
        lemma = self._lemmatizer.lemmatize(token, pos="v")
        if lemma == token:
            lemma = self._lemmatizer.lemmatize(token)
        return lemma


def preprocess(text: str, **kwargs: object) -> list[str]:
    """Convenience wrapper: build a default preprocessor and transform text."""
    return TextPreprocessor(**kwargs).transform(text)