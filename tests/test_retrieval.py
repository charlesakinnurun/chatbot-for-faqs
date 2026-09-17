"""Unit tests for the FAQ retrieval layer."""

from pathlib import Path

import pytest

from src.retrieval import FAQRetriever, RetrievalResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FAQ_CSV = DATA_DIR / "faq_dataset.csv"


@pytest.fixture(scope="module")
def retriever() -> FAQRetriever:
    return FAQRetriever.from_csv(str(FAQ_CSV), max_results=3)


class TestRetrieval:
    def test_known_faq_question_returns_expected_id(self, retriever: FAQRetriever) -> None:
        result = retriever.retrieve("How do I track my order?")
        assert result.best_hit is not None
        assert result.best_hit.faq_id == "orders_01"

    def test_similar_question_returns_expected_family(self, retriever: FAQRetriever) -> None:
        result = retriever.retrieve("Where is my order right now?")
        assert result.best_hit is not None
        assert result.best_hit.faq_id in {"orders_01", "orders_02"}

    def test_completely_unrelated_question_scores_low(self, retriever: FAQRetriever) -> None:
        result = retriever.retrieve("Tell me a joke about quantum physics")
        assert result.best_score < 0.35

    def test_multiple_competing_faqs_picks_specific_one(self, retriever: FAQRetriever) -> None:
        # Several returns_* entries overlap; the period-specific question must
        # win rather than a generic "how to return" entry.
        result = retriever.retrieve("How long do I have to return an item?")
        assert result.best_hit is not None
        assert result.best_hit.faq_id == "returns_03"

    def test_empty_query_returns_no_hits(self, retriever: FAQRetriever) -> None:
        result = retriever.retrieve("")
        assert result.hits == ()

    def test_whitespace_query_returns_no_hits(self, retriever: FAQRetriever) -> None:
        result = retriever.retrieve("    ")
        assert result.hits == ()

    def test_k_parameter_controls_number_of_hits(self, retriever: FAQRetriever) -> None:
        result = retriever.retrieve("How do I return an item?", k=5)
        assert len(result.hits) <= 5
        assert 1 <= len(result.hits)

    def test_hits_are_sorted_descending(self, retriever: FAQRetriever) -> None:
        result = retriever.retrieve("How do I get my money back?")
        scores = [hit.score for hit in result.hits]
        assert scores == sorted(scores, reverse=True)

    def test_retrieval_result_has_expected_interface(self) -> None:
        result = RetrievalResult(query="x", hits=())
        assert result.best_hit is None
        assert result.best_score == 0.0
        assert result.faq_ids() == []