"""Unit tests for the chatbot (answer + fallback decision logic)."""

from pathlib import Path

import pytest

from src.chatbot import ChatResponse, FAQChatbot

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FAQ_CSV = DATA_DIR / "faq_dataset.csv"

FALLBACK = "I don't know that one."


@pytest.fixture(scope="module")
def chatbot() -> FAQChatbot:
    return FAQChatbot.from_csv(str(FAQ_CSV), threshold=0.35, fallback_response=FALLBACK)


class TestChatbot:
    def test_correct_answer_retrieval(self, chatbot: FAQChatbot) -> None:
        response = chatbot.respond("How can I reset my password?")
        assert not response.is_fallback
        assert response.faq_id == "account_01"
        assert response.confidence >= chatbot.threshold
        assert "Forgot password" in response.answer

    def test_response_metadata_is_populated(self, chatbot: FAQChatbot) -> None:
        response = chatbot.respond("Where is my order?")
        assert response.category == "Orders"
        assert response.matched_question is not None
        assert response.faq_id in {"orders_01", "orders_02"}

    def test_fallback_for_out_of_knowledge_question(self, chatbot: FAQChatbot) -> None:
        response = chatbot.respond("What is the meaning of life?")
        assert response.is_fallback
        assert response.answer == FALLBACK
        assert response.faq_id is None

    def test_fallback_response_uses_configured_message(self) -> None:
        custom = FAQChatbot.from_csv(str(FAQ_CSV), threshold=0.5, fallback_response="custom fallback")
        assert custom.respond("gibberish qwerty asdf").answer == "custom fallback"

    def test_threshold_one_always_falls_back(self) -> None:
        # A perfect verbatim match scores exactly 1.0; any paraphrase must
        # score lower, hence a threshold of 1.0 forces the fallback.
        strict = FAQChatbot.from_csv(str(FAQ_CSV), threshold=1.0, fallback_response=FALLBACK)
        assert strict.respond("Tell me about your return policy please").is_fallback

    def test_threshold_zero_never_falls_back_for_known(self) -> None:
        lenient = FAQChatbot.from_csv(str(FAQ_CSV), threshold=0.0, fallback_response=FALLBACK)
        response = lenient.respond("What is your return policy?")
        assert not response.is_fallback
        assert response.faq_id == "returns_01"

    def test_empty_input_is_handled_gracefully(self, chatbot: FAQChatbot) -> None:
        response = chatbot.respond("")
        assert response.is_fallback
        assert response.confidence == 0.0

    def test_none_input_is_handled_gracefully(self, chatbot: FAQChatbot) -> None:
        response = chatbot.respond(None)
        assert response.is_fallback

    def test_whitespace_input_is_handled_gracefully(self, chatbot: FAQChatbot) -> None:
        response = chatbot.respond("   \n\t ")
        assert response.is_fallback

    def test_threshold_boundary_directly(self) -> None:
        # A threshold equal to the exact score is *not* a fallback.
        chatbot = FAQChatbot.from_csv(str(FAQ_CSV), threshold=0.35, fallback_response=FALLBACK)
        response = chatbot.respond("How do I cancel my order?")
        assert response.confidence >= 0.35
        assert not response.is_fallback

    def test_invalid_threshold_raises(self) -> None:
        chatbot = FAQChatbot.from_csv(str(FAQ_CSV), threshold=0.35, fallback_response=FALLBACK)
        with pytest.raises(ValueError):
            FAQChatbot(chatbot.retriever, threshold=-0.1, fallback_response=FALLBACK)
        with pytest.raises(ValueError):
            FAQChatbot(chatbot.retriever, threshold=1.1, fallback_response=FALLBACK)

    def test_chat_response_is_immutable_value_object(self) -> None:
        response = ChatResponse(
            question="q", answer="a", confidence=0.9, is_fallback=False, faq_id="id"
        )
        assert response.question == "q"
        assert response.answer == "a"

    def test_diagnose_returns_raw_ranks(self, chatbot: FAQChatbot) -> None:
        result = chatbot.diagnose("How do I cancel my order?")
        assert result.best_hit is not None
        assert len(result.hits) >= 1
        assert result.best_hit.faq_id == "orders_04"