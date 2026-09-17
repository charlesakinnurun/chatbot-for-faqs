"""Chatbot logic: turn a user question into a confident FAQ answer.

The chatbot is deliberately UI-agnostic -- it exposes a single ``respond``
method that returns a rich ``ChatResponse``.  Streamlit, CLI scripts and
tests all consume the same object, so business logic stays independent of
the presentation layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings, settings as default_settings
from src.data_loading import load_faq_dataset
from src.preprocessing import TextPreprocessor
from src.retrieval import FAQRetriever, RetrievalResult
from src.vectorizer import TfidfTextVectorizer


@dataclass(frozen=True)
class ChatResponse:
    """Result of answering one user question."""

    question: str
    answer: str
    confidence: float
    is_fallback: bool
    faq_id: str | None = None
    category: str | None = None
    matched_question: str | None = None
    top_hits: tuple = ()  # optional ranked candidates for debugging


class FAQChatbot:
    """Stateful FAQ chatbot.

    Parameters
    ----------
    retriever:
        A fitted ``FAQRetriever`` used to score user questions.
    threshold:
        Minimum cosine similarity required to answer confidently.  Scores
        below this trigger the fallback response.
    fallback_response:
        Message returned when no answer is confident enough.
    """

    def __init__(
        self,
        retriever: FAQRetriever,
        threshold: float,
        fallback_response: str,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.retriever = retriever
        self.threshold = threshold
        self.fallback_response = fallback_response

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        threshold: float = default_settings.confidence_threshold,
        fallback_response: str = default_settings.fallback_response,
        preprocessor: TextPreprocessor | None = None,
        max_results: int = default_settings.max_results,
        **vectorizer_kwargs: object,
    ) -> "FAQChatbot":
        """Build a fully wired chatbot straight from a FAQ CSV.

        Parameters
        ----------
        csv_path:
            Path to ``faq_dataset.csv``.
        threshold:
            Confidence threshold for fallback behaviour.
        fallback_response:
            Message to show when no confident answer is found.
        preprocessor:
            Optional preprocessor instance to share across the pipeline.
        max_results:
            Number of ranked candidates kept internally.
        **vectorizer_kwargs:
            Forwarded to the :class:`TfidfTextVectorizer`.

        Returns
        -------
        FAQChatbot
        """
        retriever = FAQRetriever.from_csv(
            csv_path=csv_path,
            preprocessor=preprocessor,
            max_results=max_results,
            **vectorizer_kwargs,  # type: ignore[arg-type]
        )
        return cls(
            retriever=retriever,
            threshold=threshold,
            fallback_response=fallback_response,
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "FAQChatbot":
        """Build a chatbot from a :class:`Settings` object (CLI/UI default)."""
        settings = settings or default_settings
        return cls.from_csv(
            csv_path=settings.resolve_faq_dataset(),
            threshold=settings.confidence_threshold,
            fallback_response=settings.fallback_response,
            max_results=settings.max_results,
        )

    # -- answer logic ----------------------------------------------------------
    def respond(self, question: str | None) -> ChatResponse:
        """Answer a single user question.

        Parameters
        ----------
        question:
            Raw user input.  ``None`` or whitespace-only input is handled
            gracefully and never raises.

        Returns
        -------
        ChatResponse
            Either a confident FAQ answer or the fallback response.
        """
        text = (question or "").strip()

        if not text:
            return ChatResponse(
                question=text,
                answer=self.fallback_response,
                confidence=0.0,
                is_fallback=True,
            )

        result: RetrievalResult = self.retriever.retrieve(text)
        best = result.best_hit

        if best is None or best.score < self.threshold:
            return ChatResponse(
                question=text,
                answer=self.fallback_response,
                confidence=best.score if best is not None else 0.0,
                is_fallback=True,
                top_hits=result.hits,
            )

        return ChatResponse(
            question=text,
            answer=best.answer,
            confidence=best.score,
            is_fallback=False,
            faq_id=best.faq_id,
            category=best.category,
            matched_question=best.question,
            top_hits=result.hits,
        )

    def diagnose(self, question: str) -> RetrievalResult:
        """Return the raw ranked candidates without any thresholding.

        Useful for the debug panel in the UI and for error analysis.
        """
        return self.retriever.retrieve(question)


def main() -> None:
    """Interactive REPL for using the chatbot from the terminal.

    Useful for quick experiments without launching the Streamlit app:
        python -m src.chatbot
    """
    from src.config import settings  # noqa: PLC0415

    print("Building chatbot ...")
    chatbot = FAQChatbot.from_settings(settings)
    print(settings.greeting)
    print(f"(threshold={chatbot.threshold:.2f}; type 'quit' to exit)\n")

    while True:  # pragma: no cover - interactive REPL
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if not question:
            continue
        response = chatbot.respond(question)
        print(f"Bot: {response.answer}")
        if response.is_fallback:
            print(f"     [fallback · best similarity {response.confidence:.2f}]")
        else:
            print(f"     [matched {response.faq_id} · {response.category} · confidence {response.confidence:.2f}]")


if __name__ == "__main__":
    main()