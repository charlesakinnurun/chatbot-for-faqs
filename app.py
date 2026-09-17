"""Streamlit chat interface for the FAQ Chatbot.

Run with:  streamlit run app.py

The UI is a thin presentation layer: all business logic lives in
``src.chatbot`` and is reusable without Streamlit (e.g. via the CLI).
"""

from __future__ import annotations

import streamlit as st

from src.chatbot import FAQChatbot
from src.config import settings

st.set_page_config(
    page_title="FAQ Support Assistant",
    page_icon="🤖",
    layout="centered",
)


@st.cache_resource
def build_chatbot() -> FAQChatbot:
    """Build the chatbot once per session; cached across reruns."""
    return FAQChatbot.from_settings(settings)


def _initialise() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": settings.greeting, "meta": None}
        ]


def _clear_chat() -> None:
    st.session_state.messages = [
        {"role": "assistant", "content": settings.greeting, "meta": None}
    ]


def _render_message(message: dict[str, object]) -> None:
    role = str(message["role"])
    content = str(message["content"])
    meta = message.get("meta")

    with st.chat_message(role):
        st.markdown(content)
        if meta:
            st.caption(meta)


def main() -> None:
    st.title("🤖 Frequently Asked Questions")
    st.caption(
        "A support assistant that matches your question to the most relevant "
        "FAQ using TF-IDF + cosine similarity. Answers below a confidence "
        "threshold are rejected instead of guessed."
    )

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Settings")

        threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=settings.confidence_threshold,
            step=0.05,
            help=(
                "Minimum similarity required to answer. Lower = answers more "
                "questions but risks wrong matches; higher = safer but more "
                "fallbacks."
            ),
        )
        show_debug = st.toggle("Show debug details", value=False)
        st.caption(f"Matched FAQ, category and confidence are shown when debug is on.")

        if st.button("🗑  Clear conversation", use_container_width=True):
            _clear_chat()
            st.rerun()

        st.divider()
        st.caption(f"Knowledge base: **{len(build_chatbot().retriever.faqs)} FAQs** across "
                   f"{build_chatbot().retriever.faqs['category'].nunique()} categories.")

    # ---------------------------------------------------------------- history
    _initialise()
    chatbot = build_chatbot()
    if abs(chatbot.threshold - threshold) > 1e-9:
        chatbot = FAQChatbot(
            retriever=chatbot.retriever,
            threshold=threshold,
            fallback_response=settings.fallback_response,
        )

    for message in st.session_state.messages:
        _render_message(message)

    # ---------------------------------------------------------------- input
    if prompt := st.chat_input("Ask me about orders, payments, returns, ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = chatbot.respond(prompt)

        meta = None
        if show_debug and not response.is_fallback:
            meta = (
                f"Matched: **{response.category}** · FAQ **{response.faq_id}** "
                f"· similarity **{response.confidence:.0%}**"
            )
        elif show_debug and response.is_fallback:
            meta = f"No confident match (best similarity {response.confidence:.0%}, below {threshold:.2f})."

        bot_message = {
            "role": "assistant",
            "content": response.answer,
            "meta": meta,
        }
        st.session_state.messages.append(bot_message)
        _render_message(bot_message)


if __name__ == "__main__":
    main()