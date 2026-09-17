"""Unit tests for the text preprocessing pipeline."""

import pytest

from src.preprocessing import TextPreprocessor, expand_contractions, tokenize


@pytest.fixture(scope="module")
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


class TestTokenize:
    def test_empty_input_returns_no_tokens(self) -> None:
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_normal_text(self) -> None:
        assert tokenize("Reset my password") == ["reset", "my", "password"]

    def test_capitalization_is_lowered(self) -> None:
        assert tokenize("RESET My PaSsWoRd") == ["reset", "my", "password"]

    def test_punctuation_is_removed(self) -> None:
        assert tokenize("Hello, world!!") == ["hello", "world"]

    def test_hyphenated_words_split_consistently(self) -> None:
        assert tokenize("sign-in page") == ["sign", "in", "page"]

    def test_unicode_quotes_are_normalized(self) -> None:
        assert tokenize("the \u2018quick\u2019 brown fox") == ["the", "quick", "brown", "fox"]

    def test_contraction_not_is_expanded(self) -> None:
        assert "not" in tokenize("I can't log in")
        assert expand_contractions("don't") == "do not"

    def test_numbers_can_be_filtered(self) -> None:
        assert tokenize("version 2.0") == ["version", "2", "0"]


class TestContractions:
    def test_irregular_contractions(self) -> None:
        assert expand_contractions("won't") == "will not"
        assert expand_contractions("can't") == "can not"
        assert expand_contractions("shan't") == "shall not"
        assert expand_contractions("ain't") == "is not"

    def test_regular_contractions(self) -> None:
        assert expand_contractions("shouldn't") == "should not"
        assert expand_contractions("I'll") == "I will"
        assert expand_contractions("we're") == "we are"
        assert expand_contractions("you've") == "you have"
        assert expand_contractions("I'm") == "I am"
        assert expand_contractions("she'd") == "she would"
        assert expand_contractions("Mary's account") == "Mary account"


class TestPreprocessor:
    def test_empty_input(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.transform("") == []

    def test_stop_words_removed(self, preprocessor: TextPreprocessor) -> None:
        tokens = preprocessor.transform("the and a to of")
        assert all(tok not in {"the", "and", "a", "to", "of"} for tok in tokens)

    def test_negation_not_removed(self, preprocessor: TextPreprocessor) -> None:
        tokens = preprocessor.transform("I can not log into my account")
        assert "not" in tokens

    def test_stop_words_do_not_wipe_semantics(self, preprocessor: TextPreprocessor) -> None:
        # "get" and "back" carry meaning -- they must survive preprocessing.
        tokens = preprocessor.transform("How do I get my money back")
        assert {"get", "money", "back"}.issubset(tokens)

    def test_root_words_survive_stemming(self, preprocessor: TextPreprocessor) -> None:
        tokens = preprocessor.transform("Payments and shipping options")
        assert "payment" in tokens  # payments -> payment
        assert "ship" in tokens  # shipping -> ship

    def test_lemma_mode(self) -> None:
        tokens = TextPreprocessor(normalization="lemma").transform("I forgot my password")
        assert "forget" in tokens
        assert "password" in tokens

    def test_no_normalization_mode(self) -> None:
        tokens = TextPreprocessor(normalization="none").transform("I forgot running")
        assert "forgot" in tokens

    def test_invalid_normalization_raises(self) -> None:
        with pytest.raises(ValueError):
            TextPreprocessor(normalization="silent-fail")