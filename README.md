# 🤖 Intelligent FAQ Chatbot

A production-quality NLP chatbot that answers natural-language user
questions by retrieving the most relevant entry from a curated FAQ knowledge
base — built with **text preprocessing, TF-IDF vectorization, cosine
similarity, confidence-based fallback handling, offline evaluation and a
Streamlit interface**.

This is not a keyword-matching script. It is a complete, modular ML
pipeline designed to be **evaluated, extended and replaced** piece by piece:
the TF-IDF matcher is one swappable `TextVectorizer` implementation, so the
project can grow into a semantic-search or RAG system without rewriting
anything else.

---

## Project Overview

### Problem

Support teams answer the same questions repeatedly. A chatbot that
*memorises* exact FAQ sentences fails the moment a user rephrases
anything. We need a system that understands **similar intent in different
words**.

### Solution

| Stage | What happens |
| --- | --- |
| User question | Raw text input |
| Text preprocessing | Lowercasing, punctuation cleanup, contraction expansion, stop-word handling, stemming |
| Feature representation | TF-IDF over unigrams **and** bigrams with sublinear term frequency |
| Similarity matching | Cosine similarity between the question and every FAQ question |
| Confidence check | The top score is compared to a configurable threshold |
| Response | Confident match → the FAQ answer · weak match → a friendly fallback |

If the best match is below the confidence threshold, the bot **refuses to
guess** and returns a fallback message instead of a wrong answer.

### Example

```
You:  How do I get my money back?
Bot:  Start a return from the Orders page and request a refund. The refund
      is issued to your original payment method within 5 to 7 business days
      after we receive the package.            [Returns · confidence 0.89]

You:  What is the meaning of life?
Bot:  I'm sorry, I could not find a confident answer to your question.
      Please rephrase it, or contact our support team at support@example.com.
```

---

## Features

* **Modular NLP pipeline** — preprocessing, vectorization, retrieval and
  chatbot logic are independent, reusable modules (no Streamlit required).
* **Realistic knowledge base** — 95 FAQ entries across 8 categories
  (Account, Payments, Orders, Shipping, Returns, Security, Technical
  Support, General) with paraphrased duplicate questions.
* **TF-IDF + cosine similarity** — scikit-learn vectorizer, bigram support,
  sublinear TF, configured through a central `Settings` object.
* **Confidence threshold with fallback** — below-threshold questions are
  refused gracefully instead of answered incorrectly.
* **Offline evaluation harness** — Top-1 / Top-3 accuracy, MRR, fallback
  rate, out-of-scope catch rate, and an automatic **threshold sweep** with
  a saved plot.
* **Error analysis** — `docs/error_analysis.md` documents real failure cases
  and their causes.
* **Streamlit chat UI** — chat history, confidence + debug panel, adjustable
  threshold, clear-conversation button.
* **Unit tests** — 40 pytest tests covering preprocessing, retrieval and
  chatbot behaviour.
* **Extensible design** — swap the `TextVectorizer` to upgrade to sentence
  embeddings, vector databases, RAG or LLM assistance.

## Architecture

```
User Query
    ↓
Preprocessing (src/preprocessing.py)
    ↓
TF-IDF vectorization (src/vectorizer.py)
    ↓
Cosine similarity against FAQ matrix (src/retrieval.py)
    ↓
Best FAQ
    ↓
Confidence threshold (src/config.py)
    ↓
Response / Fallback (src/chatbot.py)
```

The evaluation pipeline (`src/evaluation.py`) replays a held-out test set
(`data/test_questions.csv`) through the same path and reports retrieval
metrics.

```
src/
├── config.py          # paths, threshold, fallback text (env-overridable)
├── preprocessing.py   # tokenize, contractions, stop words, stem/lemmatize
├── vectorizer.py      # TextVectorizer interface + TF-IDF implementation
├── data_loading.py    # dataset loading & schema validation
├── retrieval.py       # FAQRetriever: indexing + cosine scoring
├── chatbot.py         # FAQChatbot: answer/fallback decision + CLI
└── evaluation.py      # metrics, threshold sweep, plots, CLI
```

## Tech Stack

* **Python 3.10+**
* **NLTK** — Snowball stemming / WordNet lemmatization
* **scikit-learn** — `TfidfVectorizer`, stop words
* **pandas** — dataset handling
* **Streamlit** — chat interface
* **pytest** — unit testing
* **matplotlib** — threshold-analysis plot

## Project Structure

```
codealpha-chatbot-for-faqs/
├── app.py                     # Streamlit chat UI (entry point)
├── data/
│   ├── faq_dataset.csv        # 95 FAQ entries across 8 categories
│   └── test_questions.csv     # 109 held-out questions + expectations
├── src/                       # core Python package (library + CLI)
│   ├── __init__.py
│   ├── config.py              # central settings (threshold, paths, fallback)
│   ├── preprocessing.py       # text -> tokens (stem/lemmatize)
│   ├── vectorizer.py          # TextVectorizer interface + TF-IDF impl
│   ├── data_loading.py        # dataset loading & schema validation
│   ├── retrieval.py           # FAQRetriever: indexing + cosine scoring
│   ├── chatbot.py             # FAQChatbot: answer/fallback decision + CLI
│   └── evaluation.py          # metrics, threshold sweep, plots, CLI
├── tests/                     # 40 unit tests (pytest)
│   ├── test_preprocessing.py
│   ├── test_retrieval.py
│   └── test_chatbot.py
├── notebooks/
│   └── faq_analysis.ipynb     # exploratory analysis notebook
├── reports/                   # generated evaluation artefacts
│   ├── evaluation_results.csv
│   ├── threshold_sweep.csv
│   └── threshold_analysis.png
├── docs/
│   └── error_analysis.md      # failure-mode analysis
├── .gitignore
├── requirements.txt
├── CONTRIBUTING.md            # contribution guidelines
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── LICENSE                    # MIT license
└── README.md
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/codealpha-chatbot-for-faqs.git
cd codealpha-chatbot-for-faqs

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

No external model downloads are required at runtime (stemming is used by
default). To use configurable lemmatization instead, install the WordNet
corpus first:

```bash
python -m nltk.downloader wordnet omw-1.4
```

## Usage

### Run the test suite

```bash
python -m pytest tests/ -q
```

### Run the offline evaluation

```bash
python -m src.evaluation
```

Outputs the metric summary and writes artefacts to `reports/`:

* `evaluation_results.csv` — per-question predictions
* `threshold_sweep.csv` — metrics across confidence thresholds
* `threshold_analysis.png` — accuracy / fallback trade-off plot

### Launch the chatbot web app

```bash
streamlit run app.py
```

### Use the chatbot as a library or CLI

```python
from src.chatbot import FAQChatbot

chatbot = FAQChatbot.from_csv("data/faq_dataset.csv", threshold=0.35)
response = chatbot.respond("How do I cancel my order?")
print(response.answer)          # -> "You can cancel from the Orders page..."
print(response.confidence)      # -> 0.93
print(response.is_fallback)     # -> False
```

Or interactively:

```bash
python -m src.chatbot
```

### Explore the analysis notebook

```bash
jupyter notebook notebooks/faq_analysis.ipynb
```

## Evaluation (measured on `data/test_questions.csv`)

```
Test questions                : 109
In-scope (expected FAQ)       : 97
Out-of-scope (fallback test)  : 12

Top-1 Accuracy    : 71.1%
Top-3 Accuracy    : 82.5%
MRR               : 76.3%
End-to-end acc.   : 66.1%
Fallback rate     : 21.1%
Out-of-scope catch: 66.7%
```

Measured at the default confidence threshold **0.35**. The threshold is an
explicit operating-point decision:

| Threshold | Fallback rate | Out-of-scope caught |
| --- | --- | --- |
| 0.20 | 6% | 33% |
| 0.35 (default) | 21% | 67% |
| 0.60 | 67% | 100% |

Lower thresholds answer more questions but leak spurious answers to
out-of-scope input; higher thresholds are safer but refuse more. Pick the
point that fits your tolerance for wrong answers (see
`reports/threshold_sweep.csv`).

> Metrics are real outputs of `python -m src.evaluation` on this repository,
> not estimates. Re-run the command to reproduce them.

## Limitations

* **Lexical matching**: TF-IDF needs shared vocabulary. A paraphrase that
  uses synonyms (`card got rejected` vs `card was declined`, `installments`
  vs `payment plans`) can score near zero.
* **Short/ambiguous queries**: a one-or-two-token question like `undo my
  order` can collapse onto a generic word (`order`) and match an unrelated
  FAQ with misleadingly high similarity.
* **Near-duplicate FAQs**: closely related entries (return policy vs.
  return window vs. how to return) are easy to confuse.
* **Out-of-scope detection**: questions that reuse common English templates
  (`How long does it take to ...`) can leak through the threshold.
* **No conversation memory** or multi-turn context; no intent beyond the FAQ
  slot matching.

A detailed, example-by-example breakdown lives in
[`docs/error_analysis.md`](docs/error_analysis.md).

## Future Improvements

The retrieval layer is deliberately an interface, not a dead end:

```
TF-IDF + Cosine Similarity   (today)
        ↓
Sentence Transformers        (swap one TextVectorizer)
        ↓
Embedding Search / Vector DB (FAISS, Qdrant, pgvector)
        ↓
RAG / LLM-based FAQ Assistant
```

Suggested roadmap:

1. **Sentence embeddings** (`sentence-transformers`) — fixes the synonymy
   failures at the top of the ranking of improvements.
2. **Vector database** — scale to tens of thousands of entries with ANN
   search.
3. **Hybrid retrieval** — combine lexical (BM25) and semantic scores.
4. **RAG / LLM answers** — generate answers from retrieved context, cite the
   source FAQ.
5. **Conversation memory** — contextual disambiguation and follow-up
   questions.
6. **Intent classification** — separate FAQ routing from a general Q&A
   model and Open-Domain refusal.
7. **Continuous evaluation** — expand `data/test_questions.csv` and gate
   releases on the metric suite.

## Testing

The suite (`tests/`) exercises exactly the behaviours that matter:

* **Preprocessing** — empty input, punctuation, capitals, stop words,
  contractions, negation preservation, stemming/lemmatization.
* **Retrieval** — known questions, similar phrasings, unrelated questions,
  competing FAQs, empty queries, ranking order.
* **Chatbot** — correct answers, fallback behaviour, threshold boundaries,
  `None` / empty / whitespace input.

```bash
python -m pytest tests/ -q
# 40 passed
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Report issues via the GitHub
issue tracker — see [`SECURITY.md`](SECURITY.md) for security disclosures.

## License

[MIT](LICENSE)

---

*Built for the CodeAlpha internship programme as a production-ready NLP
portfolio project.*