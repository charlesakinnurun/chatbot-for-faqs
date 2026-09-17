# Error Analysis — FAQ Retrieval Baseline

This report documents where the TF-IDF + cosine-similarity baseline goes
wrong, using the actual outputs of the offline evaluation
(`python -m src.evaluation`). Every example below is a real retrieval from
`data/test_questions.csv`, measured at the default confidence threshold
**0.35**.

Reference metrics at the default threshold:

| Metric | Value |
| --- | --- |
| Top-1 accuracy (in-scope) | 71.1% |
| Top-3 accuracy (in-scope) | 82.5% |
| MRR | 76.3% |
| Out-of-scope catch rate | 66.7% |

Out of 97 in-scope test questions, 28 were answered incorrectly and 15 were
refused (below threshold). Representative failures are grouped by cause below.

---

## 1. Synonymy — the dominant failure mode

Questions that express the same intent with different words share **no
tokens** with the FAQ, so TF-IDF sees pure strangers.

| User question | Retrieved | Expected | Best score |
| --- | --- | --- | --- |
| `My credit card got rejected` | `payments_02` (add card) | payments_03 (declined) | 0.50 |
| `Can I pay in installments` / `Is there a pay later option` | `payments_05` (PayPal) | payments_07 (plans) | 0.54 / 0.37 |
| `Is shipping free` (tie) | `shipping_05` (where ship to) | shipping_04 (free/expedited) | 0.60 |
| `Do you deliver to Europe` | `shipping_09` (my country) | shipping_03 (international) | 0.54 |
| `I bought the wrong size by accident` | `returns_05` (exchange size) | orders_05 (wrong item) | 0.22 |
| `How can I claim a refund` | `returns_07` (refund timing) | returns_04 (start a return) | 0.48 |

`rejected ↔ declined`, `installments ↔ payment plans`, `Europe ↔
internationally` are all lexical gaps a bag-of-words model cannot bridge.

## 2. Short queries that collapse onto generic words

Some user questions share only a highly frequent word with many FAQs
(`order`, `account`, `item`). After stop-word removal and stemming, the
query vector is a single token, and the match is effectively decided by
which FAQ happens to minimize its vector norm — a tie-break, not a decision.

| User question | Retrieved | Expected | Notes |
| --- | --- | --- | --- |
| `Undo my order` | `orders_02` (track) | orders_04 (cancel) | `undo` is out of vocabulary |
| `Where can I forward a fake email` | `account_09` (email change) | security_07 (phishing) | only shared token is nothing informational |
| `The item I received is not what I ordered` | `orders_02` (track) | orders_05 (wrong item) | noisy phrasing evaporates under stop-word removal |
| `I do not know what to order` | `orders_02` (track) | general_07 (find product) | every `order`-containing FAQ is a candidate |

These queries are genuinely ambiguous even for a human without context, but
the example shows the danger: a generic token can point at an unrelated FAQ
with artificially high similarity.

## 3. Competing FAQs that are near-duplicates

The knowledge base deliberately contains closely related entries (return
policy vs. how to return vs. return window vs. sale returns). When two
candidates differ by one or two words, the retriever flips easily.

| User question | Retrieved | Expected |
| --- | --- | --- |
| `Return window` | `returns_02` (how to return) | returns_03 (30 days) |
| `How many days can I return` | `returns_02` (how to return) | returns_03 (window) |
| `Can I send back reduced items` | `returns_10` (window) | returns_06/09 (sale items) |

Within-category lexical overlap is high; TF-IDF has no notion of which word
is the *discriminating* one.

## 4. Out-of-scope questions that leak through

Taxing exact numbers: with a 0.35 threshold, 8 of 12 out-of-scope questions
are correctly refused, but 4 are answered with a plausible-looking FAQ:

| OOS user question | Retrieved FAQ | Best score |
| --- | --- | --- |
| `How long does it take to learn Python` | `shipping_02` (delivery time) | 0.60 |
| `What is 25 times 4` | `general_10` (call support) | 0.59 |
| `Thanks for your help` | `general_07` (find product) | 0.46 |
| `When is the next lunar eclipse` | `payments_10` (save card) | 0.40 |

Each leaks through a *coincidental* shared token (`long`/`take`,
`times`, `help`, `next`). Note the linguistic trap: `How long does it take
to ...` is an extremely common English template that has nothing to do with
shipping. A frequency-weighted score this close to the in-scope distribution
means a single threshold cannot separate them cleanly — raising the
threshold to 0.60 increases the OOS catch rate to 100% but also refuses
more than two thirds of in-scope questions (see
`reports/threshold_sweep.csv`).

## 5. Preprocessing that removes information

* `When are you guys available` → best score **0.00**: after removing
  `when`/`are`/`you`, only `guys`/`available` remain, neither of which
  appears in `general_02` (customer service hours). The question word
  `when` is precisely the informative token here, but it is a stop word.
* `Show me your privacy terms` → 0.33 (refused): `terms` is not in the KB
  and `show`/`me` vanish, leaving only `privacy` — just below threshold
  even though the intent is unambiguous.
* `Follow my parcel` → 0.00: `follow` and `parcel` are both out of
  vocabulary while `track my order` uses different words entirely.

Trade-off note: keeping more function words (a less aggressive stop-word
list) improves cases like the second row but weakens discrimination in the
short-query cases above. The chosen list is deliberately conservative.

---

## Ranking of improvements (expected impact)

| Priority | Change | Expected effect |
| --- | --- | --- |
| 1 | **Sentence-embedding retriever** (e.g. sentence-transformers) | Resolves synonymy failures (section 1) — largest win |
| 2 | **Bigger knowledge base** with paraphrases | Complements embeddings; covers new intents cheaply |
| 3 | **Threshold tuning per deployment** (precision/precision trade-off) | Controls OOS leakage (section 4) at some coverage cost |
| 4 | **Intent / domain classifier front door** | Rejects `learn Python`, `stock prices` before retrieval |
| 5 | **Synonym-augmented preprocessing** (lemmatization is not enough) | Cheap partial fix for section 1 while preserving TF-IDF |
| 6 | **Answer awareness** (match on question + category) | Helps the near-duplicate cases in section 3 |

The architecture is designed so that change #1 only requires a new
`TextVectorizer` implementation (`src/vectorizer.vectorizer.py`) without
touching retrieval, chatbot or UI code.