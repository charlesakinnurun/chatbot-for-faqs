"""Offline evaluation of the retrieval pipeline.

Runs a held-out set of test questions through the retriever and reports
standard information-retrieval metrics:

* Top-1 accuracy -- the best match is an expected FAQ,
* Top-3 accuracy -- the expected FAQ appears in the top three matches,
* Mean Reciprocal Rank (MRR),
* Fallback rate -- fraction of questions rejected at a given threshold,
* Out-of-scope catch rate -- OOD questions correctly refused.

Threshold selection is treated as an explicit decision: the module sweeps
thresholds so the right operating point can be chosen for the deployment
(trading answer coverage against wrong-answer risk).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Settings, settings as default_settings
from src.data_loading import load_faq_dataset, load_test_questions
from src.preprocessing import TextPreprocessor
from src.retrieval import FAQRetriever, RetrievalResult
from src.vectorizer import TfidfTextVectorizer

TOP_K: int = 3

#: Thresholds probed by the sweep -- a fine grid across the plausible range.
DEFAULT_GRID: tuple[float, ...] = tuple(round(float(t), 2) for t in np.arange(0.10, 0.75, 0.05))


def score_question(
    retriever: FAQRetriever,
    question: str,
    threshold: float,
) -> dict[str, object]:
    """Produce a per-question record of retrieval behaviour.

    Parameters
    ----------
    retriever:
        Fitted retriever.
    question:
        Raw user question.
    threshold:
        Confidence threshold; scores below it count as "refused".

    Returns
    -------
    dict
        Keys: question, best_id, best_score, top_ids, answered, fallback.
        Hit metrics against the expected set are added by the caller.
    """
    result: RetrievalResult = retriever.retrieve(question)
    best = result.best_hit
    best_id = best.faq_id if best is not None else None
    best_score = result.best_score
    answered = best is not None and best.score >= threshold
    return {
        "question": question,
        "best_id": best_id,
        "best_score": best_score,
        "top_ids": [hit.faq_id for hit in result.hits],
        "answered": answered,
        "fallback": not answered,
    }


def build_predictions(
    retriever: FAQRetriever,
    test_df: pd.DataFrame,
    threshold: float,
    faqs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run retrieval over every test question and join with expectations.

    If ``faqs`` is provided, expected ids are expanded to include every FAQ
    entry whose answer text is identical (the knowledge base contains
    paraphrase rows that share answers), so picking a sibling paraphrase
    counts as a correct retrieval.

    Returns a DataFrame with one row per question containing the retrieved
    id/score plus ``top1_hit``, ``top3_hit`` and ``mrr`` vs. the expected set.
    """
    records: list[dict[str, object]] = []
    for question in test_df["question"]:
        records.append(score_question(retriever, question, threshold))

    preds = pd.DataFrame(records)
    preds = preds.join(test_df[["expected_id_set", "out_of_scope"]].reset_index(drop=True))

    def _expand(ids: object) -> object:
        return expand_expected_ids(faqs, tuple(ids)) if faqs is not None else ids

    preds["expected_expanded"] = preds["expected_id_set"].map(_expand)
    return preds


def expand_expected_ids(faqs: pd.DataFrame, expected_ids: tuple[str, ...]) -> set[str]:
    """Expand an expected-id set to every FAQ row sharing the same answer.

    The FAQ dataset deliberately contains multiple phrasings that resolve to
    the same answer (e.g. reset-password variants).  Retrieving any of them
    is a correct answer, so all of them count as accepting the expectation.
    """
    if not expected_ids:
        return set()
    answers = faqs.set_index("id")["answer"].astype(str).str.strip().str.lower()
    expanded: set[str] = set()
    for faq_id in expected_ids:
        if faq_id not in answers.index:
            continue
        answer_text = answers.loc[faq_id]
        matches = answers[answers == answer_text].index.tolist()
        expanded.update(str(m) for m in matches)
    return expanded


def compute_metrics(preds: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Compute headline metrics from prediction + expectation data.

    Parameters
    ----------
    preds:
        DataFrame from :func:`build_predictions`, which already carries the
        expected FAQ id set per row.
    threshold:
        The threshold used to produce ``preds`` (reported back for context).

    Returns
    -------
    dict
        Full set of metrics keyed by human readable names.
    """
    required = {"expected_expanded", "out_of_scope", "best_id", "top_ids", "answered", "fallback"}
    if not required.issubset(preds.columns):
        raise ValueError(f"preds must contain columns {sorted(required)}")

    in_scope = preds["out_of_scope"] == False  # noqa: E712
    scoped = preds[in_scope]
    ood = preds[~in_scope]

    def _hit1(best_id: object, expected: object) -> bool:
        return bool(best_id) and best_id in expected

    def _hit3(top_ids: object, expected: object) -> bool:
        return any(faq_id in expected for faq_id in top_ids if faq_id)

    def _mrr(top_ids: object, expected: object) -> float:
        for rank, faq_id in enumerate(top_ids, start=1):
            if faq_id in expected:
                return 1.0 / rank
        return 0.0

    def _scored(df: pd.DataFrame, column: str) -> float:
        return float(df[column].mean()) if len(df) else 0.0

    scoped = scoped.copy()
    scoped["hit1"] = [_hit1(b, e) for b, e in zip(scoped["best_id"], scoped["expected_expanded"])]
    scoped["hit3"] = [_hit3(t, e) for t, e in zip(scoped["top_ids"], scoped["expected_expanded"])]
    scoped["mrr"] = [_mrr(t, e) for t, e in zip(scoped["top_ids"], scoped["expected_expanded"])]

    answered = scoped["answered"] & scoped["hit1"]
    refused_in_scope = int((~scoped["answered"]).sum())
    ood_rows = preds[~in_scope]

    end_to_end = (
        int(answered.sum()) + int(ood_rows["fallback"].sum())
    ) / len(preds) if len(preds) else 0.0

    return {
        "threshold": float(threshold),
        "top1_accuracy": round(_scored(scoped, "hit1"), 4),
        "top3_accuracy": round(_scored(scoped, "hit3"), 4),
        "mrr": round(_scored(scoped, "mrr"), 4),
        "fallback_rate": round(float(preds["fallback"].mean()) if len(preds) else 0.0, 4),
        "out_of_scope_catch_rate": round(_scored(ood_rows, "fallback"), 4),
        "end_to_end_accuracy": round(float(end_to_end), 4),
        "correct_responses": int(answered.sum()),
        "incorrect_matches": int((scoped["answered"] & ~scoped["hit1"]).sum()),
        "refused_in_scope": refused_in_scope,
        "oov_refused": int(ood_rows["fallback"].sum()),
    }


def evaluate(
    retriever: FAQRetriever,
    test_df: pd.DataFrame,
    threshold: float,
    faqs: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Convenience: predictions + metrics in one shot."""
    preds = build_predictions(retriever, test_df, threshold, faqs=faqs)
    metrics = compute_metrics(preds, threshold)
    return {"metrics": metrics, "predictions": preds}


def threshold_sweep(
    retriever: FAQRetriever,
    test_df: pd.DataFrame,
    thresholds: tuple[float, ...] = DEFAULT_GRID,
    faqs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate the pipeline over a grid of confidence thresholds.

    Helps answer "how much coverage do I give up to avoid wrong answers".
    Each row is one threshold with the full metric set.
    """
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        preds = build_predictions(retriever, test_df, threshold, faqs=faqs)
        rows.append(compute_metrics(preds, threshold))
    return pd.DataFrame(rows)


def plot_threshold_analysis(
    sweep: pd.DataFrame,
    save_path: str | Path,
) -> None:
    """Plot accuracy/coverage trade-off against the threshold grid."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(sweep["threshold"], sweep["top1_accuracy"], "o-", label="Top-1 accuracy")
    ax1.plot(sweep["threshold"], sweep["top3_accuracy"], "s-", label="Top-3 accuracy")
    ax1.plot(sweep["threshold"], sweep["mrr"], "^--", label="MRR")
    ax1.plot(sweep["threshold"], sweep["end_to_end_accuracy"], "d-", label="End-to-end accuracy")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Threshold selection: accuracy vs. confidence cut-off")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(sweep["threshold"], sweep["fallback_rate"], "x--", color="crimson", label="Fallback rate")
    ax2.set_ylabel("Fallback rate", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(save_path), dpi=150)
    plt.close(fig)


def print_report(metrics: dict[str, object], test_df: pd.DataFrame) -> None:
    """Pretty-print a metric summary to the console."""
    print("=" * 62)
    print("FAQ RETRIEVAL EVALUATION")
    print("=" * 62)
    print(f"Test questions                : {len(test_df)}")
    print(f"In-scope (expected FAQ)       : {int((~test_df['out_of_scope']).sum())}")
    print(f"Out-of-scope (fallback test)  : {int(test_df['out_of_scope'].sum())}")
    print("-" * 62)
    for key in (
        "top1_accuracy",
        "top3_accuracy",
        "mrr",
        "end_to_end_accuracy",
        "fallback_rate",
        "out_of_scope_catch_rate",
    ):
        label = key.replace("_", " ").title()
        print(f"{label:<32}: {metrics[key]:.1%}")
    print("-" * 62)
    print("Response breakdown at chosen threshold:")
    print(f"  Correct responses      : {metrics['correct_responses']}")
    print(f"  Incorrect matches      : {metrics['incorrect_matches']}")
    print(f"  Refused (in-scope)     : {metrics['refused_in_scope']}")
    print(f"  Refused (out-of-scope) : {metrics['oov_refused']}")
    print("=" * 62)


def main(settings: Settings | None = None) -> None:
    """CLI entry point: evaluate the pipeline and persist reports.

    Writes ``reports/evaluation_results.csv``, ``reports/threshold_sweep.csv``
    and ``reports/threshold_analysis.png``, and prints the summary to stdout.
    """
    settings = settings or default_settings

    faqs = load_faq_dataset(settings.resolve_faq_dataset())
    test_df = load_test_questions(settings.resolve_test_questions())

    preprocessor = TextPreprocessor(normalization=settings.normalization)
    retriever = FAQRetriever(
        faqs=faqs,
        vectorizer=TfidfTextVectorizer(tokenizer=preprocessor.transform),
        max_results=settings.max_results,
    )

    threshold = settings.confidence_threshold
    result = evaluate(retriever, test_df, threshold, faqs=faqs)
    sweep = threshold_sweep(retriever, test_df, faqs=faqs)

    reports_dir = settings.resolve_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)

    preds = result["predictions"]
    preds.to_csv(reports_dir / "evaluation_results.csv", index=False)
    sweep.to_csv(reports_dir / "threshold_sweep.csv", index=False)
    plot_threshold_analysis(sweep, reports_dir / "threshold_analysis.png")

    print_report(result["metrics"], test_df)
    print(f"\nArtifacts written to: {reports_dir}")


if __name__ == "__main__":
    main()