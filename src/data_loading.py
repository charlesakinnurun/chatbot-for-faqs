"""Dataset loading and validation.

Kept in its own module so the chatbot, the retriever and the evaluation
scripts all share exactly the same schema checks and parsing logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FAQ_COLUMNS = ("id", "category", "question", "answer")
TEST_COLUMNS = ("question", "expected_ids")

_EMPTY_IDS: tuple[()] = ()


def load_faq_dataset(path: str | Path) -> pd.DataFrame:
    """Load the FAQ knowledge base and validate its schema.

    Parameters
    ----------
    path:
        Path to a CSV with columns ``id, category, question, answer``.

    Returns
    -------
    pd.DataFrame
        The validated FAQ dataset, indexed by ``id``.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    ValueError
        If the dataset is empty or missing required columns.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"FAQ dataset not found: {csv_path}")

    faqs = pd.read_csv(csv_path, dtype=str).fillna("")
    missing = [col for col in FAQ_COLUMNS if col not in faqs.columns]
    if missing:
        raise ValueError(f"FAQ dataset is missing required columns: {missing}")

    faqs = faqs[list(FAQ_COLUMNS)].drop_duplicates(subset="id")
    if faqs.empty:
        raise ValueError(f"FAQ dataset '{csv_path}' contains no rows.")

    if faqs["id"].duplicated().any():
        raise ValueError("FAQ dataset contains duplicate ids.")

    return faqs.reset_index(drop=True)


def load_test_questions(path: str | Path) -> pd.DataFrame:
    """Load the evaluation questions and parse their expected answers.

    The ``expected_ids`` column may contain one FAQ id or several ids
    separated by ``;``.  A blank value means the question is deliberately
    outside the knowledge base and should trigger the fallback response.

    Parameters
    ----------
    path:
        Path to a CSV with columns ``question, expected_ids``.

    Returns
    -------
    pd.DataFrame
        The test questions with an added ``expected_id_set`` column (tuple
        of strings) and an ``out_of_scope`` boolean column.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Test questions file not found: {csv_path}")

    test_df = pd.read_csv(csv_path, dtype=str).fillna("")
    missing = [col for col in TEST_COLUMNS if col not in test_df.columns]
    if missing:
        raise ValueError(f"Test questions file is missing required columns: {missing}")

    if test_df["question"].str.strip().eq("").any():
        raise ValueError("Test questions file contains blank questions.")

    def _parse_ids(value: str) -> tuple[str, ...]:
        ids = tuple(part.strip() for part in value.split(";") if part.strip())
        return ids or _EMPTY_IDS

    test_df["expected_id_set"] = test_df["expected_ids"].map(_parse_ids)
    test_df["out_of_scope"] = test_df["expected_id_set"].map(len) == 0
    return test_df.reset_index(drop=True)