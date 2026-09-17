"""Central configuration for the FAQ chatbot.

All paths, thresholds, model hyper-parameters and behavioural settings live
in one place so they can be tuned without touching business logic.  Every
value can be overridden through environment variables, which keeps the
project safe to deploy in containers and CI without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The root of the repository.  Because config.py lives in src/, its parent
# directory is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Directory that contains datasets and other data artifacts.
DATA_DIR = PROJECT_ROOT / "data"

#: Directory where evaluation artifacts are written (reports, charts, ...).
REPORTS_DIR = PROJECT_ROOT / "reports"

#: Words that are meaningful for FAQ matching but appear in common
#: English stop-word lists.  Negations in particular must survive so that
#: "I can not log in" is not treated as "I can log in".
NEGATION_WHITELIST: tuple[str, ...] = ("no", "not", "nor")

#: Domain words that add little signal for this FAQ corpus.
EXTRA_STOPWORDS: tuple[str, ...] = (
    "please",
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank",
    "dear",
    "kindly",
)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings.

    Attributes are deliberately plain strings/floats so the dataclass can
    stay serializable; resolve them through the provided ``resolve_*``
    helpers when a ``Path`` is needed.
    """

    # --- data -----------------------------------------------------------------
    faq_dataset_path: str = str(DATA_DIR / "faq_dataset.csv")
    test_questions_path: str = str(DATA_DIR / "test_questions.csv")
    reports_dir: str = str(REPORTS_DIR)

    # --- retrieval -------------------------------------------------------------
    #: Minimum cosine similarity for an answer to be considered confident.
    confidence_threshold: float = 0.35
    #: How many ranked candidates the retriever should return.
    max_results: int = 3

    # --- preprocessing ---------------------------------------------------------
    #: One of "stem", "lemma" or "none".
    normalization: str = "stem"
    remove_numbers: bool = True

    # --- chatbot ---------------------------------------------------------------
    fallback_response: str = (
        "I'm sorry, I could not find a confident answer to your question. "
        "Please rephrase it, or contact our support team at "
        "support@example.com for further help."
    )
    greeting: str = (
        "Hi! I am the FAQ assistant. Ask me anything about your account, "
        "orders, payments, shipping, returns, security or technical support."
    )

    # ---------------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        """Build ``Settings`` from environment variables.

        Each dataclass field maps to ``<FIELD_NAME_UPPERCASE>`` in the
        environment; ``None``/empty values are skipped.  This makes it
        trivial to override the threshold or data paths at runtime.

        Parameters
        ----------
        env:
            Environment mapping to read from.  Defaults to ``os.environ``.

        Returns
        -------
        Settings
            A new ``Settings`` instance with any env overrides applied.
        """
        if env is None:
            env = os.environ

        overrides: dict[str, str] = {}
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        for field_name, field_def in fields.items():
            env_key = field_name.upper()
            value = env.get(env_key)
            if value is not None and value.strip() != "":
                overrides[field_name] = value.strip()

        # Paths supplied through the environment are resolved against the
        # repository root when they are relative.
        for field_name in ("faq_dataset_path", "test_questions_path", "reports_dir"):
            if field_name in overrides:
                candidate = Path(overrides[field_name])
                if not candidate.is_absolute():
                    candidate = PROJECT_ROOT / candidate
                overrides[field_name] = str(candidate)

        coerced: dict[str, object] = {}
        for field_name, value in overrides.items():
            coerced[field_name] = _coerce_env_value(field_name, value, cls)

        return cls(**coerced)  # type: ignore[arg-type]

    # ---------------------------------------------------------------------------
    def resolve_faq_dataset(self) -> Path:
        """Return the FAQ dataset path as a ``Path``."""
        return Path(self.faq_dataset_path)

    def resolve_test_questions(self) -> Path:
        """Return the test questions path as a ``Path``."""
        return Path(self.test_questions_path)

    def resolve_reports_dir(self) -> Path:
        """Return the reports output directory as a ``Path``."""
        return Path(self.reports_dir)


#: Default settings instance shared across the application.  Prefer passing
#: an explicit ``Settings`` where a component needs to be configurable; this
#: singleton is a convenience for scripts and the CLI.
settings = Settings.from_env()


def _coerce_env_value(field_name: str, value: str, cls: type) -> object:
    """Convert a raw environment string to the dataclass field's type.

    Falls back to the raw string for unknown types so configuration never
    crashes on exotic values.
    """
    annotation = cls.__dataclass_fields__[field_name].type  # type: ignore[attr-defined]
    if annotation == float or annotation == "float":
        return float(value)
    if annotation == int or annotation == "int":
        return int(value)
    if annotation == bool or annotation == "bool":
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return value