"""Baselines that every later result must be quoted against.

A headline score means nothing on its own. This module assembles the floor, the trivial
strategies, a simple interpretable rule, and the three shortcut probes from phases 4 to 6,
so that phases 8 and 9 can report against all of them rather than against a majority class
figure that nobody would actually deploy.

The important number here is not the majority baseline. It is the formatting only probe at
roughly 0.70, which a model reaches without reading a single word.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

from src.explore import log_odds_ratio


class KeywordRule(BaseEstimator, ClassifierMixin):
    """A decision list built from the top scoring words per class.

    This is the "what could a person write in an afternoon" baseline. It counts how many
    of each class's keywords a document contains, and predicts the class with the most
    matches. Ties, including the very common no matches at all, fall back to the most
    frequent class in the training data.

    The keywords are learned in `fit` rather than hardcoded, which matters for honesty.
    A hand picked list chosen after looking at the whole corpus would be tuned on the test
    data by way of the author's memory. Learning them inside each cross validation fold
    keeps the baseline comparable with the models it is being compared against.

    Ranking is by document level log odds ratio, the same statistic used in phase 5, so a
    word that is merely common does not qualify.
    """

    def __init__(self, keywords_per_class: int = 10, min_df: int = 5) -> None:
        self.keywords_per_class = keywords_per_class
        self.min_df = min_df

    def fit(self, X, y):  # noqa: N803
        y = np.asarray(y)
        self.classes_ = unique_labels(y)
        self.fallback_ = max(self.classes_, key=lambda c: int((y == c).sum()))

        tokenised = [set(doc.split()) for doc in X]
        overall: dict[str, int] = {}
        per_class: dict[str, dict[str, int]] = {c: {} for c in self.classes_}
        for tokens, label in zip(tokenised, y):
            bucket = per_class[label]
            for token in tokens:
                overall[token] = overall.get(token, 0) + 1
                bucket[token] = bucket.get(token, 0) + 1

        counts = {c: int((y == c).sum()) for c in self.classes_}
        total = len(y)

        self.keywords_ = {}
        for label in self.classes_:
            n_in = counts[label]
            n_out = total - n_in
            scored = []
            for token, df_total in overall.items():
                if df_total < self.min_df:
                    continue
                df_in = per_class[label].get(token, 0)
                scored.append(
                    (log_odds_ratio(df_in, n_in, df_total - df_in, n_out), token)
                )
            scored.sort(reverse=True)
            self.keywords_[label] = [t for _, t in scored[: self.keywords_per_class]]
        return self

    def predict(self, X):  # noqa: N803
        predictions = []
        for doc in X:
            tokens = set(doc.split())
            hits = {c: len(tokens & set(words)) for c, words in self.keywords_.items()}
            best = max(hits.values())
            winners = [c for c, n in hits.items() if n == best]
            predictions.append(self.fallback_ if best == 0 or len(winners) > 1 else winners[0])
        return np.array(predictions)


def coverage_of(rule: KeywordRule, X) -> float:  # noqa: N803
    """Share of documents where the rule fired rather than falling back.

    Worth reporting alongside the score. A rule that only fires on a fifth of documents is
    doing something quite different from one that fires on nearly all of them, even if the
    two land on similar macro F1.
    """
    fired = 0
    for doc in X:
        tokens = set(doc.split())
        hits = [len(tokens & set(words)) for words in rule.keywords_.values()]
        best = max(hits)
        if best > 0 and sum(1 for h in hits if h == best) == 1:
            fired += 1
    return fired / max(len(X), 1)


# --------------------------------------------------------------- canonical figures

BASELINES_PATH = Path(__file__).resolve().parents[1] / "docs" / "phase7_baselines.json"


def load_baselines() -> dict:
    """The phase 7 baseline table, for phases 8 and 9 to quote against.

    Read from the report rather than hardcoded, so a rerun of phase 7 propagates and the
    numbers in a later write up cannot silently drift from the ones that were measured.
    """
    if not BASELINES_PATH.exists():
        raise FileNotFoundError(
            f"{BASELINES_PATH.name} not found. Run scripts/run_baselines.py first."
        )
    return json.loads(BASELINES_PATH.read_text())


def bar_to_clear() -> tuple[str, float]:
    """The highest scoring content free shortcut, and the number a model must beat.

    Returns the label and the macro F1. Beating the majority class, or even random, says
    nothing: a model has to beat the best shortcut before it has demonstrated that it is
    reading the email rather than its packaging.
    """
    report = load_baselines()
    bar = report["bar_to_clear"]
    return bar["label"], float(bar["macro_f1"])
