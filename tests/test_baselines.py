"""Tests for the baseline suite.

The keyword rule is the piece with real logic. Its guarantee is that it learns its
keywords in `fit`, so it can be cross validated honestly rather than being a hand picked
list that was really tuned on the whole corpus by way of the author's memory.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.baselines import KeywordRule, bar_to_clear, coverage_of, load_baselines

FRAUD_DOC = "beneficiary dormant fund transfer deceased"
SPAM_DOC = "viagra shops price offer unsubscribe"
NORMAL_DOC = "meeting agenda lunch conference minutes"


def make_corpus(n: int = 30) -> tuple[list[str], np.ndarray]:
    X = [FRAUD_DOC] * n + [SPAM_DOC] * n + [NORMAL_DOC] * n
    y = np.array(["FRAUD"] * n + ["SPAM"] * n + ["NORMAL"] * n)
    return X, y


# ------------------------------------------------------------------- fitting

def test_keywords_are_learned_not_hardcoded():
    X, y = make_corpus()
    rule = KeywordRule(keywords_per_class=3).fit(X, y)
    assert set(rule.keywords_["FRAUD"]) <= set(FRAUD_DOC.split())
    assert set(rule.keywords_["SPAM"]) <= set(SPAM_DOC.split())
    assert set(rule.keywords_["NORMAL"]) <= set(NORMAL_DOC.split())


def test_keywords_per_class_is_respected():
    X, y = make_corpus()
    for k in (1, 2, 3):
        rule = KeywordRule(keywords_per_class=k).fit(X, y)
        assert all(len(words) == k for words in rule.keywords_.values())


def test_more_keywords_is_never_fewer():
    X, y = make_corpus()
    small = KeywordRule(keywords_per_class=2).fit(X, y)
    large = KeywordRule(keywords_per_class=4).fit(X, y)
    for label in small.keywords_:
        assert len(large.keywords_[label]) >= len(small.keywords_[label])


def test_min_df_excludes_rare_terms():
    X, y = make_corpus(n=30)
    X = X + ["extremelyrareword padding padding padding"]
    y = np.append(y, "FRAUD")
    rule = KeywordRule(keywords_per_class=25, min_df=5).fit(X, y)
    assert "extremelyrareword" not in rule.keywords_["FRAUD"]


def test_fitting_is_deterministic():
    X, y = make_corpus()
    a = KeywordRule(keywords_per_class=3).fit(X, y).keywords_
    b = KeywordRule(keywords_per_class=3).fit(X, y).keywords_
    assert a == b


def test_fallback_is_the_most_frequent_training_class():
    X = [FRAUD_DOC] * 10 + [SPAM_DOC] * 3
    y = np.array(["FRAUD"] * 10 + ["SPAM"] * 3)
    rule = KeywordRule(keywords_per_class=2, min_df=2).fit(X, y)
    assert rule.fallback_ == "FRAUD"


# ---------------------------------------------------------------- predicting

def test_predicts_the_matching_class():
    X, y = make_corpus()
    rule = KeywordRule(keywords_per_class=3).fit(X, y)
    assert rule.predict([FRAUD_DOC])[0] == "FRAUD"
    assert rule.predict([SPAM_DOC])[0] == "SPAM"
    assert rule.predict([NORMAL_DOC])[0] == "NORMAL"


def test_falls_back_when_nothing_matches():
    X, y = make_corpus()
    rule = KeywordRule(keywords_per_class=3).fit(X, y)
    assert rule.predict(["completely unrelated vocabulary here"])[0] == rule.fallback_


def test_falls_back_on_a_tie():
    """A document carrying keywords from two classes equally is not a decision."""
    X, y = make_corpus()
    rule = KeywordRule(keywords_per_class=1).fit(X, y)
    fraud_word = rule.keywords_["FRAUD"][0]
    spam_word = rule.keywords_["SPAM"][0]
    assert rule.predict([f"{fraud_word} {spam_word}"])[0] == rule.fallback_


def test_predict_returns_one_label_per_document():
    X, y = make_corpus()
    rule = KeywordRule(keywords_per_class=3).fit(X, y)
    assert len(rule.predict(X)) == len(X)


def test_works_inside_cross_validation():
    """The point of learning keywords in fit rather than hardcoding them."""
    X, y = make_corpus(n=40)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    scores = cross_val_score(KeywordRule(keywords_per_class=3), X, y, cv=cv,
                             scoring="f1_macro")
    assert len(scores) == 4
    assert scores.mean() > 0.9  # planted signal, should be easy


# ------------------------------------------------------------------ coverage

def test_coverage_is_one_when_every_document_matches():
    X, y = make_corpus()
    rule = KeywordRule(keywords_per_class=3).fit(X, y)
    assert coverage_of(rule, X) == pytest.approx(1.0)


def test_coverage_is_zero_when_nothing_matches():
    X, y = make_corpus()
    rule = KeywordRule(keywords_per_class=3).fit(X, y)
    assert coverage_of(rule, ["nothing here matches at all"] * 5) == 0.0


def test_coverage_on_empty_input():
    X, y = make_corpus()
    rule = KeywordRule(keywords_per_class=3).fit(X, y)
    assert coverage_of(rule, []) == 0.0


# ------------------------------------------------------- the canonical table

def test_baseline_report_loads():
    report = load_baselines()
    assert report["phase"] == 7
    for key in ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "MODEL"):
        assert key in report["results"], key


def test_random_beats_the_majority_class_on_macro_f1():
    """The correction this phase makes. Macro F1's floor is random, not majority."""
    report = load_baselines()
    assert report["random_floor"] > report["majority_floor"]


def test_bar_to_clear_is_the_highest_shortcut():
    label, value = bar_to_clear()
    report = load_baselines()
    shortcuts = [report["results"][k]["macro_f1_mean"]
                 for k in ("B3", "B4", "B5", "B6", "B7", "B8")]
    assert value == pytest.approx(max(shortcuts))
    assert "formatting" in label


def test_the_model_clears_the_bar():
    _, bar = bar_to_clear()
    report = load_baselines()
    assert report["results"]["MODEL"]["macro_f1_mean"] > bar


def test_keyword_rules_improve_with_more_keywords():
    report = load_baselines()
    k5 = report["results"]["B6"]["macro_f1_mean"]
    k10 = report["results"]["B7"]["macro_f1_mean"]
    k25 = report["results"]["B8"]["macro_f1_mean"]
    assert k5 < k10 < k25
