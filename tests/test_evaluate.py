"""Tests for holdout evaluation.

The load bearing pieces are the confusion arithmetic, the deployment cost split, and the
ledger, which exists to make a repeated holdout evaluation visible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluate import (
    CLASSES,
    bare_estimator,
    boundary_errors,
    build_winning_pipeline,
    config_fingerprint,
    confusion,
    cost_weighted_errors,
    error_records,
    parse_param,
)


# ----------------------------------------------------------------- parameters

@pytest.mark.parametrize(
    "text,expected",
    [("0.1", 0.1), ("5", 5), ("(1, 2)", (1, 2)), ("True", True), ("sqrt", "sqrt"),
     ("distance", "distance"), ("uniform", "uniform")],
)
def test_parse_param_recovers_literals_and_leaves_names_alone(text, expected):
    assert parse_param(text) == expected


def test_bare_estimator_covers_every_model_the_sweep_can_pick():
    from src.models import estimators

    for name in list(estimators()) + ["voting_ensemble"]:
        assert bare_estimator(name) is not None, name


def test_bare_estimator_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        bare_estimator("transformer")


def test_build_winning_pipeline_applies_the_tuned_parameters():
    best = {
        "model": "multinomial_nb",
        "vectoriser": "tfidf",
        "best_params": {"clf__alpha": "0.1", "vect__ngram_range": "(1, 2)"},
    }
    pipeline = build_winning_pipeline(best)
    assert pipeline.named_steps["clf"].alpha == 0.1
    assert pipeline.named_steps["vect"].ngram_range == (1, 2)


def test_build_winning_pipeline_runs_end_to_end():
    best = {
        "model": "multinomial_nb",
        "vectoriser": "tfidf",
        "best_params": {"clf__alpha": "0.1", "vect__ngram_range": "(1, 1)"},
    }
    X = ["beneficiary dormant fund"] * 6 + ["viagra price offer"] * 6 + ["agenda lunch"] * 6
    y = np.array(["FRAUD"] * 6 + ["SPAM"] * 6 + ["NORMAL"] * 6)
    pipeline = build_winning_pipeline(best).fit(X, y)
    assert len(pipeline.predict(X)) == 18


# ------------------------------------------------------------------ confusion

def test_confusion_counts_rows_as_truth():
    matrix = confusion(["FRAUD", "FRAUD", "SPAM"], ["FRAUD", "SPAM", "SPAM"])
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1 and matrix[1, 1] == 1


def test_confusion_of_a_perfect_classifier_is_diagonal():
    labels = list(CLASSES) * 4
    matrix = confusion(labels, labels)
    assert matrix.sum() == len(labels)
    assert np.trace(matrix) == len(labels)


def test_boundary_errors_sums_both_directions():
    matrix = np.array([[10, 3, 0], [5, 10, 1], [0, 2, 10]])
    pairs = boundary_errors(matrix)
    assert pairs["FRAUD and SPAM"] == 8      # 3 plus 5
    assert pairs["FRAUD and NORMAL"] == 0
    assert pairs["SPAM and NORMAL"] == 3     # 1 plus 2


def test_boundary_errors_total_equals_off_diagonal():
    matrix = np.array([[10, 3, 1], [5, 10, 1], [2, 2, 10]])
    assert sum(boundary_errors(matrix).values()) == matrix.sum() - np.trace(matrix)


def test_cost_weighted_split_distinguishes_the_two_fraud_failures():
    """Fraud filed as spam still leaves the inbox. Fraud filed as normal does not.

    Macro F1 treats these identically, which is why they are reported separately.
    """
    matrix = np.array([[10, 4, 2], [0, 10, 0], [0, 0, 10]])
    costs = cost_weighted_errors(matrix)
    assert costs["fraud_caught_as_spam"] == 4
    assert costs["fraud_reaches_inbox"] == 2


def test_cost_weighted_split_tracks_lost_legitimate_mail():
    matrix = np.array([[10, 0, 0], [0, 10, 0], [3, 1, 10]])
    costs = cost_weighted_errors(matrix)
    assert costs["legitimate_mail_lost_to_fraud"] == 3
    assert costs["legitimate_mail_lost_to_spam"] == 1


# --------------------------------------------------------------------- errors

def test_error_records_only_capture_mistakes():
    raw = ["one two three", "four five six", "seven eight nine"]
    cleaned = ["one two", "four five", "seven eight"]
    records = error_records(raw, cleaned, ["FRAUD", "SPAM", "NORMAL"],
                            ["FRAUD", "FRAUD", "NORMAL"])
    assert len(records) == 1
    assert records[0]["true"] == "SPAM" and records[0]["predicted"] == "FRAUD"
    assert records[0]["index"] == 1


def test_error_records_are_empty_for_a_perfect_run():
    labels = ["FRAUD", "SPAM"]
    assert error_records(labels, labels, labels, labels) == []


def test_error_records_carry_an_excerpt_for_reading():
    raw = ["a " * 400]
    records = error_records(raw, raw, ["FRAUD"], ["SPAM"])
    assert 0 < len(records[0]["excerpt"]) <= 300


# --------------------------------------------------------------------- ledger

def test_fingerprint_is_stable_for_the_same_configuration():
    best = {"model": "m", "vectoriser": "tfidf", "best_params": {"a": "1", "b": "2"}}
    reordered = {"model": "m", "vectoriser": "tfidf", "best_params": {"b": "2", "a": "1"}}
    assert config_fingerprint(best) == config_fingerprint(reordered)


def test_fingerprint_changes_with_the_configuration():
    a = {"model": "m", "vectoriser": "tfidf", "best_params": {"alpha": "0.1"}}
    b = {"model": "m", "vectoriser": "tfidf", "best_params": {"alpha": "1.0"}}
    assert config_fingerprint(a) != config_fingerprint(b)


def test_ledger_records_only_two_distinct_configurations():
    """The discipline this project claims. Reruns are fine; new configurations are not.

    If this fails, something was scored on the test split that phase 8 did not choose.
    """
    from src.evaluate import LEDGER

    if not LEDGER.exists():
        pytest.skip("holdout not yet evaluated")
    ledger = json.loads(LEDGER.read_text())
    assert ledger["distinct_configurations"] == 2, (
        f"{ledger['distinct_configurations']} configurations have been scored on the "
        "test split; phase 8 pre committed to exactly two"
    )


def test_repeated_runs_of_a_configuration_give_identical_scores():
    """Determinism. Every rerun of the same fingerprint must record the same number."""
    from src.evaluate import LEDGER

    if not LEDGER.exists():
        pytest.skip("holdout not yet evaluated")
    ledger = json.loads(LEDGER.read_text())
    by_fingerprint: dict[str, set[float]] = {}
    for entry in ledger["evaluations"]:
        by_fingerprint.setdefault(entry["fingerprint"], set()).add(entry["test_macro_f1"])
    for fingerprint, scores in by_fingerprint.items():
        assert len(scores) == 1, f"{fingerprint} produced varying scores: {scores}"
