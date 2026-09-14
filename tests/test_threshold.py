"""Tests for threshold selection, cost weighting, prevalence projection and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from src.threshold import (
    POSITIVE,
    CostModel,
    brier_score,
    cheapest_threshold,
    expected_calibration_error,
    expected_cost,
    project_at_prevalence,
    reliability,
    sweep_thresholds,
)


def simple_case():
    """Ten fraud with high scores, ten spam mid, ten legitimate low."""
    y = np.array(["FRAUD"] * 10 + ["SPAM"] * 10 + ["NORMAL"] * 10)
    p = np.concatenate([
        np.linspace(0.60, 0.99, 10),
        np.linspace(0.10, 0.55, 10),
        np.linspace(0.00, 0.20, 10),
    ])
    fallback = np.array(["SPAM"] * 20 + ["NORMAL"] * 10)
    return y, p, fallback


# ----------------------------------------------------------------- cost model

def test_cost_model_separates_the_two_ways_of_missing_fraud():
    """The asymmetry the whole phase rests on."""
    costs = CostModel()
    assert costs.missed_fraud_to_inbox > costs.missed_fraud_to_spam
    assert costs.false_alarm_on_legitimate > costs.false_alarm_on_spam


def test_cost_model_describes_itself():
    assert len(CostModel().describe()) == 4


def test_missed_fraud_costs_more_when_it_reaches_the_inbox():
    costs = CostModel()
    y = np.array([POSITIVE])
    p = np.array([0.1])
    to_spam = expected_cost(y, p, 0.5, np.array(["SPAM"]), costs)
    to_inbox = expected_cost(y, p, 0.5, np.array(["NORMAL"]), costs)
    assert to_spam == costs.missed_fraud_to_spam
    assert to_inbox == costs.missed_fraud_to_inbox
    assert to_inbox > to_spam


def test_false_alarm_costs_more_on_legitimate_mail_than_on_spam():
    costs = CostModel()
    p = np.array([0.9])
    on_spam = expected_cost(np.array(["SPAM"]), p, 0.5, np.array(["SPAM"]), costs)
    on_legit = expected_cost(np.array(["NORMAL"]), p, 0.5, np.array(["NORMAL"]), costs)
    assert on_spam == costs.false_alarm_on_spam
    assert on_legit == costs.false_alarm_on_legitimate


def test_a_perfect_separation_costs_nothing():
    y, p, fallback = simple_case()
    assert expected_cost(y, p, 0.58, fallback, CostModel()) == 0.0


# ------------------------------------------------------------------- sweeping

def test_sweep_covers_the_threshold_range():
    y, p, fallback = simple_case()
    rows = sweep_thresholds(y, p, fallback, CostModel())
    assert rows[0]["threshold"] < 0.05 or rows[0]["threshold"] == pytest.approx(0.02)
    assert rows[-1]["threshold"] > 0.9


def test_recall_never_increases_with_the_threshold():
    y, p, fallback = simple_case()
    recalls = [r["recall"] for r in sweep_thresholds(y, p, fallback, CostModel())]
    assert all(a >= b for a, b in zip(recalls, recalls[1:]))


def test_counts_add_up_at_every_threshold():
    y, p, fallback = simple_case()
    positives = int((y == POSITIVE).sum())
    for row in sweep_thresholds(y, p, fallback, CostModel()):
        assert row["true_positives"] + row["false_negatives"] == positives


def test_a_threshold_of_zero_flags_everything():
    y, p, fallback = simple_case()
    rows = sweep_thresholds(y, p, fallback, CostModel(), grid=np.array([0.0]))
    assert rows[0]["recall"] == 1.0
    assert rows[0]["false_positives"] == 20


def test_cheapest_threshold_finds_the_minimum():
    y, p, fallback = simple_case()
    rows = sweep_thresholds(y, p, fallback, CostModel())
    best = cheapest_threshold(rows)
    assert best["cost"] == min(r["cost"] for r in rows)


def test_cheapest_threshold_breaks_ties_conservatively():
    """Cost is a step function, so ties are common. Prefer fewer false alarms."""
    rows = [
        {"threshold": 0.2, "cost": 5.0},
        {"threshold": 0.4, "cost": 5.0},
        {"threshold": 0.6, "cost": 9.0},
    ]
    assert cheapest_threshold(rows)["threshold"] == 0.4


# ---------------------------------------------------------------- prevalence

def test_precision_falls_as_prevalence_falls_with_rates_held_fixed():
    """The arithmetic behind PLAN.md 8.6, and the reason a balanced benchmark misleads."""
    balanced = project_at_prevalence(
        0.99, 0.04, 0.0, {"FRAUD": 1 / 3, "SPAM": 1 / 3, "NORMAL": 1 / 3})
    rare = project_at_prevalence(
        0.99, 0.04, 0.0, {"FRAUD": 0.005, "SPAM": 0.30, "NORMAL": 0.695})
    assert balanced["precision"] > rare["precision"]
    assert balanced["recall"] == rare["recall"]  # recall does not depend on prevalence


def test_projection_conserves_the_mailbox():
    proj = project_at_prevalence(
        0.99, 0.04, 0.01, {"FRAUD": 0.005, "SPAM": 0.30, "NORMAL": 0.695},
        mailbox_size=100_000)
    assert proj["true_positives"] + proj["false_negatives"] == pytest.approx(500.0)


def test_zero_false_alarms_on_legitimate_mail_stays_zero():
    """Phase 10 measured this rate at 0, which is why the precision collapse is cheap."""
    proj = project_at_prevalence(
        0.99, 0.04, 0.0, {"FRAUD": 0.001, "SPAM": 0.10, "NORMAL": 0.899})
    assert proj["false_positives_from_legitimate"] == 0.0
    assert proj["false_positives_from_spam"] > 0


def test_perfect_specificity_gives_perfect_precision_at_any_prevalence():
    for prevalence in (1 / 3, 0.005, 0.0001):
        proj = project_at_prevalence(
            0.99, 0.0, 0.0,
            {"FRAUD": prevalence, "SPAM": 0.3, "NORMAL": 1 - 0.3 - prevalence})
        assert proj["precision"] == pytest.approx(1.0)


# --------------------------------------------------------------- calibration

def test_brier_rewards_a_confident_correct_prediction():
    y = np.array([POSITIVE, "SPAM"])
    assert brier_score(y, np.array([1.0, 0.0])) == 0.0
    assert brier_score(y, np.array([0.5, 0.5])) == 0.25


def test_brier_punishes_confident_error():
    y = np.array([POSITIVE])
    assert brier_score(y, np.array([0.0])) == 1.0


def test_reliability_bins_partition_the_documents():
    y, p, _ = simple_case()
    bins = reliability(y, p, bins=10)
    assert sum(b["count"] for b in bins) == len(y)


def test_reliability_reports_none_for_an_empty_bin():
    y = np.array([POSITIVE, "SPAM"])
    p = np.array([0.95, 0.05])
    bins = reliability(y, p, bins=10)
    empty = [b for b in bins if b["count"] == 0]
    assert empty and all(b["mean_predicted"] is None for b in empty)


def test_calibration_error_is_zero_for_a_perfectly_calibrated_score():
    y = np.array([POSITIVE] * 50 + ["SPAM"] * 50)
    p = np.array([1.0] * 50 + [0.0] * 50)
    bins = reliability(y, p)
    assert expected_calibration_error(bins, len(y)) == pytest.approx(0.0, abs=1e-9)


def test_calibration_error_is_positive_when_confidence_is_wrong():
    y = np.array([POSITIVE] * 50 + ["SPAM"] * 50)
    p = np.array([0.6] * 50 + [0.4] * 50)
    bins = reliability(y, p)
    assert expected_calibration_error(bins, len(y)) > 0.3
