"""Fraud class threshold analysis, cost weighting and prevalence correction.

Three things macro F1 cannot tell you, all of which matter before anyone deploys this.

**Where the operating point should sit.** Macro F1 implicitly weights every error equally.
In use they are not equal, so the threshold on the fraud probability is a decision to be
made against stated costs rather than left at 0.5 by default.

**What precision means outside a balanced corpus.** The corpus is one third fraud by
construction. A real mailbox is not. Precision is the metric that depends on prevalence, so
a precision measured at 33 percent prevalence says almost nothing about a mailbox at half a
percent. This is `PLAN.md` 8.6, and it is the single most misleading thing about a balanced
benchmark.

**Whether the probabilities mean anything.** Choosing a threshold presumes the score behind
it is calibrated. That is an assumption to test, not to make.

Threshold selection happens on cross validated training predictions. The holdout is used
only to report what the chosen threshold does. Choosing it on the test split would be
selection on the test split, which is the thing phase 9 was built to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POSITIVE = "FRAUD"


# --------------------------------------------------------------------- costs

@dataclass(frozen=True)
class CostModel:
    """Explicit costs for each way the fraud decision can go wrong.

    The numbers are assumptions, not measurements, and they are stated so a reader can
    substitute their own. The ratios are what matter, not the units.

    The important asymmetry is between the two ways of missing a fraud email. One leaves it
    in the inbox in front of the user. The other files it as spam, where it is out of the
    way and the user is protected even though the label is wrong. Treating those as the same
    error, which both macro F1 and a plain binary framing do, is the mistake this class
    exists to avoid.
    """

    missed_fraud_to_inbox: float = 100.0
    missed_fraud_to_spam: float = 2.0
    false_alarm_on_legitimate: float = 20.0
    false_alarm_on_spam: float = 0.5

    def describe(self) -> list[str]:
        return [
            f"fraud left in the inbox          {self.missed_fraud_to_inbox:7.1f}",
            f"fraud misfiled as spam           {self.missed_fraud_to_spam:7.1f}",
            f"legitimate mail flagged as fraud {self.false_alarm_on_legitimate:7.1f}",
            f"spam flagged as fraud            {self.false_alarm_on_spam:7.1f}",
        ]


def expected_cost(
    y_true: np.ndarray,
    fraud_probability: np.ndarray,
    threshold: float,
    fallback: np.ndarray,
    costs: CostModel,
) -> float:
    """Total cost of flagging as fraud everything at or above the threshold.

    `fallback` is where a document goes when it is not flagged as fraud, which is the
    model's preference between the two remaining classes. That is what makes the two kinds
    of missed fraud distinguishable.
    """
    flagged = fraud_probability >= threshold
    total = 0.0
    for truth, is_flagged, other in zip(y_true, flagged, fallback):
        if truth == POSITIVE and not is_flagged:
            total += (
                costs.missed_fraud_to_spam if other == "SPAM"
                else costs.missed_fraud_to_inbox
            )
        elif truth != POSITIVE and is_flagged:
            total += (
                costs.false_alarm_on_spam if truth == "SPAM"
                else costs.false_alarm_on_legitimate
            )
    return total


def sweep_thresholds(
    y_true: np.ndarray,
    fraud_probability: np.ndarray,
    fallback: np.ndarray,
    costs: CostModel,
    grid: np.ndarray | None = None,
) -> list[dict]:
    """Precision, recall and expected cost across the threshold range."""
    if grid is None:
        grid = np.round(np.arange(0.02, 1.00, 0.02), 4)

    positive = y_true == POSITIVE
    rows = []
    for threshold in grid:
        flagged = fraud_probability >= threshold
        tp = int((flagged & positive).sum())
        fp = int((flagged & ~positive).sum())
        fn = int((~flagged & positive).sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "threshold": float(threshold),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cost": expected_cost(y_true, fraud_probability, threshold, fallback, costs),
            "missed_to_inbox": int(
                sum(1 for t, f, o in zip(y_true, flagged, fallback)
                    if t == POSITIVE and not f and o != "SPAM")
            ),
            "false_alarms_on_legitimate": int(
                sum(1 for t, f in zip(y_true, flagged) if t == "NORMAL" and f)
            ),
        })
    return rows


def cheapest_threshold(rows: list[dict]) -> dict:
    """The lowest cost row, preferring the higher threshold when costs tie.

    Ties are common because cost is a step function of the threshold: it changes only when
    a document crosses it. Preferring the higher threshold picks the most conservative
    operating point consistent with the minimum, which means fewer false alarms for the
    same cost.
    """
    lowest = min(row["cost"] for row in rows)
    return max((row for row in rows if row["cost"] == lowest), key=lambda r: r["threshold"])


# ---------------------------------------------------------------- prevalence

def project_at_prevalence(
    recall: float,
    false_alarm_rate_on_spam: float,
    false_alarm_rate_on_legitimate: float,
    prevalence: dict[str, float],
    mailbox_size: int = 100_000,
) -> dict:
    """Project the confusion counts onto a mailbox with realistic class proportions.

    The per class error rates are properties of the classifier and carry over. The
    proportions do not, and precision depends on them. This is why a balanced benchmark
    flatters precision so badly.
    """
    n_fraud = mailbox_size * prevalence["FRAUD"]
    n_spam = mailbox_size * prevalence["SPAM"]
    n_normal = mailbox_size * prevalence["NORMAL"]

    tp = recall * n_fraud
    fn = n_fraud - tp
    fp_spam = false_alarm_rate_on_spam * n_spam
    fp_normal = false_alarm_rate_on_legitimate * n_normal
    fp = fp_spam + fp_normal

    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "prevalence": prevalence,
        "mailbox_size": mailbox_size,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives_from_spam": fp_spam,
        "false_positives_from_legitimate": fp_normal,
        "precision": precision,
        "recall": recall,
        "false_alarms_per_true_fraud": (fp / tp) if tp else float("inf"),
    }


# --------------------------------------------------------------- calibration

def reliability(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> list[dict]:
    """Observed fraud rate against predicted probability, in equal width bins."""
    positive = (y_true == POSITIVE).astype(int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (probability >= low) & (probability < high if high < 1.0 else probability <= 1.0)
        n = int(in_bin.sum())
        rows.append({
            "low": float(low),
            "high": float(high),
            "count": n,
            "mean_predicted": float(probability[in_bin].mean()) if n else None,
            "observed_rate": float(positive[in_bin].mean()) if n else None,
        })
    return rows


def brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Mean squared error of the probability. Lower is better, 0.25 is a coin flip."""
    positive = (y_true == POSITIVE).astype(float)
    return float(np.mean((probability - positive) ** 2))


def expected_calibration_error(rows: list[dict], total: int) -> float:
    """Weighted mean gap between predicted probability and observed rate."""
    error = 0.0
    for row in rows:
        if row["count"] and row["mean_predicted"] is not None:
            error += (row["count"] / total) * abs(row["mean_predicted"] - row["observed_rate"])
    return error
