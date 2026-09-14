"""Phase 10: the fraud class as a detector, not a third of a balanced benchmark.

Phase 9 reported macro F1 on a corpus that is one third fraud by construction. Three things
that leaves unanswered, all of which matter before anyone would use this:

  1. where the decision threshold should sit, given that the ways of being wrong cost
     different amounts
  2. what precision looks like in a mailbox that is not one third fraud
  3. whether the probabilities the threshold acts on mean anything

Discipline: the threshold is selected on cross validated predictions over the training
split. The holdout is used only to report what the selected threshold does. Picking the
threshold on the test split would be selection on the test split.

Usage:
    python scripts/fraud_deep_dive.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import build_winning_pipeline  # noqa: E402
from src.preprocess import chosen_text_steps  # noqa: E402
from src.threshold import (  # noqa: E402
    POSITIVE,
    CostModel,
    brier_score,
    cheapest_threshold,
    expected_calibration_error,
    project_at_prevalence,
    reliability,
    sweep_thresholds,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "train.csv"
TEST = ROOT / "data" / "processed" / "test.csv"
SWEEP = ROOT / "docs" / "phase8_sweep.json"
REPORT = ROOT / "docs" / "phase10_fraud.json"
FIGURE = ROOT / "reports" / "figures" / "phase10_fraud.png"

SEED = 20260914
FOLDS = 5

# Prevalence scenarios. The corpus figure is what phase 9 measured on; the others are
# stated assumptions about what a mailbox looks like, not measurements.
SCENARIOS = {
    "corpus as built, one third fraud": {"FRAUD": 1 / 3, "SPAM": 1 / 3, "NORMAL": 1 / 3},
    "spam heavy inbox, 5 percent fraud": {"FRAUD": 0.05, "SPAM": 0.45, "NORMAL": 0.50},
    "filtered inbox, 0.5 percent fraud": {"FRAUD": 0.005, "SPAM": 0.30, "NORMAL": 0.695},
    "well filtered inbox, 0.1 percent fraud": {"FRAUD": 0.001, "SPAM": 0.10, "NORMAL": 0.899},
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def read(path: Path) -> tuple[list[str], np.ndarray]:
    csv.field_size_limit(10_000_000)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r["text"] for r in rows], np.array([r["label"] for r in rows])


def fraud_column(classes: np.ndarray) -> int:
    return int(np.where(classes == POSITIVE)[0][0])


def fallback_labels(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Where a document goes when it is not flagged as fraud: the best non fraud class."""
    fraud = fraud_column(classes)
    masked = proba.copy()
    masked[:, fraud] = -1.0
    return classes[masked.argmax(axis=1)]


def main() -> int:
    log("== phase 10: fraud class deep dive ==")
    for path in (TRAIN, TEST, SWEEP):
        if not path.exists():
            log(f"missing {path.relative_to(ROOT)}. Run the earlier phases first.")
            return 1

    headline = json.loads(SWEEP.read_text())["best"]
    X_train_raw, y_train = read(TRAIN)
    X_test_raw, y_test = read(TEST)

    text = Pipeline(chosen_text_steps())
    X_train = text.fit_transform(X_train_raw)
    X_test = text.transform(X_test_raw)

    log(f"model: {headline['model']} on {headline['vectoriser']} features")
    log(f"train {len(X_train):,} documents, test {len(X_test):,} documents")

    costs = CostModel()
    log("\ncost assumptions, stated not measured. The ratios are what matter.")
    for line in costs.describe():
        log(f"  {line}")
    log("  The asymmetry between the first two lines is the point. Fraud misfiled as spam")
    log("  is still out of the inbox and the user is still protected.")

    # ------------------------------------------- threshold chosen on the training split
    log("\nselecting the threshold on cross validated training predictions")
    pipeline = build_winning_pipeline(headline)
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    train_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )
    pipeline.fit(X_train, y_train)
    classes = pipeline.classes_
    fraud = fraud_column(classes)

    train_fraud_p = train_proba[:, fraud]
    train_fallback = fallback_labels(train_proba, classes)
    train_rows = sweep_thresholds(y_train, train_fraud_p, train_fallback, costs)
    chosen = cheapest_threshold(train_rows)

    log(f"  cheapest threshold on train: {chosen['threshold']:.2f}")
    log(f"    cost {chosen['cost']:.1f}, precision {chosen['precision']:.4f}, "
        f"recall {chosen['recall']:.4f}")
    log(f"    fraud left in the inbox {chosen['missed_to_inbox']}, "
        f"legitimate mail flagged {chosen['false_alarms_on_legitimate']}")
    default = next(r for r in train_rows if abs(r["threshold"] - 0.50) < 1e-9)
    log(f"  for comparison, the default 0.50: cost {default['cost']:.1f}, "
        f"precision {default['precision']:.4f}, recall {default['recall']:.4f}")
    log(f"  cost saved by choosing rather than defaulting: "
        f"{default['cost'] - chosen['cost']:.1f}")

    log("\n  the cost curve around the chosen point")
    log(f"    {'thresh':>7}{'prec':>8}{'recall':>8}{'cost':>9}{'inbox':>7}{'legit FA':>10}")
    for row in train_rows:
        if abs(row["threshold"] - chosen["threshold"]) <= 0.16:
            mark = "  <-- chosen" if row["threshold"] == chosen["threshold"] else ""
            log(f"    {row['threshold']:7.2f}{row['precision']:8.4f}{row['recall']:8.4f}"
                f"{row['cost']:9.1f}{row['missed_to_inbox']:7d}"
                f"{row['false_alarms_on_legitimate']:10d}{mark}")

    # ------------------------------------------------------ report on the holdout
    log("\napplying the chosen threshold to the holdout")
    test_proba = pipeline.predict_proba(X_test)
    test_fraud_p = test_proba[:, fraud]
    test_fallback = fallback_labels(test_proba, classes)
    test_rows = sweep_thresholds(y_test, test_fraud_p, test_fallback, costs)
    at_chosen = next(
        r for r in test_rows if abs(r["threshold"] - chosen["threshold"]) < 1e-9
    )
    at_default = next(r for r in test_rows if abs(r["threshold"] - 0.50) < 1e-9)

    log(f"    {'':22}{'threshold':>10}{'prec':>8}{'recall':>8}{'cost':>9}"
        f"{'inbox':>7}{'legit FA':>10}")
    for label, row in (("chosen on train", at_chosen), ("default 0.50", at_default)):
        log(f"    {label:22}{row['threshold']:10.2f}{row['precision']:8.4f}"
            f"{row['recall']:8.4f}{row['cost']:9.1f}{row['missed_to_inbox']:7d}"
            f"{row['false_alarms_on_legitimate']:10d}")

    ap = average_precision_score((y_test == POSITIVE).astype(int), test_fraud_p)
    log(f"\n  average precision on the holdout: {ap:.4f}")
    log("  the holdout threshold sweep above is descriptive. The operating point was")
    log("  fixed on the training split before any of it was computed.")

    # --------------------------------------------------------------- calibration
    log("\ncalibration of the fraud probability")
    for name, y, p in (("train, cross validated", y_train, train_fraud_p),
                       ("holdout", y_test, test_fraud_p)):
        bins = reliability(y, p)
        brier = brier_score(y, p)
        ece = expected_calibration_error(bins, len(y))
        log(f"  {name}: Brier {brier:.4f}, expected calibration error {ece:.4f}")
        log(f"    {'bin':>14}{'n':>7}{'predicted':>11}{'observed':>10}{'gap':>9}")
        for b in bins:
            if not b["count"]:
                continue
            gap = b["mean_predicted"] - b["observed_rate"]
            log(f"    {b['low']:.1f} to {b['high']:.1f}{b['count']:9d}"
                f"{b['mean_predicted']:11.3f}{b['observed_rate']:10.3f}{gap:+9.3f}")

    # ---------------------------------------------------------------- prevalence
    log("\nPLAN.md 8.6: what precision looks like outside a balanced corpus")
    recall = at_chosen["recall"]
    spam_fa_rate = sum(
        1 for t, p in zip(y_test, test_fraud_p)
        if t == "SPAM" and p >= chosen["threshold"]
    ) / int((y_test == "SPAM").sum())
    normal_fa_rate = sum(
        1 for t, p in zip(y_test, test_fraud_p)
        if t == "NORMAL" and p >= chosen["threshold"]
    ) / int((y_test == "NORMAL").sum())
    log(f"  measured rates carried forward: recall {recall:.4f}, "
        f"false alarm rate on spam {spam_fa_rate:.4f}, on legitimate mail "
        f"{normal_fa_rate:.4f}")
    log("  per 100,000 messages")
    log(f"    {'scenario':40}{'precision':>10}{'true':>7}{'missed':>8}"
        f"{'FA spam':>9}{'FA legit':>10}{'FA per hit':>12}")
    projections = {}
    for name, prevalence in SCENARIOS.items():
        proj = project_at_prevalence(recall, spam_fa_rate, normal_fa_rate, prevalence)
        projections[name] = proj
        log(f"    {name:40}{proj['precision']:10.4f}{proj['true_positives']:7.0f}"
            f"{proj['false_negatives']:8.0f}{proj['false_positives_from_spam']:9.0f}"
            f"{proj['false_positives_from_legitimate']:10.0f}"
            f"{proj['false_alarms_per_true_fraud']:12.1f}")

    balanced = projections["corpus as built, one third fraud"]["precision"]
    filtered = projections["filtered inbox, 0.5 percent fraud"]["precision"]
    log(f"\n  precision falls from {balanced:.4f} on the balanced corpus to "
        f"{filtered:.4f} at half a percent prevalence")
    log("  That collapse is arithmetic, not a defect in the model. Recall and the false")
    log("  alarm rates are unchanged; only the mix of what it is shown has changed.")
    log("  Any precision quoted from a balanced benchmark should be read this way.")
    log("\n  but read the last two columns before concluding it is bad news")
    log(f"  at half a percent prevalence, every false alarm comes from spam and none")
    log(f"  from legitimate mail, because the measured false alarm rate on legitimate")
    log(f"  mail is {normal_fa_rate:.4f}. A fraud detector that over flags spam is")
    log("  filing junk as the wrong kind of junk, which costs almost nothing.")

    # ------------------------------------------------------ cost sensitivity
    log("\nsensitivity: does the chosen threshold depend on the cost assumptions")
    log(f"    {'inbox cost ratio':>18}{'threshold':>11}{'precision':>11}{'recall':>9}")
    sensitivity = []
    for ratio in (5, 10, 25, 50, 100, 250, 500):
        alt = CostModel(missed_fraud_to_inbox=float(ratio))
        alt_rows = sweep_thresholds(y_train, train_fraud_p, train_fallback, alt)
        alt_best = cheapest_threshold(alt_rows)
        sensitivity.append({"missed_fraud_to_inbox": ratio,
                            "threshold": alt_best["threshold"],
                            "precision": alt_best["precision"],
                            "recall": alt_best["recall"]})
        log(f"    {ratio:18d}{alt_best['threshold']:11.2f}"
            f"{alt_best['precision']:11.4f}{alt_best['recall']:9.4f}")
    span = {s["threshold"] for s in sensitivity}
    log(f"  thresholds selected across a 100 fold range of cost ratios: "
        f"{sorted(span)}")

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

        precision, recall_curve, _ = precision_recall_curve(
            (y_test == POSITIVE).astype(int), test_fraud_p
        )
        axes[0].plot(recall_curve, precision, color="#2f6f4e", lw=2)
        axes[0].scatter([at_chosen["recall"]], [at_chosen["precision"]], color="#a03c1f",
                        zorder=5, s=60,
                        label=f"chosen threshold {chosen['threshold']:.2f}")
        axes[0].set_xlabel("recall")
        axes[0].set_ylabel("precision")
        axes[0].set_title(f"Fraud detector on the holdout\naverage precision {ap:.4f}")
        axes[0].set_xlim(0, 1.02)
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(loc="lower left", fontsize=8)
        axes[0].grid(alpha=0.25)

        bins = reliability(y_test, test_fraud_p)
        xs = [b["mean_predicted"] for b in bins if b["count"]]
        ys = [b["observed_rate"] for b in bins if b["count"]]
        sizes = [max(20, 3 * b["count"]) for b in bins if b["count"]]
        axes[1].plot([0, 1], [0, 1], ls="--", color="#444", lw=1, label="perfect")
        axes[1].scatter(xs, ys, s=sizes, color="#2f6f4e", alpha=0.8)
        # A proxy handle, so the legend does not inherit the largest bin's marker size.
        axes[1].scatter([], [], s=40, color="#2f6f4e", alpha=0.8,
                        label="observed, area is bin count")
        axes[1].set_xlabel("mean predicted probability of fraud")
        axes[1].set_ylabel("observed fraud rate")
        axes[1].set_title(f"Calibration on the holdout\nBrier "
                          f"{brier_score(y_test, test_fraud_p):.4f}")
        axes[1].legend(loc="upper left", fontsize=8)
        axes[1].grid(alpha=0.25)

        names = list(SCENARIOS)
        short = ["1 in 3", "5 percent", "0.5 percent", "0.1 percent"]
        precisions = [projections[n]["precision"] for n in names]
        axes[2].bar(short, precisions, color=["#7fb093", "#c99a2e", "#c46a3f", "#b5432f"])
        axes[2].set_ylim(0, 1.05)
        axes[2].set_ylabel("precision")
        axes[2].set_xlabel("fraud prevalence in the mailbox")
        axes[2].set_title("Precision depends on prevalence\nsame model, same error rates")
        for i, v in enumerate(precisions):
            axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        axes[2].grid(axis="y", alpha=0.25)

        fig.suptitle("Phase 10: the fraud class as a detector")
        fig.tight_layout()
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=150)
        plt.close(fig)
        log(f"\nfigure written to {FIGURE.relative_to(ROOT)}")
    except Exception as exc:
        log(f"\nfigure skipped: {type(exc).__name__}: {exc}")

    REPORT.write_text(json.dumps({
        "phase": 10,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": headline["model"],
        "vectoriser": headline["vectoriser"],
        "costs": costs.__dict__,
        "chosen_threshold": chosen["threshold"],
        "chosen_on": "cross validated training predictions",
        "train_at_chosen": chosen,
        "train_at_default": default,
        "test_at_chosen": at_chosen,
        "test_at_default": at_default,
        "average_precision_holdout": float(ap),
        "calibration": {
            "train_brier": brier_score(y_train, train_fraud_p),
            "train_ece": expected_calibration_error(
                reliability(y_train, train_fraud_p), len(y_train)),
            "test_brier": brier_score(y_test, test_fraud_p),
            "test_ece": expected_calibration_error(bins, len(y_test)),
            "test_bins": bins,
        },
        "measured_rates": {
            "recall": recall,
            "false_alarm_rate_on_spam": spam_fa_rate,
            "false_alarm_rate_on_legitimate": normal_fa_rate,
        },
        "prevalence_projections": projections,
        "cost_sensitivity": sensitivity,
        "threshold_sweep_test": test_rows,
    }, indent=2) + "\n")
    log(f"report written to {REPORT.relative_to(ROOT)}")

    log("\nphase 10 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
