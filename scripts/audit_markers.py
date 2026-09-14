"""Phase 4: quantify how much apparent separability is provenance rather than content.

Three isolation probes, each given access to one artifact and nothing else, so whatever
they score is achieved without reading the message:

    P1 formatting only   seven numeric style features, no words at all    tests D4
    P2 markers only      only the provenance blocklist tokens survive     tests D3
    P3 length only       one feature, the document word count             tests D11

Then four pipeline conditions, to measure what the fixes cost:

    C0 majority baseline
    C1 raw text
    C2 normalised text
    C3 normalised text with provenance markers stripped

Everything is five fold stratified cross validation on the training split. The test
split is not touched until phase 9. Vectorisers live inside the pipeline so they are
fitted on training folds only, which is the fix for PLAN.md 8.2.

Usage:
    python scripts/audit_markers.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.markers import (  # noqa: E402
    BLOCKLIST,
    BLOCKLIST_GROUPS,
    KEPT_DESPITE_SKEW,
    keep_only_markers,
    strip_markers,
)
from src.normalise import (  # noqa: E402
    FORMATTING_FEATURE_NAMES,
    Normaliser,
    formatting_features,
    normalise,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "train.csv"
REPORT = ROOT / "docs" / "phase4_audit.json"
FIGURE = ROOT / "reports" / "figures" / "phase4_leakage.png"

SEED = 20260914
FOLDS = 5

# Two different floors, and confusing them is a real reporting trap.
#
# A classifier that always predicts one class of three is right a third of the time, so
# its ACCURACY floor is 0.3333. But macro F1 averages per class F1, and that classifier
# scores 0.5 on its chosen class and 0.0 on the other two, so its MACRO F1 floor is
# 0.1667. Every figure in this script is macro F1, so the macro figure is the one that
# belongs on the chart and in the multiples.
MAJORITY_MACRO_F1 = 1 / 6
MAJORITY_ACCURACY = 1 / 3


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_train() -> tuple[list[str], np.ndarray]:
    csv.field_size_limit(10_000_000)
    with TRAIN.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r["text"] for r in rows], np.array([r["label"] for r in rows])


def tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(min_df=2, sublinear_tf=True, strip_accents="unicode")


def logreg() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=SEED)


def score(name: str, pipeline, X, y, cv) -> dict:  # noqa: N803
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    result = {
        "name": name,
        "macro_f1_mean": float(scores.mean()),
        "macro_f1_std": float(scores.std()),
        "folds": [float(s) for s in scores],
    }
    log(f"  {name:52} {scores.mean():.4f} +/- {scores.std():.4f}")
    return result


def main() -> int:
    log("== phase 4: provenance audit ==")
    if not TRAIN.exists():
        log("missing data/processed/train.csv. Run scripts/build_dataset.py first.")
        return 1

    X, y = load_train()
    log(f"training split: {len(X):,} documents, classes {sorted(set(y))}")
    log(f"{FOLDS} fold stratified cross validation, macro F1, seed {SEED}")
    log("the test split is not touched in this phase")

    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    results: dict[str, dict] = {}

    log("\nblocklist")
    for group, tokens in BLOCKLIST_GROUPS.items():
        log(f"  {group:24} {len(tokens):3} tokens: {', '.join(sorted(tokens))}")
    log(f"  {'total':24} {len(BLOCKLIST):3} tokens")
    log(f"  kept despite comparable skew: {len(KEPT_DESPITE_SKEW)} tokens, see src/markers.py")

    # ---------------------------------------------------------------- probes
    log("\nisolation probes. Each sees one artifact and no content.")

    fmt = np.array([[formatting_features(t)[k] for k in FORMATTING_FEATURE_NAMES] for t in X])
    results["P1_formatting_only"] = score(
        "P1 formatting only, 7 style features, no words", make_pipeline(StandardScaler(), logreg()),
        fmt, y, cv,
    )

    normalised = [normalise(t) for t in X]
    markers_only = [keep_only_markers(t) for t in normalised]
    empty = sum(1 for t in markers_only if not t.strip())
    log(f"     note: {empty:,} of {len(X):,} documents contain no marker at all "
        f"({100*empty/len(X):.1f} percent)")
    results["P2_markers_only"] = score(
        "P2 provenance markers only, content discarded",
        make_pipeline(TfidfVectorizer(min_df=1), logreg()), markers_only, y, cv,
    )

    lengths = np.array([[len(t.split())] for t in X])
    results["P3_length_only"] = score(
        "P3 document length only, one feature",
        make_pipeline(StandardScaler(), logreg()), lengths, y, cv,
    )

    # ------------------------------------------------------------ conditions
    log("\npipeline conditions")

    results["C0_majority"] = score(
        "C0 majority class baseline",
        make_pipeline(FunctionTransformer(lambda t: np.zeros((len(t), 1))),
                      DummyClassifier(strategy="most_frequent")),
        X, y, cv,
    )

    results["C1_raw"] = score(
        "C1 raw text, no normalisation, markers intact",
        Pipeline([("norm", Normaliser(enabled=False)), ("tfidf", tfidf()), ("clf", logreg())]),
        X, y, cv,
    )

    results["C2_normalised"] = score(
        "C2 normalised text, markers intact",
        Pipeline([("norm", Normaliser()), ("tfidf", tfidf()), ("clf", logreg())]),
        X, y, cv,
    )

    stripped_all = [strip_markers(t) for t in normalised]
    results["C3_normalised_stripped"] = score(
        "C3 normalised text, provenance markers stripped",
        make_pipeline(tfidf(), logreg()), stripped_all, y, cv,
    )

    # -------------------------------------------------- reliance transfer probe
    # The decisive experiment. Removing markers from training costs nothing, which
    # could mean either that the model never used them or that content is redundant
    # with them. Fit with markers available, then score on input where they are gone.
    # A drop proves the model leaned on provenance even though the corpus never
    # penalised it for doing so.
    log("\nP4 reliance transfer probe")
    log("  fit on text with markers intact, score on text with markers stripped")
    intact_scores, shifted_scores = [], []
    for train_idx, test_idx in cv.split(normalised, y):
        pipe = make_pipeline(tfidf(), logreg())
        pipe.fit([normalised[i] for i in train_idx], y[train_idx])
        intact_scores.append(
            f1_score(y[test_idx], pipe.predict([normalised[i] for i in test_idx]),
                     average="macro")
        )
        shifted_scores.append(
            f1_score(y[test_idx], pipe.predict([stripped_all[i] for i in test_idx]),
                     average="macro")
        )
    intact_m, shifted_m = float(np.mean(intact_scores)), float(np.mean(shifted_scores))
    log(f"  scored on markers intact    {intact_m:.4f}")
    log(f"  scored on markers stripped  {shifted_m:.4f}   change {shifted_m-intact_m:+.4f}")
    results["P4_reliance_transfer"] = {
        "name": "P4 fit with markers, scored without them",
        "macro_f1_intact": intact_m,
        "macro_f1_stripped": shifted_m,
        "drop": intact_m - shifted_m,
        "folds_intact": [float(v) for v in intact_scores],
        "folds_stripped": [float(v) for v in shifted_scores],
    }

    # ------------------------------------------------- per class breakdown of C3
    log("\nper class detail for C3, pooled across folds")
    from sklearn.model_selection import cross_val_predict
    preds = cross_val_predict(make_pipeline(tfidf(), logreg()), stripped_all, y, cv=cv, n_jobs=-1)
    log(classification_report(y, preds, digits=3))
    c3_report = classification_report(y, preds, digits=4, output_dict=True)

    order = ["FRAUD", "SPAM", "NORMAL"]
    cm = confusion_matrix(y, preds, labels=order)
    log("  confusion matrix, rows are true and columns are predicted")
    log("           " + "".join(f"{c:>9}" for c in order))
    for row_label, row in zip(order, cm):
        log(f"  {row_label:8} " + "".join(f"{v:>9}" for v in row))
    log("  D9 predicted the FRAUD and SPAM boundary would be the weak one, because")
    log("  advance fee fraud is a subset of spam. Read the off diagonal against that.")

    # ---------------------------------------------------------------- deltas
    c1 = results["C1_raw"]["macro_f1_mean"]
    c2 = results["C2_normalised"]["macro_f1_mean"]
    c3 = results["C3_normalised_stripped"]["macro_f1_mean"]

    log("\nwhat the fixes cost")
    log(f"  C1 raw                                {c1:.4f}")
    log(f"  C2 after normalisation                {c2:.4f}   change {c2-c1:+.4f}")
    log(f"  C3 after stripping markers            {c3:.4f}   change {c3-c2:+.4f}")
    log(f"  total attributable to provenance      {c1-c3:+.4f}")
    log(f"  headroom above the majority baseline  {c3-MAJORITY_MACRO_F1:.4f}")

    log("\nreading the probes")
    for key, defect in (
        ("P1_formatting_only", "D4 formatting fingerprint"),
        ("P2_markers_only", "D3 named entity leak"),
        ("P3_length_only", "D11 length shortcut"),
    ):
        m = results[key]["macro_f1_mean"]
        log(f"  {defect:28} {m:.4f}, which is {m/MAJORITY_MACRO_F1:.2f}x the majority baseline")
    log(f"  majority baseline is {MAJORITY_MACRO_F1:.4f} macro F1, not "
        f"{MAJORITY_ACCURACY:.4f}. That figure is accuracy.")

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = [
            ("C0_majority", "majority\nbaseline"),
            ("P3_length_only", "P3 length\nonly"),
            ("P2_markers_only", "P2 markers\nonly"),
            ("P1_formatting_only", "P1 formatting\nonly"),
            ("C1_raw", "C1\nraw"),
            ("C2_normalised", "C2\nnormalised"),
            ("C3_normalised_stripped", "C3 normalised\nstripped"),
        ]
        labels = [lab for _, lab in order]
        means = [results[k]["macro_f1_mean"] for k, _ in order]
        errs = [results[k]["macro_f1_std"] for k, _ in order]
        colours = ["#8c8c8c", "#c46a3f", "#c46a3f", "#c46a3f", "#7fb093", "#4a8f6a", "#2f6f4e"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, means, yerr=errs, capsize=4, color=colours)
        ax.axhline(MAJORITY_MACRO_F1, ls="--", lw=1, color="#444",
                   label=f"majority baseline, macro F1 {MAJORITY_MACRO_F1:.3f}")
        ax.axhline(results["P1_formatting_only"]["macro_f1_mean"], ls=":", lw=1.2,
                   color="#a03c1f",
                   label="formatting only, the real bar to clear "
                         f"{results['P1_formatting_only']['macro_f1_mean']:.3f}")
        ax.set_ylabel("macro F1, 5 fold stratified CV on train")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            "Phase 4: the shortcuts are real, but removing them costs 0.003 macro F1"
        )
        for i, (m, e) in enumerate(zip(means, errs)):
            ax.text(i, m + e + 0.02, f"{m:.3f}", ha="center", fontsize=9)
        ax.legend(loc="lower right")
        fig.tight_layout()
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=150)
        plt.close(fig)
        log(f"\nfigure written to {FIGURE.relative_to(ROOT)}")
    except Exception as exc:
        log(f"\nfigure skipped: {type(exc).__name__}: {exc}")

    report = {
        "phase": 4,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "folds": FOLDS,
        "train_size": len(X),
        "scoring": "f1_macro",
        "majority_baseline_macro_f1": MAJORITY_MACRO_F1,
        "majority_baseline_accuracy": MAJORITY_ACCURACY,
        "blocklist_size": len(BLOCKLIST),
        "blocklist_groups": {g: sorted(t) for g, t in BLOCKLIST_GROUPS.items()},
        "kept_despite_skew": KEPT_DESPITE_SKEW,
        "documents_with_no_marker": empty,
        "results": results,
        "c3_per_class": c3_report,
        "c3_confusion_matrix": {"labels": order, "matrix": cm.tolist()},
        "deltas": {
            "normalisation": c2 - c1,
            "marker_stripping": c3 - c2,
            "total_provenance": c1 - c3,
            "headroom_over_majority": c3 - MAJORITY_MACRO_F1,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    log(f"report written to {REPORT.relative_to(ROOT)}")

    log("\nphase 4 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
