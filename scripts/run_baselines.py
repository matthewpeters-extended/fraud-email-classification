"""Phase 7: the canonical baseline suite.

Every score reported from phase 8 onward is quoted against this table. The point is to make
it impossible to present a headline number without the reader seeing what a trivial
strategy, a ten word rule, and three content free shortcuts achieve on the same data.

Baselines:

    B0  most frequent class
    B1  stratified random, drawing from the training class distribution
    B2  uniform random
    B3  document length only, one feature, cleaned text
    B4  provenance markers only, 42 tokens, no content
    B5  formatting only, seven numeric features, no words at all
    B6  keyword rule, 5 words per class, learned inside each fold
    B7  keyword rule, 10 words per class
    B8  keyword rule, 25 words per class

For context, and clearly labelled as not a baseline, the phase 6 pipeline with logistic
regression is scored alongside.

Five fold stratified cross validation on the training split, macro F1, seed 20260914. The
test split is not touched until phase 9.

Usage:
    python scripts/run_baselines.py
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import KeywordRule, coverage_of  # noqa: E402
from src.markers import keep_only_markers  # noqa: E402
from src.normalise import (  # noqa: E402
    FORMATTING_FEATURE_NAMES,
    formatting_features,
    normalise,
)
from src.preprocess import chosen_text_steps, text_steps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "train.csv"
REPORT = ROOT / "docs" / "phase7_baselines.json"
FIGURE = ROOT / "reports" / "figures" / "phase7_baselines.png"

SEED = 20260914
FOLDS = 5


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load() -> tuple[list[str], np.ndarray]:
    csv.field_size_limit(10_000_000)
    with TRAIN.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r["text"] for r in rows], np.array([r["label"] for r in rows])


def score(pipeline, X, y, cv) -> tuple[float, float, list[float]]:  # noqa: N803
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    return float(scores.mean()), float(scores.std()), [float(s) for s in scores]


def main() -> int:
    log("== phase 7: baselines ==")
    if not TRAIN.exists():
        log("missing data/processed/train.csv. Run scripts/build_dataset.py first.")
        return 1

    X, y = load()
    log(f"training split: {len(X):,} documents, balanced across three classes")
    log(f"{FOLDS} fold stratified cross validation, macro F1, seed {SEED}")

    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    results: dict[str, dict] = {}

    def record(key: str, label: str, pipeline, data, note: str = "") -> None:
        mean, std, folds = score(pipeline, data, y, cv)
        results[key] = {
            "label": label,
            "macro_f1_mean": mean,
            "macro_f1_std": std,
            "folds": folds,
            "note": note,
        }
        log(f"  {key} {label:48} {mean:.4f} +/- {std:.4f}")

    # Precompute the representations each baseline needs.
    cleaned = Pipeline(chosen_text_steps()).fit_transform(X)
    # Phase 6 measured the length baseline on normalisation plus marker stripping only,
    # before the stopword and lemmatisation steps were chosen. Stopword removal changes
    # word counts substantially, so the two are different measurements of the same idea.
    # Both are reported rather than quietly replacing one figure with the other.
    minimally_cleaned = Pipeline(text_steps()).fit_transform(X)
    normalised = [normalise(t) for t in X]
    markers_only = [keep_only_markers(t) for t in normalised]
    lengths = np.array([[len(t.split())] for t in cleaned])
    formatting = np.array(
        [[formatting_features(t)[k] for k in FORMATTING_FEATURE_NAMES] for t in X]
    )
    ignore_text = FunctionTransformer(lambda t: np.zeros((len(t), 1)))

    log("\ntrivial strategies")
    record("B0", "most frequent class",
           make_pipeline(ignore_text, DummyClassifier(strategy="most_frequent")), X)
    record("B1", "stratified random, from the training distribution",
           make_pipeline(ignore_text,
                         DummyClassifier(strategy="stratified", random_state=SEED)), X)
    record("B2", "uniform random",
           make_pipeline(ignore_text,
                         DummyClassifier(strategy="uniform", random_state=SEED)), X)

    log("\ncontent free shortcuts, each sees one artifact and no words")
    record("B3", "document length only, on the chosen phase 6 pipeline",
           make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,
                                                              random_state=SEED)),
           lengths, "the representation the model actually sees, see D11")
    record("B3b", "document length only, normalised and stripped only",
           make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,
                                                              random_state=SEED)),
           np.array([[len(t.split())] for t in minimally_cleaned]),
           "the phase 6 measurement, kept for continuity")
    record("B4", "provenance markers only, 42 tokens",
           make_pipeline(TfidfVectorizer(min_df=1),
                         LogisticRegression(max_iter=2000, random_state=SEED)),
           markers_only, "content discarded, see D3")
    record("B5", "formatting only, seven numeric features",
           make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=SEED)),
           formatting, "no words at all, see D4")

    log("\nkeyword rules, learned inside each fold, never hand picked")
    for key, k in (("B6", 5), ("B7", 10), ("B8", 25)):
        record(key, f"keyword rule, {k} words per class",
               KeywordRule(keywords_per_class=k), cleaned,
               "decision list on word presence")
        rule = KeywordRule(keywords_per_class=k).fit(cleaned, y)
        cov = coverage_of(rule, cleaned)
        results[key]["coverage"] = cov
        results[key]["example_keywords"] = {
            c: rule.keywords_[c][:8] for c in sorted(rule.keywords_)
        }
        log(f"       fires on {100*cov:.1f} percent of documents, "
            f"falls back on the rest")

    log("\nfor context, not a baseline")
    record("MODEL", "phase 6 pipeline with logistic regression",
           Pipeline(chosen_text_steps()
                    + [("tfidf", TfidfVectorizer(min_df=2, sublinear_tf=True)),
                       ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
           X, "the thing the baselines exist to be compared against")

    # ---------------------------------------------------------------- reading
    log("\nthe floor is not where it looks")
    b0 = results["B0"]["macro_f1_mean"]
    b1 = results["B1"]["macro_f1_mean"]
    b2 = results["B2"]["macro_f1_mean"]
    log(f"  most frequent class   {b0:.4f}")
    log(f"  stratified random     {b1:.4f}")
    log(f"  uniform random        {b2:.4f}")
    log(f"  random beats always predicting one class by {b1-b0:+.4f} macro F1.")
    log("  Macro F1 averages per class F1, so a single class predictor scores 0.5 on its")
    log("  chosen class and 0 on the other two, giving 0.1667. Random guessing on a")
    log("  balanced three class problem scores about a third on every class instead.")
    log("  Earlier phases quoted 0.1667 as the floor. For macro F1 the honest floor for a")
    log("  balanced problem is random, not majority.")

    log("\nthe two length measurements")
    log(f"  B3  on the chosen pipeline, with stopwords removed   "
        f"{results['B3']['macro_f1_mean']:.4f}")
    log(f"  B3b on normalisation and stripping only              "
        f"{results['B3b']['macro_f1_mean']:.4f}")
    log("  Phase 6 reported the second, before the stopword step was chosen. Removing")
    log("  stopwords changes word counts by 43 percent, so these measure the same idea on")
    log("  different text. B3 is the one that matters, because it is what the model sees.")

    shortcut_keys = ("B3", "B4", "B5", "B6", "B7", "B8")
    bar = max(results[k]["macro_f1_mean"] for k in shortcut_keys)
    bar_key = max(shortcut_keys, key=lambda k: results[k]["macro_f1_mean"])
    model = results["MODEL"]["macro_f1_mean"]
    log(f"\nthe bar a model has to clear: {bar:.4f}, set by {bar_key}, "
        f"{results[bar_key]['label']}")
    log(f"  phase 6 pipeline reaches {model:.4f}, which is {model-bar:+.4f} above it")
    log(f"  and {model-b1:+.4f} above random")

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = ["B0", "B2", "B1", "B6", "B3", "B7", "B8", "B4", "B5", "MODEL"]
        short = {
            "B0": "most frequent\nclass",
            "B1": "stratified\nrandom",
            "B2": "uniform\nrandom",
            "B3": "length only\n1 feature",
            "B4": "markers only\n42 tokens",
            "B5": "formatting only\n7 features",
            "B6": "keyword rule\n5 per class",
            "B7": "keyword rule\n10 per class",
            "B8": "keyword rule\n25 per class",
            "MODEL": "phase 6 pipeline\nlogistic regression",
        }
        labels = [f"{k}\n{short[k]}" for k in order]
        means = [results[k]["macro_f1_mean"] for k in order]
        errs = [results[k]["macro_f1_std"] for k in order]
        colours = []
        for k in order:
            if k == "MODEL":
                colours.append("#2f6f4e")
            elif k in ("B0", "B1", "B2"):
                colours.append("#8c8c8c")
            elif k in ("B6", "B7", "B8"):
                colours.append("#c99a2e")
            else:
                colours.append("#c46a3f")

        fig, ax = plt.subplots(figsize=(13, 5.5))
        ax.bar(labels, means, yerr=errs, capsize=4, color=colours)
        ax.axhline(bar, ls=":", lw=1.4, color="#a03c1f",
                   label=f"the bar to clear: {bar:.3f}, formatting only, no words")
        ax.axhline(b1, ls="--", lw=1, color="#444",
                   label=f"random guessing {b1:.3f}")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("macro F1, 5 fold CV on train")
        ax.set_title(
            "Phase 7: what a model has to beat.\n"
            "Grey is trivial, amber is a learned keyword rule, orange sees no words at all."
        )
        for i, (m, e) in enumerate(zip(means, errs)):
            ax.text(i, m + e + 0.015, f"{m:.3f}", ha="center", fontsize=8)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("grey: trivial   amber: learned keyword rule   "
                      "orange: content free shortcut   green: the model", fontsize=9)
        fig.tight_layout()
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=150)
        plt.close(fig)
        log(f"\nfigure written to {FIGURE.relative_to(ROOT)}")
    except Exception as exc:
        log(f"\nfigure skipped: {type(exc).__name__}: {exc}")

    report = {
        "phase": 7,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "folds": FOLDS,
        "train_size": len(X),
        "scoring": "f1_macro",
        "results": results,
        "bar_to_clear": {"key": bar_key, "macro_f1": bar,
                         "label": results[bar_key]["label"]},
        "random_floor": b1,
        "majority_floor": b0,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    log(f"report written to {REPORT.relative_to(ROOT)}")

    log("\nphase 7 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
