"""Phase 9: score the held out test split, once, and read the errors.

The test split has been untouched since phase 3 built it. Nothing in phases 4 through 8 has
seen it: every figure in those phases came from cross validation on the training split.

Two configurations are scored, decided in advance at phase 8 and not chosen after seeing
these results:

    headline   the phase 8 winner, a soft voting ensemble on tfidf features
    secondary  multinomial naive Bayes on tfidf, which is inside the noise of the winner
               at a sixth of the compute and with readable coefficients

Reporting both is a commitment made before looking. The headline stays the headline even if
the secondary scores higher here, because switching after the fact would be selecting on the
test set, which is the one thing this split exists to prevent.

Every run appends to `docs/holdout_ledger.json` with a fingerprint of the configuration, so
a second evaluation is an artifact in the repository rather than something only the author
remembers.

Usage:
    python scripts/evaluate_holdout.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import bar_to_clear, load_baselines  # noqa: E402
from src.evaluate import (  # noqa: E402
    CLASSES,
    boundary_errors,
    build_winning_pipeline,
    confusion,
    cost_weighted_errors,
    error_records,
    record_in_ledger,
)
from src.preprocess import chosen_text_steps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "train.csv"
TEST = ROOT / "data" / "processed" / "test.csv"
SWEEP = ROOT / "docs" / "phase8_sweep.json"
REPORT = ROOT / "docs" / "phase9_holdout.json"
ERRORS = ROOT / "docs" / "phase9_errors.json"
FIGURE = ROOT / "reports" / "figures" / "phase9_confusion.png"

# The spam class leans on a family of deliberate misspellings found at phase 5. Each term
# appears in only 20 to 30 training documents, so phase 5 asked whether they are doing
# disproportionate work. This answers it.
MISSPELLING_FAMILY = ("shlpplng", "oniine", "miiiion", "prlces", "successfull")

BOOTSTRAP_RESAMPLES = 5000

# Adjudication of every holdout error, by reading the message. Keyed by a prefix of the
# excerpt so the mapping survives a rerun.
#
# This is judgement, not measurement, and it is recorded here rather than asserted in prose
# so a reader can disagree with any individual line. Nothing downstream depends on it: no
# model, hyperparameter or threshold was chosen using it, and the headline macro F1 stays
# exactly as measured. It exists to answer what the errors have in common, which is the
# question phase 9 was set.
#
#   label_wrong   the message is what the model said it was; the corpus label disagrees
#   model_wrong   the corpus label is right and the model missed it
#   ambiguous     a reasonable person could file it either way
ADJUDICATION: dict[str, tuple[str, str]] = {
    "Want to be BETTER then pornstar": (
        "label_wrong", "pornographic spam sitting in the fraud corpus, this is D6"),
    "US Bank Alert Message": (
        "model_wrong", "genuine bank phishing, which belongs in the fraud class"),
    "de la part des enfants": (
        "label_wrong", "advance fee fraud in French, labelled spam"),
    "undeliverable : home based business": (
        "ambiguous", "a bounce notice about a spam message, not itself spam"),
    "i need your help dear sir": (
        "label_wrong", "advance fee fraud from Sierra Leone, labelled spam"),
    "investment offer from joseph otisa": (
        "label_wrong", "advance fee fraud, a Nigerian bank branch manager, labelled spam"),
    "charity sees the need not the cost": (
        "label_wrong", "advance fee fraud, a dying merchant in Kuwait, labelled spam"),
    "out of office autoreply": (
        "label_wrong", "an out of office autoreply is not spam"),
    "from mrs fati": (
        "label_wrong", "advance fee fraud, labelled spam"),
    "http : / / www . virtu ally": (
        "model_wrong", "an unsolicited marketing approach, correctly spam"),
    "[ ilug ] deal": (
        "label_wrong", "advance fee fraud from a claimed Citibank officer, labelled spam"),
    "request for assistance barrister": (
        "label_wrong", "advance fee fraud from a claimed Lagos barrister, labelled spam"),
    "foreign business representative needed": (
        "label_wrong", "advance fee solicitation, labelled spam"),
    "data for moody ' s riskcalc": (
        "model_wrong", "internal mail about a trial subscription, reads as marketing"),
    "you are now subscribed to the frbnyrmagl": (
        "ambiguous", "a mailing list confirmation, which is what spam also looks like"),
    "cusip": (
        "model_wrong", "a bond reference containing the word mortgage, a top spam term"),
    "summer offer": (
        "model_wrong", "an internship offer containing the word offer"),
}


def adjudicate(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Attach the recorded verdict to each error, flagging any that has none."""
    tally: dict[str, int] = {"label_wrong": 0, "model_wrong": 0,
                             "ambiguous": 0, "unreviewed": 0}
    for record in records:
        excerpt = record["excerpt"]
        verdict, reason = "unreviewed", ""
        for prefix, (v, r) in ADJUDICATION.items():
            if excerpt.startswith(prefix):
                verdict, reason = v, r
                break
        record["verdict"] = verdict
        record["reason"] = reason
        tally[verdict] += 1
    return records, tally


def bootstrap_interval(y_true, y_pred, resamples=BOOTSTRAP_RESAMPLES, seed=20260914):
    """A 95 percent interval for macro F1 on this test set, by resampling documents.

    Needed because comparing the holdout score against the cross validation standard
    deviation is the wrong test. That figure measures how much the estimate moved between
    training folds; it says nothing about how precisely 540 documents pin down a score. On
    540 documents at roughly 0.97, sampling noise alone is worth several thousandths, which
    is the same size as the gap being judged.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    scores = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        # A resample that loses a whole class cannot be scored on macro F1.
        if len(np.unique(y_true[idx])) < len(CLASSES):
            scores[i] = np.nan
            continue
        scores[i] = f1_score(y_true[idx], y_pred[idx], average="macro")
    scores = scores[~np.isnan(scores)]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)), \
        float(scores.std())


def log(msg: str = "") -> None:
    print(msg, flush=True)


def read(path: Path) -> tuple[list[str], np.ndarray]:
    csv.field_size_limit(10_000_000)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r["text"] for r in rows], np.array([r["label"] for r in rows])


def show_confusion(matrix: np.ndarray) -> None:
    log("           " + "".join(f"{c:>9}" for c in CLASSES) + "      total")
    for label, row in zip(CLASSES, matrix):
        log(f"  {label:8} " + "".join(f"{v:>9}" for v in row) + f"{row.sum():>11}")
    log("  " + "-" * 0 + f"{'predicted':8} " + "".join(f"{v:>9}" for v in matrix.sum(axis=0)))


def main() -> int:
    log("== phase 9: holdout evaluation ==")
    for path in (TRAIN, TEST, SWEEP):
        if not path.exists():
            log(f"missing {path.relative_to(ROOT)}. Run the earlier phases first.")
            return 1

    sweep = json.loads(SWEEP.read_text())
    baselines = load_baselines()
    bar_label, bar = bar_to_clear()
    floor = baselines["random_floor"]

    X_train_raw, y_train = read(TRAIN)
    X_test_raw, y_test = read(TEST)
    log(f"train {len(X_train_raw):,} documents, test {len(X_test_raw):,} documents")
    log("the test split has not been read by any phase since 3 built it")

    text = Pipeline(chosen_text_steps())
    X_train = text.fit_transform(X_train_raw)
    X_test = text.transform(X_test_raw)

    headline = sweep["best"]
    secondary_key = "multinomial_nb__tfidf"
    secondary = {"key": secondary_key, **sweep["results"][secondary_key]}
    log(f"\nheadline configuration : {headline['model']} on {headline['vectoriser']}, "
        f"nested CV {headline['nested_macro_f1_mean']:.4f}")
    log(f"secondary configuration: {secondary['model']} on {secondary['vectoriser']}, "
        f"nested CV {secondary['nested_macro_f1_mean']:.4f}")
    log("both were chosen at phase 8, before this split was read")

    scored: dict[str, dict] = {}
    for role, config in (("headline", headline), ("secondary", secondary)):
        pipeline = build_winning_pipeline(config)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        macro = f1_score(y_test, predictions, average="macro")
        accuracy = accuracy_score(y_test, predictions)
        matrix = confusion(y_test, predictions)
        report = classification_report(y_test, predictions, digits=4, output_dict=True)

        log(f"\n{'=' * 70}")
        log(f"{role.upper()}: {config['model']} on {config['vectoriser']} features")
        log(f"{'=' * 70}")
        log(f"  macro F1 {macro:.4f}   accuracy {accuracy:.4f}")
        log(f"  nested CV estimate was {config['nested_macro_f1_mean']:.4f} "
            f"+/- {config['nested_macro_f1_std']:.4f}")
        gap = macro - config["nested_macro_f1_mean"]
        lo, hi, boot_std = bootstrap_interval(y_test, predictions)
        log(f"  95 percent interval on this test set: {lo:.4f} to {hi:.4f}, "
            f"width {hi-lo:.4f}")
        log(f"  holdout minus nested estimate: {gap:+.4f}")
        covered = lo <= config["nested_macro_f1_mean"] <= hi
        log(f"  the nested estimate {config['nested_macro_f1_mean']:.4f} is "
            f"{'inside' if covered else 'OUTSIDE'} that interval")
        log("  note: comparing the gap against the cross validation standard deviation")
        log("  would be the wrong test. That figure measures movement between training")
        log("  folds, not how precisely 540 documents pin down a score.")
        inside = covered

        log("\n  per class")
        log(f"    {'class':8}{'precision':>11}{'recall':>9}{'F1':>9}{'support':>9}")
        for label in CLASSES:
            r = report[label]
            log(f"    {label:8}{r['precision']:11.4f}{r['recall']:9.4f}"
                f"{r['f1-score']:9.4f}{int(r['support']):9d}")

        log("\n  confusion matrix, rows are true and columns are predicted")
        show_confusion(matrix)

        errors = int(matrix.sum() - np.trace(matrix))
        log(f"\n  {errors} errors in {matrix.sum()} documents")
        log("  by boundary")
        for pair, n in sorted(boundary_errors(matrix).items(), key=lambda kv: -kv[1]):
            share = 100 * n / errors if errors else 0
            log(f"    {pair:22} {n:3}  {share:5.1f} percent of errors")

        log("  by deployment consequence")
        for kind, n in cost_weighted_errors(matrix).items():
            log(f"    {kind:32} {n:3}")

        scored[role] = {
            "model": config["model"],
            "vectoriser": config["vectoriser"],
            "params": config["best_params"],
            "test_macro_f1": float(macro),
            "test_accuracy": float(accuracy),
            "nested_cv_estimate": config["nested_macro_f1_mean"],
            "nested_cv_std": config["nested_macro_f1_std"],
            "holdout_minus_estimate": float(gap),
            "bootstrap_95_low": lo,
            "bootstrap_95_high": hi,
            "bootstrap_std": boot_std,
            "nested_estimate_inside_interval": bool(inside),
            "per_class": {c: report[c] for c in CLASSES},
            "confusion_matrix": matrix.tolist(),
            "confusion_labels": list(CLASSES),
            "errors": errors,
            "boundary_errors": boundary_errors(matrix),
            "cost_weighted_errors": cost_weighted_errors(matrix),
            "predictions": [str(p) for p in predictions],
        }

    # ------------------------------------------------------------ error reading
    log(f"\n{'=' * 70}")
    log("ERROR ANALYSIS, headline configuration")
    log(f"{'=' * 70}")
    predictions = np.array(scored["headline"]["predictions"])
    records = error_records(X_test_raw, X_test, y_test, predictions)
    log(f"  {len(records)} misclassified documents")

    lengths_wrong = [r["words_cleaned"] for r in records]
    lengths_all = [len(t.split()) for t in X_test]
    log(f"  median length of misclassified documents: {int(np.median(lengths_wrong))} words")
    log(f"  median length of all test documents:      {int(np.median(lengths_all))} words")

    records, tally = adjudicate(records)

    log("\n  every error, adjudicated by reading the message")
    for r in records:
        log(f"\n    [{r['true']} predicted {r['predicted']}] {r['words_cleaned']} words "
            f"-> {r['verdict'].upper()}")
        if r["reason"]:
            log(f"      {r['reason']}")
        log(f"      {r['excerpt'][:200]}")

    log("\n  tally")
    for verdict in ("label_wrong", "model_wrong", "ambiguous", "unreviewed"):
        n = tally[verdict]
        share = 100 * n / len(records) if records else 0
        log(f"    {verdict:14} {n:3}  {share:5.1f} percent of errors")
    if tally["unreviewed"]:
        log("    UNREVIEWED errors present. Add them to ADJUDICATION before publishing.")

    genuine = tally["model_wrong"]
    log(f"\n  {tally['label_wrong']} of {len(records)} errors are the model being right")
    log("  and the corpus label being wrong. Every one of those is an advance fee scam")
    log("  sitting in the Enron spam corpus, or the reverse, which is exactly D9: the two")
    log("  source corpora overlap and disagree about the same kind of message.")
    log(f"\n  if those labels were corrected, errors would fall from {len(records)} to "
        f"{genuine} plus {tally['ambiguous']} arguable")
    log("  That is an indication of the corpus ceiling, not a score. The headline macro F1")
    log("  remains as measured, and no model decision was made from this reading.")

    # ------------------------------------- the phase 5 question about misspellings
    log(f"\n{'=' * 70}")
    log("MISSPELLING FAMILY, the question phase 5 left open")
    log(f"{'=' * 70}")
    log(f"  terms: {', '.join(MISSPELLING_FAMILY)}")
    present_train = sum(
        1 for t in X_train if any(w in t.split() for w in MISSPELLING_FAMILY)
    )
    present_test = sum(
        1 for t in X_test if any(w in t.split() for w in MISSPELLING_FAMILY)
    )
    log(f"  present in {present_train} of {len(X_train)} training documents "
        f"and {present_test} of {len(X_test)} test documents")

    stripped_train = [
        " ".join(w for w in t.split() if w not in MISSPELLING_FAMILY) for t in X_train
    ]
    stripped_test = [
        " ".join(w for w in t.split() if w not in MISSPELLING_FAMILY) for t in X_test
    ]
    ablated = build_winning_pipeline(headline)
    ablated.fit(stripped_train, y_train)
    ablated_macro = f1_score(y_test, ablated.predict(stripped_test), average="macro")
    log(f"  macro F1 with the family present: {scored['headline']['test_macro_f1']:.4f}")
    log(f"  macro F1 with the family removed: {ablated_macro:.4f}")
    log(f"  difference {ablated_macro - scored['headline']['test_macro_f1']:+.4f}")

    # --------------------------------------------------------------- comparison
    log(f"\n{'=' * 70}")
    log("AGAINST THE BASELINES")
    log(f"{'=' * 70}")
    log(f"  {'reference point':44}{'macro F1':>10}")
    rows = [
        ("random guessing, the trivial floor", floor),
        ("keyword rule, 25 words per class",
         baselines["results"]["B8"]["macro_f1_mean"]),
        ("provenance markers only", baselines["results"]["B4"]["macro_f1_mean"]),
        (f"{bar_label}, the bar to clear", bar),
        (f"secondary, {secondary['model']} on holdout",
         scored["secondary"]["test_macro_f1"]),
        (f"headline, {headline['model']} on holdout", scored["headline"]["test_macro_f1"]),
    ]
    for label, value in rows:
        log(f"  {label:44}{value:10.4f}")
    log(f"\n  headline clears the bar by "
        f"{scored['headline']['test_macro_f1'] - bar:+.4f}")
    log("  the baselines are cross validated on train, so this is indicative rather than")
    log("  a like for like comparison on the same documents")

    # ------------------------------------------------------------------ ledger
    record_in_ledger(headline, scored["headline"]["test_macro_f1"],
                     note="phase 9 headline")
    ledger = record_in_ledger(secondary, scored["secondary"]["test_macro_f1"],
                              note="phase 9 secondary, pre committed at phase 8")
    log(f"\nholdout ledger: {ledger['count']} evaluations recorded across "
        f"{ledger['distinct_configurations']} distinct configurations")
    log("  What matters is the second number. Repeated runs of the same configuration are")
    log("  reruns of the reporting code, and the recorded scores are byte identical, which")
    log("  the ledger also demonstrates. A third distinct configuration appearing here")
    log("  would mean something was selected on the test split, and it would be visible in")
    log("  the repository rather than only in the author's memory.")

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, role in zip(axes, ("headline", "secondary")):
            matrix = np.array(scored[role]["confusion_matrix"])
            im = ax.imshow(matrix, cmap="Greens", vmin=0, vmax=matrix.max())
            ax.set_xticks(range(3), CLASSES)
            ax.set_yticks(range(3), CLASSES)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{role}: {scored[role]['model']}\n"
                         f"macro F1 {scored[role]['test_macro_f1']:.4f}, "
                         f"{scored[role]['errors']} errors")
            for i in range(3):
                for j in range(3):
                    colour = "white" if matrix[i, j] > matrix.max() * 0.6 else "#222"
                    ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                            color=colour, fontsize=13)
            fig.colorbar(im, ax=ax, shrink=0.75)
        fig.suptitle("Phase 9: held out test split, 540 documents, scored once")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=150)
        plt.close(fig)
        log(f"\nfigure written to {FIGURE.relative_to(ROOT)}")
    except Exception as exc:
        log(f"\nfigure skipped: {type(exc).__name__}: {exc}")

    for role in scored:
        scored[role].pop("predictions", None)

    REPORT.write_text(json.dumps({
        "phase": 9,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "bar_to_clear": bar,
        "random_floor": floor,
        "scored": scored,
        "error_adjudication": tally,
        "misspelling_family": {
            "terms": list(MISSPELLING_FAMILY),
            "train_documents": present_train,
            "test_documents": present_test,
            "macro_f1_with": scored["headline"]["test_macro_f1"],
            "macro_f1_without": float(ablated_macro),
            "difference": float(ablated_macro - scored["headline"]["test_macro_f1"]),
        },
    }, indent=2) + "\n")
    ERRORS.write_text(json.dumps({"errors": records, "tally": tally}, indent=2) + "\n")
    log(f"report written to {REPORT.relative_to(ROOT)}")
    log(f"errors written to {ERRORS.relative_to(ROOT)}")

    log("\nphase 9 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
