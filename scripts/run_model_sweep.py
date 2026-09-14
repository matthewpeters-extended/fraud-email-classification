"""Phase 8: the model sweep, scored by nested cross validation.

Seven estimators crossed with two vectorisers, hyperparameters searched in an inner loop so
the outer score is never tuned on the data it reports.

Why nested. The usual approach, and the one the reference takes, is to search a grid with
cross validation and then quote the best cross validated score. That number is optimistic:
the grid was chosen by looking at those same folds, so the winner's advantage includes
whatever it happened to gain from fold noise. Nested cross validation separates the two
jobs. An inner loop picks hyperparameters on the training part of each outer fold, and the
outer loop scores the resulting model on data the search never saw.

Both numbers are reported, along with the gap between them, because the size of that gap is
itself a result worth having.

The two searches use the same fold count. An earlier version scored the non nested number
with the 3 fold inner splitter while the nested number came from 5 outer folds, which
confounded the comparison badly: a 3 fold search trains each candidate on 67 percent of the
data against 80 percent for the outer folds, so the non nested figure was depressed by
having less training data at the same time as being inflated by selection. The two effects
partly cancelled and the measured gap came out negative, which is impossible for a bias
that only ever inflates. Both now use 5 folds, so the only difference left is whether the
winner was chosen on the same data it is scored on.

A note on the preprocessing. The text steps are applied once, outside the loops, rather
than inside every fit. That is safe here and only here: those transformers hold no learned
state, which `test_steps_are_stateless_so_fitting_changes_nothing` asserts directly. The
vectorisers, which do learn vocabulary and document frequencies, stay inside the pipeline
and are therefore fitted on training folds only.

Usage:
    python scripts/run_model_sweep.py [--quick]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import bar_to_clear, load_baselines  # noqa: E402
from src.models import (  # noqa: E402
    SEED,
    VECTORISER_GRID,
    estimators,
    grid_size,
    vectorisers,
    voting_ensemble,
)
from src.preprocess import chosen_text_steps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "train.csv"
REPORT = ROOT / "docs" / "phase8_sweep.json"
FIGURE = ROOT / "reports" / "figures" / "phase8_sweep.png"

OUTER_FOLDS = 5
INNER_FOLDS = 3


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load() -> tuple[list[str], np.ndarray]:
    csv.field_size_limit(10_000_000)
    with TRAIN.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r["text"] for r in rows], np.array([r["label"] for r in rows])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="drop the slowest estimators, for a fast smoke run")
    args = parser.parse_args()

    log("== phase 8: model sweep ==")
    if not TRAIN.exists():
        log("missing data/processed/train.csv. Run scripts/build_dataset.py first.")
        return 1

    X_raw, y = load()
    log(f"training split: {len(X_raw):,} documents")
    log(f"outer {OUTER_FOLDS} fold stratified CV for scoring, "
        f"inner {INNER_FOLDS} fold for hyperparameter search, seed {SEED}")
    log("the test split is not touched until phase 9")

    try:
        bar_label, bar = bar_to_clear()
        floor = load_baselines()["random_floor"]
        log(f"\nbar to clear: {bar:.4f}, {bar_label}")
        log(f"trivial floor: {floor:.4f}, random guessing")
    except FileNotFoundError:
        log("\nphase 7 baselines not found, run scripts/run_baselines.py first")
        return 1

    log("\napplying the phase 6 text pipeline once, outside the loops")
    log("  safe because those steps hold no learned state, asserted by")
    log("  test_steps_are_stateless_so_fitting_changes_nothing")
    log("  the vectorisers do learn, and stay inside the pipeline")
    started = time.time()
    X = Pipeline(chosen_text_steps()).fit_transform(X_raw)
    log(f"  done in {time.time()-started:.1f}s")

    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)

    catalogue: dict[str, tuple[object, dict]] = dict(estimators())
    ens, ens_grid = voting_ensemble()
    catalogue["voting_ensemble"] = (ens, ens_grid)
    if args.quick:
        for slow in ("random_forest", "voting_ensemble", "knn"):
            catalogue.pop(slow, None)
        log("\nquick mode: dropped random_forest, voting_ensemble and knn")

    results: dict[str, dict] = {}
    log(f"\n{'model':22} {'vect':6} {'nested CV':>18} {'tuned on all':>13} "
        f"{'optimism':>9} {'secs':>6}")
    log("  " + "." * 84)

    for vect_name, vect in vectorisers().items():
        for model_name, (estimator, grid) in catalogue.items():
            key = f"{model_name}__{vect_name}"
            full_grid = {**grid, **VECTORISER_GRID}
            pipeline = Pipeline([("vect", vect), ("clf", estimator)])
            started = time.time()

            # Nested: the search is the estimator, so it refits inside each outer fold and
            # the outer score comes from data the search never saw.
            nested = cross_val_score(
                GridSearchCV(pipeline, full_grid, scoring="f1_macro", cv=inner,
                             n_jobs=-1, refit=True),
                X, y, cv=outer, scoring="f1_macro", n_jobs=1,
            )

            # Non nested: search the same grid over the same 5 folds, then quote the
            # winner's cross validated score. This is the number a conventional workflow
            # reports. Same folds and same training set size as the nested run, so the
            # only difference is selection on the scoring data.
            flat = GridSearchCV(pipeline, full_grid, scoring="f1_macro", cv=outer,
                                n_jobs=-1, refit=True)
            flat.fit(X, y)
            search = flat
            elapsed = time.time() - started

            optimism = float(search.best_score_) - float(nested.mean())
            results[key] = {
                "model": model_name,
                "vectoriser": vect_name,
                "nested_macro_f1_mean": float(nested.mean()),
                "nested_macro_f1_std": float(nested.std()),
                "nested_folds": [float(s) for s in nested],
                "tuned_on_all_macro_f1": float(search.best_score_),
                "optimism_gap": optimism,
                "best_params": {k: str(v) for k, v in search.best_params_.items()},
                "grid_size": grid_size(full_grid),
                "seconds": round(elapsed, 1),
            }
            log(f"  {model_name:22} {vect_name:6} {nested.mean():.4f} +/- "
                f"{nested.std():.4f} {search.best_score_:13.4f} {optimism:+9.4f} "
                f"{elapsed:6.0f}")

    # ------------------------------------------------------------------ ranking
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["nested_macro_f1_mean"])
    log("\nranked by nested cross validation, which is the honest number")
    log(f"  {'rank':<5}{'model':24}{'vect':7}{'nested':>9}{'std':>8}   best params")
    for i, (key, r) in enumerate(ranked, start=1):
        params = ", ".join(f"{k.split('__', 1)[1]}={v}" for k, v in r["best_params"].items())
        log(f"  {i:<5}{r['model']:24}{r['vectoriser']:7}"
            f"{r['nested_macro_f1_mean']:9.4f}{r['nested_macro_f1_std']:8.4f}   {params}")

    best_key, best = ranked[0]
    log(f"\nbest: {best['model']} with {best['vectoriser']} features, "
        f"{best['nested_macro_f1_mean']:.4f}")

    # Which configurations are indistinguishable from the winner
    tied = [
        k for k, r in results.items()
        if best["nested_macro_f1_mean"] - r["nested_macro_f1_mean"] <= best["nested_macro_f1_std"]
    ]
    log(f"  within one standard deviation of it: {len(tied)} of {len(results)} configurations")
    log(f"  {', '.join(sorted(tied))}")

    # ---------------------------------------------------------------- optimism
    gaps = [r["optimism_gap"] for r in results.values()]
    log("\nhow much does tuning on the scoring folds inflate the estimate")
    log("  both numbers use the same 5 folds and the same training set size, so the")
    log("  difference is selection alone")
    log(f"  mean optimism gap across {len(gaps)} configurations: {np.mean(gaps):+.4f}")
    log(f"  largest: {max(gaps):+.4f}   smallest: {min(gaps):+.4f}")
    worst = max(results.items(), key=lambda kv: kv[1]["optimism_gap"])
    log(f"  worst offender: {worst[1]['model']} with {worst[1]['vectoriser']}, "
        f"{worst[1]['tuned_on_all_macro_f1']:.4f} quoted against "
        f"{worst[1]['nested_macro_f1_mean']:.4f} honest")
    log("  The reference style workflow reports the left hand number. The gap is the")
    log("  part of it that comes from having chosen the grid on the same folds.")

    # ------------------------------------------------------------- against bar
    log("\nagainst the phase 7 baselines")
    log(f"  best model         {best['nested_macro_f1_mean']:.4f}")
    log(f"  formatting only    {bar:.4f}   model is {best['nested_macro_f1_mean']-bar:+.4f}")
    log(f"  random guessing    {floor:.4f}   model is {best['nested_macro_f1_mean']-floor:+.4f}")

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = [k for k, _ in ranked]
        labels = [f"{results[k]['model'].replace('_', chr(10))}\n({results[k]['vectoriser']})"
                  for k in order]
        nested_means = [results[k]["nested_macro_f1_mean"] for k in order]
        nested_errs = [results[k]["nested_macro_f1_std"] for k in order]
        tuned = [results[k]["tuned_on_all_macro_f1"] for k in order]

        fig, ax = plt.subplots(figsize=(max(12, len(order) * 1.0), 6))
        x = np.arange(len(order))
        ax.bar(x - 0.2, nested_means, 0.4, yerr=nested_errs, capsize=3,
               color="#2f6f4e", label="nested CV, honest")
        ax.bar(x + 0.2, tuned, 0.4, color="#c99a2e",
               label="tuned on all of train, optimistic")
        ax.axhline(bar, ls=":", lw=1.4, color="#a03c1f",
                   label=f"bar to clear {bar:.3f}, formatting only")
        ax.axhline(floor, ls="--", lw=1, color="#444", label=f"random {floor:.3f}")
        ax.set_xticks(x, labels, fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("macro F1")
        ax.set_title("Phase 8: nested cross validation against the number a tuned on "
                     "everything workflow would report")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=150)
        plt.close(fig)
        log(f"\nfigure written to {FIGURE.relative_to(ROOT)}")
    except Exception as exc:
        log(f"\nfigure skipped: {type(exc).__name__}: {exc}")

    report = {
        "phase": 8,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "train_size": len(X),
        "scoring": "f1_macro",
        "bar_to_clear": bar,
        "random_floor": floor,
        "results": results,
        "ranking": [k for k, _ in ranked],
        "best": {"key": best_key, **best},
        "tied_with_best": sorted(tied),
        "optimism": {
            "mean": float(np.mean(gaps)),
            "max": float(max(gaps)),
            "min": float(min(gaps)),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    log(f"report written to {REPORT.relative_to(ROOT)}")

    log("\nphase 8 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
