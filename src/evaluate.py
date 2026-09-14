"""Holdout evaluation and error analysis.

The test split has been untouched since phase 3. This module scores it once, with the
configuration phase 8 selected, and records that it happened.

The recording matters. The discipline of a single holdout evaluation is easy to state and
easy to erode: a disappointing number invites one more idea, and the second evaluation is
no longer a holdout. So every scoring run appends to a ledger with the configuration and a
hash of it. If the test set is scored twice with different settings, that is visible in the
repository rather than lost in a terminal.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.models import SEED, vectorisers, voting_ensemble

CLASSES = ("FRAUD", "SPAM", "NORMAL")

LEDGER = Path(__file__).resolve().parents[1] / "docs" / "holdout_ledger.json"


def parse_param(value: str):
    """Recover a parameter value from the phase 8 report.

    The sweep serialises best parameters with `str()`, so "(1, 2)" and "0.1" arrive as
    text. `literal_eval` recovers the literals and leaves bare strings such as "sqrt"
    alone, which is what it should do.
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def bare_estimator(model_name: str):
    """A fresh estimator by the name phase 8 used."""
    builders = {
        "multinomial_nb": lambda: MultinomialNB(),
        "complement_nb": lambda: ComplementNB(),
        "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
        "linear_svm": lambda: LinearSVC(random_state=SEED, max_iter=20_000),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=1
        ),
        "knn": lambda: KNeighborsClassifier(metric="cosine"),
        "voting_ensemble": lambda: voting_ensemble()[0],
    }
    if model_name not in builders:
        raise ValueError(f"unknown model {model_name!r}")
    return builders[model_name]()


def build_winning_pipeline(best: dict) -> Pipeline:
    """Rebuild the configuration phase 8 selected, with its tuned hyperparameters."""
    pipeline = Pipeline([
        ("vect", vectorisers()[best["vectoriser"]]),
        ("clf", bare_estimator(best["model"])),
    ])
    params = {k: parse_param(v) for k, v in best["best_params"].items()}
    pipeline.set_params(**params)
    return pipeline


def confusion(y_true, y_pred, labels=CLASSES) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, prediction in zip(y_true, y_pred):
        matrix[index[truth], index[prediction]] += 1
    return matrix


def boundary_errors(matrix: np.ndarray, labels=CLASSES) -> dict[str, int]:
    """Errors grouped by which pair of classes they sit between.

    Phase 4 found the FRAUD and SPAM boundary carried 57 percent of errors on the training
    folds, and D9 explains why: advance fee fraud is a kind of spam. This checks whether
    that structure survives to the holdout.
    """
    pairs: dict[str, int] = {}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i >= j:
                continue
            key = f"{a} and {b}"
            pairs[key] = int(matrix[i, j] + matrix[j, i])
    return pairs


def cost_weighted_errors(matrix: np.ndarray, labels=CLASSES) -> dict[str, int]:
    """Errors split by deployment consequence rather than by class pair.

    A fraud email classified as spam still leaves the inbox, so the user is protected. A
    fraud email classified as normal lands in front of them. Treating those two as the same
    error, which macro F1 does, understates how differently they matter.
    """
    index = {label: i for i, label in enumerate(labels)}
    fraud, spam, normal = index["FRAUD"], index["SPAM"], index["NORMAL"]
    return {
        "fraud_reaches_inbox": int(matrix[fraud, normal]),
        "fraud_caught_as_spam": int(matrix[fraud, spam]),
        "legitimate_mail_lost_to_fraud": int(matrix[normal, fraud]),
        "legitimate_mail_lost_to_spam": int(matrix[normal, spam]),
        "spam_misfiled": int(matrix[spam, fraud] + matrix[spam, normal]),
    }


def error_records(raw, cleaned, y_true, y_pred) -> list[dict]:
    """One record per misclassified document, for reading rather than counting."""
    records = []
    for i, (truth, prediction) in enumerate(zip(y_true, y_pred)):
        if truth == prediction:
            continue
        records.append({
            "index": i,
            "true": str(truth),
            "predicted": str(prediction),
            "words_raw": len(raw[i].split()),
            "words_cleaned": len(cleaned[i].split()),
            "excerpt": " ".join(raw[i].split())[:300],
        })
    return records


def config_fingerprint(best: dict) -> str:
    """A stable hash of the scored configuration, for the ledger."""
    payload = json.dumps(
        {"model": best["model"], "vectoriser": best["vectoriser"],
         "params": best["best_params"]},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def record_in_ledger(best: dict, macro_f1: float, note: str = "") -> dict:
    """Append this evaluation to the holdout ledger and return the full ledger.

    Never overwrites. A second entry is the point: it makes a repeated holdout evaluation
    an artifact in the repository instead of something only the author remembers.
    """
    entry = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": best["model"],
        "vectoriser": best["vectoriser"],
        "params": best["best_params"],
        "fingerprint": config_fingerprint(best),
        "test_macro_f1": round(float(macro_f1), 6),
        "note": note,
    }
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"evaluations": []}
    ledger["evaluations"].append(entry)
    ledger["count"] = len(ledger["evaluations"])
    ledger["distinct_configurations"] = len(
        {e["fingerprint"] for e in ledger["evaluations"]}
    )
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
    return ledger
