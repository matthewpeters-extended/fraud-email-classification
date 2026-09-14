"""Phase 3: build the deduplicated three class corpus and the train and test split.

Replaces the reference's unseeded sampling (defect D5) and removes the duplication
that would otherwise straddle the split (defect D7).

Usage:
    python scripts/build_dataset.py [--seed 20260914] [--per-class 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import (  # noqa: E402
    CLASSES,
    NEAR_DUPLICATE_THRESHOLD,
    cluster_near_duplicates,
    drop_short,
    drop_label_contradictions,
    find_cross_class_clusters,
    load_sources,
    sample_one_per_cluster,
    stratified_split,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
SPAM_NORMAL = ROOT / "data" / "raw" / "spam_normal_emails.csv"
FRAUD = ROOT / "data" / "interim" / "fraud_messages.csv"
PROCESSED = ROOT / "data" / "processed"
REPORT = PROCESSED / "phase3_build_report.json"

DEFAULT_SEED = 20260914

# The reference uses 1,000 per class. After honest deduplication that is impossible:
# the spam pool yields only 999 distinct near duplicate clusters, so 1,000 balanced
# classes can only be reached by letting copies of the same email in. 900 leaves every
# class real sampling headroom. See docs/phase3_findings.md.
DEFAULT_PER_CLASS = 900
REFERENCE_PER_CLASS = 1000

TEST_FRACTION = 0.2


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    args = parser.parse_args()

    log("== phase 3: build the corpus ==")
    log(f"seed {args.seed}, {args.per_class} per class, test fraction {TEST_FRACTION}")

    for path in (SPAM_NORMAL, FRAUD):
        if not path.exists():
            log(f"missing {path.relative_to(ROOT)}. Run the earlier phases first.")
            return 1

    docs = load_sources(SPAM_NORMAL, FRAUD)
    log(f"\nloaded {len(docs):,} documents")
    log(f"  by class: {dict(Counter(d.label for d in docs))}")

    docs, dropped = drop_short(docs)
    log(f"dropped {dropped} documents with fewer than 5 words, {len(docs):,} remain")

    log(f"\nclustering near duplicates, cosine >= {NEAR_DUPLICATE_THRESHOLD} on char 4 grams")
    log("  clustering across all classes at once, so a message appearing in two")
    log("  classes surfaces as a label contradiction rather than hiding")
    cluster_ids = cluster_near_duplicates([d.text for d in docs])
    for d, c in zip(docs, cluster_ids):
        d.cluster = int(c)

    sizes = Counter(cluster_ids.tolist())
    multi = {c: n for c, n in sizes.items() if n > 1}
    log(f"  {len(sizes):,} clusters over {len(docs):,} documents")
    log(f"  {len(multi):,} clusters hold more than one document")
    log(f"  {sum(n - 1 for n in multi.values()):,} documents are redundant copies "
        f"({100*sum(n-1 for n in multi.values())/len(docs):.1f} percent)")
    if multi:
        log(f"  largest cluster holds {max(multi.values())} documents")

    log("\n  per class, unique clusters versus raw documents")
    per_class_clusters: dict[str, int] = {}
    for label in CLASSES:
        members = [d for d in docs if d.label == label]
        uniq = len({d.cluster for d in members})
        per_class_clusters[label] = uniq
        log(f"    {label:7} {len(members):5,} documents -> {uniq:5,} clusters "
            f"({100*(1-uniq/len(members)):.1f} percent redundant)")

    contradictions = find_cross_class_clusters(docs)
    contradicted_docs = sum(1 for d in docs if d.cluster in contradictions)
    log(f"\nlabel contradictions, clusters spanning more than one class: {len(contradictions)}")
    log(f"  covering {contradicted_docs} documents")
    if contradictions:
        for c, labels in list(contradictions.items())[:10]:
            members = [d for d in docs if d.cluster == c]
            log(f"  cluster {c}: {sorted(labels)}, {len(members)} documents")
            log(f"    {members[0].text[:110]!r}")

    docs, contradictions = drop_label_contradictions(docs)
    log(f"\ndropped {len(contradictions)} contradicted clusters, {len(docs):,} documents remain")
    log("  a cluster carrying two labels cannot be sampled safely, see src/corpus.py")

    log(f"\nsampling {args.per_class} per class, at most one document per cluster")
    chosen, pool_sizes = sample_one_per_cluster(docs, args.per_class, args.seed)
    for label in CLASSES:
        headroom = pool_sizes[label]
        log(f"  {label:7} drew {args.per_class:,} from {headroom:,} available clusters "
            f"({100*args.per_class/headroom:.0f} percent of the pool)")
    tightest = min(pool_sizes, key=lambda k: pool_sizes[k])
    log(f"  binding constraint: {tightest} at {pool_sizes[tightest]:,} clusters")
    if pool_sizes[tightest] < REFERENCE_PER_CLASS:
        log(f"  note: the reference's {REFERENCE_PER_CLASS:,} per class is unreachable "
            f"without readmitting duplicates")

    assert len({d.cluster for d in chosen}) == len(chosen), "a cluster was sampled twice"
    log("  verified: every sampled document comes from a distinct cluster")

    train, test = stratified_split(chosen, TEST_FRACTION, args.seed)
    log(f"\nsplit: {len(train):,} train, {len(test):,} test")
    log(f"  train by class: {dict(Counter(d.label for d in train))}")
    log(f"  test by class:  {dict(Counter(d.label for d in test))}")

    train_clusters = {d.cluster for d in train}
    test_clusters = {d.cluster for d in test}
    overlap = train_clusters & test_clusters
    log(f"  clusters appearing on both sides of the split: {len(overlap)}")
    if overlap:
        log("  LEAKAGE. Aborting.")
        return 2
    log("  verified: no duplicate or near duplicate straddles the split")

    write_csv(PROCESSED / "corpus.csv", chosen)
    write_csv(PROCESSED / "train.csv", train)
    write_csv(PROCESSED / "test.csv", test)
    log(f"\nwrote corpus.csv, train.csv and test.csv to {PROCESSED.relative_to(ROOT)}")

    words = {label: [d.words for d in chosen if d.label == label] for label in CLASSES}
    log("\nbody length in words, by class")
    for label in CLASSES:
        w = sorted(words[label])
        log(f"  {label:7} median {w[len(w)//2]:5,}  mean {np.mean(w):7,.0f}  "
            f"min {w[0]:4,}  max {w[-1]:7,}")

    report = {
        "phase": 3,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "per_class": args.per_class,
        "reference_per_class": REFERENCE_PER_CLASS,
        "test_fraction": TEST_FRACTION,
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "documents_loaded": len(docs) + dropped,
        "dropped_short": dropped,
        "documents_clustered": len(docs),
        "clusters_total": len(sizes),
        "clusters_multi_member": len(multi),
        "redundant_documents": sum(n - 1 for n in multi.values()),
        "largest_cluster": max(multi.values()) if multi else 1,
        "clusters_per_class": per_class_clusters,
        "available_pool_per_class": pool_sizes,
        "label_contradictions": len(contradictions),
        "label_contradiction_documents": contradicted_docs,
        "label_contradiction_detail": {
            str(c): sorted(labels) for c, labels in contradictions.items()
        },
        "train_size": len(train),
        "test_size": len(test),
        "split_cluster_overlap": len(overlap),
        "words_by_class": {
            label: {
                "median": int(np.median(words[label])),
                "mean": round(float(np.mean(words[label])), 1),
                "min": int(min(words[label])),
                "max": int(max(words[label])),
            }
            for label in CLASSES
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    log(f"report written to {REPORT.relative_to(ROOT)}")

    log("\nphase 3 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
