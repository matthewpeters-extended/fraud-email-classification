"""Phase 5: exploratory analysis of the training corpus.

Phase 4 showed the content carries the signal. This describes what that content is:
length profiles, vocabulary overlap between classes, and the terms that actually
separate them once normalisation and marker stripping are applied.

The discriminative term ranking is also a second leakage sweep. Phase 4 curated its
blocklist from raw token skew; asking the same question of the cleaned representation
tests whether anything provenance shaped survived the cleanup.

Training split only. The test split is not touched until phase 9.

Usage:
    python scripts/run_eda.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explore import (  # noqa: E402
    CLASSES,
    coverage,
    discriminative_terms,
    document_frequencies,
    jaccard,
    length_profile,
    type_token_ratio,
    vocabulary,
)
from src.markers import BLOCKLIST, PHASE5_REVIEWED, strip_markers  # noqa: E402
from src.normalise import normalise  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "train.csv"
REPORT = ROOT / "docs" / "phase5_eda.json"
FIG_DIR = ROOT / "reports" / "figures"

MIN_DF_VOCAB = 5
MIN_DF_TERMS = 20
TOP_N = 25


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load() -> dict[str, list[str]]:
    csv.field_size_limit(10_000_000)
    with TRAIN.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_class: dict[str, list[str]] = {c: [] for c in CLASSES}
    for r in rows:
        by_class[r["label"]].append(r["text"])
    return by_class


def table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    log("  " + "  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    log("  " + "  ".join("." * w for w in widths))
    for r in rows:
        log("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


def main() -> int:
    log("== phase 5: exploratory analysis ==")
    if not TRAIN.exists():
        log("missing data/processed/train.csv. Run scripts/build_dataset.py first.")
        return 1

    raw = load()
    log(f"training split: {sum(len(v) for v in raw.values()):,} documents")
    log("pipeline applied: canonical normalisation, then provenance markers stripped")

    cleaned = {c: [strip_markers(normalise(t)) for t in texts] for c, texts in raw.items()}
    tokenised = {c: [t.split() for t in texts] for c, texts in cleaned.items()}

    # ------------------------------------------------------------ length profile
    log("\n1. document length, in words")
    log("   raw text as stored")
    raw_profiles = {c: length_profile(c, [len(t.split()) for t in raw[c]]) for c in CLASSES}
    table(
        ["class", "n", "min", "p25", "median", "p75", "p90", "max", "mean"],
        [
            [c, p.count, p.minimum, p.p25, p.median, p.p75, p.p90, p.maximum, f"{p.mean:.0f}"]
            for c, p in ((c, raw_profiles[c]) for c in CLASSES)
        ],
    )
    log("\n   after normalisation and marker stripping")
    clean_profiles = {c: length_profile(c, [len(t) for t in tokenised[c]]) for c in CLASSES}
    table(
        ["class", "n", "min", "p25", "median", "p75", "p90", "max", "mean"],
        [
            [c, p.count, p.minimum, p.p25, p.median, p.p75, p.p90, p.maximum, f"{p.mean:.0f}"]
            for c, p in ((c, clean_profiles[c]) for c in CLASSES)
        ],
    )
    ratio = clean_profiles["FRAUD"].median / max(clean_profiles["SPAM"].median, 1)
    log(f"\n   FRAUD median is {ratio:.2f}x the SPAM median after cleaning")
    log("   D11 measured length alone at 0.4386 macro F1, so this gap is usable signal")
    log("   and any model result has to be read against that baseline")

    # --------------------------------------------------------------- vocabulary
    log(f"\n2. vocabulary, tokens appearing in at least {MIN_DF_VOCAB} documents of a class")
    vocabs = {c: vocabulary(tokenised[c], min_df=MIN_DF_VOCAB) for c in CLASSES}
    ttr = {c: type_token_ratio(tokenised[c]) for c in CLASSES}
    table(
        ["class", "vocabulary", "total tokens", "type token ratio"],
        [
            [
                c,
                f"{len(vocabs[c]):,}",
                f"{sum(len(t) for t in tokenised[c]):,}",
                f"{ttr[c]:.4f}",
            ]
            for c in CLASSES
        ],
    )
    log("   a low type token ratio means repetitive, templated text")

    log("\n   Jaccard similarity of vocabularies, symmetric")
    table(
        [""] + list(CLASSES),
        [[a] + [f"{jaccard(vocabs[a], vocabs[b]):.3f}" if a != b else "." for b in CLASSES]
         for a in CLASSES],
    )

    log("\n   coverage: share of the row class vocabulary also present in the column class")
    table(
        [""] + list(CLASSES),
        [[a] + [f"{coverage(vocabs[a], vocabs[b]):.3f}" if a != b else "." for b in CLASSES]
         for a in CLASSES],
    )
    fs = jaccard(vocabs["FRAUD"], vocabs["SPAM"])
    fn = jaccard(vocabs["FRAUD"], vocabs["NORMAL"])
    log(f"\n   FRAUD and SPAM share {fs:.3f}, FRAUD and NORMAL share {fn:.3f}")
    log(f"   ratio {fs/max(fn,1e-9):.2f}x, which is the D9 prediction in vocabulary terms")

    unique = {
        c: sorted(vocabs[c] - set().union(*(vocabs[o] for o in CLASSES if o != c)))
        for c in CLASSES
    }
    log("\n   vocabulary exclusive to one class")
    table(
        ["class", "exclusive terms", "share of its vocabulary"],
        [[c, f"{len(unique[c]):,}", f"{100*len(unique[c])/len(vocabs[c]):.1f} percent"]
         for c in CLASSES],
    )

    # ------------------------------------------------- discriminative terms
    log(f"\n3. most discriminative terms, document level log odds ratio, min df {MIN_DF_TERMS}")
    log("   purity is the share of documents containing the term that belong to this class")
    terms = discriminative_terms(tokenised, min_df=MIN_DF_TERMS, top_n=TOP_N)
    for c in CLASSES:
        log(f"\n   {c}")
        table(
            ["term", "log odds", "df in class", "purity"],
            [[t, f"{lor:+.2f}", df, f"{100*p:.0f} percent"] for t, lor, df, p in terms[c]],
        )

    log("\n   leakage review of the above")
    log("   Asserting that no blocklisted token appears here would be tautological: the")
    log("   pipeline strips them, so they cannot appear. The real question is whether any")
    log("   term in this ranking is provenance that the phase 4 curation missed. So every")
    log("   ranked term must carry a recorded verdict in src.markers.PHASE5_REVIEWED.")

    surviving = [(c, t) for c in CLASSES for t, _, _, _ in terms[c] if t in BLOCKLIST]
    if surviving:
        log(f"   FAILURE: blocked tokens reached the ranking: {surviving}")
        return 2

    unreviewed = sorted(
        {t for c in CLASSES for t, _, _, _ in terms[c]} - set(PHASE5_REVIEWED)
    )
    reviewed = 3 * TOP_N - len(unreviewed)
    log(f"   {reviewed} of {3*TOP_N} ranked terms carry a recorded verdict")
    if unreviewed:
        log(f"   UNREVIEWED, classify each in src.markers.PHASE5_REVIEWED: {unreviewed}")
        return 3
    blocked_at_5 = sorted(t for t, v in PHASE5_REVIEWED.items() if v.startswith("blocked"))
    log(f"   {len(blocked_at_5)} terms were reclassified as provenance and blocked "
        f"at phase 5: {', '.join(blocked_at_5)}")

    # ------------------------------------------------------------------ figures
    figures: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        palette = {"FRAUD": "#b5432f", "SPAM": "#c99a2e", "NORMAL": "#2f6f4e"}

        # length distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, source, title in (
            (axes[0], raw, "raw text as stored"),
            (axes[1], cleaned, "normalised, markers stripped"),
        ):
            data = [[len(t.split()) for t in source[c]] for c in CLASSES]
            parts = ax.boxplot(
                data, tick_labels=list(CLASSES), patch_artist=True, showfliers=False
            )
            for patch, c in zip(parts["boxes"], CLASSES):
                patch.set_facecolor(palette[c])
                patch.set_alpha(0.75)
            for median in parts["medians"]:
                median.set_color("#111")
            ax.set_yscale("log")
            ax.set_ylabel("words per document, log scale")
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Phase 5: document length by class, outliers hidden")
        fig.tight_layout()
        path = FIG_DIR / "phase5_length.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path.name)

        # discriminative terms
        fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
        for ax, c in zip(axes, CLASSES):
            rows = terms[c][:15][::-1]
            ax.barh([t for t, _, _, _ in rows], [lor for _, lor, _, _ in rows],
                    color=palette[c], alpha=0.85)
            ax.set_title(c)
            ax.set_xlabel("log odds ratio")
            ax.grid(axis="x", alpha=0.25)
        fig.suptitle(
            "Phase 5: terms that separate each class, after normalisation and marker stripping"
        )
        fig.tight_layout()
        path = FIG_DIR / "phase5_terms.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path.name)

        # vocabulary overlap
        fig, ax = plt.subplots(figsize=(6.5, 5))
        matrix = [[jaccard(vocabs[a], vocabs[b]) if a != b else float("nan") for b in CLASSES]
                  for a in CLASSES]
        im = ax.imshow(matrix, cmap="YlOrBr", vmin=0, vmax=max(
            v for row in matrix for v in row if v == v) * 1.15)
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        for i in range(3):
            for j in range(3):
                if i != j:
                    ax.text(j, i, f"{matrix[i][j]:.3f}", ha="center", va="center", fontsize=12)
                else:
                    ax.text(j, i, ".", ha="center", va="center", fontsize=12, color="#888")
        ax.set_title("Phase 5: vocabulary Jaccard overlap\nFRAUD and SPAM overlap most")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Jaccard")
        fig.tight_layout()
        path = FIG_DIR / "phase5_vocab_overlap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path.name)

        log(f"\nfigures written: {', '.join(figures)}")
    except Exception as exc:
        log(f"\nfigures skipped: {type(exc).__name__}: {exc}")

    report = {
        "phase": 5,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_size": sum(len(v) for v in raw.values()),
        "min_df_vocab": MIN_DF_VOCAB,
        "min_df_terms": MIN_DF_TERMS,
        "length_raw": {c: raw_profiles[c].as_dict() for c in CLASSES},
        "length_cleaned": {c: clean_profiles[c].as_dict() for c in CLASSES},
        "fraud_to_spam_median_ratio": round(ratio, 3),
        "vocabulary_size": {c: len(vocabs[c]) for c in CLASSES},
        "type_token_ratio": {c: round(ttr[c], 5) for c in CLASSES},
        "jaccard": {
            a: {b: round(jaccard(vocabs[a], vocabs[b]), 4) for b in CLASSES if b != a}
            for a in CLASSES
        },
        "coverage": {
            a: {b: round(coverage(vocabs[a], vocabs[b]), 4) for b in CLASSES if b != a}
            for a in CLASSES
        },
        "exclusive_vocabulary": {c: len(unique[c]) for c in CLASSES},
        "discriminative_terms": {
            c: [
                {"term": t, "log_odds": round(lor, 4), "df_in_class": df, "purity": round(p, 4)}
                for t, lor, df, p in terms[c]
            ]
            for c in CLASSES
        },
        "blocklisted_terms_surviving": len(surviving),
        "ranked_terms_unreviewed": unreviewed,
        "blocked_at_phase5": blocked_at_5,
        "figures": figures,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    log(f"report written to {REPORT.relative_to(ROOT)}")

    log("\nphase 5 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
