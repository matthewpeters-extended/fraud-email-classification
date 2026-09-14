"""Check that every headline figure in README.md matches the report that produced it.

The README is the deliverable, so its numbers must be traceable rather than remembered.
This reads each phase's machine readable report and asserts the corresponding figure appears
in the README, which catches the two ways a write up rots:

  * a number retyped wrong when the README was drafted
  * a number that was right, and then a rerun changed the underlying result

Run it after any rerun of an analysis phase, and before publishing.

Usage:
    python scripts/verify_readme.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def load(relative: str) -> dict:
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f"{relative} missing. Run the phase that produces it.")
    return json.loads(path.read_text())


def build_expectations() -> list[tuple[str, str]]:
    p2 = load("data/interim/phase2_parse_report.json")
    p3 = load("data/processed/phase3_build_report.json")
    p4 = load("docs/phase4_audit.json")
    p6 = load("docs/phase6_ablation.json")
    p7 = load("docs/phase7_baselines.json")
    p8 = load("docs/phase8_sweep.json")
    p9 = load("docs/phase9_holdout.json")
    p10 = load("docs/phase10_fraud.json")
    ledger = load("docs/holdout_ledger.json")

    head = p9["scored"]["headline"]
    second = p9["scored"]["secondary"]
    ens = p8["results"]["voting_ensemble__tfidf"]
    nb = p8["results"]["multinomial_nb__tfidf"]

    expectations: list[tuple[str, str]] = [
        ("headline test macro F1", f"{head['test_macro_f1']:.4f}"),
        ("bootstrap interval low", f"{head['bootstrap_95_low']:.4f}"),
        ("bootstrap interval high", f"{head['bootstrap_95_high']:.4f}"),
        ("secondary test macro F1", f"{second['test_macro_f1']:.4f}"),
        ("headline total errors", str(head["errors"])),
        ("fraud reaching the inbox", str(head["cost_weighted_errors"]["fraud_reaches_inbox"])),
        ("legitimate mail destroyed",
         str(head["cost_weighted_errors"]["legitimate_mail_lost_to_fraud"])),
        ("test split size", str(p9["test_size"])),
        ("train split size", f"{p9['train_size']:,}"),
        ("errors adjudicated label wrong", str(p9["error_adjudication"]["label_wrong"])),
        ("errors adjudicated model wrong", str(p9["error_adjudication"]["model_wrong"])),
        ("misspelling family cost", f"{abs(p9['misspelling_family']['difference']):.4f}"),
        ("formatting only baseline", f"{p7['results']['B5']['macro_f1_mean']:.4f}"),
        ("markers only baseline", f"{p7['results']['B4']['macro_f1_mean']:.4f}"),
        ("keyword rule 25 baseline", f"{p7['results']['B8']['macro_f1_mean']:.4f}"),
        ("length only baseline", f"{p7['results']['B3']['macro_f1_mean']:.4f}"),
        ("random floor", f"{p7['random_floor']:.4f}"),
        ("majority floor", f"{p7['majority_floor']:.4f}"),
        ("ensemble nested macro F1", f"{ens['nested_macro_f1_mean']:.4f}"),
        ("ensemble nested std", f"{ens['nested_macro_f1_std']:.4f}"),
        ("naive Bayes nested macro F1", f"{nb['nested_macro_f1_mean']:.4f}"),
        ("optimism mean", f"{p8['optimism']['mean']:.4f}"),
        ("optimism max", f"{p8['optimism']['max']:.4f}"),
        ("provenance total cost", f"{p4['deltas']['total_provenance']:.4f}"),
        ("reliance transfer drop", f"{p4['results']['P4_reliance_transfer']['drop']:.4f}"),
        ("blocklist size", str(p4["blocklist_size"])),
        ("preprocessing spread", f"{p6['spread']:.4f}"),
        ("preprocessing fold std", f"{p6['mean_fold_std']:.4f}"),
        ("average precision", f"{p10['average_precision_holdout']:.4f}"),
        ("chosen threshold", str(p10["chosen_threshold"])),
        ("recall carried forward", f"{p10['measured_rates']['recall']:.4f}"),
        ("fraud mailbox envelopes", f"{p2['envelope_splits']:,}"),
        ("spam cluster pool", str(p3["available_pool_per_class"]["SPAM"])),
        ("documents per class", str(p3["per_class"])),
        ("ledger distinct configurations", str(ledger["distinct_configurations"])),
        ("ledger entry count", str(ledger["count"])),
    ]

    for label, key in (
        ("balanced", "corpus as built, one third fraud"),
        ("five percent", "spam heavy inbox, 5 percent fraud"),
        ("half a percent", "filtered inbox, 0.5 percent fraud"),
        ("a tenth of a percent", "well filtered inbox, 0.1 percent fraud"),
    ):
        projection = p10["prevalence_projections"][key]
        expectations.append(
            (f"precision at {label}", f"{projection['precision']:.4f}")
        )
        expectations.append(
            (f"spam false alarms at {label}",
             f"{projection['false_positives_from_spam']:,.0f}")
        )

    for label in ("FRAUD", "SPAM", "NORMAL"):
        expectations.append(
            (f"{label} F1", f"{head['per_class'][label]['f1-score']:.4f}")
        )
        expectations.append(
            (f"{label} precision", f"{head['per_class'][label]['precision']:.4f}")
        )
        expectations.append(
            (f"{label} recall", f"{head['per_class'][label]['recall']:.4f}")
        )

    return expectations


def main() -> int:
    if not README.exists():
        print("README.md not found")
        return 1

    text = README.read_text()
    expectations = build_expectations()

    missing = [(label, value) for label, value in expectations if value not in text]
    for label, value in expectations:
        status = "MISSING" if (label, value) in missing else "ok     "
        print(f"  {status} {label:34} {value}")

    print()
    if missing:
        print(f"FAIL: {len(missing)} of {len(expectations)} figures are not in README.md")
        print("Either the README is stale or a rerun changed the result. Check which.")
        return 1
    print(f"PASS: all {len(expectations)} figures traced to the report that produced them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
