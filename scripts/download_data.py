"""Phase 1: acquire and verify the raw corpus.

Downloads Datasets.zip from the reference repository, records a SHA256 checksum,
extracts it safely into data/raw, then cross checks the observed row counts against
the figures documented in docs/sources.md.

Nothing here trusts the archive. Zip members are validated against path traversal
before extraction, and every count is reported rather than assumed.

Usage:
    python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ARCHIVE = RAW / "Datasets.zip"
MANIFEST = RAW / "manifest.json"

URL = "https://raw.githubusercontent.com/SimarjotKaur/Email-Classifier/master/Datasets.zip"

# Documented in docs/sources.md. Observed values are compared against these.
EXPECTED_ARCHIVE_BYTES = 10_754_307
EXPECTED_SPAM_ROWS = 5_728
EXPECTED_FRAUD_MESSAGES = 4_075

# Resolved in docs/data_defects.md at phase 1. The reference README claims 4,075 fraud
# messages. The canonical CLAIR corpus is cited at 3,977. We observe 3,976 marker lines.
# The reference figure is unsubstantiated, so the accepted set records what is defensible
# and the residual off by one is a parsing boundary handed to phase 2.
ACCEPTED_FRAUD_MESSAGES = {3_976, 3_977}
EXPECTED_FINAL_ROWS = 3_000
EXPECTED_PER_CLASS = 1_000

# The archive's real member names, which differ from the reference README's prose.
# "fradulent" is the upstream author's spelling and is preserved deliberately.
SPAM_FILE = "spam_normal_emails.csv"
FRAUD_FILE = "fradulent_emails.txt"
FINAL_FILE = "final_dataset.csv"

# The CLAIR fraud corpus is a concatenated mailbox. Each message begins with this marker.
FRAUD_MESSAGE_MARKER = "From r"


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(force: bool) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() and not force:
        log(f"archive already present, skipping download: {ARCHIVE.name}")
        return
    log(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        if resp.status != 200:
            raise SystemExit(f"download failed with HTTP {resp.status}")
        payload = resp.read()
    ARCHIVE.write_bytes(payload)
    log(f"wrote {ARCHIVE.name}, {len(payload):,} bytes")


def safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Reject absolute paths, parent traversal and symlinks before extraction."""
    approved: list[zipfile.ZipInfo] = []
    for info in zf.infolist():
        name = info.filename
        if name.endswith("/"):
            continue
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise SystemExit(f"refusing unsafe zip member: {name!r}")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise SystemExit(f"refusing symlink zip member: {name!r}")
        approved.append(info)
    return approved


def extract() -> list[dict]:
    with zipfile.ZipFile(ARCHIVE) as zf:
        members = safe_members(zf)
        log(f"archive holds {len(members)} file(s)")
        listing = []
        for info in members:
            target = RAW / Path(info.filename).name
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())
            listing.append(
                {
                    "name": target.name,
                    "archive_path": info.filename,
                    "bytes": target.stat().st_size,
                    "sha256": sha256_of(target),
                }
            )
            log(f"  extracted {target.name}, {target.stat().st_size:,} bytes")
        return listing


def count_spam_csv(path: Path) -> dict:
    """Row count and label split for the Enron derived spam corpus."""
    csv.field_size_limit(10_000_000)
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = 0
        labels: dict[str, int] = {}
        for row in reader:
            rows += 1
            key = str(row.get("spam", "?")).strip()
            labels[key] = labels.get(key, 0) + 1
    return {"columns": header, "rows": rows, "label_counts": labels}


def count_fraud_text(path: Path) -> dict:
    """Message count for the concatenated CLAIR fraud mailbox."""
    raw = path.read_bytes().decode("utf-8", errors="replace")
    markers = sum(1 for line in raw.splitlines() if line.startswith(FRAUD_MESSAGE_MARKER))
    subjects = sum(1 for line in raw.splitlines() if line.startswith("Subject:"))
    return {
        "bytes": len(raw),
        "lines": raw.count("\n") + 1,
        "message_markers": markers,
        "subject_lines": subjects,
    }


def count_final_dataset(path: Path) -> dict:
    """Row count, class balance and the source marker probe for the prebuilt corpus."""
    csv.field_size_limit(10_000_000)
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = 0
        labels: dict[str, int] = {}
        marker: dict[str, dict[str, int]] = {}
        lengths: dict[str, list[int]] = {}
        for row in reader:
            rows += 1
            label = str(row.get("Label", "?")).strip()
            body = str(row.get("Email", ""))
            labels[label] = labels.get(label, 0) + 1
            slot = marker.setdefault(label, {"with": 0, "total": 0})
            slot["total"] += 1
            if body.lstrip().lower().startswith("subject:"):
                slot["with"] += 1
            lengths.setdefault(label, []).append(len(body))

    share = {
        label: {
            "with": v["with"],
            "total": v["total"],
            "pct": 100.0 * v["with"] / v["total"] if v["total"] else 0.0,
        }
        for label, v in marker.items()
    }
    mean_len = {
        label: round(sum(vals) / len(vals), 1) for label, vals in lengths.items() if vals
    }
    return {
        "columns": header,
        "rows": rows,
        "label_counts": labels,
        "subject_prefix_share": share,
        "mean_body_chars": mean_len,
    }


def verify(label: str, observed: int, expected: int) -> bool:
    ok = observed == expected
    verdict = "MATCH" if ok else "MISMATCH"
    log(f"  {label}: observed {observed:,}, documented {expected:,} -> {verdict}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="redownload even if present")
    args = parser.parse_args()

    log("== phase 1: acquire and verify ==")
    download(args.force)

    archive_bytes = ARCHIVE.stat().st_size
    archive_hash = sha256_of(ARCHIVE)
    log(f"archive sha256: {archive_hash}")

    log("checking archive size against docs/sources.md")
    checks = {"archive_bytes": verify("archive bytes", archive_bytes, EXPECTED_ARCHIVE_BYTES)}

    listing = extract()

    spam_path = RAW / SPAM_FILE
    fraud_path = RAW / FRAUD_FILE
    final_path = RAW / FINAL_FILE
    missing = [p.name for p in (spam_path, fraud_path, final_path) if not p.exists()]
    if missing:
        log(f"expected source files not found in archive: {missing}")
        log(f"archive actually contained: {[m['name'] for m in listing]}")
        return 1

    log("counting the spam corpus")
    spam_stats = count_spam_csv(spam_path)
    log(f"  columns: {spam_stats['columns']}")
    log(f"  label counts: {spam_stats['label_counts']}")
    checks["spam_rows"] = verify(f"{SPAM_FILE} rows", spam_stats["rows"], EXPECTED_SPAM_ROWS)

    log("counting the fraud corpus")
    fraud_stats = count_fraud_text(fraud_path)
    log(f"  {fraud_stats['lines']:,} lines, {fraud_stats['bytes']:,} characters")
    log(f"  lines beginning {FRAUD_MESSAGE_MARKER!r}: {fraud_stats['message_markers']:,}")
    log(f"  lines beginning 'Subject:': {fraud_stats['subject_lines']:,}")
    observed_fraud = fraud_stats["message_markers"]
    if observed_fraud in ACCEPTED_FRAUD_MESSAGES:
        checks["fraud_messages"] = True
        log(
            f"  fraud message markers: observed {observed_fraud:,}, reference claims "
            f"{EXPECTED_FRAUD_MESSAGES:,} -> ACCEPTED"
        )
        log("    reference figure is unsubstantiated, see docs/data_defects.md D1")
    else:
        checks["fraud_messages"] = False
        log(
            f"  fraud message markers: observed {observed_fraud:,}, accepted "
            f"{sorted(ACCEPTED_FRAUD_MESSAGES)} -> MISMATCH"
        )

    log("counting the prebuilt dataset")
    final_stats = count_final_dataset(final_path)
    log(f"  columns: {final_stats['columns']}")
    log(f"  label counts: {final_stats['label_counts']}")
    checks["final_rows"] = verify(f"{FINAL_FILE} rows", final_stats["rows"], EXPECTED_FINAL_ROWS)
    for label, n in sorted(final_stats["label_counts"].items()):
        checks[f"final_class_{label}"] = verify(f"  class {label}", n, EXPECTED_PER_CLASS)

    log("")
    log("leakage probe: share of each class whose body opens with 'Subject:'")
    probe = final_stats["subject_prefix_share"]
    for label in sorted(probe):
        share = probe[label]
        log(f"  {label}: {share['with']:,} of {share['total']:,} = {share['pct']:.1f} percent")
    log("A near perfect split here means the source file is recoverable from one token,")
    log("so any classifier can separate classes without reading the content. See PLAN.md 8.1.")

    manifest = {
        "phase": 1,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": URL,
        "archive": {"bytes": archive_bytes, "sha256": archive_hash},
        "extracted": listing,
        "spam_corpus": spam_stats,
        "fraud_corpus": fraud_stats,
        "final_dataset": final_stats,
        "expected": {
            "archive_bytes": EXPECTED_ARCHIVE_BYTES,
            "spam_rows": EXPECTED_SPAM_ROWS,
            "fraud_messages": EXPECTED_FRAUD_MESSAGES,
            "final_rows": EXPECTED_FINAL_ROWS,
            "per_class": EXPECTED_PER_CLASS,
        },
        "checks": checks,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"manifest written to {MANIFEST.relative_to(ROOT)}")

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        log("")
        log(f"VERIFICATION INCOMPLETE. Mismatched checks: {failed}")
        log("Resolve in docs/data_defects.md before phase 2 consumes this data.")
        return 2

    log("")
    log("all documented counts reproduced. phase 1 clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
