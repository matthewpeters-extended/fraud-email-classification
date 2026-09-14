"""Phase 2: parse the fraud mailbox into individual messages.

Splits fradulent_emails.txt on mbox envelope lines, parses each message with the
standard library email module, extracts the best available text body, and writes
one row per message to data/interim/fraud_messages.csv.

Reports parse health rather than assuming it. Every message that fails to yield a
usable body is counted and characterised.

Usage:
    python scripts/parse_fraud.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fraud_mailbox import parse_corpus, read_corpus, split_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "fradulent_emails.txt"
OUT_CSV = ROOT / "data" / "interim" / "fraud_messages.csv"
OUT_JSON = ROOT / "data" / "interim" / "phase2_parse_report.json"

# Established at phase 2 by counting mbox envelope lines. The reference README claims
# 4,075 and the commonly cited CLAIR figure is 3,977. Both are wrong for this file.
EXPECTED_MESSAGES = 3_978

MIN_WORDS = 5


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    log("== phase 2: parse the fraud mailbox ==")
    if not SOURCE.exists():
        log(f"missing {SOURCE.relative_to(ROOT)}. Run scripts/download_data.py first.")
        return 1

    raw = read_corpus(SOURCE)
    log(f"read {len(raw):,} characters")

    blocks = split_messages(raw)
    log(f"envelope splits: {len(blocks):,}, expected {EXPECTED_MESSAGES:,}")
    if len(blocks) != EXPECTED_MESSAGES:
        log("  MISMATCH. Investigate before proceeding.")
        return 2
    log("  MATCH")

    messages = list(parse_corpus(raw))

    usable = [m for m in messages if m.ok]
    unusable = [m for m in messages if not m.ok]
    note_counts = Counter(n.split(":")[0] for m in messages for n in m.notes)
    charsets = Counter(m.charset.lower() for m in messages if m.charset)
    lengths = sorted(m.body_chars for m in usable)

    log()
    log("parse health")
    log(f"  usable bodies, at least {MIN_WORDS} words: {len(usable):,}")
    log(f"  unusable: {len(unusable):,} ({100*len(unusable)/len(messages):.2f} percent)")
    log(f"  multipart: {sum(1 for m in messages if m.was_multipart):,}")
    log(f"  html only, tags stripped: {sum(1 for m in messages if m.was_html):,}")
    log(f"  notes: {dict(note_counts)}")
    log(f"  top declared charsets: {dict(charsets.most_common(6))}")

    log()
    log("body length in characters, usable messages only")
    if lengths:
        log(f"  min {lengths[0]:,}  median {int(median(lengths)):,}  max {lengths[-1]:,}")
        log(f"  mean {sum(lengths)/len(lengths):,.0f}")

    # Direct check that MIME decoding did its job. Raw quoted printable artifacts
    # were one half of defect D4.
    raw_artifacts = raw.count("=20")
    body_artifacts = sum(m.body.count("=20") for m in messages)
    log()
    log("MIME decoding check, defect D4")
    log(f"  '=20' occurrences in the raw mailbox: {raw_artifacts:,}")
    log(f"  '=20' occurrences in parsed bodies:  {body_artifacts:,}")
    if raw_artifacts:
        removed = 100 * (1 - body_artifacts / raw_artifacts)
        log(f"  resolved by quoted printable decoding: {removed:.1f} percent")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "index",
                "date",
                "from_addr",
                "subject",
                "content_type",
                "charset",
                "was_multipart",
                "was_html",
                "body_chars",
                "notes",
                "body",
            ]
        )
        for m in usable:
            writer.writerow(
                [
                    m.index,
                    m.date,
                    m.from_addr,
                    m.subject,
                    m.content_type,
                    m.charset,
                    int(m.was_multipart),
                    int(m.was_html),
                    m.body_chars,
                    ";".join(m.notes),
                    m.body,
                ]
            )
    log()
    log(f"wrote {len(usable):,} rows to {OUT_CSV.relative_to(ROOT)}")

    report = {
        "phase": 2,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": SOURCE.name,
        "source_chars": len(raw),
        "envelope_splits": len(blocks),
        "expected_messages": EXPECTED_MESSAGES,
        "usable": len(usable),
        "unusable": len(unusable),
        "min_words_for_usable": MIN_WORDS,
        "multipart": sum(1 for m in messages if m.was_multipart),
        "html_only": sum(1 for m in messages if m.was_html),
        "notes": dict(note_counts),
        "charsets": dict(charsets),
        "body_chars": {
            "min": lengths[0] if lengths else 0,
            "median": int(median(lengths)) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "max": lengths[-1] if lengths else 0,
        },
        "mime_artifacts": {"raw_eq20": raw_artifacts, "parsed_eq20": body_artifacts},
        "unusable_sample": [
            {"index": m.index, "subject": m.subject[:80], "notes": m.notes, "chars": m.body_chars}
            for m in unusable[:15]
        ],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    log(f"report written to {OUT_JSON.relative_to(ROOT)}")

    log()
    log("phase 2 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
