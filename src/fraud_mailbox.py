"""Parse the CLAIR advance fee fraud corpus into individual messages.

`fradulent_emails.txt` is a single concatenated mailbox, not a CSV. Messages are
separated by mbox envelope lines. Header blocks are then parsed with the standard
library email module, which decodes quoted printable and base64 payloads properly,
so MIME artifacts such as "=20" are resolved rather than left in the text.

Splitting on the literal prefix "From r" loses messages. Two envelope lines in this
corpus do not carry that prefix. See docs/data_defects.md D1.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from email import message_from_string, policy
from email.message import Message
from typing import Iterator

# An mbox envelope line: "From <sender> <Day> <Mon> <dd> <hh:mm:ss> <yyyy>".
# Requiring the weekday and month keeps ordinary body lines that begin with the word
# "From" from being mistaken for message boundaries.
DAYS = "Mon|Tue|Wed|Thu|Fri|Sat|Sun"
MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
ENVELOPE_RE = re.compile(
    rf"^From\s+\S*\s*(?:{DAYS})\s+(?:{MONTHS})\s+\d{{1,2}}\s+\d{{1,2}}:\d{{2}}:\d{{2}}\s+\d{{4}}",
    re.MULTILINE,
)

TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
WS_RE = re.compile(r"[ \t]+")
BLANKS_RE = re.compile(r"\n{3,}")


@dataclass
class ParsedMessage:
    """One fraud email, reduced to the fields this project needs."""

    index: int
    subject: str = ""
    from_addr: str = ""
    date: str = ""
    content_type: str = ""
    charset: str = ""
    was_multipart: bool = False
    was_html: bool = False
    body: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def body_chars(self) -> int:
        return len(self.body)

    @property
    def ok(self) -> bool:
        """A message is usable if it has a body with some actual words in it."""
        return len(self.body.split()) >= 5


def read_corpus(path) -> str:
    """Decode the mailbox. The corpus mixes encodings, so replacement is deliberate."""
    return path.read_bytes().decode("utf-8", errors="replace")


def split_messages(raw: str) -> list[str]:
    """Cut the mailbox into raw message blocks on envelope lines."""
    starts = [m.start() for m in ENVELOPE_RE.finditer(raw)]
    if not starts:
        return []
    bounds = starts + [len(raw)]
    return [raw[bounds[i] : bounds[i + 1]] for i in range(len(starts))]


def strip_html(text: str) -> str:
    text = SCRIPT_STYLE_RE.sub(" ", text)
    text = re.sub(r"<br\s*/?>|</p>|</div>|</tr>", "\n", text, flags=re.IGNORECASE)
    text = TAG_RE.sub(" ", text)
    return html.unescape(text)


def tidy(text: str) -> str:
    """Collapse runaway whitespace without touching casing or punctuation.

    Normalisation of casing and punctuation is a modelling decision and belongs in the
    preprocessing pipeline at phase 6, not here. This function only makes the stored
    text readable.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return BLANKS_RE.sub("\n\n", text).strip()


def _payload_to_text(part: Message, notes: list[str]) -> str:
    """Decode one MIME part to text, falling back progressively."""
    charset = part.get_content_charset() or "utf-8"
    payload = part.get_payload(decode=True)
    if payload is None:
        value = part.get_payload()
        if isinstance(value, str):
            notes.append("undecoded_payload")
            return value
        return ""
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        notes.append(f"unknown_charset:{charset}")
        return payload.decode("utf-8", errors="replace")


def extract_body(msg: Message, notes: list[str]) -> tuple[str, bool, bool]:
    """Return the best available text body, plus multipart and html flags.

    Prefers text/plain. Falls back to text/html with tags stripped, since a handful of
    these messages are html only and discarding them would bias the corpus toward the
    older plain text scams.
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []
    multipart = msg.is_multipart()

    for part in msg.walk() if multipart else [msg]:
        if part.get_content_maintype() == "multipart":
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain":
            plain_parts.append(_payload_to_text(part, notes))
        elif ctype == "text/html":
            html_parts.append(_payload_to_text(part, notes))

    if plain_parts:
        return tidy("\n".join(plain_parts)), multipart, False
    if html_parts:
        notes.append("html_only")
        return tidy(strip_html("\n".join(html_parts))), multipart, True

    notes.append("no_text_part")
    return "", multipart, False


def parse_message(block: str, index: int) -> ParsedMessage:
    """Parse one raw mailbox block into a ParsedMessage."""
    notes: list[str] = []

    # Drop the envelope line. It is a mailbox artifact, not an email header.
    newline = block.find("\n")
    body_block = block[newline + 1 :] if newline != -1 else ""

    try:
        msg = message_from_string(body_block, policy=policy.compat32)
    except Exception as exc:  # a malformed header block should not kill the run
        notes.append(f"header_parse_failed:{type(exc).__name__}")
        return ParsedMessage(index=index, body="", notes=notes)

    body, multipart, was_html = extract_body(msg, notes)

    return ParsedMessage(
        index=index,
        subject=tidy(str(msg.get("Subject", "") or "")),
        from_addr=tidy(str(msg.get("From", "") or "")),
        date=tidy(str(msg.get("Date", "") or "")),
        content_type=msg.get_content_type(),
        charset=msg.get_content_charset() or "",
        was_multipart=multipart,
        was_html=was_html,
        body=body,
        notes=notes,
    )


def parse_corpus(raw: str) -> Iterator[ParsedMessage]:
    for i, block in enumerate(split_messages(raw), start=1):
        yield parse_message(block, i)
