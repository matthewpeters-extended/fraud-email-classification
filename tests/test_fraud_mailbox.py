"""Tests for the fraud mailbox parser.

The envelope regex is the load bearing piece. Splitting on the literal "From r"
lost two messages in the real corpus, so these tests pin the boundary behaviour.
"""

from __future__ import annotations

import pytest

from src.fraud_mailbox import (
    ENVELOPE_RE,
    extract_body,
    parse_message,
    split_messages,
    strip_html,
    tidy,
)


ENVELOPES_SEEN_IN_CORPUS = [
    "From r  Wed Oct 30 21:41:56 2002",
    "From R@T  Fri May 14 07:49:12 2004",
    "From joykem@centrum.sk Fri Dec 22 13:07:36 2006",
]

NOT_ENVELOPES = [
    "From the desk of Mr James Ngola",
    "From: james@example.com",
    "From my heart I thank you",
    "Fromage is not an envelope",
    "  From r  Wed Oct 30 21:41:56 2002",  # indented, so not a boundary
]


@pytest.mark.parametrize("line", ENVELOPES_SEEN_IN_CORPUS)
def test_envelope_regex_matches_real_forms(line):
    assert ENVELOPE_RE.match(line), f"should be an envelope: {line!r}"


@pytest.mark.parametrize("line", NOT_ENVELOPES)
def test_envelope_regex_rejects_prose(line):
    assert not ENVELOPE_RE.match(line), f"should not be an envelope: {line!r}"


def test_literal_from_r_prefix_would_lose_messages():
    """Documents the bug this parser exists to avoid. See docs/data_defects.md D1."""
    missed = [e for e in ENVELOPES_SEEN_IN_CORPUS if not e.startswith("From r")]
    assert len(missed) == 2
    assert all(ENVELOPE_RE.match(e) for e in missed)


def build_mailbox(*bodies: str) -> str:
    """Assemble a synthetic mailbox.

    Built by explicit concatenation rather than textwrap.dedent, because a multiline
    body with an unindented second line destroys dedent's common prefix and would
    leave the envelope lines indented.
    """
    parts = []
    for i, body in enumerate(bodies, start=1):
        parts.append(
            "\n".join(
                [
                    f"From r  Wed Oct 30 21:41:5{i} 2002",
                    f"From: sender{i}@example.com",
                    "To: victim@example.com",
                    f"Subject: message {i}",
                    'Content-Type: text/plain; charset="us-ascii"',
                    "",
                    body,
                    "",
                ]
            )
        )
    return "".join(parts)


def test_build_mailbox_helper_produces_unindented_envelopes():
    """Guards the test harness itself. A previous version indented the envelopes."""
    mailbox = build_mailbox("a body with several words in it")
    assert mailbox.startswith("From r  Wed")
    assert "\n    From r" not in mailbox


def test_split_counts_messages():
    mailbox = build_mailbox("first body here", "second body here", "third body here")
    assert len(split_messages(mailbox)) == 3


def test_split_does_not_break_on_from_inside_a_body():
    mailbox = build_mailbox("From the desk of a person\nmore text follows here")
    blocks = split_messages(mailbox)
    assert len(blocks) == 1
    assert "From the desk of a person" in blocks[0]


def test_split_returns_empty_for_text_with_no_envelope():
    assert split_messages("just some text with no mailbox structure") == []


def test_parse_message_extracts_headers_and_body():
    block = split_messages(build_mailbox("hello there this is the body text"))[0]
    msg = parse_message(block, index=1)
    assert msg.subject == "message 1"
    assert msg.from_addr == "sender1@example.com"
    assert "hello there this is the body text" in msg.body
    assert msg.ok


def test_envelope_line_is_not_part_of_the_body():
    block = split_messages(build_mailbox("a body with at least five words"))[0]
    msg = parse_message(block, index=1)
    assert "From r" not in msg.body


def test_quoted_printable_is_decoded():
    """Defect D4: raw '=20' artifacts must not survive into the parsed body."""
    raw = (
        "From r  Wed Oct 30 21:41:56 2002\n"
        "From: a@example.com\n"
        "Subject: test\n"
        "Content-Type: text/plain; charset=iso-8859-1\n"
        "Content-Transfer-Encoding: quoted-printable\n"
        "\n"
        "Bank Manager of ABN AMRO Bank 101 Moorgate,=20\n"
        "London EC2M 6SB with urgent business\n"
    )
    msg = parse_message(split_messages(raw)[0], index=1)
    assert "=20" not in msg.body
    assert "Moorgate," in msg.body


def test_short_body_is_not_usable():
    msg = parse_message(split_messages(build_mailbox("too short"))[0], index=1)
    assert not msg.ok


def test_empty_body_is_not_usable():
    raw = (
        "From r  Fri Sep  5 06:15:36 2003\n"
        "From: esamakingdom@fsmail.net\n"
        "Subject: URGENT ASSISTANCE\n"
        "Content-Type: text/plain; charset=iso-8859-1\n"
        "\n"
        "\n"
    )
    msg = parse_message(split_messages(raw)[0], index=1)
    assert msg.body_chars == 0
    assert not msg.ok


def test_html_only_message_falls_back_to_stripped_html():
    raw = (
        "From r  Wed Oct 30 21:41:56 2002\n"
        "From: a@example.com\n"
        "Subject: test\n"
        'Content-Type: text/html; charset="us-ascii"\n'
        "\n"
        "<html><body><div>Your earliest response will be appreciated</div>"
        "<script>var x=1;</script></body></html>\n"
    )
    msg = parse_message(split_messages(raw)[0], index=1)
    assert msg.was_html
    assert "Your earliest response will be appreciated" in msg.body
    assert "<div>" not in msg.body
    assert "var x=1" not in msg.body


def test_unknown_charset_is_noted_not_fatal():
    raw = (
        "From r  Wed Oct 30 21:41:56 2002\n"
        "From: a@example.com\n"
        "Subject: test\n"
        "Content-Type: text/plain; charset=not-a-real-charset\n"
        "\n"
        "this body should still be recovered intact\n"
    )
    msg = parse_message(split_messages(raw)[0], index=1)
    assert "this body should still be recovered intact" in msg.body
    assert any(n.startswith("unknown_charset") for n in msg.notes)


def test_strip_html_unescapes_entities():
    assert "Regard, Mr" in strip_html("<p>Regard,&nbsp;Mr</p>").replace("\xa0", " ")


def test_tidy_preserves_case_and_punctuation():
    """Casing is a modelling decision for phase 6, not a parsing decision."""
    out = tidy("  Mr. Mark Boland   of  ABN AMRO.  ")
    assert out == "Mr. Mark Boland of ABN AMRO."


def test_tidy_collapses_runaway_blank_lines():
    assert tidy("a line\n\n\n\n\nanother line") == "a line\n\nanother line"
