"""Tests for the provenance marker blocklist, the fix for defect D3."""

from __future__ import annotations

from src.markers import (
    BLOCKLIST,
    BLOCKLIST_GROUPS,
    KEPT_DESPITE_SKEW,
    keep_only_markers,
    strip_markers,
)


def test_blocklist_is_the_union_of_its_groups():
    union = set()
    for tokens in BLOCKLIST_GROUPS.values():
        union |= set(tokens)
    assert union == set(BLOCKLIST)


def test_blocklist_and_kept_lists_do_not_overlap():
    """A token cannot be both blocked as provenance and kept as content."""
    assert not (set(BLOCKLIST) & set(KEPT_DESPITE_SKEW))


def test_blocklist_covers_the_measured_leaks():
    """The tokens the phase 4 skew analysis flagged as provenance."""
    for token in ("enron", "kaminski", "vince", "hou", "ect", "houston", "cc", "pm"):
        assert token in BLOCKLIST, token


def test_blocklist_covers_the_spam_collection_tools():
    """The leak is not one sided. Spam leaks how it was gathered."""
    assert "spamassassin" in BLOCKLIST
    assert "projecthoneypot" in BLOCKLIST


def test_genuine_content_signals_are_not_blocked():
    """These are just as class skewed but describe the message, so they stay."""
    for token in ("beneficiary", "deceased", "viagra", "abidjan", "mortgage", "derivatives"):
        assert token not in BLOCKLIST, token


def test_strip_markers_removes_only_blocked_tokens():
    text = "vince kaminski at enron discussed derivatives valuation in houston"
    out = strip_markers(text)
    assert "enron" not in out and "vince" not in out and "houston" not in out
    assert "derivatives" in out and "valuation" in out and "discussed" in out


def test_keep_only_markers_is_the_complement():
    text = "vince kaminski at enron discussed derivatives valuation in houston"
    kept = keep_only_markers(text).split()
    stripped = strip_markers(text).split()
    assert set(kept) == {"vince", "kaminski", "enron", "houston"}
    assert not (set(kept) & set(stripped))


def test_a_document_with_no_markers_strips_to_itself():
    text = "dear sir i am the beneficiary of a dormant account"
    assert strip_markers(text).split() == text.split()
    assert keep_only_markers(text) == ""


def test_marker_matching_is_whole_token_only():
    """Substrings must not match. 'enronxgate' is blocked, 'enroute' is not."""
    assert strip_markers("enroute to the meeting") == "enroute to the meeting"
    assert "enronxgate" not in strip_markers("enronxgate mail system")


def test_strip_and_keep_partition_the_text():
    text = "enron and vince discussed a dormant beneficiary account in houston"
    total = len(text.split())
    assert len(strip_markers(text).split()) + len(keep_only_markers(text).split()) == total


def test_no_blocked_token_is_a_common_english_word():
    """Regression. "am" was blocked for AM and PM timestamps, but it is also the verb
    that opens almost every advance fee email, so blocking it deleted content. It had
    never appeared in the skew evidence; it was added by assumption."""
    from src.markers import COMMON_ENGLISH_WORDS

    collisions = set(BLOCKLIST) & COMMON_ENGLISH_WORDS
    assert not collisions, f"these block real content: {sorted(collisions)}"


def test_the_verb_am_survives_stripping():
    text = "i am the beneficiary of a dormant account"
    assert strip_markers(text).split() == text.split()
