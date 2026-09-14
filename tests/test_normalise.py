"""Tests for canonical normalisation, the fix for defect D4.

The central guarantee is that the same sentence arriving in Enron style and in fraud
mailbox style normalises to the same string. If that holds, the formatting fingerprint
cannot identify the source file.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.normalise import (
    FORMATTING_FEATURE_NAMES,
    Normaliser,
    formatting_features,
    normalise,
)


def test_the_two_source_styles_collapse_to_the_same_text():
    """The whole point of D4. Same content, two preprocessing conventions."""
    fraud_style = "Contact Mr. Boland at mark.boland@abnamro.com about the transfer."
    enron_style = "contact mr . boland at mark . boland @ abnamro . com about the transfer ."
    assert normalise(fraud_style) == normalise(enron_style)


def test_casing_is_removed():
    assert normalise("URGENT BUSINESS ASSISTANCE") == normalise("urgent business assistance")


def test_mime_artifacts_are_removed():
    out = normalise("Bank 101 Moorgate,=20 London")
    assert "20" not in out.split()
    assert "moorgate" in out


def test_decode_artifacts_are_removed():
    """Both the literal escape from the source and the replacement char we introduced."""
    assert "fffd" not in normalise("The all new My Yahoo! \\ufffd Get yours free")
    assert "fffd" not in normalise("The all new My Yahoo! � Get yours free")


def test_meaningful_symbols_survive_as_words():
    out = normalise("please send $5000 to admin@example.com, a 50% share")
    assert "currencysign" in out
    assert "atsign" in out
    assert "percentsign" in out


def test_numbers_are_masked_as_one_token_not_several():
    """Regression: stripping punctuation first split 25,000,000.00 into four tokens."""
    out = normalise("a transfer of $25,000,000.00 is available")
    assert out.split().count("numbertoken") == 1


def test_number_masking_can_be_disabled():
    assert "numbertoken" not in normalise("call 713 853 5290", mask_numbers=False)
    assert "713" in normalise("call 713 853 5290", mask_numbers=False)


def test_punctuation_is_dropped():
    assert normalise("hello, world! (really)") == "hello world really"


def test_whitespace_is_collapsed():
    assert normalise("too    many\n\n\nspaces\there") == "too many spaces here"


def test_empty_input_is_safe():
    assert normalise("") == ""
    assert normalise("   \n\t ") == ""


# ------------------------------------------------------------------ transformer

def test_transformer_normalises_when_enabled():
    out = Normaliser().fit_transform(["Hello, WORLD!"])
    assert out == ["hello world"]


def test_transformer_is_a_passthrough_when_disabled():
    raw = ["Hello, WORLD!"]
    assert Normaliser(enabled=False).fit_transform(raw) == raw


def test_transformer_works_inside_a_pipeline():
    """It must be usable as a pipeline step so it is applied inside folds, not on disk."""
    pipe = make_pipeline(Normaliser(), TfidfVectorizer())
    out = pipe.fit_transform(["Dear SIR, urgent!", "dear sir urgent"])
    assert out.shape[0] == 2
    # Both documents reduce to identical text, so their vectors must be identical.
    assert np.allclose(out[0].toarray(), out[1].toarray())


# -------------------------------------------------------------- formatting probe

def test_formatting_features_separate_the_two_styles():
    fraud_style = "I am Mr. Mark Boland, Bank Manager of ABN AMRO Bank."
    enron_style = "i am mr . mark boland , bank manager of abn amro bank ."
    a = formatting_features(fraud_style)
    b = formatting_features(enron_style)
    assert a["uppercase_ratio"] > b["uppercase_ratio"]
    assert b["spaced_punct_per_kchar"] > a["spaced_punct_per_kchar"]


def test_formatting_features_never_read_content():
    """Two totally different messages with the same styling must look alike."""
    a = formatting_features("dear sir , i need your help with a transfer .")
    b = formatting_features("gas curve validation meeting is at noon today .")
    assert a["uppercase_ratio"] == b["uppercase_ratio"] == 0.0


def test_formatting_feature_names_are_stable():
    """The probe's feature order is used to build a matrix, so it must not drift."""
    assert FORMATTING_FEATURE_NAMES == (
        "uppercase_ratio",
        "spaced_punct_per_kchar",
        "tight_punct_per_kchar",
        "newline_ratio",
        "mime_artifacts",
        "apostrophe_ratio",
        "tab_ratio",
    )


@pytest.mark.parametrize("text", ["", "a", "\n", "!!!"])
def test_formatting_features_do_not_divide_by_zero(text):
    feats = formatting_features(text)
    assert all(np.isfinite(v) for v in feats.values())
