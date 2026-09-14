"""Tests for the preprocessing pipeline.

The lemmatiser carries most of the risk here. WordNet's noun lemmatiser strips any
trailing "s" it reads as a plural, which produced several non words before the ordering
and the length guard were fixed.
"""

from __future__ import annotations

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.preprocess import (
    CHOSEN_CONFIG,
    MIN_NOUN_LEMMA_LENGTH,
    Normalise,
    ReduceWordForms,
    RemoveStopwords,
    SIGNAL_STOPWORDS,
    StripMarkers,
    chosen_text_steps,
    default_stopwords,
    lemmatise_token,
    stem_token,
    text_steps,
)


# ------------------------------------------------------------------ lemmatiser

@pytest.mark.parametrize(
    "token,expected",
    [
        # verb step
        ("died", "die"),
        ("transferred", "transfer"),
        ("inherited", "inherit"),
        ("funds", "fund"),
        ("proceeds", "proceed"),
        # noun step
        ("beneficiaries", "beneficiary"),
        ("mailings", "mailing"),
        ("modalities", "modality"),
        ("widows", "widow"),
        ("banks", "bank"),
    ],
)
def test_lemmatiser_handles_both_parts_of_speech(token, expected):
    assert lemmatise_token(token) == expected


@pytest.mark.parametrize("token", ["was", "has", "us", "as", "mrs"])
def test_lemmatiser_does_not_produce_non_words(token):
    """Regression. Noun first lemmatisation gave was->wa, has->ha, us->u, as->a, mrs->mr.

    The first four are not words. The last conflated a top fraud term, "mrs" at 126
    training documents and 96 percent purity, with its masculine form.
    """
    result = lemmatise_token(token)
    assert result in {token, "be", "have"}, f"{token} became {result}"
    assert len(result) >= 2


def test_noun_lemma_length_guard_is_the_documented_value():
    assert MIN_NOUN_LEMMA_LENGTH == 3


def test_lemmatiser_is_idempotent():
    for token in ("beneficiary", "fund", "die", "mailing"):
        assert lemmatise_token(lemmatise_token(token)) == lemmatise_token(token)


def test_stemmer_is_more_aggressive_than_the_lemmatiser():
    """Documents the tradeoff that decided phase 6: stems are not readable words."""
    assert stem_token("beneficiaries") == "beneficiari"
    assert lemmatise_token("beneficiaries") == "beneficiary"


# ------------------------------------------------------------------- stopwords

def test_default_stopwords_excludes_the_signal_pronouns():
    assert not (default_stopwords() & SIGNAL_STOPWORDS)


def test_remove_stopwords_defaults_to_the_full_list():
    """Phase 6 measured the signal exemption as no better, so it is off by default."""
    assert RemoveStopwords().keep_signal is False
    out = RemoveStopwords(enabled=True).fit_transform(["i am the beneficiary of the fund"])
    assert "i" not in out[0].split()
    assert "beneficiary" in out[0].split()


def test_remove_stopwords_can_hold_back_the_signal_pronouns():
    out = RemoveStopwords(enabled=True, keep_signal=True).fit_transform(
        ["i am the beneficiary of the fund"]
    )
    assert "i" in out[0].split() and "am" in out[0].split()
    assert "the" not in out[0].split()


def test_remove_stopwords_is_a_passthrough_when_disabled():
    text = ["i am the beneficiary of the fund"]
    assert RemoveStopwords(enabled=False).fit_transform(text) == text


# --------------------------------------------------------------- word forms

def test_reduce_word_forms_none_is_a_passthrough():
    text = ["beneficiaries inherited funds"]
    assert ReduceWordForms(method="none").fit_transform(text) == text


def test_reduce_word_forms_lemmatise():
    out = ReduceWordForms(method="lemmatise").fit_transform(["beneficiaries inherited funds"])
    assert out == ["beneficiary inherit fund"]


def test_reduce_word_forms_stem():
    out = ReduceWordForms(method="stem").fit_transform(["beneficiaries inherited funds"])
    assert out == ["beneficiari inherit fund"]


def test_reduce_word_forms_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        ReduceWordForms(method="porter2").fit_transform(["anything at all"])


# ----------------------------------------------------------------- pipeline

def test_text_steps_are_named_and_ordered():
    names = [name for name, _ in text_steps()]
    assert names == ["normalise", "strip_markers", "stopwords", "word_forms"]


def test_normalisation_runs_before_marker_stripping():
    """Marker matching is defined on lowercased alphanumeric tokens, so order matters."""
    out = Pipeline(text_steps()).fit_transform(["Vince Kaminski at ENRON discussed funds"])
    assert "enron" not in out[0] and "vince" not in out[0]
    assert "funds" in out[0] or "fund" in out[0]


def test_full_chosen_pipeline_end_to_end():
    out = Pipeline(chosen_text_steps()).fit_transform(
        ["I am Mrs Fatima, the sole beneficiary of $25,000,000.00 held in a dormant "
         "Enron account in Houston"]
    )
    tokens = out[0].split()
    assert "enron" not in tokens and "houston" not in tokens   # markers gone
    assert "the" not in tokens                                  # stopwords gone
    assert "beneficiary" in tokens and "dormant" in tokens      # content kept
    assert "currencysign" in tokens and "numbertoken" in tokens  # symbols kept as words
    assert "mrs" in tokens                                       # not conflated with mr


def test_chosen_config_matches_the_phase_6_decision():
    assert CHOSEN_CONFIG == {
        "normalise_text": True,
        "strip_provenance": True,
        "remove_stopwords": True,
        "reduce_forms": "lemmatise",
    }


def test_chosen_pipeline_works_with_a_vectoriser():
    pipe = Pipeline(chosen_text_steps() + [("tfidf", TfidfVectorizer(min_df=1))])
    matrix = pipe.fit_transform(["the beneficiary inherited funds", "viagra shops online"])
    assert matrix.shape[0] == 2 and matrix.shape[1] > 0


def test_steps_are_stateless_so_fitting_changes_nothing():
    """These transformers cannot leak: fitting on other data must not alter output."""
    pipe = Pipeline(chosen_text_steps())
    first = pipe.fit_transform(["the beneficiary inherited funds"])
    pipe.fit(["completely different training text about meetings and agendas"])
    assert pipe.transform(["the beneficiary inherited funds"]) == first


def test_disabled_steps_leave_text_untouched():
    steps = text_steps(
        normalise_text=False, strip_provenance=False, remove_stopwords=False,
        reduce_forms="none",
    )
    text = ["Vince at ENRON, $5,000.00!"]
    assert Pipeline(steps).fit_transform(text) == text


def test_normalise_and_strip_markers_are_individually_switchable():
    assert Normalise(enabled=False).fit_transform(["ABC"]) == ["ABC"]
    assert StripMarkers(enabled=False).fit_transform(["enron"]) == ["enron"]
    assert StripMarkers(enabled=True).fit_transform(["enron funds"]) == ["funds"]
