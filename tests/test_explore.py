"""Tests for the exploratory analysis helpers."""

from __future__ import annotations

import math

import pytest

from src.explore import (
    coverage,
    discriminative_terms,
    document_frequencies,
    jaccard,
    length_profile,
    log_odds_ratio,
    percentile,
    type_token_ratio,
    vocabulary,
)


# ------------------------------------------------------------------ percentiles

@pytest.mark.parametrize(
    "q,expected", [(0.0, 1), (0.25, 2), (0.5, 3), (0.75, 4), (1.0, 5)]
)
def test_percentile_on_a_simple_series(q, expected):
    assert percentile([1, 2, 3, 4, 5], q) == expected


def test_percentile_on_empty_input():
    assert percentile([], 0.5) == 0


def test_length_profile_orders_its_quantiles():
    p = length_profile("FRAUD", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert p.minimum <= p.p25 <= p.median <= p.p75 <= p.p90 <= p.maximum
    assert p.count == 10
    assert p.mean == pytest.approx(55.0)


def test_length_profile_serialises():
    p = length_profile("SPAM", [5, 10, 15])
    assert set(p.as_dict()) == {"count", "min", "p25", "median", "p75", "p90", "max", "mean"}


# ------------------------------------------------------------------ vocabulary

def test_document_frequency_counts_presence_not_repetition():
    """A word repeated inside one document still counts once."""
    df = document_frequencies([["fraud", "fraud", "fraud"], ["fraud", "spam"]])
    assert df["fraud"] == 2
    assert df["spam"] == 1


def test_vocabulary_applies_the_minimum_document_frequency():
    docs = [["a", "b"], ["a", "c"], ["a", "d"]]
    assert vocabulary(docs, min_df=3) == {"a"}
    assert vocabulary(docs, min_df=1) == {"a", "b", "c", "d"}


def test_jaccard_bounds():
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 0.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_coverage_is_asymmetric():
    """The point of reporting coverage alongside Jaccard."""
    small, large = {"a", "b"}, {"a", "b", "c", "d"}
    assert coverage(small, large) == 1.0
    assert coverage(large, small) == 0.5
    assert jaccard(small, large) == 0.5


def test_coverage_on_empty_input():
    assert coverage(set(), {"a"}) == 0.0


def test_type_token_ratio_detects_repetition():
    repetitive = [["a", "a", "a", "a"]]
    varied = [["a", "b", "c", "d"]]
    assert type_token_ratio(repetitive) == 0.25
    assert type_token_ratio(varied) == 1.0


def test_type_token_ratio_on_empty_input():
    assert type_token_ratio([]) == 0.0


# ----------------------------------------------------------- log odds ratio

def test_log_odds_is_positive_for_an_in_class_term():
    assert log_odds_ratio(200, 720, 2, 1440) > 4


def test_log_odds_is_near_zero_for_a_neutral_term():
    """Equal rates inside and outside the class must not score as discriminative."""
    assert abs(log_odds_ratio(100, 720, 200, 1440)) < 0.01


def test_log_odds_is_negative_for_an_out_of_class_term():
    # Computes to about -3.84: 2 of 720 in class against 200 of 1440 outside it.
    assert log_odds_ratio(2, 720, 200, 1440) < -3.5


def test_log_odds_is_antisymmetric_under_swapping_the_classes():
    forward = log_odds_ratio(200, 720, 2, 1440)
    backward = log_odds_ratio(2, 1440, 200, 720)
    assert forward == pytest.approx(-backward, abs=1e-9)


def test_log_odds_survives_zero_counts():
    """The continuity correction must keep this finite."""
    assert math.isfinite(log_odds_ratio(0, 720, 0, 1440))
    assert math.isfinite(log_odds_ratio(720, 720, 0, 1440))


def test_log_odds_prefers_the_purer_term_at_equal_frequency():
    pure = log_odds_ratio(50, 720, 0, 1440)
    mixed = log_odds_ratio(50, 720, 50, 1440)
    assert pure > mixed


# --------------------------------------------------- discriminative terms

def test_discriminative_terms_finds_the_planted_signal():
    by_class = {
        "FRAUD": [["beneficiary", "funds"] for _ in range(30)],
        "SPAM": [["viagra", "offer"] for _ in range(30)],
        "NORMAL": [["meeting", "agenda"] for _ in range(30)],
    }
    terms = discriminative_terms(by_class, min_df=20, top_n=2)
    assert {t for t, *_ in terms["FRAUD"]} == {"beneficiary", "funds"}
    assert {t for t, *_ in terms["SPAM"]} == {"viagra", "offer"}
    assert {t for t, *_ in terms["NORMAL"]} == {"meeting", "agenda"}


def test_discriminative_terms_ignores_a_term_shared_by_every_class():
    by_class = {
        "FRAUD": [["shared", "beneficiary"] for _ in range(30)],
        "SPAM": [["shared", "viagra"] for _ in range(30)],
        "NORMAL": [["shared", "meeting"] for _ in range(30)],
    }
    terms = discriminative_terms(by_class, min_df=20, top_n=1)
    for label in by_class:
        assert terms[label][0][0] != "shared"


def test_discriminative_terms_respects_min_df():
    by_class = {
        "FRAUD": [["rare"]] + [["common", "beneficiary"] for _ in range(30)],
        "SPAM": [["viagra"] for _ in range(30)],
        "NORMAL": [["meeting"] for _ in range(30)],
    }
    terms = discriminative_terms(by_class, min_df=20, top_n=10)
    assert "rare" not in {t for t, *_ in terms["FRAUD"]}


def test_discriminative_terms_reports_purity():
    by_class = {
        "FRAUD": [["beneficiary"] for _ in range(30)],
        "SPAM": [["viagra"] for _ in range(30)],
        "NORMAL": [["meeting"] for _ in range(30)],
    }
    terms = discriminative_terms(by_class, min_df=20, top_n=1)
    _, _, df_in, purity = terms["FRAUD"][0]
    assert df_in == 30
    assert purity == pytest.approx(1.0)
