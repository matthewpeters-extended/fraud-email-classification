"""Exploratory analysis of the training corpus.

Phase 4 established that the content, not the provenance shortcuts, carries the signal.
This module characterises that content: how long the documents are, how much vocabulary
the classes share, and which terms actually separate them once normalisation and marker
stripping have been applied.

The discriminative term ranking doubles as a second sweep for leakage. Phase 4 curated a
blocklist from raw token skew; running the same question against the cleaned
representation asks whether anything provenance shaped survived.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

CLASSES = ("FRAUD", "SPAM", "NORMAL")


# --------------------------------------------------------------- length profile

@dataclass
class LengthProfile:
    label: str
    count: int
    minimum: int
    p25: int
    median: int
    p75: int
    p90: int
    maximum: int
    mean: float

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "min": self.minimum,
            "p25": self.p25,
            "median": self.median,
            "p75": self.p75,
            "p90": self.p90,
            "max": self.maximum,
            "mean": round(self.mean, 1),
        }


def percentile(sorted_values: list[int], q: float) -> int:
    """Nearest rank percentile. No numpy dependency so this stays easy to test."""
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def length_profile(label: str, word_counts: list[int]) -> LengthProfile:
    values = sorted(word_counts)
    return LengthProfile(
        label=label,
        count=len(values),
        minimum=values[0] if values else 0,
        p25=percentile(values, 0.25),
        median=percentile(values, 0.50),
        p75=percentile(values, 0.75),
        p90=percentile(values, 0.90),
        maximum=values[-1] if values else 0,
        mean=sum(values) / len(values) if values else 0.0,
    )


# ----------------------------------------------------------------- vocabulary

def document_frequencies(tokenised: list[list[str]]) -> Counter:
    """How many documents each token appears in. Presence, not raw count."""
    df: Counter = Counter()
    for tokens in tokenised:
        df.update(set(tokens))
    return df


def vocabulary(tokenised: list[list[str]], min_df: int = 5) -> set[str]:
    df = document_frequencies(tokenised)
    return {token for token, n in df.items() if n >= min_df}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def coverage(a: set[str], b: set[str]) -> float:
    """Share of a's vocabulary that also appears in b. Asymmetric on purpose.

    Jaccard alone hides asymmetry. A small class whose vocabulary is wholly contained in
    a larger one scores a low Jaccard while being entirely covered, and that distinction
    matters for the Fraud against Spam boundary.
    """
    if not a:
        return 0.0
    return len(a & b) / len(a)


def type_token_ratio(tokenised: list[list[str]]) -> float:
    """Vocabulary richness. Templated text repeats itself and scores low."""
    types: set[str] = set()
    total = 0
    for tokens in tokenised:
        types.update(tokens)
        total += len(tokens)
    return len(types) / total if total else 0.0


# ------------------------------------------------------- discriminative terms

def log_odds_ratio(
    df_in_class: int, n_in_class: int, df_out_class: int, n_out_class: int, prior: float = 0.5
) -> float:
    """Document level log odds ratio with a continuity correction.

    Positive means the term is more likely to appear in a document of this class than
    outside it. Working at document level rather than token level stops a single long
    document that repeats a word from dominating the ranking.
    """
    a = df_in_class + prior
    b = n_in_class - df_in_class + prior
    c = df_out_class + prior
    d = n_out_class - df_out_class + prior
    return math.log(a / b) - math.log(c / d)


def discriminative_terms(
    tokenised_by_class: dict[str, list[list[str]]],
    min_df: int = 20,
    top_n: int = 25,
) -> dict[str, list[tuple[str, float, int, float]]]:
    """Rank terms by how strongly their presence indicates each class.

    Returns per class a list of (term, log odds ratio, document frequency in class,
    share of all documents containing the term that are in this class).
    """
    df_by_class = {
        label: document_frequencies(docs) for label, docs in tokenised_by_class.items()
    }
    n_by_class = {label: len(docs) for label, docs in tokenised_by_class.items()}
    total_df: Counter = Counter()
    for df in df_by_class.values():
        total_df.update(df)

    results: dict[str, list[tuple[str, float, int, float]]] = {}
    for label in tokenised_by_class:
        in_df = df_by_class[label]
        n_in = n_by_class[label]
        n_out = sum(n for other, n in n_by_class.items() if other != label)
        scored = []
        for term, overall in total_df.items():
            if overall < min_df:
                continue
            df_in = in_df.get(term, 0)
            df_out = overall - df_in
            lor = log_odds_ratio(df_in, n_in, df_out, n_out)
            purity = df_in / overall if overall else 0.0
            scored.append((term, lor, df_in, purity))
        scored.sort(key=lambda row: -row[1])
        results[label] = scored[:top_n]
    return results
