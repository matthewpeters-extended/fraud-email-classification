"""Tests for corpus construction.

The load bearing guarantees are that no duplicate straddles the train and test split,
and that clustering uses complete linkage so shared boilerplate cannot chain unrelated
documents into one cluster.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.corpus import (
    NEAR_DUPLICATE_THRESHOLD,
    Document,
    UnionFind,
    _complete_linkage_subclusters,
    cluster_near_duplicates,
    content_hash,
    drop_label_contradictions,
    drop_short,
    find_cross_class_clusters,
    normalise_for_matching,
    sample_one_per_cluster,
    stratified_split,
    strip_subject_prefix,
)


# ---------------------------------------------------------------- text handling

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Subject: hello there", "hello there"),
        ("subject: hello there", "hello there"),
        ("  Subject:   hello there  ", "hello there"),
        ("SUBJECT:hello there", "hello there"),
        ("no prefix at all", "no prefix at all"),
        ("body mentions Subject: midway", "body mentions Subject: midway"),
    ],
)
def test_strip_subject_prefix(raw, expected):
    assert strip_subject_prefix(raw) == expected


def test_normalise_for_matching_ignores_case_and_punctuation():
    a = normalise_for_matching("Dear Sir, I am Mr. Smith!")
    b = normalise_for_matching("dear sir i am mr smith")
    assert a == b


def test_content_hash_agrees_with_normalisation():
    assert content_hash("Dear Sir, hello!") == content_hash("dear sir hello")
    assert content_hash("one thing") != content_hash("another thing")


def test_drop_short_removes_below_threshold():
    docs = [
        Document(text="one two three four five", label="SPAM", source="s", source_row=1),
        Document(text="too short", label="SPAM", source="s", source_row=2),
    ]
    kept, dropped = drop_short(docs, min_words=5)
    assert len(kept) == 1 and dropped == 1


# ---------------------------------------------------------------- union find

def test_union_find_merges_transitively():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(3)


# ------------------------------------------------- complete linkage behaviour

def test_complete_linkage_refuses_to_chain():
    """The defect this replaced. A and C must not join through B.

    Single linkage put 232 fraud emails in one cluster whose minimum pairwise cosine
    was 0.563, because shared boilerplate linked them pair by pair.
    """
    sim = np.array(
        [
            [1.00, 0.90, 0.40],
            [0.90, 1.00, 0.90],
            [0.40, 0.90, 1.00],
        ]
    )
    labels = _complete_linkage_subclusters(sim, threshold=0.85)
    assert labels[0] != labels[2], "A and C are dissimilar and must not share a cluster"
    assert len(set(labels)) > 1


def test_complete_linkage_keeps_a_genuinely_tight_group_together():
    sim = np.full((4, 4), 0.95)
    np.fill_diagonal(sim, 1.0)
    labels = _complete_linkage_subclusters(sim, threshold=0.85)
    assert len(set(labels)) == 1


def test_complete_linkage_separates_everything_below_threshold():
    sim = np.array([[1.0, 0.2], [0.2, 1.0]])
    labels = _complete_linkage_subclusters(sim, threshold=0.85)
    assert labels[0] != labels[1]


def test_complete_linkage_handles_single_member():
    assert len(_complete_linkage_subclusters(np.array([[1.0]]), threshold=0.85)) == 1


# ---------------------------------------------------------------- clustering

def test_identical_texts_cluster_together():
    text = "dear sir i am a barrister acting for a deceased client with funds to transfer"
    ids = cluster_near_duplicates([text, text, "wholly unrelated content about smile curves"])
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]


def test_clustering_is_deterministic():
    texts = [
        "dear sir i am a barrister acting for a deceased client with funds",
        "dear sir i am a barrister acting for a deceased client with funds",
        "quarterly gas curve validation meeting notes for the trading desk",
        "quarterly gas curve validation meeting notes for the trading floor",
    ]
    assert (cluster_near_duplicates(texts) == cluster_near_duplicates(texts)).all()


def test_every_document_receives_a_cluster():
    texts = [f"message number {i} with some filler words to pad it out" for i in range(12)]
    ids = cluster_near_duplicates(texts)
    assert len(ids) == 12
    assert (ids >= 0).all()


# ------------------------------------------------------- label contradictions

def make(label: str, cluster: int, row: int, text: str = "some words in here at all") -> Document:
    d = Document(text=text, label=label, source="s", source_row=row)
    d.cluster = cluster
    return d


def test_find_cross_class_clusters_spots_disagreement():
    docs = [make("FRAUD", 1, 1), make("SPAM", 1, 2), make("NORMAL", 2, 3)]
    found = find_cross_class_clusters(docs)
    assert set(found) == {1}
    assert found[1] == {"FRAUD", "SPAM"}


def test_drop_label_contradictions_removes_the_whole_cluster():
    docs = [make("FRAUD", 1, 1), make("SPAM", 1, 2), make("NORMAL", 2, 3)]
    kept, found = drop_label_contradictions(docs)
    assert [d.cluster for d in kept] == [2]
    assert len(found) == 1


def test_drop_label_contradictions_is_a_noop_when_clean():
    docs = [make("FRAUD", 1, 1), make("SPAM", 2, 2)]
    kept, found = drop_label_contradictions(docs)
    assert len(kept) == 2 and found == {}


# ---------------------------------------------------------------- sampling

def build_pool(per_class: int) -> list[Document]:
    docs = []
    cluster = 0
    for label in ("FRAUD", "SPAM", "NORMAL"):
        for i in range(per_class):
            docs.append(make(label, cluster, i, text=f"{label} body {i} with padding words"))
            cluster += 1
    return docs


def test_sample_draws_one_per_cluster():
    chosen, pools = sample_one_per_cluster(build_pool(50), per_class=20, seed=1)
    assert len(chosen) == 60
    assert len({d.cluster for d in chosen}) == 60
    assert pools == {"FRAUD": 50, "SPAM": 50, "NORMAL": 50}


def test_sample_raises_when_pool_too_small():
    with pytest.raises(ValueError, match="cannot sample"):
        sample_one_per_cluster(build_pool(5), per_class=10, seed=1)


def test_sample_is_reproducible_under_a_seed():
    a = sample_one_per_cluster(build_pool(50), per_class=20, seed=7)[0]
    b = sample_one_per_cluster(build_pool(50), per_class=20, seed=7)[0]
    assert [d.source_row for d in a] == [d.source_row for d in b]


def test_sample_changes_with_the_seed():
    a = sample_one_per_cluster(build_pool(50), per_class=20, seed=7)[0]
    b = sample_one_per_cluster(build_pool(50), per_class=20, seed=8)[0]
    assert [d.source_row for d in a] != [d.source_row for d in b]


def test_sample_picks_the_longest_member_of_a_cluster():
    docs = [
        make("FRAUD", 0, 1, text="short version only here now"),
        make("FRAUD", 0, 2, text="much longer version of the very same templated scam email body"),
        make("SPAM", 1, 3),
        make("NORMAL", 2, 4),
    ]
    chosen, _ = sample_one_per_cluster(docs, per_class=1, seed=1)
    fraud = [d for d in chosen if d.label == "FRAUD"][0]
    assert fraud.source_row == 2


# ---------------------------------------------------------------- splitting

def test_split_is_stratified_and_disjoint_by_cluster():
    chosen, _ = sample_one_per_cluster(build_pool(100), per_class=100, seed=3)
    train, test = stratified_split(chosen, test_fraction=0.2, seed=3)
    assert len(train) == 240 and len(test) == 60
    for label in ("FRAUD", "SPAM", "NORMAL"):
        assert sum(1 for d in train if d.label == label) == 80
        assert sum(1 for d in test if d.label == label) == 20
    assert not ({d.cluster for d in train} & {d.cluster for d in test})


def test_split_is_reproducible_under_a_seed():
    chosen, _ = sample_one_per_cluster(build_pool(100), per_class=100, seed=3)
    t1, _ = stratified_split(chosen, 0.2, seed=5)
    t2, _ = stratified_split(chosen, 0.2, seed=5)
    assert [d.source_row for d in t1] == [d.source_row for d in t2]


def test_threshold_is_the_documented_value():
    """Pinned because every duplication figure in the write up depends on it."""
    assert NEAR_DUPLICATE_THRESHOLD == 0.85


def test_clustering_does_not_collapse_on_a_tiny_input():
    """Regression: min_df of 3 on three documents left a vocabulary of shared
    whitespace, so every vector looked alike and unrelated texts merged silently."""
    ids = cluster_near_duplicates(
        [
            "dear sir i am a barrister acting for a deceased client with funds",
            "quarterly gas curve validation notes for the trading desk meeting",
            "cheap replica watches available now at our online store today",
        ]
    )
    assert len(set(ids)) == 3
