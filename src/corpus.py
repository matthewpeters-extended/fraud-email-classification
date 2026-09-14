"""Build the three class corpus from the two source files.

Construction order matters and is deliberate:

  1. strip the Subject: prefix, which is a source file marker and not content
  2. drop empty and near empty bodies
  3. cluster near duplicates across the whole pool, not per class, so that a
     message appearing in two classes is caught as a label contradiction
  4. sample one representative per cluster, so the corpus contains no duplicates
  5. split train and test

Because step 4 takes a single representative per cluster, no duplicate can straddle
the train and test split. That is a stronger guarantee than splitting by cluster
afterwards, and it is the fix for defect D7.

Casing and punctuation are left untouched. Normalising them is a modelling decision
and belongs inside the pipeline at phase 6, where it applies to all classes alike.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SUBJECT_PREFIX = re.compile(r"^\s*subject\s*:\s*", re.IGNORECASE)
NON_WORD = re.compile(r"\W+")

CLASSES = ("FRAUD", "SPAM", "NORMAL")

# Cosine similarity on character 4 grams. Two messages at or above this are treated
# as the same message for sampling purposes.
NEAR_DUPLICATE_THRESHOLD = 0.85
MIN_WORDS = 5

# Character n grams seen in fewer than this many documents are dropped, which keeps the
# vocabulary bounded on the full corpus. It has to scale down on small inputs: with
# min_df of 3 and only a handful of documents, a feature must appear in nearly all of
# them to survive, the vocabulary collapses to shared whitespace, every vector looks
# alike and unrelated documents merge. Below this corpus size, keep every n gram.
MIN_DF_CORPUS_SIZE = 300
MIN_DF_LARGE = 3
MIN_DF_SMALL = 1


@dataclass
class Document:
    text: str
    label: str
    source: str
    source_row: int
    cluster: int = -1

    @property
    def words(self) -> int:
        return len(self.text.split())


def strip_subject_prefix(text: str) -> str:
    """Remove the leading Subject: marker carried by every row of the Enron source.

    Measured at phase 3: present on 1,368 of 1,368 spam rows and 4,360 of 4,360 ham
    rows, and on 1 of 3,976 fraud messages. Leaving it in hands any classifier a
    single token that identifies the source file. See docs/data_defects.md D2.
    """
    return SUBJECT_PREFIX.sub("", text).strip()


def normalise_for_matching(text: str) -> str:
    """Aggressive normalisation used only for duplicate detection, never for modelling."""
    return NON_WORD.sub(" ", text.lower()).strip()


def content_hash(text: str) -> str:
    return hashlib.sha1(normalise_for_matching(text).encode()).hexdigest()


def load_sources(spam_normal_csv: Path, fraud_csv: Path) -> list[Document]:
    """Read both sources into a single document list."""
    csv.field_size_limit(10_000_000)
    docs: list[Document] = []

    with spam_normal_csv.open(newline="", encoding="utf-8", errors="replace") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=1):
            label = "SPAM" if str(row["spam"]).strip() == "1" else "NORMAL"
            docs.append(
                Document(
                    text=strip_subject_prefix(row["text"]),
                    label=label,
                    source=spam_normal_csv.name,
                    source_row=i,
                )
            )

    with fraud_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            docs.append(
                Document(
                    text=strip_subject_prefix(row["body"]),
                    label="FRAUD",
                    source=fraud_csv.name,
                    source_row=int(row["index"]),
                )
            )

    return docs


def drop_short(docs: list[Document], min_words: int = MIN_WORDS) -> tuple[list[Document], int]:
    kept = [d for d in docs if d.words >= min_words]
    return kept, len(docs) - len(kept)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _complete_linkage_subclusters(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Split one connected component so that every pair inside a cluster clears threshold.

    Complete linkage is the right semantics for duplicate detection. Single linkage
    chains A to B to C on shared boilerplate even when A and C are unrelated, which
    on this corpus merged 232 fraud emails whose median pairwise cosine was only
    0.767 and whose minimum was 0.563. Complete linkage cannot do that.
    """
    n = sim.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)

    dist = np.clip(1.0 - sim, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce symmetry against float drift
    tree = linkage(squareform(dist, checks=False), method="complete")
    return fcluster(tree, t=1.0 - threshold, criterion="distance")


def cluster_near_duplicates(
    texts: list[str],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
    block: int = 512,
) -> np.ndarray:
    """Group texts into near duplicate clusters by cosine similarity on char 4 grams.

    Two stages, because complete linkage on the full corpus would need a dense
    matrix of every pair:

      1. Cheap: find connected components of the graph of pairs at or above the
         threshold. Single linkage over sparse blocks. Components are a superset of
         the final clusters, since a complete linkage cluster can never span two
         disconnected components.
      2. Exact: inside each component, run complete linkage agglomerative clustering
         on the small dense sub matrix, so every pair within a final cluster is at or
         above the threshold.

    Returns an array of cluster ids, one per text.
    """
    n = len(texts)
    uf = UnionFind(n)

    # Cheap pass: exact matches after normalisation.
    by_hash: dict[str, int] = {}
    for i, t in enumerate(texts):
        h = content_hash(t)
        if h in by_hash:
            uf.union(by_hash[h], i)
        else:
            by_hash[h] = i

    min_df = MIN_DF_LARGE if n >= MIN_DF_CORPUS_SIZE else MIN_DF_SMALL
    vectoriser = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 4),
        min_df=min_df,
        max_features=300_000,
        lowercase=True,
        dtype=np.float32,
    )
    matrix = normalize(vectoriser.fit_transform(texts))

    # Stage 1: connected components.
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = (matrix[start:stop] @ matrix.T).toarray()
        for local, global_i in enumerate(range(start, stop)):
            row = sims[local]
            row[: global_i + 1] = 0.0
            for j in np.nonzero(row >= threshold)[0]:
                uf.union(global_i, int(j))

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(uf.find(i), []).append(i)

    # Stage 2: complete linkage inside each component.
    cluster_ids = np.full(n, -1, dtype=int)
    next_id = 0
    for members in components.values():
        if len(members) == 1:
            cluster_ids[members[0]] = next_id
            next_id += 1
            continue
        sub = matrix[members]
        sim = (sub @ sub.T).toarray().astype(np.float64)
        labels = _complete_linkage_subclusters(sim, threshold)
        remap: dict[int, int] = {}
        for member, lab in zip(members, labels):
            if lab not in remap:
                remap[lab] = next_id
                next_id += 1
            cluster_ids[member] = remap[lab]

    assert (cluster_ids >= 0).all(), "some document was left unclustered"
    return cluster_ids


def find_cross_class_clusters(docs: list[Document]) -> dict[int, set[str]]:
    """Clusters whose members disagree on the label. A label contradiction."""
    seen: dict[int, set[str]] = {}
    for d in docs:
        seen.setdefault(d.cluster, set()).add(d.label)
    return {c: labels for c, labels in seen.items() if len(labels) > 1}


def drop_label_contradictions(
    docs: list[Document],
) -> tuple[list[Document], dict[int, set[str]]]:
    """Remove clusters whose members carry more than one label.

    These are real: a handful of advance fee scams appear in both the fraud mailbox and
    the Enron spam corpus, because the same 419 campaigns reached Enron employees. The
    two sources therefore disagree on the label of the same email.

    They must go before sampling. A contradicted cluster yields one representative per
    label, so the same near identical text would enter the corpus twice under different
    labels and could land on both sides of the train and test split.

    Dropping rather than taking a majority label is the honest choice. If two sources
    disagree about what an email is, we have no grounds to pick a winner. It also makes
    a real point about the task: advance fee fraud is a subset of spam, so the boundary
    between those two classes is genuinely fuzzy at the edges.
    """
    contradictions = find_cross_class_clusters(docs)
    if not contradictions:
        return docs, {}
    bad = set(contradictions)
    return [d for d in docs if d.cluster not in bad], contradictions


def sample_one_per_cluster(
    docs: list[Document], per_class: int, seed: int
) -> tuple[list[Document], dict[str, int]]:
    """Draw per_class documents for each label, at most one from any cluster.

    The longest member of a cluster is chosen as its representative, so that when a
    templated scam appears in several truncated forms the fullest one survives.
    """
    rng = np.random.default_rng(seed)

    representatives: dict[tuple[str, int], Document] = {}
    for d in docs:
        key = (d.label, int(d.cluster))
        current = representatives.get(key)
        if current is None or d.words > current.words:
            representatives[key] = d

    available: dict[str, list[Document]] = {c: [] for c in CLASSES}
    for (label, _), d in representatives.items():
        available[label].append(d)

    chosen: list[Document] = []
    pool_sizes: dict[str, int] = {}
    for label in CLASSES:
        pool = sorted(available[label], key=lambda d: (d.source, d.source_row))
        pool_sizes[label] = len(pool)
        if len(pool) < per_class:
            raise ValueError(
                f"class {label} has only {len(pool)} unique clusters, "
                f"cannot sample {per_class}"
            )
        picks = rng.choice(len(pool), size=per_class, replace=False)
        chosen.extend(pool[int(i)] for i in sorted(picks))

    return chosen, pool_sizes


def stratified_split(
    docs: list[Document], test_fraction: float, seed: int
) -> tuple[list[Document], list[Document]]:
    """Stratified split. Every document is already from a distinct cluster."""
    rng = np.random.default_rng(seed)
    train: list[Document] = []
    test: list[Document] = []
    for label in CLASSES:
        members = sorted(
            [d for d in docs if d.label == label], key=lambda d: (d.source, d.source_row)
        )
        order = rng.permutation(len(members))
        cut = int(round(len(members) * test_fraction))
        for rank, idx in enumerate(order):
            (test if rank < cut else train).append(members[int(idx)])
    return train, test


def write_csv(path: Path, docs: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "source", "source_row", "cluster", "words", "chars", "text"])
        for d in docs:
            writer.writerow(
                [d.label, d.source, d.source_row, d.cluster, d.words, len(d.text), d.text]
            )
