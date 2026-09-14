"""The model sweep: estimators, vectorisers and their hyperparameter grids.

Kept separate from the driver script so the search space is reviewable on its own, and so
phase 9 can rebuild the winning configuration without re running the sweep.

Every grid is deliberately small. A large grid searched by nested cross validation on 2,160
documents would spend most of its budget distinguishing configurations that differ by less
than the fold to fold noise, which phase 6 measured at roughly 0.005 macro F1.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

SEED = 20260914

# Shared across every model. Unigrams against unigrams plus bigrams is the one vectoriser
# choice worth searching; min_df is fixed at 2 to drop hapax terms, which on a corpus this
# size are almost all typos and mail fragments.
VECTORISER_GRID = {
    "vect__ngram_range": [(1, 1), (1, 2)],
}


def vectorisers() -> dict[str, object]:
    """The two feature representations named in the brief."""
    return {
        "count": CountVectorizer(min_df=2),
        "tfidf": TfidfVectorizer(min_df=2, sublinear_tf=True),
    }


def estimators() -> dict[str, tuple[object, dict]]:
    """Model name to (estimator, hyperparameter grid).

    Grids are expressed against the pipeline step name "clf".
    """
    return {
        "multinomial_nb": (
            MultinomialNB(),
            {"clf__alpha": [0.01, 0.1, 0.5, 1.0]},
        ),
        "complement_nb": (
            ComplementNB(),
            {"clf__alpha": [0.1, 0.5, 1.0]},
        ),
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"clf__C": [0.5, 1.0, 5.0, 20.0]},
        ),
        "linear_svm": (
            # max_iter is raised well above the default 1000. On CountVectorizer features
            # the default does not converge: raw term counts are unscaled and span orders
            # of magnitude, so the margin problem is badly conditioned and liblinear stops
            # early. A model that stopped early is reporting an arbitrary point on its
            # optimisation path, not a fitted model, so its score would be meaningless.
            # TF IDF features are L2 normalised and converge without help.
            LinearSVC(random_state=SEED, max_iter=20_000),
            {"clf__C": [0.05, 0.25, 1.0, 5.0]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
            {"clf__max_features": ["sqrt", 0.05], "clf__min_samples_leaf": [1, 3]},
        ),
        "knn": (
            KNeighborsClassifier(metric="cosine"),
            {"clf__n_neighbors": [3, 7, 15], "clf__weights": ["uniform", "distance"]},
        ),
    }


def voting_ensemble() -> tuple[VotingClassifier, dict]:
    """Soft voting over three estimators that disagree in useful ways.

    LinearSVC is excluded despite being a strong individual model, because it has no
    `predict_proba` and soft voting needs one. Wrapping it in calibration would add a
    nested fit inside an already nested search for no expected gain over logistic
    regression, which occupies the same part of the model space.
    """
    ensemble = VotingClassifier(
        estimators=[
            ("nb", MultinomialNB(alpha=0.1)),
            ("lr", LogisticRegression(max_iter=2000, random_state=SEED)),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)),
        ],
        voting="soft",
    )
    return ensemble, {"clf__lr__C": [1.0, 5.0], "clf__nb__alpha": [0.1, 0.5]}


def grid_size(grid: dict) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size
