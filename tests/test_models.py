"""Tests for the model sweep configuration.

These guard the search space itself. A grid that silently references the wrong step name,
or an estimator that cannot fit the representation it is paired with, would fail deep
inside a nested cross validation run after several minutes of compute.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import ClassifierMixin, is_classifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.models import (
    SEED,
    VECTORISER_GRID,
    estimators,
    grid_size,
    vectorisers,
    voting_ensemble,
)

EXPECTED_MODELS = {
    "multinomial_nb",
    "complement_nb",
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "knn",
}


def tiny_corpus() -> tuple[list[str], np.ndarray]:
    X = (
        ["beneficiary dormant fund transfer deceased estate"] * 12
        + ["viagra shops price offer unsubscribe discount"] * 12
        + ["meeting agenda lunch conference minutes review"] * 12
    )
    y = np.array(["FRAUD"] * 12 + ["SPAM"] * 12 + ["NORMAL"] * 12)
    return X, y


# --------------------------------------------------------------- the catalogue

def test_every_model_named_in_the_plan_is_present():
    """The brief names Naive Bayes, Random Forest and SVM explicitly."""
    assert EXPECTED_MODELS <= set(estimators())


def test_both_vectorisers_named_in_the_plan_are_present():
    assert set(vectorisers()) == {"count", "tfidf"}


def test_vectorisers_drop_hapax_terms():
    for vect in vectorisers().values():
        assert vect.min_df == 2


def test_every_entry_is_a_classifier():
    for name, (estimator, _) in estimators().items():
        assert is_classifier(estimator), name


def test_every_grid_is_non_empty():
    for name, (_, grid) in estimators().items():
        assert grid, name
        assert grid_size(grid) >= 2, name


def test_grids_address_the_pipeline_step_name():
    """A grid key that does not match the step name fails only at search time."""
    for name, (_, grid) in estimators().items():
        for key in grid:
            assert key.startswith("clf__"), f"{name}: {key}"
    for key in VECTORISER_GRID:
        assert key.startswith("vect__")


def test_grid_size_multiplies_the_axes():
    assert grid_size({"a": [1, 2], "b": [3, 4, 5]}) == 6
    assert grid_size({}) == 1


def test_vectoriser_grid_searches_unigrams_against_bigrams():
    assert VECTORISER_GRID["vect__ngram_range"] == [(1, 1), (1, 2)]


def test_seeded_estimators_carry_the_project_seed():
    for name, (estimator, _) in estimators().items():
        if "random_state" in estimator.get_params():
            assert estimator.get_params()["random_state"] == SEED, name


# ------------------------------------------------------------------- ensemble

def test_voting_ensemble_uses_soft_voting():
    ensemble, _ = voting_ensemble()
    assert isinstance(ensemble, VotingClassifier)
    assert ensemble.voting == "soft"


def test_voting_ensemble_members_all_expose_predict_proba():
    """Soft voting needs probabilities. LinearSVC has none, which is why it is excluded."""
    ensemble, _ = voting_ensemble()
    for name, member in ensemble.estimators:
        assert hasattr(member, "predict_proba"), name


def test_voting_ensemble_excludes_linear_svm():
    ensemble, _ = voting_ensemble()
    names = {name for name, _ in ensemble.estimators}
    assert "svm" not in names and "linear_svm" not in names


def test_voting_ensemble_grid_addresses_its_members():
    _, grid = voting_ensemble()
    for key in grid:
        assert key.startswith("clf__"), key
        assert key.count("__") == 2, f"{key} should address a member parameter"


# -------------------------------------------------------- everything actually runs

@pytest.mark.parametrize("model_name", sorted(EXPECTED_MODELS))
@pytest.mark.parametrize("vect_name", ["count", "tfidf"])
def test_each_model_fits_with_each_vectoriser(model_name, vect_name):
    X, y = tiny_corpus()
    estimator, _ = estimators()[model_name]
    pipeline = Pipeline([("vect", vectorisers()[vect_name]), ("clf", estimator)])
    pipeline.fit(X, y)
    assert len(pipeline.predict(X)) == len(X)


def test_a_grid_search_completes_on_the_smallest_model():
    X, y = tiny_corpus()
    estimator, grid = estimators()["multinomial_nb"]
    search = GridSearchCV(
        Pipeline([("vect", vectorisers()["tfidf"]), ("clf", estimator)]),
        {**grid, **VECTORISER_GRID},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
    )
    search.fit(X, y)
    assert set(search.best_params_) <= set(grid) | set(VECTORISER_GRID)
    assert search.best_score_ > 0.9  # planted signal


def test_voting_ensemble_fits_through_the_pipeline():
    X, y = tiny_corpus()
    ensemble, _ = voting_ensemble()
    pipeline = Pipeline([("vect", vectorisers()["tfidf"]), ("clf", ensemble)])
    pipeline.fit(X, y)
    assert len(pipeline.predict(X)) == len(X)


def test_estimators_are_fresh_objects_each_call():
    """The sweep reuses names across vectorisers, so shared mutable state would leak
    fitted attributes from one configuration into the next."""
    a = estimators()["logistic_regression"][0]
    b = estimators()["logistic_regression"][0]
    assert a is not b
