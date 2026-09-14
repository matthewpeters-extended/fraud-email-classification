"""The preprocessing pipeline, assembled as scikit learn transformers.

Every step lives inside a `Pipeline` rather than being applied to the data on disk. The
steps here are stateless, so they cannot leak by themselves, but keeping them in the
pipeline means the preprocessing choice travels with the fitted model and is applied
identically at scoring time. That is the fix for `PLAN.md` 8.2.

Order is fixed: normalise, strip provenance markers, then optionally remove stopwords and
reduce word forms. Normalisation must come first because marker matching is defined on
lowercased alphanumeric tokens.

Nothing here is assumed to help. Phase 6 ablates each step and keeps only what earns its
place on the training folds.
"""

from __future__ import annotations

from functools import lru_cache

from sklearn.base import BaseEstimator, TransformerMixin

from src.markers import strip_markers
from src.normalise import normalise

# The hypothesis was that these pronouns carry genuine signal here, because advance fee
# fraud is written in the first person about a named relative: "I am Mrs X, my late
# husband ...". Holding them back from the stopword list was expected to help.
#
# Measured at phase 6, it does not. Keeping them scores 0.9657 macro F1 against 0.9676 for
# the plain NLTK list, so the exemption is very slightly worse, and in any case both sit
# inside the fold to fold noise. The mechanism is kept because the reasoning is worth
# preserving and the comparison is reproducible, but `keep_signal` now defaults to False.
# See docs/phase6_findings.md.
SIGNAL_STOPWORDS = frozenset({"i", "am", "my", "me", "your", "you", "he", "she", "his",
                              "her", "not", "no"})


# A noun lemma shorter than this is rejected. WordNet's noun lemmatiser strips any
# trailing "s" it reads as a plural, which turns "was" into "wa", "has" into "ha", "us"
# into "u", "as" into "a" and "mrs" into "mr". The first four are not words and the last
# conflates a top fraud term with its masculine form.
MIN_NOUN_LEMMA_LENGTH = 3


@lru_cache(maxsize=200_000)
def lemmatise_token(token: str) -> str:
    """Lemmatise one token: verb form first, then noun form behind a length guard.

    WordNet needs a part of speech tag and defaults to noun. Neither tag alone is enough:

        noun only    died -> died,          transferred -> transferred     (misses verbs)
        verb only    beneficiaries -> beneficiaries,  mailings -> mailings (misses nouns)

    Verb first, then noun only if the verb step changed nothing, handles both. The length
    guard stops the noun step from mangling short words it misreads as plurals:

        was           -> be           (verb step)
        has           -> have         (verb step)
        died          -> die          (verb step)
        transferred   -> transfer     (verb step)
        beneficiaries -> beneficiary  (noun step)
        mailings      -> mailing      (noun step)
        us            -> us           (noun lemma "u" rejected, too short)
        mrs           -> mrs          (noun lemma "mr" rejected, too short)

    Cached because the vocabulary is small relative to the token count and the ablation
    runs this over the corpus many times.
    """
    lemmatiser = _wordnet()

    as_verb = lemmatiser.lemmatize(token, "v")
    if as_verb != token:
        return as_verb

    as_noun = lemmatiser.lemmatize(token, "n")
    if as_noun != token and len(as_noun) < MIN_NOUN_LEMMA_LENGTH:
        return token
    return as_noun


@lru_cache(maxsize=1)
def _wordnet():
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _porter():
    from nltk.stem import PorterStemmer

    return PorterStemmer()


@lru_cache(maxsize=200_000)
def stem_token(token: str) -> str:
    return _porter().stem(token)


@lru_cache(maxsize=1)
def default_stopwords() -> frozenset[str]:
    """English stopwords, less the ones that carry signal in this corpus."""
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english")) - SIGNAL_STOPWORDS


class _StatelessTransformer(BaseEstimator, TransformerMixin):
    """Base for the text steps, which hold no learned state.

    `fit` still has to record that it happened. scikit learn's `check_is_fitted` looks for
    an attribute ending in an underscore, and without one `Pipeline.transform` raises
    NotFittedError even though the step needs nothing from the data. That breaks the fit
    once then transform many times pattern used when scoring a held out set, while
    `fit_transform` keeps working, so the failure only appears at scoring time.
    """

    def fit(self, X, y=None):  # noqa: N803
        self.fitted_ = True
        return self


class Normalise(_StatelessTransformer):
    """Canonical normalisation. See src/normalise.py."""

    def __init__(self, enabled: bool = True, mask_numbers: bool = True) -> None:
        self.enabled = enabled
        self.mask_numbers = mask_numbers

    def transform(self, X):  # noqa: N803
        if not self.enabled:
            return list(X)
        return [normalise(t, mask_numbers=self.mask_numbers) for t in X]


class StripMarkers(_StatelessTransformer):
    """Remove provenance markers. See src/markers.py."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def transform(self, X):  # noqa: N803
        if not self.enabled:
            return list(X)
        return [strip_markers(t) for t in X]


class RemoveStopwords(_StatelessTransformer):
    """Drop common English words.

    `keep_signal` holds back the pronouns in SIGNAL_STOPWORDS. It defaults to False
    because the phase 6 ablation found the exemption did not help.
    """

    def __init__(self, enabled: bool = True, keep_signal: bool = False) -> None:
        self.enabled = enabled
        self.keep_signal = keep_signal

    def _stopwords(self) -> frozenset[str]:
        if self.keep_signal:
            return default_stopwords()
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("english"))

    def transform(self, X):  # noqa: N803
        if not self.enabled:
            return list(X)
        stop = self._stopwords()
        return [" ".join(w for w in t.split() if w not in stop) for t in X]


class ReduceWordForms(_StatelessTransformer):
    """Collapse inflected forms, by lemmatisation or by stemming.

    Lemmatisation returns real words, which keeps the discriminative term rankings
    readable. Stemming is more aggressive and returns fragments such as "beneficiari".
    Both are offered so the ablation can choose on evidence.
    """

    def __init__(self, method: str = "none") -> None:
        self.method = method

    def transform(self, X):  # noqa: N803
        if self.method == "none":
            return list(X)
        if self.method == "lemmatise":
            fn = lemmatise_token
        elif self.method == "stem":
            fn = stem_token
        else:
            raise ValueError(f"unknown method {self.method!r}, expected none, lemmatise or stem")
        return [" ".join(fn(w) for w in t.split()) for t in X]


def text_steps(
    normalise_text: bool = True,
    strip_provenance: bool = True,
    remove_stopwords: bool = False,
    reduce_forms: str = "none",
) -> list[tuple[str, BaseEstimator]]:
    """The text side of the pipeline, as a list of named steps.

    Returned as steps rather than a built Pipeline so callers can append a vectoriser and
    a classifier without unpacking anything.
    """
    return [
        ("normalise", Normalise(enabled=normalise_text)),
        ("strip_markers", StripMarkers(enabled=strip_provenance)),
        ("stopwords", RemoveStopwords(enabled=remove_stopwords)),
        ("word_forms", ReduceWordForms(method=reduce_forms)),
    ]


# The configuration chosen at phase 6, and the single source of truth for every later
# phase. Do not inline these settings elsewhere.
#
# All nine ablated conditions tied within the fold to fold noise, spread 0.0042 against a
# mean standard deviation of 0.0049, so macro F1 could not choose. The decision was made
# on stated secondary criteria: readable features, a smaller vocabulary and fewer tokens
# per document. Stemming was rejected despite the smallest vocabulary because it turns
# "beneficiary" into "beneficiari" and would make the phase 8 feature tables and the phase
# 9 error analysis unreadable.
#
# The honest summary: none of this improved the model. It was kept because it removes 15.5
# percent of the vocabulary and 43 percent of the tokens at no measurable cost.
CHOSEN_CONFIG: dict[str, object] = {
    "normalise_text": True,
    "strip_provenance": True,
    "remove_stopwords": True,
    "reduce_forms": "lemmatise",
}


def chosen_text_steps() -> list[tuple[str, BaseEstimator]]:
    """The phase 6 pipeline, for use by every phase from 7 onward."""
    return text_steps(**CHOSEN_CONFIG)  # type: ignore[arg-type]
