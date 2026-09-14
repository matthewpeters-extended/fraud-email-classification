"""Canonical text normalisation, the fix for defect D4.

The two source corpora were preprocessed differently before they were ever combined.
The Enron derived rows are lowercased with whitespace padded around punctuation, so
"amitava.dhar@enron.com" arrives as "amitava . dhar @ enron . com". The fraud mailbox
retains original casing and ordinary punctuation. Left alone, that difference lets any
classifier identify the source file without reading a word, and it also splits the
token inventory: "e-mail" appears in 100 percent FRAUD documents purely because the
Enron rows tokenise the same word as "e - mail".

Normalising both to one canonical form removes the fingerprint symmetrically. That is
preferred over mimicking the Enron style, which would mean reproducing an artifact.

Punctuation is dropped, with one exception. A few symbols carry genuine meaning for this
task, notably the currency sign, which appears in 74.6 percent of fraud documents against
7.5 percent of normal ones. Those are mapped to word tokens so the signal survives
normalisation rather than being thrown out with the formatting.
"""

from __future__ import annotations

import re

from sklearn.base import BaseEstimator, TransformerMixin

# Quoted printable remnants that survived MIME decoding, plus soft line breaks.
MIME_ARTIFACTS = re.compile(r"=(?:20|0A|0D|3D|2C|09)|=\n")

# Two decoding artifacts, both 100 percent Fraud and neither of them content.
# The literal escape "\ufffd" appears in 62 training documents, carried in from Yahoo
# footers in the source. The replacement character itself appears in 32, and that one is
# ours: phase 2 decoded the mailbox with errors="replace". Removing both here rather
# than blocklisting them keeps the fix with the artifact.
DECODE_ARTIFACTS = re.compile(r"\\?u?fffd|\ufffd", re.IGNORECASE)

# Symbols worth keeping as words. Dropping them with the rest of the punctuation would
# discard real signal along with the formatting fingerprint.
SYMBOL_WORDS = {
    "$": " currencysign ",
    "@": " atsign ",
    "%": " percentsign ",
}

NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
DIGITS = re.compile(r"\b\d[\d,.]*\b")
WHITESPACE = re.compile(r"\s+")


def normalise(text: str, mask_numbers: bool = True) -> str:
    """Reduce text to a canonical form shared by all three classes.

    Lowercase, strip MIME remnants, promote meaningful symbols to words, drop remaining
    punctuation, and optionally replace runs of digits with a single token.

    Numbers are masked by default because specific amounts and phone fragments are
    provenance rather than content: the Houston area code 713 appears in 102 documents,
    all of them Normal. The fact that a message quotes a large sum is signal; the exact
    digits are not.
    """
    text = text.lower()
    text = DECODE_ARTIFACTS.sub(" ", text)
    text = MIME_ARTIFACTS.sub(" ", text)
    for symbol, word in SYMBOL_WORDS.items():
        text = text.replace(symbol, word)
    # Numbers are masked before punctuation is stripped. The other order splits
    # "25,000,000.00" into four separate number tokens once its separators are gone.
    if mask_numbers:
        text = DIGITS.sub(" numbertoken ", text)
    text = NON_ALNUM.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


class Normaliser(BaseEstimator, TransformerMixin):
    """Normalisation as a pipeline step, so it is fitted and applied inside folds.

    It is stateless, but living in the pipeline keeps the preprocessing choice attached
    to the model rather than applied ad hoc to the data on disk.
    """

    def __init__(self, enabled: bool = True, mask_numbers: bool = True) -> None:
        self.enabled = enabled
        self.mask_numbers = mask_numbers

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        if not self.enabled:
            return list(X)
        return [normalise(t, mask_numbers=self.mask_numbers) for t in X]


# ----------------------------------------------------------- formatting probe

def formatting_features(text: str) -> dict[str, float]:
    """Measure only how a document is formatted, never what it says.

    Used to quantify defect D4. A classifier fed nothing but these numbers has no access
    to content, so whatever accuracy it reaches is pure source provenance.
    """
    n = max(len(text), 1)
    letters = sum(c.isalpha() for c in text) or 1
    spaced_punct = len(re.findall(r"\s[.,:;!?]\s", text))
    tight_punct = len(re.findall(r"[a-z0-9][.,:;!?][a-z0-9\s]", text, re.IGNORECASE))
    return {
        "uppercase_ratio": sum(c.isupper() for c in text) / letters,
        "spaced_punct_per_kchar": 1000 * spaced_punct / n,
        "tight_punct_per_kchar": 1000 * tight_punct / n,
        "newline_ratio": text.count("\n") / n,
        "mime_artifacts": len(MIME_ARTIFACTS.findall(text)),
        "apostrophe_ratio": 1000 * text.count("'") / n,
        "tab_ratio": 1000 * text.count("\t") / n,
    }


FORMATTING_FEATURE_NAMES = tuple(formatting_features("a").keys())
