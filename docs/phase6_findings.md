# Phase 6 findings: the preprocessing pipeline, decided by ablation

Stopword removal and lemmatisation are standard NLP boilerplate. The reference solution
applies both without testing either, as most tutorials do. Neither is assumed here.

Entry point `scripts/run_preprocessing_ablation.py`, transformers in `src/preprocess.py`,
tests in `tests/test_preprocess.py`. Numbers in `docs/phase6_ablation.json`, figure in
`reports/figures/phase6_ablation.png`.

Five fold stratified cross validation on the training split, macro F1, seed 20260914. The
test split is not touched until phase 9.

## Result: nothing helps, and that is the finding

<table>
<tr><th>Condition</th><th>Macro F1</th><th>Vocabulary</th><th>Tokens per document</th><th>Readable features</th></tr>
<tr><td>S0 baseline: normalised, markers stripped</td><td>0.9671</td><td>27,632</td><td>297</td><td>yes</td></tr>
<tr><td>S1 stopwords removed, signal pronouns kept</td><td>0.9657</td><td>27,496</td><td>191</td><td>yes</td></tr>
<tr><td>S2 stopwords removed, full NLTK list</td><td>0.9676</td><td>27,484</td><td>169</td><td>yes</td></tr>
<tr><td>S3 lemmatised</td><td>0.9658</td><td>23,480</td><td>297</td><td>yes</td></tr>
<tr><td>S4 stemmed</td><td>0.9658</td><td>21,321</td><td>297</td><td>no</td></tr>
<tr><td>S5 stopwords kept signal, lemmatised</td><td>0.9662</td><td>23,367</td><td>191</td><td>yes</td></tr>
<tr><td>S6 stopwords kept signal, stemmed</td><td>0.9662</td><td>21,228</td><td>191</td><td>no</td></tr>
<tr><td><b>S7 stopwords full list, lemmatised</b></td><td><b>0.9662</b></td><td><b>23,356</b></td><td><b>169</b></td><td><b>yes</b></td></tr>
<tr><td>S8 stopwords full list, stemmed</td><td>0.9634</td><td>21,224</td><td>169</td><td>no</td></tr>
</table>

Spread across all nine conditions: **0.0042**. Mean fold to fold standard deviation:
**0.0049**. The entire range of outcomes is smaller than the noise, and every condition
sits within one standard deviation of the best.

So macro F1 cannot choose between these. Saying "we selected S2 because it scored highest"
would be reading a 0.0004 difference as a result, which it is not.

## The hypothesis that failed

`SIGNAL_STOPWORDS` was built on a specific claim: advance fee fraud is written in the first
person about a named relative, "I am Mrs X, my late husband ...", so the pronouns `i`,
`am`, `my`, `your`, `he`, `she`, `not` and friends are register rather than noise, and
holding them back from the stopword list should help.

<table>
<tr><th>Condition</th><th>Macro F1</th></tr>
<tr><td>S1, twelve signal pronouns held back</td><td>0.9657</td></tr>
<tr><td>S2, plain NLTK list, nothing held back</td><td>0.9676</td></tr>
</table>

The exemption is 0.0018 **worse**, not better. Within noise, so the honest reading is that
it makes no difference, but the direction is opposite to the prediction and there is no
evidence for it at all.

Decision: the mechanism stays in the code because the reasoning is worth preserving and
the comparison is reproducible, but `keep_signal` now defaults to False and the chosen
pipeline uses the plain list.

## A defect in the lemmatiser, caught before it reached any result

WordNet needs a part of speech tag and defaults to noun. Neither tag alone is sufficient:

<table>
<tr><th>Strategy</th><th>Fails on</th></tr>
<tr><td>noun only</td><td>died to died, transferred to transferred</td></tr>
<tr><td>verb only</td><td>beneficiaries to beneficiaries, mailings to mailings</td></tr>
</table>

The first implementation chained noun then verb, which handled both but introduced a worse
problem. WordNet's noun lemmatiser strips any trailing `s` it reads as a plural:

<table>
<tr><th>Token</th><th>Noun first result</th><th>Problem</th></tr>
<tr><td>was</td><td>wa</td><td>not a word</td></tr>
<tr><td>has</td><td>ha</td><td>not a word</td></tr>
<tr><td>us</td><td>u</td><td>not a word</td></tr>
<tr><td>as</td><td>a</td><td>not a word</td></tr>
<tr><td>mrs</td><td>mr</td><td>conflates a top fraud term with its masculine form</td></tr>
</table>

`mrs` is the one that would have mattered. It appears in 126 training documents at 96
percent fraud purity, and merging it into `mr` would have destroyed that signal silently.

Fixed by reversing the order and guarding the noun step: lemmatise as a verb first, and
only fall back to the noun form if the verb step changed nothing and the noun lemma is at
least three characters. All five bad cases are corrected and all ten good cases preserved,
pinned by parametrised tests.

## A second defect, which would have surfaced only at scoring time

The transformers hold no learned state, so their `fit` methods originally did nothing but
return `self`. scikit learn's `check_is_fitted` looks for an attribute ending in an
underscore, and without one `Pipeline.transform` raises `NotFittedError`.

The failure mode is nasty: `fit_transform` works fine, so every cross validation run
passes. Only the fit once then transform many times pattern breaks, which is exactly what
phase 9 does when scoring the held out test set. A test caught it here rather than there.

## The decision, and the criteria that made it

Since score cannot decide, the criteria are stated explicitly rather than implied:

1. **Interpretability.** Phases 8 and 9 report top features and read misclassified emails.
   Stemming returns `beneficiari`, `busi` and `wit`. Those phases would be unreadable.
2. **Vocabulary size.** A smaller feature space is less to overfit during the phase 8
   hyperparameter sweep.
3. **Tokens per document.** Lower is faster for that sweep.
4. **Simplicity**, where nothing else separates the options.

**Chosen: S7, normalise, strip markers, remove stopwords with the full list, lemmatise.**

Macro F1 0.9662, tied with everything. Vocabulary 23,356, down 15.5 percent on the
baseline. Tokens per document 169, down 43 percent. Every feature is a real word.

Stemming would remove a further 2,132 terms and is rejected on criterion 1 alone. That
trade is not close: the extra reduction buys nothing measurable and costs the readability
that two later phases depend on.

The configuration lives in `src.preprocess.CHOSEN_CONFIG` with a `chosen_text_steps()`
factory, so phases 7 onward cannot drift from it. A test asserts its contents.

### The honest sentence for the README

None of this preprocessing improved the model. It was kept because it removes 15.5 percent
of the vocabulary and 43 percent of the tokens at no measurable cost, not because it
helped. The value of the phase is knowing that, rather than applying the steps by reflex
and implying they contributed.

## D11 settled: length only on cleaned text

Phase 5 predicted the length shortcut would look stronger after cleaning, because the
Enron derived rows pad whitespace around punctuation so a raw word count credits them with
tokens that are only ` . ` and ` , `.

<table>
<tr><th>Length only baseline</th><th>Macro F1</th></tr>
<tr><td>On raw text, measured at phase 4</td><td>0.4386</td></tr>
<tr><td>On cleaned text</td><td>0.4698</td></tr>
</table>

Confirmed. Phase 7 uses 0.4698 as the length baseline.

## Baselines carried into phase 7

<table>
<tr><th>Baseline</th><th>Macro F1</th><th>What it means</th></tr>
<tr><td>Majority class</td><td>0.1667</td><td>the floor, and not 0.333 which is accuracy</td></tr>
<tr><td>Document length only</td><td>0.4698</td><td>one feature, no words</td></tr>
<tr><td>Provenance markers only</td><td>0.6082</td><td>42 tokens, no content</td></tr>
<tr><td>Formatting only</td><td>0.7013</td><td>seven numeric features, no words at all</td></tr>
</table>

A model has to clear 0.7013 before it has demonstrated anything about reading email.

## Tests

143 tests pass. New coverage in `tests/test_preprocess.py`: the lemmatiser across both
parts of speech, the five non word regressions, the length guard value, idempotence,
stemming being more aggressive than lemmatising, the stopword default now being the full
list, the signal exemption still being available, each step switchable, step naming and
ordering, normalisation running before marker matching, the full chosen pipeline end to
end, `CHOSEN_CONFIG` matching the decision recorded here, and statelessness verified by
fitting on unrelated text and confirming the output does not change.
