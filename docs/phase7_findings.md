# Phase 7 findings: baselines, and a correction to the floor

Every score reported from phase 8 onward is quoted against this table. The purpose is to
make it impossible to present a headline number without the reader seeing what a trivial
strategy, a ten word rule, and three content free shortcuts achieve on the same data.

Entry point `scripts/run_baselines.py`, logic in `src/baselines.py`, tests in
`tests/test_baselines.py`. Numbers in `docs/phase7_baselines.json`, figure in
`reports/figures/phase7_baselines.png`.

Five fold stratified cross validation on the training split, macro F1, seed 20260914. The
test split is not touched until phase 9.

## The canonical table

<table>
<tr><th>Key</th><th>Baseline</th><th>Macro F1</th><th>Standard deviation</th></tr>
<tr><td>B0</td><td>most frequent class</td><td>0.1667</td><td>0.0000</td></tr>
<tr><td>B2</td><td>uniform random</td><td>0.3147</td><td>0.0000</td></tr>
<tr><td>B1</td><td>stratified random</td><td>0.3355</td><td>0.0000</td></tr>
<tr><td>B6</td><td>keyword rule, 5 words per class</td><td>0.4181</td><td>0.0174</td></tr>
<tr><td>B3</td><td>document length only, one feature</td><td>0.4453</td><td>0.0409</td></tr>
<tr><td>B7</td><td>keyword rule, 10 words per class</td><td>0.4600</td><td>0.0135</td></tr>
<tr><td>B8</td><td>keyword rule, 25 words per class</td><td>0.5737</td><td>0.0141</td></tr>
<tr><td>B4</td><td>provenance markers only, 42 tokens</td><td>0.6082</td><td>0.0269</td></tr>
<tr><td><b>B5</b></td><td><b>formatting only, seven numeric features</b></td><td><b>0.7013</b></td><td><b>0.0202</b></td></tr>
<tr><td>MODEL</td><td>phase 6 pipeline with logistic regression, not a baseline</td><td>0.9662</td><td>0.0045</td></tr>
</table>

**The bar to clear is 0.7013.** Beating the majority class, or even beating random, says
nothing at all. A model has to beat the best content free shortcut before it has
demonstrated that it is reading the email rather than its packaging.

The phase 6 pipeline clears that bar by 0.2649.

## Correction: the floor is random, not majority

Phases 4 through 6 quoted 0.1667 as the macro F1 floor. That figure is correct for a
classifier that always predicts one class, but it is not the floor.

<table>
<tr><th>Trivial strategy</th><th>Macro F1</th></tr>
<tr><td>Most frequent class</td><td>0.1667</td></tr>
<tr><td>Uniform random</td><td>0.3147</td></tr>
<tr><td>Stratified random</td><td>0.3355</td></tr>
</table>

Random guessing beats always predicting one class by 0.1688 macro F1, which is more than
double. The reason is in the metric. Macro F1 averages per class F1, and a single class
predictor scores 0.5 on its chosen class and 0.0 on the other two, giving 0.1667. Random
guessing on a balanced three class problem instead scores roughly a third on all three
classes, giving about 0.335.

So on this problem the majority class baseline is a strange thing to quote: it is not the
worst a sensible strategy can do, and it is not what an untrained system would achieve.
The honest trivial floor is **0.3355**.

This matters beyond bookkeeping. Phase 4 reported the formatting probe as "4.21 times the
majority baseline". Against the correct floor it is 2.09 times, which is a materially
different claim. Multiples against a badly chosen denominator flatter every result, and the
denominator here was badly chosen.

Decision: from here on, quote random at 0.3355 as the trivial floor and formatting only at
0.7013 as the bar. Both appear in every later table. The majority figure stays in the table
for completeness, labelled as what it is.

## The keyword rule

The "what could a person write in an afternoon" baseline: count how many of each class's
keywords a document contains, predict the class with the most matches, fall back to the
most frequent training class on a tie or on no matches at all.

The keywords are **learned in `fit`**, ranked by document level log odds ratio, not
hardcoded. That distinction is the difference between a baseline and a cheat. A hand picked
list chosen after reading the whole corpus would have been tuned on the test data by way of
the author's memory, and could not be honestly cross validated. Learning them inside each
fold keeps the rule comparable with the models it is compared against.

<table>
<tr><th>Keywords per class</th><th>Macro F1</th><th>Documents where the rule fires</th></tr>
<tr><td>5</td><td>0.4181</td><td>22.8 percent</td></tr>
<tr><td>10</td><td>0.4600</td><td>30.6 percent</td></tr>
<tr><td>25</td><td>0.5737</td><td>51.5 percent</td></tr>
</table>

Coverage is reported alongside the score because the two tell different stories. At five
keywords per class the rule declines to decide on 77 percent of documents and falls back to
a fixed class, so its 0.4181 is mostly the fallback, not the rule. At twenty five keywords
it actually fires on half the corpus.

Seventy five words, twenty five per class, gets 0.5737. That is a useful anchor: it is most
of the way to the provenance marker probe and it is genuinely interpretable. Anyone
proposing a machine learning system for this task should be asked why it beats writing down
seventy five words.

## Two measurements of the length baseline

<table>
<tr><th>Representation</th><th>Macro F1</th></tr>
<tr><td>B3, the chosen phase 6 pipeline, stopwords removed and lemmatised</td><td>0.4453</td></tr>
<tr><td>B3b, normalisation and marker stripping only</td><td>0.4698</td></tr>
</table>

Phase 6 reported the second figure, measured before the stopword step had been chosen.
Removing stopwords cuts tokens per document by 43 percent, so these are the same idea
measured on different text rather than a disagreement.

Both are reported. B3 is the one that matters, because it is the representation the model
actually sees. The phase 6 figure is kept so the earlier write up remains traceable rather
than silently overwritten.

Note also that B3 carries the widest standard deviation of any baseline at 0.0409, roughly
nine times that of the model. A single feature is unstable across folds, which is itself a
reason not to read too much into small differences among the weaker baselines.

## What this says about the reference solution

The reference reports accuracy against no baseline at all. On this corpus:

* A classifier that reads nothing but seven formatting statistics reaches 0.7013
* A list of seventy five words reaches 0.5737
* Random guessing reaches 0.3355

Any accuracy figure presented without those numbers beside it is uninterpretable. That is
not a criticism of the modelling in the reference, which is conventional and fine. It is a
criticism of the reporting, and it is the single easiest thing to fix in a project like
this one.

## Tests

162 tests pass. New coverage in `tests/test_baselines.py`: keywords being learned rather
than hardcoded, the per class count being respected, the minimum document frequency
excluding rare terms, determinism, the fallback being the most frequent training class,
correct prediction on matching documents, fallback on no match and on a tie, the rule
working inside `cross_val_score` at all, coverage at one, zero and on empty input, the
report loading, random beating majority, the bar being the highest shortcut, the model
clearing the bar, and the keyword rule improving monotonically with more keywords.
