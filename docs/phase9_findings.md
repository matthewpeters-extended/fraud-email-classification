# Phase 9 findings: the holdout, and what the errors actually are

The test split has been untouched since phase 3 built it. Every figure in phases 4 through 8
came from cross validation on the training split.

Entry point `scripts/evaluate_holdout.py`, logic in `src/evaluate.py`, tests in
`tests/test_evaluate.py`. Numbers in `docs/phase9_holdout.json`, every error in
`docs/phase9_errors.json`, evaluation ledger in `docs/holdout_ledger.json`, figure in
`reports/figures/phase9_confusion.png`.

Two configurations were scored, both chosen at phase 8 before this split was read. The
headline stays the headline regardless of which scores higher here, because switching after
the fact would be selecting on the test set.

## Result

<table>
<tr><th>Configuration</th><th>Test macro F1</th><th>Accuracy</th><th>Errors in 540</th><th>Nested CV estimate</th></tr>
<tr><td><b>Headline: soft voting ensemble, tfidf</b></td><td><b>0.9684</b></td><td>0.9685</td><td>17</td><td>0.9740</td></tr>
<tr><td>Secondary: multinomial naive Bayes, tfidf</td><td>0.9628</td><td>0.9630</td><td>20</td><td>0.9721</td></tr>
</table>

Per class, headline configuration:

<table>
<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
<tr><td>FRAUD</td><td>0.9570</td><td>0.9889</td><td>0.9727</td><td>180</td></tr>
<tr><td>SPAM</td><td>0.9657</td><td>0.9389</td><td>0.9521</td><td>180</td></tr>
<tr><td>NORMAL</td><td>0.9832</td><td>0.9778</td><td>0.9805</td><td>180</td></tr>
</table>

Confusion matrix, rows are true and columns predicted:

<table>
<tr><th></th><th>FRAUD</th><th>SPAM</th><th>NORMAL</th></tr>
<tr><td><b>FRAUD</b></td><td>178</td><td>2</td><td>0</td></tr>
<tr><td><b>SPAM</b></td><td>8</td><td>169</td><td>3</td></tr>
<tr><td><b>NORMAL</b></td><td>0</td><td>4</td><td>176</td></tr>
</table>

Against the baselines: random guessing 0.3355, a seventy five word keyword rule 0.5737,
provenance markers only 0.6082, formatting only 0.7013. The headline clears the bar by
0.2671. The baselines are cross validated on train, so this is indicative rather than a like
for like comparison on the same documents.

## The number that matters for deployment

Macro F1 treats every error as equivalent. In use they are not. A fraud email filed as spam
still leaves the inbox and the user is protected. A fraud email filed as normal lands in
front of them.

<table>
<tr><th>Failure mode</th><th>Count</th></tr>
<tr><td><b>Fraud reaching the inbox, filed as normal</b></td><td><b>0</b></td></tr>
<tr><td>Fraud caught but misfiled as spam</td><td>2</td></tr>
<tr><td>Legitimate mail destroyed as fraud</td><td>0</td></tr>
<tr><td>Legitimate mail misfiled as spam</td><td>4</td></tr>
<tr><td>Spam misfiled</td><td>11</td></tr>
</table>

Not one of the 180 fraud emails in the holdout was classified as legitimate mail, and not one
of the 180 legitimate emails was classified as fraud. The FRAUD against NORMAL boundary
produces zero errors in 360 opportunities.

That is the shape of error you want and it is not an accident of this run. Phase 4 measured
the same structure on the training folds, where FRAUD against NORMAL accounted for 3 percent
of errors, and D9 explained why: advance fee fraud is a kind of spam, so it sits close to
spam and far from a colleague's meeting invitation. The holdout confirms it more sharply.

## The holdout agrees with cross validation, and my first test of that was wrong

The headline scores 0.0056 below its nested cross validation estimate, and the secondary
0.0094 below. The first version of this script compared those gaps against the cross
validation standard deviation of 0.0027 and reported both as **outside one standard
deviation**, which reads as a discrepancy needing explanation.

That comparison is invalid. The cross validation standard deviation measures how much the
estimate moved between training folds. It says nothing about how precisely 540 documents pin
down a score. Bootstrapping the test set, 5,000 resamples:

<table>
<tr><th>Configuration</th><th>Test macro F1</th><th>95 percent interval</th><th>Width</th><th>Nested estimate inside</th></tr>
<tr><td>Headline</td><td>0.9684</td><td>0.9526 to 0.9819</td><td>0.0293</td><td>yes</td></tr>
<tr><td>Secondary</td><td>0.9628</td><td>0.9456 to 0.9777</td><td>0.0321</td><td>yes</td></tr>
</table>

The interval is roughly eleven times wider than the fold standard deviation, and both nested
estimates sit comfortably inside it. So the honest conclusion is that the holdout agrees with
cross validation within sampling error, and the apparent discrepancy was an artifact of
holding a 540 document score to a yardstick built for something else.

Worth stating as a general point, because it is a common error: a cross validation standard
deviation is not a confidence interval on a test score.

## What the errors actually are

Phase 9 was set the task of reading the misclassified emails rather than counting them. All
seventeen were read and adjudicated. The verdicts are recorded in
`ADJUDICATION` in `scripts/evaluate_holdout.py` so that a reader can disagree with any
individual line, and the script fails if an error has no recorded verdict.

<table>
<tr><th>Verdict</th><th>Count</th><th>Share</th></tr>
<tr><td><b>Label wrong, the model was right</b></td><td><b>10</b></td><td>58.8 percent</td></tr>
<tr><td>Model wrong</td><td>5</td><td>29.4 percent</td></tr>
<tr><td>Ambiguous</td><td>2</td><td>11.8 percent</td></tr>
</table>

**Ten of seventeen errors are the model being correct and the corpus label being wrong.**
Every one is an advance fee scam that lives in the Enron spam corpus and is therefore
labelled SPAM, which the model reads as FRAUD. A sample of what it was marked wrong for:

```
[labelled SPAM, predicted FRAUD]
  "investment offer from joseph otisa ... the branch manager of allstates
   trust bank of nigeria plc lagos state branch"

[labelled SPAM, predicted FRAUD]
  "request for assistance barrister adewale coker chambers legal
   practitioners / notary public ... lagos nigeria"

[labelled SPAM, predicted FRAUD]
  "de la part des enfants ... abidjan republique de cote d ' ivoire
   proposition d ? affaires"

[labelled SPAM, predicted NORMAL]
  "out of office autoreply : ... i am on vacation week 29 + 30 + 31"
```

This is D9 arriving as a measured quantity rather than a caveat. Phase 3 found five clusters
where the two source corpora disagreed about the same email and dropped them. Those five were
the cases where the *same message* appeared in both files. The ten errors here are the larger
population that deduplication could never catch: different messages of the same kind, filed
under different labels by two different corpus builders.

The five genuine model errors are more instructive than the score:

<table>
<tr><th>True</th><th>Predicted</th><th>Why it failed</th></tr>
<tr><td>FRAUD</td><td>SPAM</td><td>a US Bank phishing alert, which is fraud by intent but reads in the register of bulk mail</td></tr>
<tr><td>NORMAL</td><td>SPAM</td><td>a bond reference containing the word <code>mortgage</code>, a top spam term</td></tr>
<tr><td>NORMAL</td><td>SPAM</td><td>an internship offer containing the word <code>offer</code></td></tr>
<tr><td>NORMAL</td><td>SPAM</td><td>internal mail about a trial software subscription, which reads as marketing</td></tr>
<tr><td>SPAM</td><td>NORMAL</td><td>a genuine unsolicited marketing approach written in a personal register</td></tr>
</table>

Three of the five are legitimate business mail containing a single strong spam term. That is
the failure mode of a bag of words model stated plainly, and it is the argument for phrase
level or contextual features rather than for more hyperparameter search.

The one genuine fraud failure is the interesting one: a bank phishing message, filed as spam.
It is fraud by intent and spam by form. Notably the corpus contains almost no phishing, since
the CLAIR collection predates its rise, so the model had little chance to learn it. That is a
limitation of the training data and it belongs in the README as one.

### An indication of the ceiling, and not a score

If the ten mislabelled documents were corrected, errors would fall from seventeen to five,
plus two arguable. That points at a corpus ceiling somewhere near 0.99 rather than 1.00, and
it means the residual difference between the top models in phase 8 is smaller than the label
noise they are being ranked on.

Stated carefully: **that is not a result and it is not the headline.** The headline macro F1
remains 0.9684 exactly as measured. No model, hyperparameter or threshold was chosen using
the adjudication, and reporting a label corrected score as the outcome would be exactly the
selection on the test set this phase exists to avoid.

## The misspelling question phase 5 left open

Phase 5 flagged that the spam class leaned on a family of deliberate misspellings,
`shlpplng`, `oniine`, `miiiion`, `prlces` and `successfull`, each appearing in only 20 to 30
training documents, and asked whether they were doing disproportionate work.

<table>
<tr><th>Condition</th><th>Test macro F1</th></tr>
<tr><td>Family present</td><td>0.9684</td></tr>
<tr><td>Family removed from train and test</td><td>0.9665</td></tr>
</table>

Present in 63 of 2,160 training and 12 of 540 test documents. Removing them costs 0.0019,
which is well inside the bootstrap interval. Answer: no. They are strong individually and
redundant collectively, which is the same pattern phase 4 found for the provenance markers.

## The holdout ledger

Every evaluation appends to `docs/holdout_ledger.json` with a fingerprint of the
configuration. The current state: **12 evaluations across 2 distinct configurations.**

The second number is the one that matters. The twelve entries are reruns of the reporting code
as the bootstrap interval and the error adjudication were added, and the ledger shows every
rerun of a given fingerprint recording a byte identical score, which also demonstrates
determinism. A third distinct configuration appearing there would mean something was chosen
using the test split, and it would be visible in the repository rather than resting on the
author's assurance. A test asserts the count is two.

This is the mechanism that makes the single holdout claim checkable instead of merely stated.

## Tests

214 tests pass. New coverage in `tests/test_evaluate.py`: parameter recovery from the sweep
report, every model the sweep can select being rebuildable, the winning pipeline applying its
tuned parameters and running end to end, confusion arithmetic and its diagonal, boundary
errors summing both directions and totalling the off diagonal, the deployment cost split
distinguishing fraud caught as spam from fraud reaching the inbox, error records capturing
only mistakes, fingerprint stability under key reordering and sensitivity to parameter
changes, the ledger holding exactly two distinct configurations, and every rerun of a
fingerprint recording an identical score.
