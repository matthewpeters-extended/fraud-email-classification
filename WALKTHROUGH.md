# Walkthrough: how this was built, including the wrong turns

`README.md` reports what the project found. This document records how it got there, in
order, with the mistakes left in. Eleven phases, each one committed separately, each one
written up in `docs/`.

The reason to keep the mistakes is that they are the most transferable part. Seven of the
twenty two logged defects were in our own method rather than in the data, and several of
them would have produced a confident, wrong, publishable number.

## Phase 1: acquire and verify

The reference repository ships its data as `Datasets.zip`, which is why the dataset looks
missing from the file listing. The download is scripted rather than committed, checksummed
at `7d3d07a1bc58ecbdcf03533515e5256d61dda4fd44a4902bd35fdf0910db22fe`, and its zip members
are validated against path traversal and symlinks before extraction.

The verification script failed on its first run, which is what it was for. The archive
contains three files under different names from the ones the reference README describes,
including the author's own misspelling of `fradulent_emails.txt`, and it also ships a
prebuilt corpus the README never mentions.

Probing for provenance leaks immediately found two, and the first prediction we made was
wrong. We had expected every Normal and Spam body to open with the literal string
`Subject:`, inherited from the source CSV. It appears in zero of the 3,000 prebuilt rows:
the reference author had already stripped it. Credit where due, hypothesis withdrawn.

What the probe found instead was worse. The token `enron` appears in 590 of 1,000 Normal
emails and zero of the others, and the two source corpora were preprocessed so differently
that casing alone separates Fraud from the rest.

## Phase 2: parse the fraud mailbox

`fradulent_emails.txt` is a 17.3 MB concatenated mailbox, not a CSV. Splitting it on the
literal prefix `From r` gives 3,976 messages, which is the figure most code uses. It is
wrong. Two envelope lines in this corpus do not carry that prefix, at lines 72,425 and
285,198, so the naive rule welds two messages onto their predecessors and loses two real
ones. The true count is **3,978**.

Every figure on record was wrong: the reference README claims 4,075, the commonly cited
CLAIR figure is 3,977, the naive split gives 3,976. Of the 3,978, exactly 3,976 yield a
usable body, which coincidentally equals the naive figure. We noted that the agreement is a
coincidence rather than a validation, because someone would otherwise read it as one.

Parsing through the standard library `email` module rather than as plain text decoded
quoted printable payloads properly, dropping `=20` occurrences from 6,423 to 9.

A test failed here and it was the test harness, not the parser: `textwrap.dedent` silently
left the synthetic mailbox envelopes indented once a multiline body introduced an unindented
line, so a working parser looked broken. There is now a test guarding the harness.

## Phase 3: build the corpus

The first clustering implementation used connected components, which is single linkage. It
reported 32.3 percent redundancy and produced a 232 document cluster.

Checking pairwise similarity inside the largest clusters showed most of them were not
duplicate groups at all but chains. Shared boilerplate linked A to B to C where A and C were
unrelated. One 85 member Enron cluster had a minimum pairwise cosine of **0.045**. Accepting
it would have discarded 84 genuinely distinct emails.

Replaced with complete linkage, where every pair inside a cluster must clear the threshold,
implemented in two stages so the dense matrix of 47 million pairs is never built. Largest
cluster fell from 232 to 112.

Then an assertion caught something real. Five clusters carry two labels, every one FRAUD
against SPAM, because the same advance fee campaigns reached Enron employees and both corpora
collected them. Those clusters produce one representative per label, which would have put
near identical text on both sides of the train and test split under contradictory labels.

Deduplication also made the reference's corpus size unreachable. The spam pool of 1,368 rows
reduces to 994 usable clusters, so a balanced 1,000 per class can only be built by leaving
copies in. The corpus is 900 per class, 2,160 train and 540 test, with zero clusters spanning
the split.

## Phase 4: quantify the leaks

Three probes, each given one artifact and no access to content. Whatever a probe scores, it
scores without reading the message.

<table>
<tr><th>Probe</th><th>Macro F1</th></tr>
<tr><td>Formatting only, seven numeric features, zero words</td><td>0.7013</td></tr>
<tr><td>Provenance markers only, 42 curated tokens</td><td>0.6082</td></tr>
<tr><td>Document length only, one feature</td><td>0.4386</td></tr>
</table>

Then the result that inverted our prediction. `PLAN.md` 8.1 expected the gap between raw and
cleaned scores to be the headline finding. It is **0.0014**, smaller than the fold to fold
standard deviation. The leaks are real and individually powerful but redundant with the
content, so removing them costs nothing.

That could mean the model ignored them or that content covered for them, so we fit with the
markers available and scored on input where they were removed. Macro F1 fell 0.0078. The
model does lean on the shortcuts, and nothing in a conventional train and test split
penalises it, because the leak and the genuine signal agree on this corpus. **A leak that
correlates with the true signal is invisible to holdout validation.** Only an ablation finds
it.

Our own error here was instructive. The blocklist included `am` for AM and PM timestamps. It
is also the English verb that opens almost every advance fee email, and it had never appeared
in the skew evidence. That single wrong token inflated the measured leak from 0.576 to
**0.894**, which is a third of a macro F1 point, and halved the count of documents containing
no marker. Ablation probes are sensitive to blocklist errors in both directions.

## Phase 5: characterise the content

After cleaning, the terms that separate each class are interpretable: West African geography
and dormant accounts for fraud, product vocabulary and deliberate misspellings such as
`shlpplng` and `oniine` for spam, and the vocabulary of doing a job for normal mail.

Fraud is the most templated class, type token ratio 0.0401 against spam's 0.0905, which
independently confirms the phase 3 measurement that 38.4 percent of the fraud corpus is
redundant. Two methods at two phases, same conclusion.

The leakage check in this phase was worthless and passed anyway. It asserted that no
blocklisted token appeared in the discriminative ranking. The pipeline strips those tokens
before the ranking is computed, so their absence was guaranteed by construction. It printed
"confirmed: 0 of 75 ranked terms are blocklisted" while eight genuine provenance markers sat
in its own output, including `forwarded` at 183 documents and `corp` at 111, both missed by
phase 4 for being only 89 and 90 percent class pure.

Fixed by inverting the question: every ranked term must now carry a recorded verdict, and the
script exits non zero listing any that does not. **A verification that can only pass is worse
than none, because it manufactures confidence.**

## Phase 6: choose the preprocessing by ablation

Nine conditions. The spread across all of them is 0.0042 against a mean fold standard
deviation of 0.0049, so nothing is distinguishable and macro F1 cannot choose.

The pipeline was therefore selected on stated secondary criteria: readable features, because
phases 8 and 9 report top features and read misclassified emails and stemming returns
`beneficiari` and `busi`; a 15.5 percent smaller vocabulary; and 43 percent fewer tokens per
document. Not on score.

Another hypothesis of ours failed. `SIGNAL_STOPWORDS` held back `i`, `am`, `my` and similar,
on the theory that first person register distinguishes advance fee fraud. It scores 0.9657
against 0.9676 for the plain list. No evidence, and the direction is opposite to the
prediction.

Two defects caught before they touched a result. The lemmatiser produced `wa`, `ha`, `u`, `a`
and turned `mrs` into `mr`, which would have silently destroyed a term appearing in 126
training documents at 96 percent fraud purity. And the transformers never set a fitted
attribute, so `fit_transform` worked and every cross validation run passed clean while the fit
then transform pattern phase 9 uses to score the test set would have raised `NotFittedError`.

## Phase 7: baselines

Nine baselines, and a correction we owed. Phases 4 to 6 quoted 0.1667 as the macro F1 floor.
Random guessing scores **0.3355**, more than double, because macro F1 punishes a single class
predictor harder than it punishes chance. Phase 4's claim that formatting alone is "4.21 times
the majority baseline" is 2.09 times against the correct floor. A multiple against a badly
chosen denominator flatters every result.

The keyword rule learns its words inside each fold rather than being hand picked, which is the
difference between a baseline and a cheat. Reported with coverage, because at five words per
class it declines to decide on 77 percent of documents, so its score is mostly fallback. At
twenty five words per class, seventy five words total, it reaches **0.5737**.

## Phase 8: the model sweep

Fourteen configurations scored by nested cross validation. Winner is a soft voting ensemble
on tfidf at 0.9740, with multinomial naive Bayes at 0.9721 inside the noise for a sixth of
the compute.

`LinearSVC` never converged on count features at the default 1,000 iteration cap, for all
eight grid points, so a quarter of the sweep was reporting an arbitrary point on an
optimisation path rather than a fitted model. scikit learn emits one warning per fit, so
hundreds of identical lines scrolled past inside the search, and they were the only evidence.

Our optimism measurement came out **negative**, which is impossible for a bias that only
inflates. The nested arm trained on 80 percent of the data and the non nested arm on 67
percent, so one effect we had not intended to measure cancelled the one we had. When a
measured effect has the wrong sign, suspect the experiment before concluding the effect is
absent. Corrected, optimism is +0.0007 on average.

## Phase 9: the holdout, read once

Headline macro F1 **0.9684** on 540 documents untouched since phase 3, with a bootstrap
interval of 0.9526 to 0.9819. Both nested estimates fall inside their intervals.

The first version of this script compared the holdout gap against the cross validation
standard deviation of 0.0027 and reported both models as outside one standard deviation,
which reads as a discrepancy requiring explanation. A fold standard deviation is not a
confidence interval on a 540 document score; the bootstrap interval is eleven times wider.

Then the best finding in the project, and it came from reading the errors rather than counting
them. **Ten of seventeen errors are the model being right and the label being wrong**: advance
fee scams that live in the Enron spam corpus and are therefore labelled SPAM. Phase 3 dropped
five clusters where the same message appeared in both files; these are the larger population
of different messages of the same kind, filed differently by two corpus builders, which no
deduplication can reach.

## Phase 10: the fraud class as a detector

Threshold selected on training folds under stated costs. It barely matters: 0.40 against the
default 0.50, worth 3.5 cost units out of 150, and the same threshold is chosen across a fifty
fold range of cost ratios.

The result that matters is the prevalence correction. Precision falls from 0.9570 on the
balanced corpus to 0.1820 at 0.1 percent fraud prevalence, with recall and both false alarm
rates unchanged. That is the case against quoting precision from a balanced benchmark. But
every false alarm lands on spam and none on legitimate mail, so the collapse describes a
detector filing junk as the wrong kind of junk. Both halves have to be said together.

## Phase 11: the write up

`WALKTHROUGH.md` first, then `README.md` with the measured numbers. The README leads on the
deployment claim rather than on a precision figure, and states the prevalence assumption
behind every precision number it quotes.

## Reproducing all of it

```bash
source .venv/bin/activate && python scripts/download_data.py
```

```bash
source .venv/bin/activate && python scripts/parse_fraud.py && python scripts/build_dataset.py
```

```bash
source .venv/bin/activate && python scripts/audit_markers.py && python scripts/run_eda.py
```

```bash
source .venv/bin/activate && python scripts/run_preprocessing_ablation.py && python scripts/run_baselines.py
```

```bash
source .venv/bin/activate && python scripts/run_model_sweep.py
```

```bash
source .venv/bin/activate && python scripts/evaluate_holdout.py && python scripts/fraud_deep_dive.py
```

Environment setup is in `docs/SETUP.md`. Every seed is fixed at 20260914 and every phase
writes a machine readable report into `docs/` or `data/`, so any number in the README can be
traced to the run that produced it.
