# PLAN: Fraud Email Classification

Multi class text classification that sorts raw email bodies into Normal, Spam and Fraud, where
the Fraud class is advance fee and wire transfer solicitation. Built as a resume portfolio piece
with an honest evaluation protocol rather than a headline accuracy number.

Status: phases 1 to 7 complete and verified. Phases 8 to 11 outstanding.
Created 2026 09 14.
Scope locked with Matthew: three classes, scripts as the primary artefact.

## 1. What this project is

Take three thousand real emails across three categories, clean the text, turn it into numeric
features with CountVectorizer and TF IDF, and train several classical classifiers to tell the
categories apart. Then check honestly whether the model learned the *content* of a fraud email
or merely learned which file each email arrived in.

The fintech relevance is the Fraud class. Advance fee fraud, fake wire instructions and payment
redirection attempts are the email surface of payment fraud and business email compromise. A
bank, a payments processor or a corporate treasury function all run some version of this triage.
That is the story this project tells.

The resume value is not "I trained a Naive Bayes model". Any bootcamp exercise does that. The
value is that this corpus is assembled from two separate source files, which creates a specific
and severe leakage trap, and this project finds it, measures it, and reports the result both with
and without the trap removed.

## 2. Reference solution and attribution

The starting specification is the public repository `SimarjotKaur/Email-Classifier`, which
assembles a three class email corpus and compares SVM, KNN, Naive Bayes, Decision Trees,
Logistic Regression and ensemble methods across CountVectorizer and TF IDF features.

That repository is credited explicitly in the README and in `docs/sources.md`.

We are not copying it. We reuse its corpus construction, port the method to a cleaner pipeline,
and repair the evaluation gaps listed in section 8.

## 3. The brief we are building to

From the project description:

* Use a dataset of manually classified emails across categories
* Preprocess with NLP techniques to identify key features such as the presence of certain words
  or phrases
* Implement several algorithms including Naive Bayes, Random Forest and Support Vector Machine
* Classify emails into categories based on the identified features

All of that is in scope and will be delivered. Sections 8 and 9 add the rigour on top.

## 4. Data decision

One acquisition: `Datasets.zip` from the reference repository. Verified live at 10,754,307 bytes,
content type application/zip.

It contains two source corpora:

Verified contents, as observed at phase 1 rather than as described by the reference README:

<table>
<tr><th>File</th><th>What it is</th><th>Observed rows</th><th>Sampled</th><th>Class it supplies</th></tr>
<tr><td><code>fradulent_emails.txt</code></td><td>CLAIR advance fee fraud mailbox. Upstream spelling preserved</td><td>3,976 messages</td><td>1,000</td><td>Fraud</td></tr>
<tr><td><code>spam_normal_emails.csv</code></td><td>Enron derived corpus, columns text and spam</td><td>5,728 rows, 1,368 spam and 4,360 ham</td><td>1,000 plus 1,000</td><td>Spam and Normal</td></tr>
<tr><td><code>final_dataset.csv</code></td><td>The author's prebuilt corpus. Cross check only, not used for training</td><td>3,000, balanced</td><td>not used</td><td>none</td></tr>
</table>

Why this rather than pulling the two primaries separately: one download, no Kaggle account or API
token, and both corpora arrive in exactly the form the reference consumed them, so any divergence
in our numbers is attributable to our method and not to a different data pull.

Upstream provenance and fallback links live in `docs/sources.md`. Before trusting the zip we
verify its contents, record a checksum, and cross check the row counts against the documented
upstream figures. If they disagree we stop and pull the primaries instead.

Minimum data requirement for this task is only raw email body text plus a category label. No
headers, no sender fields, no metadata.

## 5. What we need to start

Environment:

* Homebrew Python 3.12 at `/opt/homebrew/opt/python@3.12/bin/python3.12`, local `.venv`
  The system Python 3.9 at `/usr/bin/python3` is not used
* Packages: pandas, numpy, scikit learn, nltk, matplotlib, seaborn, joblib, pytest
* NLTK corpora downloaded once: `stopwords`, `punkt`, `wordnet`

Accounts and access:

* None. This is the whole point of the data decision in section 4. No Kaggle login, no API keys,
  no paid services.
* A GitHub repository under `matthewpeters-extended` for the push. Note that the `gh` CLI is not
  installed on this laptop, so the remote gets created in the browser and added by hand.

Compute:

* Trivial. Three thousand documents. Everything runs on the laptop in seconds to low minutes.
  No GPU, no cloud.

## 6. Repository layout

```
fraud-email-classification/
  PLAN.md              this file
  README.md            written LAST, with measured numbers
  WALKTHROUGH.md       the build narrative
  data/raw/            Datasets.zip and its extracted contents, git ignored
  data/interim/        parsed and deduped intermediates
  data/processed/      final_dataset.csv and the train and test splits
  docs/sources.md      attribution and provenance
  docs/SETUP.md        environment reproduction steps
  docs/*_findings.md   per phase findings
  notebooks/NN_*.ipynb exploration, numbered
  reports/figures/     confusion matrices, learning curves, top feature plots
  scripts/             download, build, run_experiment entry points
  src/                 importable modules
  tests/               pytest suite
  requirements.txt     plus a pip freeze lock
```

## 7. Phase plan

**Phase 1. Acquire and verify.** Download `Datasets.zip`, checksum it, extract to `data/raw`,
confirm the two source files and their row counts against section 4.

**Phase 2. Parse the fraud corpus.** Complete. `fradulent_emails.txt` is a single concatenated
mailbox, not a CSV. Split on the mbox envelope pattern, not on the literal string "From r" which
loses two messages. True count is 3,978, of which 3,976 yield usable bodies. Detail in
`docs/phase2_findings.md`.

**Phase 3. Build the corpus.** Complete. Strip the source marker prefix, drop short bodies,
cluster near duplicates with complete linkage, drop label contradictions, then sample one
document per cluster with a fixed seed. 900 per class rather than 1,000, because the spam pool
yields only 994 usable clusters. Detail in `docs/phase3_findings.md`.

**Phase 4. Provenance audit.** Complete. Three isolation probes quantify each shortcut,
four pipeline conditions measure what the fixes cost, and a transfer probe tests reliance.
Formatting alone reaches 0.701 macro F1 with no words at all; removing every shortcut costs
0.0028. Detail in `docs/phase4_findings.md`.

**Phase 5. Exploratory analysis.** Complete. Length profiles, vocabulary overlap and
coverage, type token ratios, and the terms that separate each class after cleaning. The
signal is genuine and interpretable: 419 geography and narrative for fraud, product
vocabulary and deliberate misspellings for spam, the vocabulary of doing a job for normal.
Detail in `docs/phase5_findings.md`.

**Phase 6. Preprocessing pipeline.** Complete. Every step is a scikit learn transformer
inside the pipeline. Nine conditions ablated; none changes macro F1. The chosen pipeline is
normalise, strip markers, remove stopwords, lemmatise, selected on interpretability and
feature space size rather than score. Detail in `docs/phase6_findings.md`.

**Phase 7. Baselines.** Complete. Nine baselines from most frequent class through a learned
keyword rule to the three content free shortcuts, plus the phase 6 pipeline for context.
The trivial floor is random at 0.3355, not majority at 0.1667. The bar to clear is
formatting only at 0.7013. Detail in `docs/phase7_findings.md`.

**Phase 8. Model sweep.** Naive Bayes, Logistic Regression, Linear SVM, Random Forest, KNN and a
voting ensemble, crossed with CountVectorizer and TF IDF. Stratified five fold cross validation
on the training split only. Hyperparameters tuned inside the folds.

**Phase 9. Holdout evaluation and error analysis.** Single final scoring on the untouched test
split. Confusion matrix, per class precision, recall and F1, macro F1. Then read the actual
misclassified emails and write up what they have in common.

**Phase 10. Fraud class deep dive.** Treat Fraud as the positive class. Precision and recall
tradeoff, threshold selection under an asymmetric cost assumption, and a stated view on where
the operating point should sit.

**Phase 11. Write up.** WALKTHROUGH.md, then README.md last with real numbers.

## 8. Defects we inherit and fix

This section is the heart of the project. Each item is a real flaw in the reference approach.

**8.1 Source provenance leakage. Severity: critical. Measured at phase 1.**
The original prediction was that every Normal and Spam body would open with the literal string
`Subject:`, inherited from the source CSV. Measured, that is false. The reference author already
stripped the prefix, and it appears in zero of the 3,000 built rows. That hypothesis is
withdrawn and recorded as D2 in `docs/data_defects.md`.

Two stronger leaks were found in its place, both quantified at phase 1.

*Named entity leak, D3.* The Normal class is real internal Enron corporate mail. The token
`enron` appears in 590 of 1,000 Normal emails and in zero Spam and zero Fraud emails. One token
recovers 59 percent of the class with no false positives, and it says nothing whatsoever about
whether a message is legitimate.

*Formatting fingerprint, D4.* The two source corpora were preprocessed differently before being
combined. Normal and Spam are lowercased with whitespace padded around punctuation. Fraud
retains original casing, ordinary punctuation and MIME artifacts such as `=20`. Any uppercase
character, or any full stop without a preceding space, separates Fraud from the other two
classes almost perfectly.

Fix: normalise all three classes to one text representation before any model sees them, and
strip a blocklist of organisation and person named entities drawn from the Enron source. Report
every headline result twice, before and after, and treat the gap as a reported finding.

**8.2 Vectorizer fitted before the split. Severity: high.**
Fitting CountVectorizer or TF IDF on the full corpus lets test set vocabulary and test set
document frequencies inform the training representation. Fix: every vectorizer lives inside a
`Pipeline` and is fitted on training folds only.

**8.3 Duplicate and near duplicate emails. Severity: high.**
Spam corpora are full of repeated and lightly mutated messages. If a near duplicate pair
straddles the split, the test score is partly a memorisation score. Fix: exact deduplication,
then near duplicate clustering by character n gram similarity, then split by cluster so that no
cluster spans the split.

**8.4 No baseline reported. Severity: high.**
With a balanced three class corpus the majority class baseline is 33.3 percent. A reported
accuracy means nothing without that number beside it. Fix: baselines are phase 7 and appear in
the same table as every model.

**8.5 Accuracy as the headline metric. Severity: medium.**
On a balanced three class problem accuracy hides which class is failing, and the Fraud class is
the only one anybody cares about. Fix: macro F1 as the headline, full per class breakdown, and a
confusion matrix always shown.

**8.6 Artificial class balance. Severity: medium.**
A real inbox is not one third fraud. Sampling 1,000 per class inflates fraud precision relative
to deployment. Fix: report a second evaluation at a realistic prevalence assumption and state the
assumption explicitly.

**8.7 Unseeded sampling and a single split. Severity: medium.**
The reference samples randomly with no seed and scores on one split, so its numbers are not
reproducible and carry no error bar. Fix: fixed seeds throughout, cross validation for model
selection, and a reported spread rather than a point estimate.

**8.8 No calibration. Severity: low.**
Threshold selection for the fraud class needs probabilities that mean something. Fix: reliability
check on the chosen model, and calibration applied if it is badly off.

## 9. Evaluation protocol

Decided in advance so it cannot be tuned after seeing results.

* Split: stratified, 80 percent train and 20 percent test. Every document comes from a
  distinct near duplicate cluster, so no duplicate can straddle the split. Seed 20260914,
  recorded in `data/processed/phase3_build_report.json`
* Test split is touched exactly once, in phase 9
* Model selection: stratified five fold cross validation on the training split, macro F1
* Headline metric: macro F1 on the holdout, always quoted next to the trivial floor of
  0.3355, which is random guessing, and next to the formatting only probe at 0.7013, which
  is the real bar a model has to clear
* Secondary: per class precision, recall and F1, confusion matrix, and Fraud class recall at a
  fixed precision target. Note two separate traps here. The 0.333 figure often quoted is
  accuracy, not macro F1. And the majority class score of 0.1667 is not the macro F1 floor
  either, because random guessing scores 0.3355. Quote random as the floor
* Everything reported twice, once on the raw corpus and once with source marker tokens removed

Success is not a high number. Success is a defensible number with the leakage quantified and the
failure cases described.

## 10. Resume framing

One line for the resume, to be finalised with real figures once phase 11 lands:

> Built a three class email triage model separating legitimate mail, spam and advance fee fraud
> from 3,000 messages, identifying and quantifying a corpus construction leak that inflated
> apparent accuracy, and reporting macro F1 against a stated majority baseline.

The interview story is section 8.1. Finding that the model was reading the file it came from
rather than the content is the kind of thing an interviewer actually wants to hear about.

## 11. Decisions taken

1. Repository: `matthewpeters-extended/fraud-email-classification`. Pushed per phase via
   `scripts/push.sh`.
2. Three classes: FRAUD, SPAM, NORMAL. A fourth class would need another corpus. Shipping three
   classes well instead.
3. Scripts as the primary artefact, `src/` modules plus entry points under `scripts/`. Notebooks
   are for exploration only and are not the deliverable.

## 12. Phase results

### Phase 1

Complete and verified. Archive checksummed, contents corrected against the reference README,
counts reproduced, and five data defects logged in `docs/data_defects.md` including two critical
leaks found before any model was trained. Detail in D0 through D5.

### Phase 2

Complete and verified. Mailbox parsed to 3,976 usable messages from 3,978 envelopes, 23 tests
passing. Every message count previously on record was wrong, including the reference README's
4,075 and the commonly cited 3,977. Quoted printable decoding resolved 99.9 percent of the MIME
artifact half of D4. Two new defects logged: D6 label noise at 0.1 percent, accepted, and D7
duplication at 17.1 percent exact matches, which is the measurement confirming 8.3 and changes
how phase 3 must sample. Detail in `docs/phase2_findings.md`.

### Phase 3

Complete and verified. 2,700 document corpus at 900 per class, 2,160 train and 540 test, seed
20260914, zero near duplicate clusters straddling the split, 52 tests passing.

Near duplicate matching raises fraud duplication from the 17.1 percent exact figure to 38.4
percent, and the combined pool is 28.6 percent redundant. A balanced 1,000 per class turns out
to be unreachable after honest deduplication, because the spam pool holds only 994 usable
clusters, so the reference's corpus size depended on leaving duplicates in.

One defect was our own: the first clustering pass used single linkage and chained unrelated
documents through shared boilerplate, producing a 232 member cluster whose minimum pairwise
similarity was 0.563. Replaced with two stage complete linkage. Detail in
`docs/phase3_findings.md`.

### Phase 4

Complete and verified. 84 tests passing. Section 8.1 is revised by measurement, and the
revision runs against what we predicted.

The shortcuts are real. Formatting alone, seven numeric features and no words, reaches 0.7013
macro F1 against a majority baseline of 0.1667. Thirty four provenance tokens alone reach
0.5763 with two thirds of documents containing none of them. Length alone reaches 0.4386.

But removing all of it costs 0.0028 macro F1, less than the fold to fold standard deviation.
The leaks are redundant with the content, not additive to it, so the reference's headline was
never materially inflated. Reported as a negative result because that is what the measurement
says.

The finding worth more than the score: fitting with the shortcuts available and scoring
without them drops macro F1 by 0.0078, so a model does lean on them, and a holdout drawn from
the same corpus cannot detect it, because the leak and the signal agree. Only an ablation can.
Logged as D13.

One defect was ours again: `am` was blocked as a timestamp marker despite being the English
verb that opens almost every advance fee email, which overstated the measured leak by 0.32
macro F1. Logged as D12 and guarded by a test. Detail in `docs/phase4_findings.md`.

Revised at phase 5: the blocklist grew to 42 tokens, the markers only probe rose to 0.6082,
and the total attributable to provenance halved to 0.0014.

### Phase 5

Complete and verified. 109 tests passing.

The content signal is genuine and interpretable. Fraud separates on the geography and
narrative furniture of advance fee fraud, spam on product vocabulary and on deliberate
misspellings such as `shlpplng` and `oniine` that exist to defeat keyword filters, and normal
on the vocabulary of doing a job. No company name, employee name or header fragment survives
into the ranking, so the phase 4 cleanup held.

Fraud is the most templated class, type token ratio 0.0401 against spam's 0.0905, which
independently confirms the phase 3 finding that 38.4 percent of the fraud corpus is
redundant. Two methods at two phases agree that this is a genre written from scripts.

Two corrections to earlier claims. D11 is amplified by the cleanup rather than mitigated: the
fraud to spam median length ratio is 3.32 raw and 4.35 cleaned, because the Enron rows pad
whitespace around punctuation so a raw word count credits them with punctuation as words. And
D9 is downgraded to partially supported: the confusion matrix backs it strongly, vocabulary
Jaccard contradicts it, and asymmetric coverage backs it, because shared vocabulary and
confusability are not the same measurement.

One defect was ours for the third time. The phase 5 leakage check asserted that no
blocklisted token appeared in the discriminative ranking, which cannot fail because the
pipeline strips them first. It passed, reported success, and missed eight genuine provenance
markers sitting in plain sight in its own output. Logged as D14. Detail in
`docs/phase5_findings.md`.

### Phase 6

Complete and verified. 143 tests passing. The phase level result is negative and worth
stating plainly: **no preprocessing step changes macro F1.** The spread across nine ablated
conditions is 0.0042 against a mean fold standard deviation of 0.0049, so every condition
sits inside the noise.

The chosen pipeline is normalise, strip markers, remove stopwords with the plain list, and
lemmatise. It was selected on stated secondary criteria, readable features for phases 8 and
9 plus a 15.5 percent smaller vocabulary and 43 percent fewer tokens, not on score.
Stemming was rejected despite the smallest vocabulary because `beneficiari` and `busi`
would make the later write ups unreadable. The configuration lives in
`src.preprocess.CHOSEN_CONFIG` so later phases cannot drift from it.

A hypothesis of ours failed: holding pronouns such as `i`, `am` and `my` back from the
stopword list, on the theory that first person register distinguishes advance fee fraud,
scores 0.9657 against 0.9676 for the plain list. No evidence for it, and the direction is
opposite to the prediction. Logged as D17.

Two defects caught before they reached a result. D15: the lemmatiser produced `wa`, `ha`,
`u`, `a` and turned `mrs` into `mr`, which would have silently destroyed a term appearing
in 126 training documents at 96 percent fraud purity. D16: the transformers never set a
fitted attribute, so `fit_transform` worked and every cross validation passed, while the fit
then transform pattern phase 9 uses to score the test set would have raised NotFittedError.

D11 is closed: length only is 0.4698 on cleaned text against 0.4386 on raw, confirming the
phase 5 prediction. Detail in `docs/phase6_findings.md`.

### Phase 7

Complete and verified. 162 tests passing. The canonical baseline table is in
`docs/phase7_baselines.json` and is loaded by later phases through
`src.baselines.load_baselines()`, so no figure can drift by being retyped.

<table>
<tr><th>Baseline</th><th>Macro F1</th></tr>
<tr><td>most frequent class</td><td>0.1667</td></tr>
<tr><td>stratified random</td><td>0.3355</td></tr>
<tr><td>keyword rule, 25 words per class</td><td>0.5737</td></tr>
<tr><td>provenance markers only</td><td>0.6082</td></tr>
<tr><td><b>formatting only, the bar to clear</b></td><td><b>0.7013</b></td></tr>
<tr><td>phase 6 pipeline, for context</td><td>0.9662</td></tr>
</table>

The keyword rule learns its words inside each fold rather than being hand picked, which is
what makes it a baseline rather than a cheat, and it is reported with coverage: at 25 words
per class it fires on 51.5 percent of documents and reaches 0.5737. Seventy five words get
most of the way to the marker probe.

A correction we owed: phases 4 to 6 quoted 0.1667 as the floor, but random guessing scores
0.3355, more than double. Macro F1 punishes a single class predictor harder than it punishes
chance. Phase 4's claim that formatting alone is "4.21 times the majority baseline" is 2.09
times against the correct floor. Logged as D18. Detail in `docs/phase7_findings.md`.
