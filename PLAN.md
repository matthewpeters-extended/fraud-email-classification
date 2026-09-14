# PLAN: Fraud Email Classification

Multi class text classification that sorts raw email bodies into Normal, Spam and Fraud, where
the Fraud class is advance fee and wire transfer solicitation. Built as a resume portfolio piece
with an honest evaluation protocol rather than a headline accuracy number.

Status: phase 1 complete and verified. Phases 2 to 11 outstanding.
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

**Phase 2. Parse the fraud corpus.** `fraudulent_emails.txt` is a single concatenated mailbox
style text file, not a CSV. Split it into individual messages, strip the headers, keep the body.
Record how many messages parse cleanly and how many are malformed.

**Phase 3. Build the corpus.** Sample 1,000 per class with a fixed random seed, assemble
`final_dataset.csv` with columns for text and label, and write a data dictionary.

**Phase 4. Data defects audit.** The most important phase. Deduplication, near duplicate
detection, and the source marker investigation described in section 8. Findings go to
`docs/data_defects.md`.

**Phase 5. Exploratory analysis.** Length distributions per class, vocabulary overlap between
classes, and the most discriminative terms. Establishes whether the classes are genuinely
separable on content.

**Phase 6. Preprocessing pipeline.** Lowercasing, punctuation and digit removal, stopword
removal, lemmatisation. Implemented as a scikit learn transformer so it lives inside the
pipeline and cannot leak.

**Phase 7. Baselines.** Majority class, stratified random, and a single keyword rule. Nothing
gets reported later without these sitting next to it.

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

* Split: stratified, 80 percent train and 20 percent test, split by near duplicate cluster,
  seed fixed and recorded
* Test split is touched exactly once, in phase 9
* Model selection: stratified five fold cross validation on the training split, macro F1
* Headline metric: macro F1 on the holdout, always quoted next to the 33.3 percent majority
  baseline
* Secondary: per class precision, recall and F1, confusion matrix, and Fraud class recall at a
  fixed precision target
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

## 12. Phase 1 result

Complete and verified. Archive checksummed, contents corrected against the reference README,
counts reproduced, and five data defects logged in `docs/data_defects.md` including two critical
leaks found before any model was trained. Detail in D0 through D5.
