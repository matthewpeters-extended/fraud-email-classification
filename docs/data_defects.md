# Data defects log

Findings that affect how the data may be used. Each entry carries an id, the evidence, and
the decision taken. Referenced from `PLAN.md` section 8 and from the README.

Machine readable counts for phase 1 live in `data/raw/manifest.json`.

## Phase 1: acquire and verify

Archive fetched 2026 09 14 from the reference repository.

* URL: https://raw.githubusercontent.com/SimarjotKaur/Email-Classifier/master/Datasets.zip
* Size: 10,754,307 bytes, matching the documented figure
* SHA256: `7d3d07a1bc58ecbdcf03533515e5256d61dda4fd44a4902bd35fdf0910db22fe`

Zip members were validated against absolute paths, parent traversal and symlinks before
extraction. All three members passed.

### D0. Archive contents do not match the reference README. Severity: low, resolved.

The README describes `fraudulent_emails.txt` and `emails.csv`. The archive actually contains
three files under different names. The first verification run failed on this and was correct
to do so.

<table>
<tr><th>Actual member</th><th>Bytes</th><th>What it is</th></tr>
<tr><td><code>spam_normal_emails.csv</code></td><td>8,954,755</td><td>Enron derived corpus, columns text and spam, 5,728 rows, 4,360 ham and 1,368 spam</td></tr>
<tr><td><code>fradulent_emails.txt</code></td><td>17,344,435</td><td>CLAIR advance fee fraud mailbox, 335,027 lines. Note the upstream spelling of the filename</td></tr>
<tr><td><code>final_dataset.csv</code></td><td>5,213,165</td><td>The author's prebuilt corpus, columns Email, Label, Length, 3,000 rows, exactly 1,000 each of FRAUD, SPAM and NORMAL</td></tr>
</table>

Decision: filenames corrected in `scripts/download_data.py` and in `docs/sources.md`. The
upstream misspelling of `fradulent_emails.txt` is preserved rather than silently renamed, so
the download stays reproducible.

### D1. Fraud corpus message count is overstated by the reference. Severity: low, resolved.

The reference README claims 4,075 fraud emails. We observe 3,976 lines beginning with the
mailbox marker `From r`. The canonical CLAIR corpus is commonly cited at 3,977.

Decision: the reference figure is treated as unsubstantiated and is not used. The verification
script accepts 3,976 or 3,977 and records the reason inline. The residual difference of one
message is a parsing boundary question, handed to phase 2, which will reconcile marker lines
against successfully parsed message bodies.

Corroborating count: 3,986 lines beginning `Subject:`, which exceeds the marker count because
quoted and forwarded material inside message bodies also carries subject lines. Phase 2 must
therefore split on the mailbox marker and not on subject lines.

### D2. The Subject prefix leak does not survive into the built corpus. Severity: none. Hypothesis withdrawn.

`PLAN.md` section 8.1 predicted that every Normal and Spam body would open with the literal
string `Subject:`, inherited from the source CSV, giving a single token that identifies the
source file. The raw source does carry that prefix on every row.

Measured on the built corpus, the prefix is absent from all three classes:

<table>
<tr><th>Class</th><th>Bodies opening with Subject:</th></tr>
<tr><td>FRAUD</td><td>0 of 1,000</td></tr>
<tr><td>NORMAL</td><td>0 of 1,000</td></tr>
<tr><td>SPAM</td><td>0 of 1,000</td></tr>
</table>

Decision: the reference author stripped the prefix during corpus construction. Credit where it
is due. This specific hypothesis is withdrawn. Two stronger leaks were found in its place.

### D3. Named entity leak. The token "enron" identifies the Normal class. Severity: critical, open.

The Normal emails are real internal Enron corporate mail. The Spam and Fraud emails are not.
The company name therefore acts as a class label.

<table>
<tr><th>Token</th><th>NORMAL</th><th>SPAM</th><th>FRAUD</th></tr>
<tr><td>enron</td><td>590 of 1,000</td><td>0 of 1,000</td><td>0 of 1,000</td></tr>
<tr><td>nigeria</td><td>0 of 1,000</td><td>12 of 1,000</td><td>214 of 1,000</td></tr>
<tr><td>urgent</td><td>9 of 1,000</td><td>19 of 1,000</td><td>361 of 1,000</td></tr>
<tr><td>$</td><td>75 of 1,000</td><td>283 of 1,000</td><td>746 of 1,000</td></tr>
</table>

A single token recovers 59 percent of the Normal class with zero false positives. Nothing about
the string "enron" indicates that a message is legitimate. A model leaning on it would collapse
the moment it met legitimate mail from any other company, which is every real deployment.

Note the contrast with the row below it. "urgent" and "$" are also skewed toward Fraud, but
those are genuine fraud signals, the actual rhetoric of advance fee solicitation. The
distinction between a leak and a signal is whether the feature would still hold on mail from a
different organisation.

Decision: phase 4 builds a blocklist of organisation and person named entities drawn from the
Enron source and strips them. Every headline result is reported twice, before and after
stripping, and the gap is a reported finding.

### D4. Formatting fingerprint separates Fraud from the other two classes. Severity: critical, open.

The two source corpora were preprocessed differently before the author ever combined them, and
the difference is visible in every single row.

Normal and Spam, from the Enron derived CSV, are lowercased throughout with whitespace inserted
around punctuation:

```
request submitted : access request for amitava . dhar @ enron . com
save your money by getting an oem software !  need in software for your pc ?
```

Fraud, from the mailbox dump, retains original casing, ordinary punctuation and MIME artifacts:

```
I am the wife of Mr. Nicolai Sainovic one of the  people of the former Yugoslavia
I am Mr Mark Boland the Bank Manager of ABN AMRO Bank 101 Moorgate,=20
```

So the presence of any uppercase character, or the absence of a space before a full stop, or the
string `=20`, each separate Fraud from the other two classes almost perfectly. This is the
severe form of the leak predicted in `PLAN.md` 8.1. The model can identify the file a message
came from without reading a word of it.

Decision: normalise all three classes to a single text representation before any model sees
them. Lowercase everything, collapse punctuation spacing to one convention, and strip MIME
quoted printable artifacts. This normalisation belongs in the preprocessing pipeline at phase 6,
and phase 4 will quantify how much apparent accuracy it removes.

### D5. The prebuilt corpus was sampled without a recorded seed. Severity: medium, open.

`final_dataset.csv` is exactly 1,000 per class, drawn from pools of 4,360 ham, 1,368 spam and
roughly 3,976 fraud. The reference selects those rows with unseeded randomness, so the specific
sample is not reproducible by anyone, including its author.

Decision: we build our own corpus at phase 3 from `spam_normal_emails.csv` and
`fradulent_emails.txt` with a fixed recorded seed. `final_dataset.csv` is retained only as a
cross check, to confirm our construction lands in the same region as the reference. It is not
used for training or scoring.

Note also that 1,000 of 1,368 is 73 percent of the entire spam pool, so the spam class has very
little headroom. Any future increase in class size is constrained by spam, not by the other two.

### Phase 1 verdict

Data is present, checksummed and counted. Two critical leaks are identified and quantified
before a single model has been trained, which is the correct order to do this in. Phase 2 may
proceed to parsing the fraud mailbox.

## Phase 2: parse the fraud mailbox

Detail and tables in `docs/phase2_findings.md`. Counts in
`data/interim/phase2_parse_report.json`.

### D1 update. Message count resolved at 3,978. Closed.

Phase 1 left the count open. Phase 2 settles it. Splitting on the full mbox envelope pattern
yields 3,978 messages. Two envelope lines do not carry the `From r` prefix, at lines 72,425 and
285,198, so any parser keyed on that literal loses two messages and corrupts two more.

The reference claim of 4,075 is wrong. The commonly cited 3,977 is wrong by one. The naive
prefix split at 3,976 is wrong by two. Of those 3,978, exactly 3,976 yield a usable body, which
coincidentally equals the naive figure. The agreement is a coincidence and not a validation.

### D4 update. MIME artifact half resolved.

Parsing through the standard library email module decodes quoted printable payloads. Occurrences
of `=20` fall from 6,423 in the raw mailbox to 9 in the parsed bodies, a 99.9 percent reduction,
and the survivors are literal body text rather than artifacts. The casing and punctuation
spacing half of D4 remains open and is handled by normalisation at phase 6.

### D6. Label noise inside the fraud corpus. Severity: low, accepted.

The corpus is presented as advance fee fraud but carries a little ordinary spam. Across the 3,976
usable messages, measured by keyword signature: 93.8 percent advance fee, 0.1 percent pharmacy or
replica goods, 0.2 percent both, 5.9 percent neither strong signature. Two messages are
unambiguously pharmacy spam, one of which is among the two unusable records.

Decision: accepted without action. At 0.1 percent this sits far below the resolution of any
metric we will report. Recorded so the README does not overclaim corpus purity.

### D7. Heavy duplication inside the fraud corpus. Severity: high, open.

The measurement behind `PLAN.md` 8.3.

<table>
<tr><th>Match rule</th><th>Unique</th><th>Redundant copies</th><th>Share</th><th>Largest group</th></tr>
<tr><td>Exact body</td><td>3,298</td><td>678</td><td>17.1 percent</td><td>13</td></tr>
<tr><td>Case and punctuation normalised</td><td>3,257</td><td>719</td><td>18.1 percent</td><td>16</td></tr>
<tr><td>First 200 normalised characters</td><td>2,985</td><td>991</td><td>24.9 percent</td><td>20</td></tr>
</table>

One in six fraud messages is byte identical to another. The reference draws 1,000 messages from
this pool and then splits train and test, so copies of the same message sit on both sides and
part of its reported score is memorisation.

Decision: phase 3 deduplicates before sampling, and clusters near duplicates so that a cluster
never spans the train and test split. The size of the correction is reported.

### Phase 2 verdict

3,978 messages located, 3,976 parsed to usable bodies, 23 tests passing. One defect closed, one
half resolved, two new ones logged. Phase 3 may proceed to corpus construction.

## Phase 3: build the corpus

Detail and tables in `docs/phase3_findings.md`. Counts in
`data/processed/phase3_build_report.json`.

### D2 update. The Subject prefix concern was correct about the source. Closed.

Phase 1 withdrew this after finding the prefix absent from the reference's built corpus.
Measured against the source files it is present on 1,368 of 1,368 spam rows and 4,360 of
4,360 ham rows, and on 1 of 3,976 fraud messages. So the leak is real in the source and the
reference had already handled it. We strip it too, in `src.corpus.strip_subject_prefix`.

### D7 update. Duplication is worse than exact matching showed.

Phase 2 measured 17.1 percent byte identical duplication in the fraud corpus. Near duplicate
matching at cosine 0.85 raises that to 38.4 percent. Across all three classes, 2,776 of 9,699
documents are redundant, or 28.6 percent. Resolved by sampling one document per cluster, which
leaves the corpus with no duplicates at all and therefore none able to straddle the split. The
build asserts zero cluster overlap rather than assuming it.

### D8. Single linkage chaining in our own clustering. Severity: high, fixed.

Not a defect in the data. A defect in our first implementation.

Connected components clustering is single linkage, which merges A and C whenever both connect
to B. On this corpus shared boilerplate produced chains rather than duplicate groups. The
largest cluster held 232 fraud emails with a median pairwise cosine of 0.767 and a minimum of
0.563. An 85 member Enron cluster had a minimum pairwise similarity of 0.045.

Fixed by switching to complete linkage, under which every pair inside a cluster clears the
threshold. Implemented in two stages so the full dense matrix of 47 million pairs is never
needed: cheap connected components first, then exact complete linkage inside each component.
Largest cluster fell from 232 to 112, redundancy from 32.3 percent to 28.6 percent. Pinned by
`test_complete_linkage_refuses_to_chain`.

### D9. The two sources disagree on the label of the same email. Severity: medium, resolved.

Five clusters covering 12 documents carry two labels, every one of them FRAUD against SPAM.
They are advance fee scams present in both the CLAIR fraud mailbox and the Enron spam corpus,
because the same campaigns reached Enron employees.

Decision: drop the contradicted clusters. A majority vote would invent information we do not
have. More importantly this is a fact about the task rather than a data cleaning nuisance:
advance fee fraud is a subset of spam, so the FRAUD and SPAM boundary is genuinely fuzzy and a
perfect score on this problem would be a warning sign.

### D10. A balanced 1,000 per class is unreachable after deduplication. Severity: medium, resolved.

The spam pool of 1,368 rows reduces to 999 distinct clusters, and to 994 once contradictions
are dropped. The reference's 1,000 per class was therefore only achievable by leaving
duplicates in, and 26.9 percent of its spam class is redundant with the rest of that class.

Decision: 900 per class, giving 2,700 documents, 2,160 train and 540 test. Spam draws 91
percent of its pool, fraud 37 percent and normal 26 percent.

### D11. Body length differs sharply by class. Severity: medium, open.

<table>
<tr><th>Class</th><th>Median words</th><th>Mean</th></tr>
<tr><td>FRAUD</td><td>424</td><td>434</td></tr>
<tr><td>SPAM</td><td>126</td><td>251</td></tr>
<tr><td>NORMAL</td><td>204</td><td>313</td></tr>
</table>

Fraud is more than three times the median length of spam. Part of that is genuine, since an
advance fee pitch needs room to tell a story, and part is an artifact of corpus provenance.

Decision: phase 5 measures how far document length alone gets you, and phase 7 includes a
length only baseline. If that baseline scores well, every later number must be read against
it rather than against the majority class alone.

### Phase 3 verdict

2,700 document corpus, 900 per class, seed 20260914, zero cluster overlap across the split,
52 tests passing. D2 and D7 closed, four new defects logged of which three are resolved and
D11 is handed forward. Phase 4 may proceed to the source marker audit.

## Phase 4: provenance audit

Detail, tables and the full method in `docs/phase4_findings.md`. Numbers in
`docs/phase4_audit.json`. Figure in `reports/figures/phase4_leakage.png`.

All figures are macro F1, five fold cross validation on the training split only. Note the
majority class baseline for macro F1 is 0.1667, not 0.333. The 0.333 figure is accuracy.

### D3 update. The named entity leak is real but redundant. Severity downgraded to low.

Thirty four curated provenance tokens, given to a classifier with all other content
removed, reach 0.5763 macro F1, which is 3.46 times the majority baseline, and two thirds
of training documents contain none of them. The leak is real.

But stripping those tokens from the pipeline changes macro F1 by 0.0000, and normalisation
plus stripping together costs 0.0028, which is smaller than the fold to fold standard
deviation. The leak is redundant with the content rather than additive to it.

Decision: markers stay stripped in all later phases. The cost is negligible and the
shortcut is removed. Severity downgraded because the headline number was never inflated by
it, which is the opposite of what `PLAN.md` 8.1 predicted.

### D4 update. Closed, and it was the largest single artifact.

Formatting alone, seven numeric style features and not one word, reaches 0.7013 macro F1.
That is 4.21 times the majority baseline and the largest shortcut found in this project. A
model trained on the raw corpus has a route to roughly seventy percent of the task without
reading any text.

An unanticipated component: D4 changes the token inventory, not just the appearance.
`e-mail` appears in 100 percent Fraud documents purely because the Enron rows tokenise the
same word as `e - mail`. Same for `co-operation`, `don't`, `yahoo.com`, `u.s` and `000.00`.

Closed by canonical normalisation in `src.normalise`, which collapses both source styles to
one representation. A test asserts that the same sentence in Enron style and in fraud
mailbox style normalises to an identical string.

### D11 update. Length alone reaches 0.4386. Carried into phase 7.

2.63 times the majority baseline from a single feature. Real but the weakest of the three
shortcuts. Decision: phase 7 baselines include length only and formatting only, so a model
must beat 0.7013 rather than 0.1667 before it has demonstrated anything.

### D12. Over blocking in our own curation. Severity: high, fixed.

Not a defect in the data. A defect in our blocklist.

`am` was blocked as an AM and PM timestamp marker. It is also the English verb, and it opens
almost every advance fee email as "I am Mr ...". It had never appeared in the skew evidence
and was added by assumption.

Effect on the measurement: the markers only probe read 0.894 macro F1 with `am` wrongly
blocked and 0.576 once corrected, an overstatement of 0.32. Documents containing no marker
rose from 34.3 percent to 66.1 percent. Handing an ablation probe one very common content
word lets it recover class information that has nothing to do with provenance.

Fixed by removal. `test_no_blocked_token_is_a_common_english_word` now asserts the blocklist
carries no ordinary English word. The lesson is that ablation probes are sensitive to
blocklist errors in both directions.

Residual judgement, flagged rather than resolved: `jeff`, `kevin` and `tanya` are blocked as
named Enron individuals but are also common given names that would appear in mail from
anywhere. This is the weakest part of the curation.

### D13. A leak that agrees with the signal is invisible to holdout validation. Severity: informational.

The most useful finding of the phase, and not a defect in this corpus so much as a fact
about the method.

Fitting with markers available and then scoring on input where they are removed drops macro
F1 from 0.9658 to 0.9580. So a model handed the shortcuts does lean on them, and nothing in
a conventional train and test split penalises it, because the leak and the genuine signal
agree on this corpus. A holdout drawn from the same corpus cannot detect the dependence.
Only an ablation can.

Decision: record it prominently in the README. It is the transferable lesson of the project
and it is worth more than the headline score.

### Phase 4 verdict

D4 closed. D3 downgraded to low. D11 carried to phase 7 as a required baseline. Two new
entries: D12, an over blocking error of our own, fixed; and D13, informational. 84 tests
passing. Phase 5 may proceed to exploratory analysis.

## Phase 5: exploratory analysis

Detail and tables in `docs/phase5_findings.md`. Numbers in `docs/phase5_eda.json`.

### D3 update. Eight more provenance markers found. Blocklist 34 to 42 tokens.

The phase 4 sweep used document frequency 25 or more and purity 95 percent or more. Ranking
the cleaned representation by log odds ratio surfaced markers that fell outside both
thresholds: `forwarded` at 183 documents and 89 percent purity, and `corp` at 111 and 90
percent, were the two highest frequency provenance markers in the corpus and both sat below
the purity cut. `anjam`, `donna`, `wharton`, `baylor`, `hsb` and `listinfo` sat below the
frequency cut.

Rerunning the phase 4 audit with 42 tokens: the markers only probe rises from 0.5763 to
0.6082 macro F1, documents containing no marker fall from 66.1 to 63.2 percent, and the
total attributable to provenance halves from 0.0028 to 0.0014. Conclusion unchanged.

### D9 update. Partially supported. Vocabulary overlap is weak evidence.

The phase 4 confusion matrix supported D9 strongly: FRAUD against SPAM carried 57 percent of
errors against 3 percent for FRAUD against NORMAL. Vocabulary Jaccard does not agree, ranking
SPAM and NORMAL highest at 0.419 against 0.384 for FRAUD and SPAM.

Jaccard is confounded by vocabulary size, and fraud's vocabulary is the largest at 3,218
terms. Asymmetric coverage does support D9: spam's vocabulary is 62.1 percent contained
inside fraud's, the highest containment of any ordered pair.

The deeper point is that shared vocabulary and confusability are different measurements. Two
classes can share most of their ordinary English and still separate trivially if the
discriminating terms are strong. Recorded as partial rather than confirmed.

### D11 update. Amplified by the cleanup, not mitigated. Still open.

The fraud to spam median length ratio is 3.32 on raw text and 4.35 after normalisation. The
cleanup widened the gap, because the Enron derived rows pad whitespace around punctuation so
a raw word count credits spam and normal with tokens that are only punctuation. Fraud, with
ordinary punctuation, was never credited that way.

So the 0.4386 macro F1 measured for length alone was computed on raw counts and is a floor.
Decision: phase 7 measures its length baseline on cleaned text.

### D14. A leakage check that cannot fail is not a check. Severity: high, fixed.

Not a defect in the data. A defect in our verification.

The first version of the phase 5 script asserted that no blocklisted token appeared in the
discriminative term ranking. The pipeline strips those tokens before the ranking is
computed, so their absence was guaranteed by construction. The check passed, reported
"confirmed: 0 of 75 ranked terms are blocklisted", and told us nothing. Meanwhile eight
genuine provenance markers sat in the ranking, visible to anyone reading the output.

Fixed by inverting the question. Every term in the ranking must now carry a recorded verdict
in `src.markers.PHASE5_REVIEWED`, and the script exits non zero listing any term that does
not. That forced explicit classification of all 75 ranked terms across two passes, since
removing the first batch reshuffled the ranking and surfaced 24 more.

The general lesson, and it belongs in the README next to D13: a verification that can only
pass is worse than no verification, because it manufactures confidence. Ask whether a check
could fail before trusting the fact that it passed.

### Phase 5 verdict

D3 updated with eight further markers. D9 downgraded to partially supported. D11 amplified
and carried to phase 7. One new defect, D14, in our own verification, fixed. 109 tests
passing. Phase 6 may proceed to the preprocessing pipeline.

## Phase 6: preprocessing ablation

Detail and tables in `docs/phase6_findings.md`. Numbers in `docs/phase6_ablation.json`.

### D11 closed. Length only baseline is 0.4698 on cleaned text.

Phase 5 predicted the shortcut would strengthen after cleaning, because the Enron rows pad
whitespace around punctuation so a raw word count credits them with punctuation as words.
Confirmed: 0.4386 on raw text against 0.4698 on cleaned. Phase 7 uses the cleaned figure.

### D15. Lemmatiser produced non words. Severity: high, fixed before it affected any result.

WordNet's noun lemmatiser strips any trailing `s` it reads as a plural. The first
implementation chained noun then verb and produced `was` to `wa`, `has` to `ha`, `us` to
`u`, `as` to `a`, and `mrs` to `mr`.

The last one is the one that mattered. `mrs` appears in 126 training documents at 96 percent
fraud purity, and merging it into `mr` would have destroyed that signal without any error
being raised.

Fixed by reversing the order and guarding: verb lemma first, falling back to the noun lemma
only when the verb step changed nothing and the result is at least three characters. Five
bad cases corrected, ten good cases preserved, pinned by parametrised tests.

### D16. Transformers were not fitted by scikit learn's definition. Severity: medium, fixed.

The text steps hold no learned state, so their `fit` methods returned `self` without
recording anything. `check_is_fitted` looks for an attribute ending in an underscore, so
`Pipeline.transform` raised `NotFittedError`.

The failure mode is why this matters. `fit_transform` works, so every cross validation run
passes clean. Only fit once then transform many times breaks, which is exactly the pattern
phase 9 uses to score the held out test set. A test caught it at phase 6 instead.

Fixed with a shared `_StatelessTransformer` base that sets `fitted_` in `fit`.

### D17. The signal pronoun exemption did not earn its place. Severity: informational.

The hypothesis was that `i`, `am`, `my`, `your` and similar carry fraud register, since
advance fee fraud is written in the first person about a named relative, so holding them
back from the stopword list should help. Measured: 0.9657 with the exemption against 0.9676
without it. Within noise, but the direction is opposite to the prediction and there is no
evidence for the claim.

Decision: `keep_signal` defaults to False. The mechanism stays so the comparison remains
reproducible.

### Phase 6 verdict

D11 closed. Three new entries: D15 and D16 fixed, D17 informational. 143 tests passing.

The phase level finding is a negative one worth stating plainly: no preprocessing step
changes macro F1. The spread across nine conditions is 0.0042 against a fold standard
deviation of 0.0049. The chosen pipeline was selected on stated secondary criteria,
interpretability and feature space size, not on score. Phase 7 may proceed to baselines.

## Phase 7: baselines

Detail and the canonical table in `docs/phase7_findings.md`. Numbers in
`docs/phase7_baselines.json`.

### D18. We quoted the wrong floor for four phases. Severity: medium, corrected.

Phases 4 through 6 reported 0.1667 as the macro F1 floor, being the score of a classifier
that always predicts one class. It is not the floor.

<table>
<tr><th>Trivial strategy</th><th>Macro F1</th></tr>
<tr><td>Most frequent class</td><td>0.1667</td></tr>
<tr><td>Uniform random</td><td>0.3147</td></tr>
<tr><td>Stratified random</td><td>0.3355</td></tr>
</table>

Random guessing beats always predicting one class by 0.1688, more than double. Macro F1
averages per class F1: a single class predictor scores 0.5 on its class and 0.0 on the other
two, while random guessing scores about a third on all three.

The consequence is not cosmetic. Phase 4 described the formatting probe as 4.21 times the
majority baseline. Against the correct floor it is 2.09 times. A multiple against a badly
chosen denominator flatters every result.

Decision: quote random at 0.3355 as the trivial floor and formatting only at 0.7013 as the
bar in every later table. The majority figure stays for completeness, labelled as what it
is. A test asserts that the random floor exceeds the majority floor so the point cannot be
quietly lost.

### D11 note. Two length measurements, both reported.

B3 measures length on the chosen phase 6 pipeline at 0.4453. B3b measures it on
normalisation and stripping only at 0.4698, which is the figure phase 6 reported before the
stopword step was chosen. Stopword removal cuts tokens per document by 43 percent, so these
are the same idea on different text.

B3 is the operative figure because it is what the model sees. B3b is retained so the phase 6
write up stays traceable. B3 also carries the widest standard deviation of any baseline at
0.0409, about nine times the model's, which is a reason not to over read small differences
among the weak baselines.

### Phase 7 verdict

The baseline suite is fixed and loadable by later phases through
`src.baselines.load_baselines()` and `bar_to_clear()`, read from the report rather than
hardcoded so numbers cannot drift. One new entry, D18, corrected. 162 tests passing.

The bar a model must clear is 0.7013, set by formatting alone. The phase 6 pipeline clears
it by 0.2649. Phase 8 may proceed to the model sweep.

## Phase 8: model sweep

Detail and the ranking in `docs/phase8_findings.md`. Numbers in `docs/phase8_sweep.json`.

### D19. Linear SVM did not converge, so a quarter of the sweep reported noise. Severity: high, fixed.

On CountVectorizer features, `LinearSVC` at its default 1,000 iteration cap failed to
converge for all eight grid points. A model that stopped early is reporting an arbitrary
point on its optimisation path rather than a fitted model, so those scores measured nothing.

Cause: raw term counts are unscaled and span orders of magnitude, which conditions the margin
problem badly. TF IDF is L2 normalised and converged without help, so the failure appeared on
one representation only.

Fixed by raising the cap to 20,000, verified by promoting `ConvergenceWarning` to an error
and refitting all eight configurations. The sweep was rerun.

Worth recording how nearly this was missed. scikit learn emits one warning per fit, so a
nested search produced hundreds of identical lines that scrolled past as noise. They were the
only signal that a quarter of the results were meaningless.

### D20. Our optimism measurement was confounded. Severity: medium, fixed.

The first attempt to quantify selection optimism produced a mean gap of 0.0019 in the
negative direction. A bias that only ever inflates cannot produce a negative mean, so the
measurement was wrong rather than the bias being absent.

Cause: the nested figure used five outer folds, training on 80 percent of the data, while the
non nested figure used the three fold inner splitter, training on 67 percent. The non nested
number was depressed by less training data at the same time as being inflated by selection,
and the larger of the two effects was the one we had not intended to measure.

Fixed by giving both searches the same five folds, so training set size is identical and the
only difference left is selection. The corrected gap is +0.0007 on average and +0.0033 at
worst, which is the right direction and a plausible magnitude for small grids on a strong
signal.

The lesson generalises: when a measured effect has the wrong sign, suspect the experiment
before concluding the effect is absent.

### Phase 8 verdict

Winner is a soft voting ensemble on tfidf features at 0.9740 macro F1, though it beats
multinomial naive Bayes by 0.0019 against a standard deviation of 0.0027, costs roughly six
times the compute, and gives up interpretability. Phase 9 reports both.

Two new defects, D19 and D20, both ours and both fixed. 190 tests passing. Phase 9 may
proceed to the holdout.
