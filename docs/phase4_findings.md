# Phase 4 findings: how much separability is provenance rather than content

Goal: put a number on defects D3, D4 and D11 instead of asserting them, and measure what
fixing them costs.

Entry point `scripts/audit_markers.py`, logic in `src/normalise.py` and `src/markers.py`,
tests in `tests/test_normalise.py` and `tests/test_markers.py`. Numbers in
`docs/phase4_audit.json`, figure in `reports/figures/phase4_leakage.png`.

All figures are macro F1 under five fold stratified cross validation on the **training
split only**, seed 20260914. The test split is not touched until phase 9. Vectorisers sit
inside pipelines so they are fitted on training folds only, which is the fix for `PLAN.md`
8.2.

## Method

Three isolation probes, each given one artifact and no access to content. Whatever a probe
scores, it scores without reading the message.

<table>
<tr><th>Probe</th><th>What it sees</th><th>Tests</th></tr>
<tr><td>P1 formatting only</td><td>seven numeric style features, zero words</td><td>D4 formatting fingerprint</td></tr>
<tr><td>P2 markers only</td><td>only blocklisted provenance tokens, content discarded</td><td>D3 named entity leak</td></tr>
<tr><td>P3 length only</td><td>one feature, the word count</td><td>D11 length shortcut</td></tr>
</table>

Then four pipeline conditions, to measure what the fixes cost, and a transfer probe to
test reliance.

## Finding the markers

Token skew was measured on the training split: document frequency 25 or more, with 95
percent or more of occurrences in a single class. That produced 136 candidates, which fall
into four groups.

<table>
<tr><th>Group</th><th>Examples with document frequency</th><th>Decision</th></tr>
<tr><td>Enron identity</td><td>vince 445, enron 404, kaminski 300, ect 251, hou 234, houston 137, shirley 119, crenshaw 84, vkamins 50</td><td>block</td></tr>
<tr><td>Header furniture</td><td>cc 348, pm 279, doc 75, edu 96, fyi 40</td><td>block</td></tr>
<tr><td>Spam collection tools</td><td>spamassassin 28, projecthoneypot 27</td><td>block</td></tr>
<tr><td>Tokenisation artifacts</td><td><code>e-mail</code> 144, <code>co-operation</code> 103, <code>don't</code> 90, <code>yahoo.com</code> 81, <code>u.s</code> 50, <code>000.00</code> 179</td><td>fixed by normalisation, not blocked</td></tr>
<tr><td>Genuine content</td><td>died 285, deposited 241, kin 214, beneficiary 202, foreigner 169, viagra 55, mortgage 29, derivatives 36</td><td>keep</td></tr>
</table>

Two groups deserve comment.

The tokenisation artifacts were not expected. `e-mail` appears in 100 percent Fraud
documents not because fraudsters say it more often, but because the Enron rows have
whitespace padded around punctuation and therefore tokenise the same word as `e - mail`.
So D4 does not merely change casing, it changes the token inventory. Normalisation removes
this group for free; a blocklist would have been the wrong tool.

The spam collection tools are a leak nobody had flagged. SpamAssassin and Project Honeypot
are the software used to *gather* the spam corpus, and their names appear in 28 and 27
training documents, all Spam. That makes the leak two sided rather than a Normal class
problem, and Fraud contributed a third: the Unicode replacement character appears in 32
documents, all Fraud, because phase 2 decoded the mailbox with `errors="replace"`. That
one we introduced ourselves, and it is now stripped in normalisation.

The block versus keep decision is semantic and cannot be made statistically. On this
training split `enron` and `beneficiary` are equally class pure. Only one of them would
still hold on mail from a different company. The stated rule is therefore:

> Block a token if it names a specific organisation, person, place, phone fragment, mail
> infrastructure component, or corpus collection tool. Keep every token that describes the
> content, intent or rhetoric of the message.

Final blocklist: 34 tokens in five groups. The sixteen tokens kept despite comparable skew
are listed with a reason each in `src/markers.py`, so the curation is auditable rather than
implicit.

## An error in our own curation

The first blocklist had 35 tokens, because `am` was included for AM and PM timestamps. It
is also the English verb, and it opens almost every advance fee email as "I am Mr ...". It
had never appeared in the skew evidence; it was added by assumption.

The effect on the measurement was large:

<table>
<tr><th>P2 markers only</th><th>Macro F1</th><th>Documents containing no marker</th></tr>
<tr><td>With "am" wrongly blocked</td><td>0.894</td><td>34.3 percent</td></tr>
<tr><td>Corrected</td><td>0.576</td><td>66.1 percent</td></tr>
</table>

Handing the probe one very common content word let it recover class information that has
nothing to do with provenance, and inflated the apparent leak by 0.32 macro F1. The lesson
is that an ablation probe is sensitive to blocklist errors in both directions: over
blocking overstates the leak just as under blocking hides it. A test now asserts that no
blocklisted token is an ordinary English word.

The weakest remaining part of the curation is the three common given names, `jeff`, `kevin`
and `tanya`. In this corpus they are specific Enron individuals, but they would also appear
in mail from anywhere else. They are kept blocked, and flagged here as the judgement most
open to challenge.

## Results

<table>
<tr><th>Run</th><th>Macro F1</th><th>Standard deviation</th><th>Multiple of majority baseline</th></tr>
<tr><td>C0 majority class baseline</td><td>0.1667</td><td>0.0000</td><td>1.00x</td></tr>
<tr><td>P3 length only, one feature</td><td>0.4386</td><td>0.0068</td><td>2.63x</td></tr>
<tr><td>P2 provenance markers only, 34 tokens</td><td>0.5763</td><td>0.0272</td><td>3.46x</td></tr>
<tr><td>P1 formatting only, seven features, no words</td><td>0.7013</td><td>0.0202</td><td>4.21x</td></tr>
<tr><td>C1 raw text</td><td>0.9686</td><td>0.0043</td><td>5.81x</td></tr>
<tr><td>C2 normalised</td><td>0.9658</td><td>0.0057</td><td>5.79x</td></tr>
<tr><td>C3 normalised, markers stripped</td><td>0.9658</td><td>0.0054</td><td>5.79x</td></tr>
</table>

Note the majority baseline is 0.1667 macro F1, not 0.333. Macro F1 averages per class F1,
and a classifier that always predicts one class of three scores 0.5 on that class and 0
on the other two. Accuracy for that baseline is 0.333. Quoting the accuracy figure as
though it were the macro F1 floor would understate the model by a wide margin, and this is
the sort of thing that gets misreported.

## What the fixes cost: essentially nothing

<table>
<tr><th>Step</th><th>Macro F1</th><th>Change</th></tr>
<tr><td>C1 raw</td><td>0.9686</td><td></td></tr>
<tr><td>C2 after normalisation</td><td>0.9658</td><td>0.0028 lower</td></tr>
<tr><td>C3 after stripping markers</td><td>0.9658</td><td>no change</td></tr>
<tr><td>Total attributable to provenance</td><td></td><td>0.0028 lower</td></tr>
</table>

This inverts the prediction in `PLAN.md` 8.1, which expected the gap between the raw and
cleaned figures to be the headline finding. The gap is 0.0028, which is smaller than the
fold to fold standard deviation of either condition. Stated plainly: **the reference's
headline accuracy is not materially inflated by the provenance leaks.**

That is not the same as saying the leaks do not exist. Every probe says they do:

* Formatting alone reaches 0.701 with no access to words at all
* Thirty four provenance tokens alone reach 0.576, with two thirds of documents containing
  none of them
* Document length alone reaches 0.439

The resolution is that the leaks are **redundant with the content rather than additive to
it**. These three classes are separable on what the messages actually say, so a model does
not need the shortcuts, and removing them costs nothing.

## The reliance probe, and why this is still dangerous

If removing markers from training is free, that could mean either that the model ignored
them or that the content covered for them. P4 settles it by fitting with markers available
and then scoring on input where they have been removed:

<table>
<tr><th>P4, fitted with markers intact</th><th>Macro F1</th></tr>
<tr><td>Scored on markers intact</td><td>0.9658</td></tr>
<tr><td>Scored on markers stripped</td><td>0.9580</td></tr>
<tr><td>Drop</td><td>0.0078</td></tr>
</table>

So a model handed the shortcuts does lean on them, mildly, and nothing in a conventional
train and test split penalises it for doing so. The leak and the genuine signal agree on
this corpus, so a holdout drawn from the same corpus cannot detect the dependence. Only an
ablation can.

That is the transferable lesson from this phase, and it is worth more than the headline
score: **a leak that correlates with the true signal is invisible to holdout validation.**
The 0.701 figure for formatting alone is the number to quote. A model trained on the raw
corpus has a route to roughly seventy percent of the task without reading a word, and the
first time it meets mail from a different company, preprocessed by a different pipeline,
that route disappears.

## Error structure confirms the phase 3 prediction

C3, pooled across folds. Rows are true, columns predicted.

<table>
<tr><th></th><th>predicted FRAUD</th><th>predicted SPAM</th><th>predicted NORMAL</th></tr>
<tr><td><b>true FRAUD</b></td><td>705</td><td>14</td><td>1</td></tr>
<tr><td><b>true SPAM</b></td><td>28</td><td>684</td><td>8</td></tr>
<tr><td><b>true NORMAL</b></td><td>1</td><td>22</td><td>697</td></tr>
</table>

<table>
<tr><th>Boundary</th><th>Errors</th><th>Share of the 74 errors</th></tr>
<tr><td>FRAUD against SPAM</td><td>42</td><td>57 percent</td></tr>
<tr><td>NORMAL against SPAM</td><td>30</td><td>41 percent</td></tr>
<tr><td>FRAUD against NORMAL</td><td>2</td><td>3 percent</td></tr>
</table>

D9 predicted this exactly. Advance fee fraud is a subset of spam, so that boundary is
genuinely fuzzy, while fraud against legitimate corporate mail is nearly trivial with two
errors in 1,440 opportunities. Per class F1: FRAUD 0.970, NORMAL 0.978, SPAM 0.950. Spam is
the weakest class, and it is weak specifically where it abuts fraud.

For a deployment framing this is the right shape of error. Classifying a fraud email as
spam still gets it out of the inbox. Classifying it as normal does not, and that happens
once in 720.

## Decisions carried forward

1. Normalisation and marker stripping stay in the pipeline for all later phases. They cost
   0.0028 macro F1 and they remove a 0.701 shortcut. That trade is obviously worth taking
   even though the headline number barely moves.
2. Phase 7 baselines must include the length only and formatting only probes, not just the
   majority class. A model must beat 0.701, not 0.167, before it has demonstrated anything.
3. The majority baseline for macro F1 is 0.1667. Every later table quotes that, not 0.333.
4. Phase 9 error analysis concentrates on the FRAUD and SPAM boundary, which carries 57
   percent of all errors.
5. `PLAN.md` 8.1 is revised: the leak is real and individually sufficient, but it is
   redundant with content and does not inflate the headline. Reported as a negative result,
   because that is what the measurement says.

## Tests

84 tests pass. New coverage: the two source styles collapsing to identical normalised text,
MIME and decode artifact removal, symbol words surviving normalisation, the number masking
order, the normaliser working as a pipeline step, formatting features separating the styles
while never reading content, blocklist and keep lists being disjoint, whole token matching
so `enroute` is not caught by `enron`, strip and keep partitioning the text exactly, and
the regression guard that no blocklisted token is a common English word.
