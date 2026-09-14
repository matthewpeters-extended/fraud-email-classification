# Phase 5 findings: what the content signal actually is

Phase 4 established that the content, not the provenance shortcuts, carries the signal.
This phase describes that content: length profiles, vocabulary overlap between classes,
and the terms that separate them once normalisation and marker stripping are applied.

Entry point `scripts/run_eda.py`, logic in `src/explore.py`, tests in
`tests/test_explore.py`. Numbers in `docs/phase5_eda.json`. Figures
`phase5_length.png`, `phase5_terms.png` and `phase5_vocab_overlap.png` in
`reports/figures`.

Training split only, 2,160 documents. The test split is not touched until phase 9.

## Document length, and D11 gets worse rather than better

<table>
<tr><th>Class</th><th colspan="3">raw text, words</th><th colspan="3">normalised and stripped, words</th></tr>
<tr><th></th><th>p25</th><th>median</th><th>p75</th><th>p25</th><th>median</th><th>p75</th></tr>
<tr><td>FRAUD</td><td>309</td><td>422</td><td>538</td><td>325</td><td>439</td><td>563</td></tr>
<tr><td>SPAM</td><td>74</td><td>127</td><td>274</td><td>59</td><td>101</td><td>225</td></tr>
<tr><td>NORMAL</td><td>113</td><td>208</td><td>385</td><td>84</td><td>150</td><td>271</td></tr>
</table>

The fraud to spam median ratio is 3.32 on raw text and **4.35 after cleaning**. The gap
widens, and the reason is instructive: the Enron derived rows pad whitespace around
punctuation, so a token count on raw text counts every ` . ` and ` , ` as a word. Spam and
Normal were being credited with words that were only punctuation. Fraud, with ordinary
punctuation, was not. Normalisation removes that inflation from two classes and not the
third.

So D11 is amplified by the very cleanup that closed D4. The length only probe measured
0.4386 macro F1 on raw word counts, and the cleaned gap is wider, so that figure is a
floor rather than a ceiling.

Decision: phase 7 measures its length baseline on cleaned text, not raw, or it will
understate the shortcut it exists to expose.

## Vocabulary

<table>
<tr><th>Class</th><th>Vocabulary at df 5 or more</th><th>Total tokens</th><th>Type token ratio</th><th>Exclusive vocabulary</th></tr>
<tr><td>FRAUD</td><td>3,218</td><td>328,014</td><td>0.0401</td><td>1,385, or 43.0 percent</td></tr>
<tr><td>SPAM</td><td>2,600</td><td>156,736</td><td>0.0905</td><td>682, or 26.2 percent</td></tr>
<tr><td>NORMAL</td><td>2,422</td><td>157,507</td><td>0.0725</td><td>720, or 29.8 percent</td></tr>
</table>

Fraud is the most repetitive class by a wide margin: its type token ratio is 0.0401 against
spam's 0.0905, so it reuses its vocabulary roughly 2.3 times as heavily. That is an
independent confirmation of the phase 3 measurement, which found 38.4 percent of the fraud
corpus redundant under near duplicate matching against 26.9 percent for spam. Two different
methods, at two different phases, agree that advance fee fraud is the most templated of the
three classes. It is a genre written from scripts.

Fraud also has the largest vocabulary in absolute terms despite being the most repetitive,
because its documents are four times longer.

## Vocabulary overlap does not support D9 as cleanly as the confusion matrix did

<table>
<tr><th>Pair</th><th>Jaccard</th></tr>
<tr><td>SPAM and NORMAL</td><td>0.419</td></tr>
<tr><td>FRAUD and SPAM</td><td>0.384</td></tr>
<tr><td>FRAUD and NORMAL</td><td>0.330</td></tr>
</table>

D9 predicted that the FRAUD and SPAM boundary would be the fuzziest, and the phase 4
confusion matrix agreed emphatically: that pair carried 57 percent of all errors against 3
percent for FRAUD and NORMAL. Jaccard overlap puts SPAM and NORMAL highest instead.

Two things are going on, and both are worth stating rather than picking whichever supports
the story.

First, Jaccard is confounded by vocabulary size. Fraud's vocabulary is 3,218 terms against
2,422 for Normal, and Jaccard divides by the union, so the larger set is penalised.
Coverage, which is asymmetric, tells a different story:

<table>
<tr><th>Share of the row class vocabulary also present in the column class</th><th>FRAUD</th><th>SPAM</th><th>NORMAL</th></tr>
<tr><td>FRAUD</td><td>.</td><td>0.502</td><td>0.434</td></tr>
<tr><td>SPAM</td><td><b>0.621</b></td><td>.</td><td>0.570</td></tr>
<tr><td>NORMAL</td><td>0.577</td><td>0.612</td><td>.</td></tr>
</table>

Spam's vocabulary is 62.1 percent contained inside fraud's, the highest containment of any
ordered pair. Read that way, spam is the class most subsumed by fraud, which is what D9
claimed.

Second, and more importantly, shared vocabulary and confusability are not the same
measurement. Two classes can share most of their common words and still be trivially
separable if the discriminating terms are strong and frequent. Normal and Spam share a lot
of ordinary English while differing sharply on `viagra`, `mortgage` and `mailings`. Fraud
and Spam share less vocabulary overall but overlap precisely where it matters, because
advance fee fraud *is* a kind of spam.

Conclusion: the confusion matrix is the stronger evidence for D9 and vocabulary overlap is
weak evidence either way. Recorded as a partial rather than a confirmation.

## The terms that separate the classes

Ranked by document level log odds ratio, minimum document frequency 20, computed after
normalisation and marker stripping. Purity is the share of documents containing the term
that belong to that class.

<table>
<tr><th>FRAUD</th><th>SPAM</th><th>NORMAL</th></tr>
<tr><td>senegal, dakar, dormant, hospital, ghana, ivory, liberia, dearest, donate, inheritance, chamber, frozen, seized, inherit, burkina, beloved, abidjan, faso, accra</td><td>andmanyother, viagra, mailings, shlpplng, mortgage, shops, oniine, opt, sightings, miiiion, prlces, advertisement, solicitation, sex, proven</td><td>dinner, derivatives, speaker, agenda, lunch, valuation, interview, resume, conference, chair, modeling, model, weather, professor</td></tr>
</table>

This is the answer to the question phase 4 raised. The signal is genuine and it is
interpretable:

* **Fraud** separates on the geography and narrative furniture of advance fee fraud. West
  African place names, dormant accounts, frozen and seized funds, inheritance, a deceased
  relative, a hospital. This is the 419 script, and a model learning it is learning the
  genre rather than the corpus.
* **Spam** separates on product vocabulary and, notably, on **deliberate misspellings**.
  `shlpplng`, `oniine`, `miiiion` and `prlces` substitute the letter l for i and vice
  versa to defeat keyword filters. Those are not noise, they are the adversarial signature
  of the genre, and a classifier picking them up is doing exactly the right thing.
* **Normal** separates on the vocabulary of doing a job: meetings, agendas, lunch, dinner,
  conferences, interviews, resumes, and the specific quantitative finance content that this
  company worked on, including weather derivatives and valuation modelling.

Note what is *not* here. No company name, no employee name, no header furniture, no phone
fragment. The phase 4 cleanup held.

## The leakage check was tautological, and fixing it found eight more markers

The first version of this phase asserted that no blocklisted token appeared in the
discriminative ranking. That check cannot fail: the pipeline strips those tokens before the
ranking is computed, so their absence is guaranteed by construction rather than evidence of
anything.

The real question is whether any term in the ranking is provenance that the phase 4
curation missed. Reading the output by eye, several were:

<table>
<tr><th>Term</th><th>df</th><th>Purity</th><th>What it is</th></tr>
<tr><td>forwarded</td><td>183</td><td>89 percent</td><td>the "Forwarded by ..." mail separator</td></tr>
<tr><td>corp</td><td>111</td><td>90 percent</td><td>the tail of the company name</td></tr>
<tr><td>anjam</td><td>24</td><td>100 percent</td><td>a named Enron individual</td></tr>
<tr><td>listinfo</td><td>24</td><td>92 percent</td><td>a Mailman URL fragment, so a collection artifact</td></tr>
<tr><td>wharton</td><td>22</td><td>100 percent</td><td>a business school recurring in recruiting mail</td></tr>
<tr><td>baylor</td><td>22</td><td>96 percent</td><td>a university local to the company</td></tr>
<tr><td>hsb</td><td>20</td><td>100 percent</td><td>an internal Enron acronym</td></tr>
<tr><td>donna</td><td>20</td><td>95 percent</td><td>a named individual</td></tr>
</table>

Phase 4 swept at document frequency 25 or more, which is why the individuals appearing in
20 to 24 documents were missed. `forwarded` and `corp` were missed for a different reason:
at 89 and 90 percent purity they fell below the 95 percent threshold, despite being the two
highest frequency provenance markers in the corpus.

The check is now non tautological. Every term in the ranking must carry a recorded verdict
in `src.markers.PHASE5_REVIEWED`, and the script exits non zero listing any term without
one. That forced 75 of 75 ranked terms to be classified explicitly, in two passes, because
removing `forwarded` and `corp` reshuffled the ranking and surfaced 24 further terms.

Blocklist grows from 34 to 42 tokens. Every kept term carries a stated reason, including
the three the curation is least sure about: `rice` is Rice University locally but also a
common noun, `jul` is a date abbreviation that number masking does not catch, and `friday`
is a day name where blocking would risk content in the manner of D12.

## Effect on the phase 4 numbers

The audit was rerun with the 42 token blocklist.

<table>
<tr><th>Measurement</th><th>34 tokens</th><th>42 tokens</th></tr>
<tr><td>P2 provenance markers only</td><td>0.5763</td><td>0.6082</td></tr>
<tr><td>Documents containing no marker</td><td>66.1 percent</td><td>63.2 percent</td></tr>
<tr><td>C3 normalised, markers stripped</td><td>0.9658</td><td>0.9671</td></tr>
<tr><td>Total attributable to provenance</td><td>0.0028</td><td>0.0014</td></tr>
</table>

The eight added markers raise the measured leak to 0.6082 macro F1 from provenance alone.
Stripping them now scores marginally *above* the unstripped condition, and the total
attributable to provenance halves to 0.0014.

The phase 4 conclusion is unchanged and slightly strengthened: the shortcuts are real and
substantial in isolation, and removing them costs nothing measurable.

## Decisions carried forward

1. Phase 7 measures the length baseline on cleaned text, where the fraud to spam ratio is
   4.35 rather than 3.32. The 0.4386 figure is a floor.
2. D9 is recorded as partially supported. The confusion matrix supports it strongly,
   vocabulary Jaccard does not, and coverage does. Vocabulary overlap and confusability
   measure different things.
3. The 42 token blocklist is final unless a later phase surfaces more. The review registry
   makes any future addition auditable.
4. Phase 9 error analysis should check whether the misspelling family, `shlpplng`, `oniine`,
   `miiiion` and `prlces`, is doing disproportionate work in the spam class, since those
   terms are strong but each appears in only 20 to 30 documents.

## Tests

109 tests pass. New coverage in `tests/test_explore.py` for percentile edges, quantile
ordering in the length profile, document frequency counting presence rather than
repetition, the minimum document frequency filter, Jaccard bounds, the asymmetry of
coverage against the symmetry of Jaccard, type token ratio detecting repetition, log odds
ratio sign and antisymmetry and finiteness under zero counts, preference for the purer term
at equal frequency, and the discriminative ranking finding planted signal while ignoring a
term shared by every class.
