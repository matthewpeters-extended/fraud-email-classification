# Phase 3 findings: building the corpus

Goal: replace the reference's unseeded sample of 1,000 per class with a deduplicated,
reproducible corpus whose train and test split cannot leak.

Entry point `scripts/build_dataset.py`, logic in `src/corpus.py`, tests in
`tests/test_corpus.py`. Counts in `data/processed/phase3_build_report.json`.

## Construction order, and why it is that order

1. Strip the `Subject:` prefix, which is a source file marker rather than content
2. Drop bodies with fewer than five words
3. Cluster near duplicates across all three classes at once
4. Drop clusters that carry more than one label
5. Sample one document per cluster, seeded
6. Split train and test

Step 3 runs across all classes rather than within each class, so that a message appearing
in two source files surfaces as a label contradiction instead of hiding inside one class.
Step 5 takes a single representative per cluster, which is a stronger guarantee than
splitting by cluster afterwards: if the corpus contains no duplicates at all, none can
straddle the split.

## The Subject prefix question, settled

Phase 1 withdrew this hypothesis after finding the prefix absent from the reference's built
corpus. Measured now against the source files, the original concern was correct and the
reference simply handled it:

<table>
<tr><th>Class</th><th>Source rows carrying a leading Subject: prefix</th></tr>
<tr><td>SPAM</td><td>1,368 of 1,368, 100.0 percent</td></tr>
<tr><td>NORMAL</td><td>4,360 of 4,360, 100.0 percent</td></tr>
<tr><td>FRAUD</td><td>1 of 3,976, 0.0 percent</td></tr>
</table>

Left in, that single token identifies the source file and therefore separates Fraud from the
other two classes perfectly. We strip it, as the reference did.

## A defect in our own method: single linkage chaining

The first clustering implementation used connected components, which is single linkage. It
reported 32.3 percent redundancy and produced a 232 document cluster. Checking pairwise
similarity inside the largest clusters showed that most of them were not duplicates at all,
but chains: shared boilerplate linked A to B to C even where A and C were unrelated.

<table>
<tr><th>Cluster</th><th>Size</th><th>Median pair cosine</th><th>Min pair</th><th>Pairs at or above 0.85</th><th>Verdict</th></tr>
<tr><td>0</td><td>113</td><td>0.957</td><td>0.781</td><td>98.5 percent</td><td>genuine duplication</td></tr>
<tr><td>4357</td><td>81</td><td>0.832</td><td>0.464</td><td>38.4 percent</td><td>partly chained</td></tr>
<tr><td>4341</td><td>232</td><td>0.767</td><td>0.563</td><td>10.6 percent</td><td>chained</td></tr>
<tr><td>975</td><td>85</td><td>0.695</td><td>0.045</td><td>14.5 percent</td><td>chained</td></tr>
</table>

Cluster 975 is the clearest case. Eighty five Enron emails held together by signatures,
subject line prefixes and internal addresses, with a minimum pairwise similarity of 0.045.
Treating those as one message would have discarded 84 genuinely distinct emails.

The fix is complete linkage, under which every pair inside a cluster must clear the
threshold. Running it on the full corpus would need a dense matrix of all 47 million pairs,
so the implementation is two stage: find connected components cheaply over sparse blocks,
then run complete linkage agglomerative clustering inside each component. Components are a
superset of the final clusters, because a complete linkage cluster can never span two
disconnected components, so the result is exact rather than approximate.

Effect: the largest cluster falls from 232 to 112, and measured redundancy from 32.3 percent
to 28.6 percent. A regression test pins the refusal to chain.

## Duplication in the combined pool

At cosine 0.85 on character 4 grams, complete linkage, across 9,699 documents:

<table>
<tr><th>Class</th><th>Documents</th><th>Distinct clusters</th><th>Redundant</th></tr>
<tr><td>FRAUD</td><td>3,976</td><td>2,450</td><td>38.4 percent</td></tr>
<tr><td>SPAM</td><td>1,367</td><td>999</td><td>26.9 percent</td></tr>
<tr><td>NORMAL</td><td>4,356</td><td>3,479</td><td>20.1 percent</td></tr>
<tr><td><b>Combined</b></td><td><b>9,699</b></td><td><b>6,923</b></td><td><b>28.6 percent</b></td></tr>
</table>

Fraud is the worst affected, which fits: advance fee campaigns are sent from templates with
names and figures swapped. Phase 2 measured 17.1 percent byte identical duplication in the
fraud corpus, and near duplicate matching more than doubles that to 38.4 percent.

## The reference's 1,000 per class is unreachable

The spam pool holds 1,368 rows, which reduce to 999 distinct clusters, and to 994 after
dropping label contradictions. A balanced corpus of 1,000 per class therefore cannot be built
without readmitting copies of the same email.

This is not a small technicality. It means the reference's headline corpus size was only
achievable because it did not deduplicate, and that 26.9 percent of its spam class is
redundant with the rest of that class.

Decision: 900 per class, which leaves every class genuine sampling headroom rather than
consuming a pool whole. Spam draws 900 of 994, or 91 percent. Fraud draws 37 percent and
Normal 26 percent.

## Label contradictions between the two sources

Five clusters, covering 12 documents, carry two labels. Every one of them is FRAUD against
SPAM, and reading them explains why: they are advance fee scams that appear in both the CLAIR
fraud mailbox and the Enron spam corpus, because the same campaigns reached Enron employees.

```
cluster 73:  letter from : daniel kabila  investment offer ...
cluster 187: i am prince fayad w . bolkiah , the eldest son of prince jefri bolkiah ...
cluster 330: i am mrs . fatima rasheed khalifa a widow to late sheik mohammed ...
cluster 413: i am david wood the bank manager ...
cluster 886: greetings from u . a . e  hello my dear ...
```

Decision: drop the contradicted clusters. Taking a majority label would be inventing
information, since if two sources disagree about what an email is we have no grounds to pick
a winner.

Worth stating plainly in the README: this is not merely a data cleaning note, it is a fact
about the task. Advance fee fraud is a subset of spam, so the boundary between those two
classes is genuinely fuzzy, and a perfect score on this three class problem would be a
warning sign rather than an achievement.

## The corpus

<table>
<tr><th>Split</th><th>FRAUD</th><th>SPAM</th><th>NORMAL</th><th>Total</th></tr>
<tr><td>Train</td><td>720</td><td>720</td><td>720</td><td>2,160</td></tr>
<tr><td>Test</td><td>180</td><td>180</td><td>180</td><td>540</td></tr>
</table>

Seed 20260914, recorded in the report. Clusters appearing on both sides of the split: zero,
asserted by the build rather than hoped for.

Written to `data/processed/corpus.csv`, `train.csv` and `test.csv`, with columns label,
source, source_row, cluster, words, chars and text.

## A new shortcut to watch: body length

<table>
<tr><th>Class</th><th>Median words</th><th>Mean</th><th>Min</th><th>Max</th></tr>
<tr><td>FRAUD</td><td>424</td><td>434</td><td>24</td><td>2,266</td></tr>
<tr><td>SPAM</td><td>126</td><td>251</td><td>5</td><td>6,128</td></tr>
<tr><td>NORMAL</td><td>204</td><td>313</td><td>5</td><td>4,834</td></tr>
</table>

Fraud messages are more than three times the length of spam messages at the median, and that
gap is large enough that document length alone is a usable classifier. Some of it is real,
since advance fee fraud needs space to tell a story, but some is an artifact of which corpus
each class came from. Logged as D11 and handed to phase 5, which will measure how far length
alone gets you. If a length only baseline scores well, every later result has to be read
against it.

## Tests

52 tests pass. New coverage in `tests/test_corpus.py` for prefix stripping, normalisation,
the refusal to chain under complete linkage, determinism of clustering and of sampling under
a fixed seed, the longest member being chosen as a cluster representative, the error raised
when a pool is too small, stratification, and the absence of cluster overlap across the split.

One regression test exists because of a latent bug found while writing the tests. The
vectoriser used `min_df=3`, tuned for 9,700 documents. On a handful of documents that
requires an n gram to appear in nearly all of them, so the vocabulary collapsed to shared
whitespace, every vector looked alike, and three unrelated texts merged into one cluster with
no error raised. `min_df` now scales with corpus size. The full corpus run is unaffected, and
the figures above are unchanged by the fix.
