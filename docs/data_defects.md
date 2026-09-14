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
