# Phase 2 findings: parsing the fraud mailbox

Goal: turn `fradulent_emails.txt`, a single 17.3 MB concatenated mailbox, into one row per
message with a clean text body, and report parse health honestly rather than assuming it.

Entry point `scripts/parse_fraud.py`, logic in `src/fraud_mailbox.py`, tests in
`tests/test_fraud_mailbox.py`. Machine readable output in
`data/interim/phase2_parse_report.json`.

## The message count is 3,978, and every figure on record was wrong

<table>
<tr><th>Source of the figure</th><th>Count</th><th>Status</th></tr>
<tr><td>Reference repository README</td><td>4,075</td><td>wrong, unsubstantiated</td></tr>
<tr><td>Commonly cited CLAIR figure</td><td>3,977</td><td>wrong by one</td></tr>
<tr><td>Naive split on the literal string "From r"</td><td>3,976</td><td>wrong by two</td></tr>
<tr><td><b>Split on the mbox envelope pattern</b></td><td><b>3,978</b></td><td>correct</td></tr>
</table>

Two envelope lines in this corpus do not begin with `From r`:

```
L72425:  From R@T  Fri May 14 07:49:12 2004
L285198: From joykem@centrum.sk Fri Dec 22 13:07:36 2006
```

Any parser keying on that literal prefix silently welds those two messages onto the end of
their predecessors, producing two corrupted records and losing two real ones. The parser
instead requires a full envelope signature: the word `From` at the start of a line, an optional
sender token, then a weekday, a month, a day, a time and a year. That pattern is specific enough
that ordinary prose beginning with the word "From" is not mistaken for a message boundary, which
matters because advance fee emails very often open with "From the desk of". A regression test
pins both behaviours.

A related trap: the file contains 3,986 lines beginning `Subject:`, more than the message count,
because quoted and forwarded material inside bodies carries its own subject lines. Splitting on
subject lines would have been wrong in the other direction.

## Parse health

<table>
<tr><th>Measure</th><th>Result</th></tr>
<tr><td>Envelope splits</td><td>3,978</td></tr>
<tr><td>Usable bodies, at least five words</td><td>3,976, or 99.95 percent</td></tr>
<tr><td>Unusable</td><td>2</td></tr>
<tr><td>Multipart messages</td><td>876</td></tr>
<tr><td>HTML only, tags stripped to recover the text</td><td>141</td></tr>
<tr><td>Declared charsets unknown to Python</td><td>63, recovered by falling back to utf8</td></tr>
<tr><td>Body length, characters</td><td>min 89, median 2,546, mean 2,639, max 123,388</td></tr>
</table>

Declared charsets are led by us ascii at 1,389, iso 8859 1 at 846, utf8 at 215, then a long tail
including iso 8859 15, iso 8859 2 and windows 1251.

The two unusable messages are both genuine and correctly excluded:

* Index 406, subject `URGENT ASSISTANCE`. Headers are intact, the body is empty. A truncated
  record in the source corpus.
* Index 1537, subject encoded as base64, decoding to `Cialis. Only 3 usd per dose!`. A
  multipart message carrying images and 24 characters of text. See D6 below.

## Quoted printable decoding resolves half of defect D4

Parsing through the standard library email module, rather than reading the file as plain text,
decodes Content Transfer Encoding properly.

<table>
<tr><th>Measure</th><th>Occurrences</th></tr>
<tr><td>"=20" in the raw mailbox</td><td>6,423</td></tr>
<tr><td>"=20" surviving into parsed bodies</td><td>9</td></tr>
</table>

That is 99.9 percent resolved. The nine survivors are literal text inside message bodies rather
than encoding artifacts. This removes the MIME artifact half of defect D4. The casing and
punctuation spacing half remains and is handled by normalisation at phase 6.

## Two new defects found

Both recorded in `docs/data_defects.md`.

**D6, label noise in the fraud corpus. Low severity, accepted.** The corpus is billed as advance
fee fraud but contains a small amount of ordinary spam. Measured by keyword signature across the
3,976 usable messages: 93.8 percent carry an advance fee signature, 0.1 percent carry a pharmacy
or replica goods signature instead, 0.2 percent carry both, and 5.9 percent carry neither strong
signature and are mostly short or unusual messages. Two messages are unambiguously pharmacy
spam. At 0.1 percent this is below the noise floor of anything we will measure, so it is recorded
and not acted upon.

**D7, heavy duplication inside the fraud corpus. High severity, open.** This is the measurement
that confirms the concern raised as 8.3 in `PLAN.md`.

<table>
<tr><th>Match rule</th><th>Unique</th><th>Repeated groups</th><th>Redundant copies</th><th>Share of corpus</th><th>Largest group</th></tr>
<tr><td>Exact body</td><td>3,298</td><td>475</td><td>678</td><td>17.1 percent</td><td>13</td></tr>
<tr><td>Body normalised for case and punctuation</td><td>3,257</td><td>498</td><td>719</td><td>18.1 percent</td><td>16</td></tr>
<tr><td>First 200 normalised characters</td><td>2,985</td><td>608</td><td>991</td><td>24.9 percent</td><td>20</td></tr>
</table>

One in six fraud messages is a byte identical copy of another, and one in four shares an opening
with another. The single worst case is one message appearing twenty times.

This matters because the reference samples 1,000 messages from this pool and then splits into
train and test. With 17 percent duplication, copies of the same message land on both sides of
that split, so part of the reported test score is memorisation rather than generalisation. The
effect is not subtle at this rate.

Decision: phase 3 deduplicates before sampling, and phase 3 assigns near duplicate clusters to
one side of the split as a unit so that no cluster spans it. The scale of the correction gets
reported.

## Output

`data/interim/fraud_messages.csv`, 3,976 rows, columns index, date, from_addr, subject,
content_type, charset, was_multipart, was_html, body_chars, notes, body.

Casing and punctuation are deliberately left untouched at this stage. Normalising them is a
modelling decision that belongs inside the pipeline at phase 6, where it can be applied to all
three classes identically and its effect measured.

## Tests

23 tests pass. They cover the envelope regex against all three real envelope forms found in the
corpus and against five prose forms that must not match, the loss of two messages under the naive
prefix rule, quoted printable decoding, HTML fallback with script stripping, unknown charset
recovery, empty and short body rejection, and the preservation of casing through the tidy step.

One test guards the test harness itself. The first version of the synthetic mailbox builder used
`textwrap.dedent`, which silently left the envelope lines indented once a multiline body
introduced an unindented line, so a passing parser looked broken. That is now pinned.
