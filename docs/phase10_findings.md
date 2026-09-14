# Phase 10 findings: the fraud class as a detector

Phase 9 reported macro F1 on a corpus that is one third fraud by construction. This phase
answers the three things that leaves open, all of which matter before anyone would use the
model.

Entry point `scripts/fraud_deep_dive.py`, logic in `src/threshold.py`, tests in
`tests/test_threshold.py`. Numbers in `docs/phase10_fraud.json`, figure in
`reports/figures/phase10_fraud.png`.

The threshold was selected on cross validated predictions over the **training** split. The
holdout is used only to report what the selected threshold does. Nothing in this phase was
chosen by looking at test results, and no new configuration was scored, so the holdout
ledger still records two.

## The cost assumptions, stated rather than implied

<table>
<tr><th>Outcome</th><th>Assumed cost</th></tr>
<tr><td>Fraud left in the inbox</td><td>100</td></tr>
<tr><td>Fraud misfiled as spam</td><td>2</td></tr>
<tr><td>Legitimate mail flagged as fraud</td><td>20</td></tr>
<tr><td>Spam flagged as fraud</td><td>0.5</td></tr>
</table>

These are assumptions, not measurements, and only the ratios matter. The important one is
the first against the second. Both are a fraud email the model failed to label fraud, and
both count identically in macro F1, but one is sitting in front of the user and the other is
in the spam folder where it can do nothing. A framing that cannot tell those apart cannot
say anything useful about deployment.

## Where the threshold should sit: almost anywhere

Selected on the training folds, the cheapest threshold is **0.40** rather than the default
0.50. What that buys is close to nothing.

<table>
<tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>Cost on train</th><th>Fraud in inbox</th><th>Legitimate flagged</th></tr>
<tr><td>0.30</td><td>0.9596</td><td>0.9889</td><td>148.5</td><td>1</td><td>1</td></tr>
<tr><td><b>0.40, chosen</b></td><td>0.9609</td><td>0.9889</td><td><b>148.0</b></td><td>1</td><td>1</td></tr>
<tr><td>0.50, default</td><td>0.9621</td><td>0.9861</td><td>151.5</td><td>1</td><td>1</td></tr>
<tr><td>0.54</td><td>0.9620</td><td>0.9833</td><td>253.5</td><td>2</td><td>1</td></tr>
</table>

The cost curve is flat from 0.26 to 0.50 and moves by 3.5 out of roughly 150. On the
holdout, at the chosen threshold against the default:

<table>
<tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>Cost</th><th>Fraud reaching the inbox</th><th>Legitimate mail flagged</th></tr>
<tr><td>0.40, chosen on train</td><td>0.9570</td><td>0.9889</td><td>8.0</td><td>0</td><td>0</td></tr>
<tr><td>0.50, default</td><td>0.9568</td><td>0.9833</td><td>10.0</td><td>0</td><td>0</td></tr>
</table>

Both give zero fraud in the inbox and zero legitimate mail destroyed. Average precision on
the holdout is 0.9905, and the precision recall curve sits against the top right corner,
which is why the threshold has so little to do.

The honest conclusion: **threshold tuning is not where the value is on this problem.** It is
worth doing because you cannot know that without doing it, and because the exercise produces
the cost framing that the rest of the phase depends on. But reporting a carefully tuned
threshold as an achievement here would be overselling a move worth two units of cost out of
ten.

### The threshold is also robust to the cost assumptions

Because the numbers above are invented, the sensible check is whether the answer depends on
them.

<table>
<tr><th>Cost of fraud reaching the inbox</th><th>Selected threshold</th><th>Precision</th><th>Recall</th></tr>
<tr><td>5</td><td>0.40</td><td>0.9609</td><td>0.9889</td></tr>
<tr><td>50</td><td>0.40</td><td>0.9609</td><td>0.9889</td></tr>
<tr><td>250</td><td>0.40</td><td>0.9609</td><td>0.9889</td></tr>
<tr><td>500</td><td>0.12</td><td>0.8938</td><td>0.9931</td></tr>
</table>

The same threshold is selected across a fifty fold range of cost ratios, moving only when
missing a fraud email is made 250 times worse than a false alarm on legitimate mail. So the
operating point does not rest on the specific numbers chosen, which is the useful thing to
know about an assumption you had to invent.

## Calibration

<table>
<tr><th>Split</th><th>Brier score</th><th>Expected calibration error</th></tr>
<tr><td>Train, cross validated</td><td>0.0197</td><td>0.0331</td></tr>
<tr><td>Holdout</td><td>0.0188</td><td>0.0299</td></tr>
</table>

Well calibrated in aggregate, and a coin flip would score 0.25 on Brier. But the bins show a
consistent shape rather than random scatter:

<table>
<tr><th>Predicted probability bin</th><th>Documents</th><th>Mean predicted</th><th>Observed rate</th><th>Gap</th></tr>
<tr><td>0.0 to 0.1</td><td>337</td><td>0.025</td><td>0.006</td><td>0.019 too high</td></tr>
<tr><td>0.1 to 0.2</td><td>15</td><td>0.145</td><td>0.000</td><td>0.145 too high</td></tr>
<tr><td>0.2 to 0.3</td><td>2</td><td>0.255</td><td>0.000</td><td>0.255 too high</td></tr>
<tr><td>0.8 to 0.9</td><td>34</td><td>0.858</td><td>0.853</td><td>0.005 too high</td></tr>
<tr><td>0.9 to 1.0</td><td>141</td><td>0.953</td><td>0.993</td><td>0.040 too low</td></tr>
</table>

The model is slightly over confident at the bottom and slightly under confident at the top:
its probabilities are pulled toward the middle. That is the expected signature of a soft
voting ensemble, which averages three members and so cannot produce a probability more
extreme than its most extreme member. It is a reason to prefer the naive Bayes model if
calibrated probabilities mattered downstream, and a reason not to bother here, since the
distortion is small and the threshold is insensitive anyway.

## PLAN 8.6: precision outside a balanced corpus

This is the most important result in the phase.

Recall and the per class false alarm rates are properties of the classifier and carry over
to any mailbox. The class proportions do not, and precision depends on them. Measured rates
carried forward: recall 0.9889, false alarm rate on spam 0.0444, false alarm rate on
legitimate mail **0.0000**.

Per 100,000 messages:

<table>
<tr><th>Mailbox</th><th>Precision</th><th>True fraud caught</th><th>Fraud missed</th><th>False alarms from spam</th><th>False alarms from legitimate mail</th><th>False alarms per catch</th></tr>
<tr><td>Corpus as built, 1 in 3 fraud</td><td>0.9570</td><td>32,963</td><td>370</td><td>1,481</td><td>0</td><td>0.0</td></tr>
<tr><td>Spam heavy inbox, 5 percent</td><td>0.7120</td><td>4,944</td><td>56</td><td>2,000</td><td>0</td><td>0.4</td></tr>
<tr><td>Filtered inbox, 0.5 percent</td><td>0.2705</td><td>494</td><td>6</td><td>1,333</td><td>0</td><td>2.7</td></tr>
<tr><td>Well filtered inbox, 0.1 percent</td><td>0.1820</td><td>99</td><td>1</td><td>444</td><td>0</td><td>4.5</td></tr>
</table>

**Precision falls from 0.9570 to 0.1820.** Nothing about the model changed. Recall is
identical, both false alarm rates are identical, and only the mix of what it is shown has
changed. The collapse is arithmetic.

That is the case against quoting a precision figure from a balanced benchmark, and it
applies to the reference solution and to almost every tutorial on this dataset. A model
reported at 96 percent precision would flag five messages for every real fraud it caught in
a well filtered mailbox.

### But read the last two columns before concluding it is bad news

Every single false alarm comes from spam. None comes from legitimate mail, because the
measured false alarm rate on legitimate mail is zero: not one of the 180 legitimate emails in
the holdout was flagged as fraud, at either threshold.

So what the precision collapse actually describes is a detector that files junk as the wrong
kind of junk. In a triage system where both fraud and spam leave the inbox, that costs
almost nothing. Under the cost model above, 444 spam false alarms at 0.5 each cost 222, while
the one missed fraud reaching an inbox costs 100.

The two statements have to be made together. Quoting 0.957 without the prevalence correction
is misleading. Quoting 0.182 without saying where the false alarms land is equally
misleading in the other direction.

## What this phase changes about the project's conclusions

1. The headline macro F1 of 0.9684 stands, and the threshold behind it barely matters.
2. Precision claims must carry a prevalence assumption. The README will state the operating
   assumption explicitly and give the projection table.
3. The deployment relevant claim is not the precision figure. It is that zero fraud emails
   reached the inbox and zero legitimate emails were destroyed, and that every false alarm
   landed on spam.
4. If calibrated probabilities were needed downstream, prefer multinomial naive Bayes over
   the ensemble. The ensemble's averaging pulls probabilities toward the middle.
5. The corpus cannot answer how this behaves on phishing, which phase 9 already flagged: the
   one genuine fraud miss was a bank phishing alert, and the CLAIR collection predates the
   rise of that genre. That is a limitation to state, not to paper over.

## Tests

235 tests pass. New coverage in `tests/test_threshold.py`: the cost model separating both
kinds of missed fraud and both kinds of false alarm, perfect separation costing nothing,
recall never rising with the threshold, counts reconciling at every threshold, a zero
threshold flagging everything, the cheapest threshold finding the minimum and breaking ties
conservatively, precision falling with prevalence while recall does not, the projection
conserving the mailbox, a zero false alarm rate on legitimate mail staying zero, perfect
specificity giving perfect precision at any prevalence, Brier score behaviour at the extremes,
reliability bins partitioning the documents and reporting nothing for empty bins, and
calibration error being zero for a perfectly calibrated score.
