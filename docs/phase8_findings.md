# Phase 8 findings: the model sweep, scored by nested cross validation

Seven estimators crossed with two vectorisers, hyperparameters searched in an inner loop so
that no reported score was tuned on the data it reports.

Entry point `scripts/run_model_sweep.py`, search space in `src/models.py`, tests in
`tests/test_models.py`. Numbers in `docs/phase8_sweep.json`, figure in
`reports/figures/phase8_sweep.png`.

Outer five fold stratified cross validation for scoring, inner three fold for the
hyperparameter search, seed 20260914, training split only. The test split is not touched
until phase 9.

## Ranking

<table>
<tr><th>Rank</th><th>Model</th><th>Features</th><th>Nested macro F1</th><th>Std</th><th>Tuned parameters</th></tr>
<tr><td>1</td><td>voting ensemble</td><td>tfidf</td><td><b>0.9740</b></td><td>0.0027</td><td>lr C=5.0, nb alpha=0.1, bigrams</td></tr>
<tr><td>2</td><td>multinomial naive Bayes</td><td>tfidf</td><td>0.9721</td><td>0.0049</td><td>alpha=0.1, bigrams</td></tr>
<tr><td>3</td><td>linear SVM</td><td>tfidf</td><td>0.9703</td><td>0.0050</td><td>C=1.0, bigrams</td></tr>
<tr><td>4</td><td>complement naive Bayes</td><td>tfidf</td><td>0.9702</td><td>0.0054</td><td>alpha=0.1, bigrams</td></tr>
<tr><td>5</td><td>logistic regression</td><td>tfidf</td><td>0.9690</td><td>0.0067</td><td>C=20.0, bigrams</td></tr>
<tr><td>6</td><td>voting ensemble</td><td>count</td><td>0.9685</td><td>0.0067</td><td>lr C=5.0, nb alpha=0.1, bigrams</td></tr>
<tr><td>7</td><td>multinomial naive Bayes</td><td>count</td><td>0.9661</td><td>0.0038</td><td>alpha=0.1, bigrams</td></tr>
<tr><td>8</td><td>k nearest neighbours</td><td>tfidf</td><td>0.9643</td><td>0.0079</td><td>k=7, uniform, bigrams</td></tr>
<tr><td>9</td><td>complement naive Bayes</td><td>count</td><td>0.9632</td><td>0.0035</td><td>alpha=1.0, bigrams</td></tr>
<tr><td>10</td><td>logistic regression</td><td>count</td><td>0.9523</td><td>0.0110</td><td>C=20.0, bigrams</td></tr>
<tr><td>11</td><td>random forest</td><td>count</td><td>0.9517</td><td>0.0182</td><td>sqrt features, leaf=1, bigrams</td></tr>
<tr><td>12</td><td>linear SVM</td><td>count</td><td>0.9509</td><td>0.0110</td><td>C=0.05, bigrams</td></tr>
<tr><td>13</td><td>random forest</td><td>tfidf</td><td>0.9507</td><td>0.0157</td><td>sqrt features, leaf=1, unigrams</td></tr>
<tr><td>14</td><td>k nearest neighbours</td><td>count</td><td>0.8943</td><td>0.0154</td><td>k=3, uniform, bigrams</td></tr>
</table>

Only two of fourteen configurations sit within one standard deviation of the winner: the
ensemble itself and multinomial naive Bayes on tfidf.

## Four things worth saying about that table

**TF IDF beats raw counts for every model except random forest.** The margin is small for
the naive Bayes variants, which are relatively insensitive to feature scaling, and large
for the margin and distance based models: linear SVM gains 0.0194 and k nearest neighbours
gains 0.0700. That is the expected direction and the size of the gap tracks how much each
model cares about feature magnitude.

**k nearest neighbours is the clearest illustration.** On raw counts it is the worst
configuration in the sweep at 0.8943; on tfidf it reaches 0.9643, a gain of 0.07. Cosine
distance between unnormalised count vectors is dominated by document length, and phase 5
measured fraud documents at 4.35 times the median length of spam. So on count features the
nearest neighbours of a document are largely the documents of similar length, which is
D11's length shortcut arriving by a different route. TF IDF is L2 normalised, which removes
it.

**Random forest is second worst on both representations**, at 0.9517 and 0.9507, and it
carries the widest fold to fold variance in the sweep at 0.0182. Tree ensembles do poorly
on high dimensional sparse text: each split can consider only a handful of the 23,000
features, and the signal here is spread thinly across many terms rather than concentrated
in a few. It is also by far the most expensive model in the sweep at roughly 25 seconds per
configuration against 4 for naive Bayes.

**The ensemble's win is not really a win.** It beats multinomial naive Bayes by 0.0019,
which is well inside a standard deviation of 0.0027, while costing about six times the
compute and giving up the ability to read a coefficient. Naive Bayes with a four value grid
is the honest recommendation for this problem, and phase 9 reports both.

## Linear SVM did not converge, and the first run's scores were meaningless

On CountVectorizer features, `LinearSVC` at its default 1,000 iteration cap failed to
converge for every grid point, emitting a `ConvergenceWarning` on each fit. A model that
stopped early is reporting an arbitrary point on its optimisation path rather than a fitted
model, so those scores were not measurements of anything.

The cause is conditioning. Raw term counts are unscaled and span orders of magnitude, so the
margin problem is badly conditioned. TF IDF features are L2 normalised and converge without
help, which is why the warning appeared on one representation and not the other.

Raising the cap to 20,000 iterations resolves it for all eight grid points, verified
explicitly by promoting `ConvergenceWarning` to an error and refitting each one. The whole
sweep was rerun afterwards.

The general point: a convergence warning inside a cross validation loop is easy to scroll
past, and scikit learn emits one per fit so there were hundreds of them. They were the only
evidence that a quarter of the sweep was reporting noise.

## Optimism, and a confounded first attempt at measuring it

The usual workflow, and the one the reference follows, searches a grid with cross validation
and quotes the winner's cross validated score. That number is optimistic, because the winner
was chosen by looking at those same folds and its advantage includes whatever it gained from
fold noise. Nested cross validation separates the two jobs.

<table>
<tr><th>Optimism gap, corrected measurement</th><th>Macro F1</th></tr>
<tr><td>Mean across fourteen configurations</td><td>+0.0007</td></tr>
<tr><td>Largest, random forest on tfidf</td><td>+0.0033</td></tr>
<tr><td>Smallest</td><td>0.0019 lower</td></tr>
</table>

The first version of this measurement was wrong, and the way it announced itself is worth
recording. It produced a **mean gap of 0.0019 in the negative direction**, which is
impossible for a bias that only ever inflates.

The cause was a confound of our own making. The nested figure came from five outer folds, so
each model trained on 80 percent of the data, while the non nested figure came from the
three fold inner splitter, training on 67 percent. The non nested number was therefore being
depressed by having less training data at the same moment as it was being inflated by
selection. The two effects partly cancelled and the smaller one won.

Both searches now use the same five folds, so training set size is identical and the only
remaining difference is whether the winner was chosen on the data it is scored on. The gap
turns positive, as it must.

Honest reading of the corrected number: optimism here is real but small, at most 0.0033.
That is what small grids and a strong signal produce. It would matter far more with a large
grid, a weak signal, or a smaller corpus, and the reason to measure it is that you cannot
know which case you are in without doing so.

## Against the phase 7 baselines

<table>
<tr><th>Reference point</th><th>Macro F1</th><th>Best model margin</th></tr>
<tr><td>Best model, voting ensemble on tfidf</td><td>0.9740</td><td></td></tr>
<tr><td>Formatting only, the bar to clear</td><td>0.7013</td><td>0.2727 above</td></tr>
<tr><td>Keyword rule, 25 words per class</td><td>0.5737</td><td>0.4003 above</td></tr>
<tr><td>Random guessing</td><td>0.3355</td><td>0.6385 above</td></tr>
</table>

The model clears the real bar by a wide margin, which is the result phase 7 existed to make
interpretable. Worth noting that every one of the fourteen configurations, including the
worst, clears 0.7013 comfortably.

## Tests

190 tests pass. New coverage in `tests/test_models.py`: every model named in the brief being
present, both vectorisers present and dropping hapax terms, every entry being a classifier,
every grid non empty and addressed to the correct pipeline step, grid size arithmetic, the
project seed reaching every seeded estimator, the ensemble using soft voting with members
that all expose `predict_proba`, linear SVM being excluded from it for that reason, and every
model actually fitting with every vectoriser so a configuration error surfaces in two seconds
rather than after several minutes of nested search.
