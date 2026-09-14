# Sources and attribution

## Reference solution

The method specification for this project comes from the public repository
`SimarjotKaur/Email-Classifier`.

* https://github.com/SimarjotKaur/Email-Classifier

That repository assembles a three class email corpus of 3,000 messages and compares SVM, KNN,
Naive Bayes, Decision Trees, Logistic Regression and ensemble methods across CountVectorizer and
TF IDF feature extraction. Its files are `Email_Classification.py`, `Extract_email.py`,
`Datasets.zip` and a README.

This project reuses the corpus construction and reports against the same task. The pipeline,
the evaluation protocol and the leakage analysis are our own. Defects we inherit and repair are
enumerated in section 8 of `PLAN.md`.

## Dataset as acquired

Single acquisition, `Datasets.zip` from the reference repository.

* https://github.com/SimarjotKaur/Email-Classifier/raw/master/Datasets.zip
* Verified 2026 09 14: HTTP 200, content type application/zip, 10,754,307 bytes
* SHA256 `7d3d07a1bc58ecbdcf03533515e5256d61dda4fd44a4902bd35fdf0910db22fe`, recorded in `docs/data_defects.md` and `data/raw/manifest.json`

Contents as verified at phase 1 on 2026 09 14. The member names differ from the reference
README's prose, which described `fraudulent_emails.txt` and `emails.csv`. These are the real
names, including the upstream author's misspelling of "fradulent", preserved so the download
stays reproducible.

<table>
<tr><th>Member</th><th>Bytes</th><th>Description</th><th>Observed count</th><th>Class</th></tr>
<tr><td><code>fradulent_emails.txt</code></td><td>17,344,435</td><td>CLAIR advance fee fraud mailbox, concatenated</td><td>3,976 messages</td><td>Fraud</td></tr>
<tr><td><code>spam_normal_emails.csv</code></td><td>8,954,755</td><td>Enron derived corpus, columns text and spam</td><td>5,728 rows, 1,368 spam and 4,360 ham</td><td>Spam and Normal</td></tr>
<tr><td><code>final_dataset.csv</code></td><td>5,213,165</td><td>The author's prebuilt corpus, columns Email, Label, Length</td><td>3,000, balanced</td><td>cross check only</td></tr>
</table>

The reference README claims 4,075 fraud emails. We observe 3,976 and the canonical CLAIR figure
is cited at 3,977, so the reference figure is treated as unsubstantiated. Recorded as D1 in
`docs/data_defects.md`.

## Upstream primaries

These are the original publications of the two corpora. They are the fallback if the reference
zip becomes unavailable, and they are the figures we cross check the zip against.

* Fraudulent E mail Corpus, published by Rachael Tatman on Kaggle, the CLAIR collection of
  advance fee fraud emails
  https://www.kaggle.com/datasets/rtatman/fraudulent-email-corpus
* Spam email Dataset, published by jackksoncsie on Kaggle, `emails.csv` with columns text and
  spam, 5,728 rows
  https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset

Both require a free Kaggle account to download, which is the reason the reference zip is the
primary acquisition path.

Related public corpora considered and not used, recorded so the choice is legible:

* Enron Spam Data, a cleaned single file version of the same underlying Enron spam corpus
  https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data
* Phishing Email Dataset, larger and more modern, but its label taxonomy does not separate spam
  from fraud, which is the distinction this project is built on
  https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

## Licensing note

The underlying corpora are published for research use. Before the public push, phase 1 records
the stated licence terms found on each Kaggle page above, and the README states them. No email
content is redistributed in this repository. `data/` is git ignored in full, and the download is
reproduced by script rather than committed.

## Tooling

* scikit learn, for vectorisers, models, pipelines and metrics
* NLTK, for stopwords, tokenisation and lemmatisation
* pandas and numpy, for data handling
* matplotlib and seaborn, for figures
