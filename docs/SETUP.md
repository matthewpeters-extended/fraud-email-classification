# Setup

Reproduces the environment from a clean checkout. macOS, Homebrew Python 3.12.

## 1. Create the virtual environment

The system Python 3.9 at `/usr/bin/python3` is not used. Homebrew Python 3.12 is.

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
```

```bash
source .venv/bin/activate && python --version
```

Expect `Python 3.12.x`.

## 2. Install packages

```bash
source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
```

## 3. Download the NLTK corpora

One time, into the virtual environment.

```bash
source .venv/bin/activate && python -c "import nltk; [nltk.download(p) for p in ('stopwords','punkt','punkt_tab','wordnet','omw-1.4')]"
```

## 4. Fetch the data

`data/` is git ignored, so a fresh checkout has no data. The download is scripted.

```bash
source .venv/bin/activate && python scripts/download_data.py
```

This fetches `Datasets.zip` from the reference repository, verifies its checksum, extracts to
`data/raw`, and prints the row counts for cross checking against `docs/sources.md`.

## 5. Build the corpus and run the experiment

```bash
source .venv/bin/activate && python scripts/build_dataset.py
```

```bash
source .venv/bin/activate && python scripts/run_experiment.py
```

## 6. Tests

```bash
source .venv/bin/activate && python -m pytest -q
```

## 7. Prose style check

House style for this project is markdown prose with no dash characters. The check enforces
it, allowing dashes only inside code fences, inline code, HTML code elements, URLs and
markdown link targets.

```bash
source .venv/bin/activate && python scripts/check_prose.py
```

## 8. Verify the write up

The README is the deliverable, so its numbers are checked against the reports that produced
them rather than trusted. Run this after any rerun of an analysis phase.

```bash
source .venv/bin/activate && python scripts/verify_readme.py
```

## Lock file

After any dependency change, refresh the lock.

```bash
source .venv/bin/activate && pip freeze > requirements.lock.txt
```
