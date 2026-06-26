# Data

This directory is for dataset notes, small examples, and data-loading helpers.

Only lightweight, reviewable files should be committed here. Downloaded datasets
and generated artifacts should stay local.

## Tracked Structure

```text
data/
  README.md
  utils.py
  examples/
```

- `README.md` documents the data directory.
- `utils.py` holds small shared data helpers.
- `examples/` holds small checked-in examples.

Everything else under `data/` should stay untracked.

## Train and Eval Splits

Generated train and eval splits should stay local:

```text
data/train/
data/eval/
```

Rows should contain only the canonical Sudoku fields:

```json
{"sudoku": "...", "solution": "..."}
```

Prompt and completion formatting should happen outside the data files.

## Radcliffe Kaggle Dataset

Download the Kaggle dataset into a local, untracked directory:

```bash
uv sync
source .venv/bin/activate

mkdir -p data/radcliffe-3-million-sudoku-puzzles-with-ratings

kaggle datasets download \
  -d radcliffe/3-million-sudoku-puzzles-with-ratings \
  -f sudoku-3m.csv \
  -p data/radcliffe-3-million-sudoku-puzzles-with-ratings

unzip data/radcliffe-3-million-sudoku-puzzles-with-ratings/sudoku-3m.csv.zip \
  -d data/radcliffe-3-million-sudoku-puzzles-with-ratings

rm data/radcliffe-3-million-sudoku-puzzles-with-ratings/sudoku-3m.csv.zip
```

The extracted CSV is:

```text
data/radcliffe-3-million-sudoku-puzzles-with-ratings/sudoku-3m.csv
```

## Rohan Rao Kaggle Dataset

Download the Kaggle dataset into a local, untracked directory:

```bash
uv sync
source .venv/bin/activate

mkdir -p data/rohanrao-sudoku

kaggle datasets download \
  -d rohanrao/sudoku \
  -f sudoku.csv \
  -p data/rohanrao-sudoku

unzip data/rohanrao-sudoku/sudoku.csv.zip \
  -d data/rohanrao-sudoku

rm data/rohanrao-sudoku/sudoku.csv.zip
```

The extracted CSV is:

```text
data/rohanrao-sudoku/sudoku.csv
```

## Bryan Park Kaggle Dataset

Download the Kaggle dataset into a local, untracked directory:

```bash
uv sync
source .venv/bin/activate

mkdir -p data/bryanpark-sudoku

kaggle datasets download \
  -d bryanpark/sudoku \
  -f sudoku.csv \
  -p data/bryanpark-sudoku

unzip data/bryanpark-sudoku/sudoku.csv.zip \
  -d data/bryanpark-sudoku

rm data/bryanpark-sudoku/sudoku.csv.zip
```

The extracted CSV is:

```text
data/bryanpark-sudoku/sudoku.csv
```
