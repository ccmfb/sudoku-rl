# Data

This directory contains small data helpers, checked-in examples, and local-only Sudoku datasets.

Large raw datasets and generated outputs are ignored by Git. Keep them local and regenerate them from the scripts here.

## Canonical Format

Sudoku boards are represented as 81-character strings.

- `sudoku`: digits `1`-`9` plus `.` for missing cells
- `solution`: digits `1`-`9` only
- JSONL rows contain only canonical fields:

```json
{"sudoku": "...", "solution": "..."}
```

Prompt and completion formatting should happen outside the data files.

## Committed Files

```text
data/
  README.md
  utils.py
  split.py
  sample.py
  mask.py
  examples/
    radcliffe_10.jsonl
```

`utils.py` validates and normalizes source rows. `split.py`, `sample.py`, and `mask.py` build the local train and eval artifacts.

## Local Source Datasets

These paths are ignored by Git and may exist only in a local checkout.

| Path | Source | CSV fields | Blank cells | Rows |
| --- | --- | --- | --- | --- |
| `data/radcliffe-3-million-sudoku-puzzles-with-ratings/sudoku-3m.csv` | `radcliffe/3-million-sudoku-puzzles-with-ratings` | `id,puzzle,solution,clues,difficulty` | `.` | 3,000,000 |
| `data/rohanrao-sudoku/sudoku.csv` | `rohanrao/sudoku` | `puzzle,solution` | `0`, converted to `.` | 9,000,000 |
| `data/bryanpark-sudoku/sudoku.csv` | `bryanpark/sudoku` | `quizzes,solutions` | `0`, converted to `.` | 1,000,000 |

## Download Sources

Download the source datasets with the Kaggle CLI after `uv sync` and Kaggle authentication.

Radcliffe:

```bash
mkdir -p data/radcliffe-3-million-sudoku-puzzles-with-ratings

uv run kaggle datasets download \
  -d radcliffe/3-million-sudoku-puzzles-with-ratings \
  -f sudoku-3m.csv \
  -p data/radcliffe-3-million-sudoku-puzzles-with-ratings

unzip data/radcliffe-3-million-sudoku-puzzles-with-ratings/sudoku-3m.csv.zip \
  -d data/radcliffe-3-million-sudoku-puzzles-with-ratings

rm data/radcliffe-3-million-sudoku-puzzles-with-ratings/sudoku-3m.csv.zip
```

Rohan Rao:

```bash
mkdir -p data/rohanrao-sudoku

uv run kaggle datasets download \
  -d rohanrao/sudoku \
  -f sudoku.csv \
  -p data/rohanrao-sudoku

unzip data/rohanrao-sudoku/sudoku.csv.zip \
  -d data/rohanrao-sudoku

rm data/rohanrao-sudoku/sudoku.csv.zip
```

Bryan Park:

```bash
mkdir -p data/bryanpark-sudoku

uv run kaggle datasets download \
  -d bryanpark/sudoku \
  -f sudoku.csv \
  -p data/bryanpark-sudoku

unzip data/bryanpark-sudoku/sudoku.csv.zip \
  -d data/bryanpark-sudoku

rm data/bryanpark-sudoku/sudoku.csv.zip
```

## Build Flow

Build the full train/eval split from whatever source CSVs are present locally:

```bash
uv run python -m data.split
```

This writes:

```text
data/train/all.jsonl
data/eval/all.jsonl
```

With all three source datasets present, the current output has 12,348,899 train rows and 651,101 eval rows.

Train/eval assignment is deterministic by completed solution grid. A solution is assigned to eval when its BLAKE2b hash falls in the held-out 5 percent. Eval keeps one row per held-out solution.

Build the deterministic 100-row eval sample:

```bash
uv run python -m data.sample
```

This reads `data/eval/all.jsonl` and writes:

```text
data/eval/sample_100.jsonl
```

Build near-solved diagnostic eval sets:

```bash
uv run python -m data.mask
```

This reads `data/eval/sample_100.jsonl`, masks completed solutions, and writes 100-row files with exactly 1, 2, 5, 10, 20, or 40 missing cells:

```text
data/eval/missing_1_100.jsonl
data/eval/missing_2_100.jsonl
data/eval/missing_5_100.jsonl
data/eval/missing_10_100.jsonl
data/eval/missing_20_100.jsonl
data/eval/missing_40_100.jsonl
```

## Notes

`data/__pycache__/`, raw Kaggle folders, `data/train/`, and `data/eval/` are generated or local-only artifacts and should not be committed.
