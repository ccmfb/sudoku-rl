import json

import data.split as split


def test_is_eval_solution_is_deterministic():
    solution = "1" * 81

    assert split.is_eval_solution(solution) == split.is_eval_solution(solution)


def test_build_splits_assigns_eval_by_solution_and_dedupes(tmp_path, monkeypatch):
    train_path = tmp_path / "train" / "all.jsonl"
    eval_path = tmp_path / "eval" / "all.jsonl"
    first_solution = "1" * 81
    second_solution = "2" * 81
    rows = [
        {"sudoku": "." * 81, "solution": first_solution},
        {"sudoku": "1" + "." * 80, "solution": first_solution},
        {"sudoku": "2" + "." * 80, "solution": second_solution},
    ]

    def load_rows(path):
        yield from rows

    monkeypatch.setattr(split, "TRAIN_PATH", train_path)
    monkeypatch.setattr(split, "EVAL_PATH", eval_path)
    monkeypatch.setattr(split, "EVAL_PERCENT", 100)
    monkeypatch.setattr(split, "SOURCES", [(tmp_path, load_rows)])

    counts = split.build_splits()

    assert counts == {"train": 0, "eval": 2, "eval_duplicate_solutions": 1}
    assert train_path.read_text().splitlines() == []
    assert [json.loads(line) for line in eval_path.read_text().splitlines()] == [
        {"sudoku": "." * 81, "solution": first_solution},
        {"sudoku": "2" + "." * 80, "solution": second_solution},
    ]


def test_build_splits_keeps_train_rows_with_the_same_solution(tmp_path, monkeypatch):
    train_path = tmp_path / "train" / "all.jsonl"
    eval_path = tmp_path / "eval" / "all.jsonl"
    solution = "1" * 81
    rows = [
        {"sudoku": "." * 81, "solution": solution},
        {"sudoku": "1" + "." * 80, "solution": solution},
    ]

    def load_rows(path):
        yield from rows

    monkeypatch.setattr(split, "TRAIN_PATH", train_path)
    monkeypatch.setattr(split, "EVAL_PATH", eval_path)
    monkeypatch.setattr(split, "EVAL_PERCENT", 0)
    monkeypatch.setattr(split, "SOURCES", [(tmp_path, load_rows)])

    counts = split.build_splits()

    assert counts == {"train": 2, "eval": 0, "eval_duplicate_solutions": 0}
    assert [json.loads(line) for line in train_path.read_text().splitlines()] == rows
    assert eval_path.read_text().splitlines() == []
