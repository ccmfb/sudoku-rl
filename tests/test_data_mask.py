import json

import pytest

import data.mask as mask


def test_mask_positions_is_deterministic():
    solution = "1" * 81

    assert mask.mask_positions(solution, 5) == mask.mask_positions(solution, 5)


def test_masked_sudoku_masks_exactly_the_requested_count():
    solution = "1" * 81
    sudoku = mask.masked_sudoku(solution, 5)

    assert len(sudoku) == 81
    assert sudoku.count(".") == 5
    assert sudoku.replace(".", "1") == solution


def test_build_masked_sets_writes_disjoint_missing_count_windows(tmp_path, monkeypatch):
    input_path = tmp_path / "data" / "train" / "all.jsonl"
    rows = [
        {"sudoku": "." * 81, "solution": "1" * 81},
        {"sudoku": "." * 81, "solution": "2" * 81},
        {"sudoku": "." * 81, "solution": "3" * 81},
        {"sudoku": "." * 81, "solution": "4" * 81},
    ]
    input_path.parent.mkdir(parents=True)
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setattr(mask, "ROOT", tmp_path)
    monkeypatch.setattr(mask, "MISSING_COUNTS", [1, 3])

    counts = mask.build_masked_sets("train", set_size=2)

    assert counts == {"split": "train", "input": 4, "sets": 2}

    seen_solutions = set()
    for missing_count, expected_solutions in [(1, ["1" * 81, "2" * 81]), (3, ["3" * 81, "4" * 81])]:
        path = tmp_path / "data" / "train" / f"missing_{missing_count}_2.jsonl"
        written_rows = [json.loads(line) for line in path.read_text().splitlines()]

        assert len(written_rows) == 2
        assert [row["solution"] for row in written_rows] == expected_solutions
        assert all(row["sudoku"].count(".") == missing_count for row in written_rows)
        assert seen_solutions.isdisjoint(row["solution"] for row in written_rows)
        seen_solutions.update(row["solution"] for row in written_rows)


def test_build_masked_sets_rejects_small_eval_file(tmp_path, monkeypatch):
    input_path = tmp_path / "data" / "eval" / "all.jsonl"
    rows = [{"sudoku": "." * 81, "solution": "1" * 81}]
    input_path.parent.mkdir(parents=True)
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setattr(mask, "ROOT", tmp_path)
    monkeypatch.setattr(mask, "MISSING_COUNTS", [1])

    with pytest.raises(ValueError, match="Need 2 rows for missing_1"):
        mask.build_masked_sets("eval", set_size=2)


def test_build_masked_sets_rejects_invalid_set_size():
    with pytest.raises(ValueError, match="set_size must be positive"):
        mask.build_masked_sets(set_size=0)
