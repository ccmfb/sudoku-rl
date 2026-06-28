import json

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


def test_build_masked_sets_writes_each_missing_count(tmp_path, monkeypatch):
    input_path = tmp_path / "eval" / "sample_100.jsonl"
    output_dir = tmp_path / "eval"
    rows = [
        {"sudoku": "." * 81, "solution": "1" * 81},
        {"sudoku": "." * 81, "solution": "2" * 81},
    ]
    input_path.parent.mkdir(parents=True)
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setattr(mask, "INPUT_PATH", input_path)
    monkeypatch.setattr(mask, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(mask, "MISSING_COUNTS", [1, 3])

    counts = mask.build_masked_sets()

    assert counts == {"input": 2, "sets": 2}

    for missing_count in [1, 3]:
        path = output_dir / f"missing_{missing_count}_100.jsonl"
        written_rows = [json.loads(line) for line in path.read_text().splitlines()]

        assert len(written_rows) == 2
        assert [row["solution"] for row in written_rows] == ["1" * 81, "2" * 81]
        assert all(row["sudoku"].count(".") == missing_count for row in written_rows)
