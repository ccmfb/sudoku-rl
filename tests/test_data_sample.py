import json

import pytest

import data.sample as sample


def test_sample_score_is_deterministic():
    row = {"sudoku": "." * 81, "solution": "1" * 81}

    assert sample.sample_score(row) == sample.sample_score(row)


def test_build_sample_writes_deterministic_rows(tmp_path, monkeypatch):
    input_path = tmp_path / "eval" / "all.jsonl"
    sample_path = tmp_path / "eval" / "sample_100.jsonl"
    rows = [
        {"sudoku": "." * 80 + str(index + 1), "solution": str(index + 1) * 81}
        for index in range(5)
    ]
    input_path.parent.mkdir(parents=True)
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setattr(sample, "INPUT_PATH", input_path)
    monkeypatch.setattr(sample, "SAMPLE_PATH", sample_path)
    monkeypatch.setattr(sample, "SAMPLE_COUNT", 3)

    counts = sample.build_sample()
    written_rows = [json.loads(line) for line in sample_path.read_text().splitlines()]

    assert counts == {"input": 5, "sample": 3}
    assert written_rows == sorted(rows, key=sample.sample_score)[:3]


def test_build_sample_rejects_small_eval_file(tmp_path, monkeypatch):
    input_path = tmp_path / "eval" / "all.jsonl"
    sample_path = tmp_path / "eval" / "sample_100.jsonl"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(json.dumps({"sudoku": "." * 81, "solution": "1" * 81}) + "\n")

    monkeypatch.setattr(sample, "INPUT_PATH", input_path)
    monkeypatch.setattr(sample, "SAMPLE_PATH", sample_path)
    monkeypatch.setattr(sample, "SAMPLE_COUNT", 2)

    with pytest.raises(ValueError, match="Need 2 rows"):
        sample.build_sample()
