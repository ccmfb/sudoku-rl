import argparse
import json
from datetime import datetime, timezone

from scripts import eval_qwen


def test_make_result_path_uses_utc_datetime_filename(monkeypatch) -> None:
    monkeypatch.setattr(eval_qwen, "RESULTS_DIR", eval_qwen.PROJECT_ROOT / "tmp-results")
    created_at = datetime(2026, 7, 1, 15, 30, 44, 238941, tzinfo=timezone.utc)

    assert eval_qwen.make_result_path(created_at) == eval_qwen.PROJECT_ROOT / "tmp-results" / "20260701T153044.238941Z.jsonl"


def test_write_results_writes_summary_then_question_rows(tmp_path) -> None:
    path = tmp_path / "results" / "20260701T153044.238941Z.jsonl"
    args = argparse.Namespace(
        model="Qwen/Qwen3-8B",
        adapter=None,
        data=tmp_path / "data" / "eval.jsonl",
        limit=None,
        thinking=False,
        max_new_tokens=256,
        batch_size=16,
    )
    result_rows = [
        {
            "index": 0,
            "sudoku": "1" * 81,
            "solution": "2" * 81,
            "attempt": "3" * 81,
            "score": 0.5,
        }
    ]
    created_at = datetime(2026, 7, 1, 15, 30, 44, 238941, tzinfo=timezone.utc)

    eval_qwen.write_results(path, args, result_rows, 0.5, created_at)

    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert rows == [
        {
            "type": "summary",
            "created_at": "2026-07-01T15:30:44.238941Z",
            "model": "Qwen/Qwen3-8B",
            "adapter": None,
            "data": str(args.data),
            "limit": None,
            "rows": 1,
            "thinking": False,
            "max_new_tokens": 256,
            "batch_size": 16,
            "score": 0.5,
        },
        {
            "type": "question",
            "index": 0,
            "sudoku": "1" * 81,
            "solution": "2" * 81,
            "attempt": "3" * 81,
            "score": 0.5,
        },
    ]
