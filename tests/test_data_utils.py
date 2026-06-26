import json

from data.utils import extract_splits, load_bryanpark_csv, load_radcliffe_csv, load_rohanrao_csv


def test_load_radcliffe_csv(tmp_path):
    sudoku = "1..5.37..6.3..8.9......98...1.......8761..........6...........7.8.9.76.47...6.312"
    solution = "198543726643278591527619843914735268876192435235486179462351987381927654759864312"
    path = tmp_path / "radcliffe.csv"
    path.write_text(
        "id,puzzle,solution,clues,difficulty\n"
        f"1,{sudoku},{solution},27,2.2\n"
        f"2,{'.' * 81},{'1' * 81},81,0\n"
    )

    assert list(load_radcliffe_csv(path, limit=1)) == [{"sudoku": sudoku, "solution": solution}]


def test_load_rohanrao_csv(tmp_path):
    raw_sudoku = "070000043040009610800634900094052000358460020000800530080070091902100005007040802"
    solution = "679518243543729618821634957794352186358461729216897534485276391962183475137945862"
    path = tmp_path / "rohanrao.csv"
    path.write_text(f"puzzle,solution\n{raw_sudoku},{solution}\n")

    assert list(load_rohanrao_csv(path)) == [{"sudoku": raw_sudoku.replace("0", "."), "solution": solution}]


def test_load_bryanpark_csv(tmp_path):
    raw_sudoku = "004300209005009001070060043006002087190007400050083000600000105003508690042910300"
    solution = "864371259325849761971265843436192587198657432257483916689734125713528694542916378"
    path = tmp_path / "bryanpark.csv"
    path.write_text(f"quizzes,solutions\n{raw_sudoku},{solution}\n")

    assert list(load_bryanpark_csv(path)) == [{"sudoku": raw_sudoku.replace("0", "."), "solution": solution}]


def test_extract_splits_writes_train_and_eval_jsonl(tmp_path):
    sudoku = "." * 81
    solution = "1" * 81
    rows = (
        {"sudoku": sudoku, "solution": solution, "source_index": index}
        for index in range(5)
    )
    train_path = tmp_path / "data" / "train" / "radcliffe.jsonl"
    eval_path = tmp_path / "data" / "eval" / "radcliffe.jsonl"

    extract_splits(rows, train_path, eval_path, train_count=3, eval_count=2)

    train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    eval_rows = [json.loads(line) for line in eval_path.read_text().splitlines()]

    assert train_rows == [
        {"sudoku": sudoku, "solution": solution},
        {"sudoku": sudoku, "solution": solution},
        {"sudoku": sudoku, "solution": solution},
    ]
    assert eval_rows == [
        {"sudoku": sudoku, "solution": solution},
        {"sudoku": sudoku, "solution": solution},
    ]
