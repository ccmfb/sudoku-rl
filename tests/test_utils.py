import pytest

from sudoku_rl.utils import print_sudoku


def test_print_sudoku_formats_grid(capsys: pytest.CaptureFixture[str]) -> None:
    print_sudoku("1..5.37..6.3..8.9......98...1.......8761..........6...........7.8.9.76.47...6.312")

    assert capsys.readouterr().out == (
        "+-------+-------+-------+\n"
        "| 1 . . | 5 . 3 | 7 . . |\n"
        "| 6 . 3 | . . 8 | . 9 . |\n"
        "| . . . | . . 9 | 8 . . |\n"
        "+-------+-------+-------+\n"
        "| . 1 . | . . . | . . . |\n"
        "| 8 7 6 | 1 . . | . . . |\n"
        "| . . . | . . 6 | . . . |\n"
        "+-------+-------+-------+\n"
        "| . . . | . . . | . . 7 |\n"
        "| . 8 . | 9 . 7 | 6 . 4 |\n"
        "| 7 . . | . 6 . | 3 1 2 |\n"
        "+-------+-------+-------+\n"
    )


def test_print_sudoku_rejects_invalid_characters() -> None:
    with pytest.raises(ValueError, match="Invalid sudoku characters"):
        print_sudoku("0" + "." * 80)


def test_print_sudoku_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="Expected 81 cells"):
        print_sudoku("." * 80)
