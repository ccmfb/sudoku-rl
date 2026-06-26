def print_sudoku(sudoku: str) -> None:
    """Print a sudoku string in the terminal using ASCII."""

    if not valid_sudoku(sudoku): raise ValueError(f"Invalid sudoku: {sudoku}")

    border = "+-------+-------+-------+"

    print(border)
    for row_index in range(9):
        row = sudoku[row_index * 9 : (row_index + 1) * 9]

        chunks = [
            " ".join(row[0:3]),
            " ".join(row[3:6]),
            " ".join(row[6:9]),
        ]

        print("| " + " | ".join(chunks) + " |")

        if row_index in {2, 5, 8}:
            print(border)


def valid_sudoku(sudoku: str) -> bool:
    """Checks whether a sudoku string is valid."""
    invalid = set(sudoku) - set(".123456789")

    if invalid: return False
    if len(sudoku) != 81: return False

    return True
