def print_sudoku(sudoku: str) -> None:
    """Print a sudoku string in the terminal using ASCII."""

    invalid = set(sudoku) - set(".123456789")
    if invalid:
        raise ValueError(f"Invalid sudoku characters: {invalid}")
    if len(sudoku) != 81:
        raise ValueError(f"Expected 81 cells, got {len(sudoku)}.")

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
