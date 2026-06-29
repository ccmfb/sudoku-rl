import re


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> str:
    """Extract the last valid 81-digit tagged Sudoku answer."""
    answers = ANSWER_PATTERN.findall(text)

    for answer_text in reversed(answers):
        answer = re.sub(r"\D", "", answer_text)
        if len(answer) == 81: return answer

    return ""


def format_prompt(sudoku: str) -> str:
    """Format a sudoku prompt for a model."""
    return f"Solve this Sudoku:\n{sudoku}\nReturn only the completed 81-character solution."


def valid_sudoku(sudoku: str) -> bool:
    """Checks whether a sudoku string is valid."""
    invalid = set(sudoku) - set(".123456789")

    if invalid: return False
    if len(sudoku) != 81: return False

    return True


def valid_solution(solution: str) -> bool:
    """Checks whether a solution string to a sudoku is valid."""
    invalid = set(solution) - set("123456789")

    if invalid: return False
    if len(solution) != 81: return False

    return True


def print_sudoku(sudoku: str) -> None:
    """Print a sudoku string in the terminal using ASCII."""
    if not valid_sudoku(sudoku):
        print(f"INVALID: {sudoku}")
        return


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
