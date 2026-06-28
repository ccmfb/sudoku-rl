from sudoku_rl.methods.sft import format_training_example
from sudoku_rl.tasks.sudoku import format_prompt


SUDOKU = "1..5.37..6.3..8.9......98...1.......8761..........6...........7.8.9.76.47...6.312"
SOLUTION = "198543726643278591527619843914735268876192435235486179462351987381927654759864312"


def test_format_training_example_uses_prompt_completion_chat_shape() -> None:
    example = format_training_example({"sudoku": SUDOKU, "solution": SOLUTION})

    assert example == {
        "prompt": [{"role": "user", "content": format_prompt(SUDOKU)}],
        "completion": [{"role": "assistant", "content": SOLUTION}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
