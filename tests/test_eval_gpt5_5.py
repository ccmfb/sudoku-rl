import os
import sys
import types

import pytest

from scripts import eval_gpt5_5
from scripts.eval_gpt5_5 import GPT55Policy, MODEL, REASONING


def test_gpt55_policy_uses_fixed_no_reasoning_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(output_text=" " + "123456789" * 9 + " ")

    class OpenAI:
        def __init__(self) -> None:
            self.responses = Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))

    policy = GPT55Policy()
    attempt = policy.attempt("solve this")

    assert attempt == "123456789" * 9
    assert calls == [
        {
            "model": MODEL,
            "input": "solve this",
            "reasoning": REASONING,
            "text": {"verbosity": "low"},
            "store": False,
        }
    ]


def test_gpt55_policy_loads_repo_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n")
    keys = []

    class Responses:
        def create(self, **kwargs):
            return types.SimpleNamespace(output_text="")

    class OpenAI:
        def __init__(self) -> None:
            keys.append(os.environ["OPENAI_API_KEY"])
            self.responses = Responses()

    monkeypatch.setattr(eval_gpt5_5, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))

    GPT55Policy()

    assert keys == ["from-dotenv"]


def test_gpt55_policy_exits_when_openai_sdk_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(SystemExit, match="uv run --with openai"):
        GPT55Policy()
