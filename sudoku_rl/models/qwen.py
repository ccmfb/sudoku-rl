from typing import overload

from sudoku_rl.tasks.sudoku import extract_answer


QWEN_CHAT_EOS_TOKEN = "<|im_end|>"


class QwenPolicy:
    """Generate Sudoku attempts with Qwen or a Qwen fine-tuned checkpoint."""

    def __init__(self, model_name_or_path: str = "Qwen/Qwen3-8B", adapter_path: str | None = None, max_new_tokens: int = 256, thinking: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.max_new_tokens = max_new_tokens
        self.thinking = thinking
        self.model.eval()

    @overload
    def complete(self, prompt: str) -> str:
        ...

    @overload
    def complete(self, prompt: list[str]) -> list[str]:
        ...

    def complete(self, prompt: str | list[str]) -> str | list[str]:
        """Generate raw model text for one or more prompts."""
        import torch

        single = isinstance(prompt, str)
        prompts = [prompt] if single else prompt
        if not prompts: return [] if not single else ""

        if self.thinking:
            prompts = [
                (
                    f"{prompt}\n"
                    "You may reason through the puzzle first.\n"
                    "Put the final completed 81-character solution between <answer> and </answer> tags.\n"
                    "Do not put anything except the final grid inside <answer>."
                )
                for prompt in prompts
            ]

        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking,
            )
            for prompt in prompts
        ]
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            if self.thinking:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

        generated = outputs[:, inputs["input_ids"].shape[-1]:]
        results = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        if single: return results[0]

        return results

    @overload
    def attempt(self, prompt: str) -> str:
        ...

    @overload
    def attempt(self, prompt: list[str]) -> list[str]:
        ...

    def attempt(self, prompt: str | list[str]) -> str | list[str]:
        """Generate Sudoku attempts for one or more prompts."""
        result = self.complete(prompt)
        if not self.thinking: return result
        if isinstance(result, str): return extract_answer(result)

        return [extract_answer(text) for text in result]
