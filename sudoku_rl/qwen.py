import re


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> str:
    """Extract the last valid 81-digit tagged Sudoku answer."""
    answers = ANSWER_PATTERN.findall(text)

    for answer_text in reversed(answers):
        answer = re.sub(r"\D", "", answer_text)
        if len(answer) == 81: return answer

    return ""


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

    def complete(self, prompt: str) -> str:
        """Generate raw model text for a prompt."""
        import torch

        if self.thinking:
            prompt = (
                f"{prompt}\n"
                "You may reason through the puzzle first.\n"
                "Put the final completed 81-character solution between <answer> and </answer> tags.\n"
                "Do not put anything except the final grid inside <answer>."
            )

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

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

        generated = outputs[0][inputs["input_ids"].shape[-1]:]

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def attempt(self, prompt: str) -> str:
        """Generate a Sudoku attempt."""
        result = self.complete(prompt)
        if self.thinking: return extract_answer(result)

        return result
