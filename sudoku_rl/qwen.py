import re
from collections.abc import Callable


def extract_attempt(text: str) -> str:
    """Extract an 81-digit sudoku attempt from model output."""
    compact = re.sub(r"\s+", "", text)
    match = re.search(r"(?<!\d)[1-9]{81}(?!\d)", compact)

    if not match:
        return ""

    return match.group(0)

def load_qwen_generator(model_name: str = "Qwen/Qwen3-8B") -> Callable[[str, str], str]:
    """Load a Qwen model and return a sudoku attempt generator."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    def generate_attempt(sudoku: str, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        return extract_attempt(response)

    return generate_attempt
