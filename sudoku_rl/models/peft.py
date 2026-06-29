def make_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Build the default LoRA config for causal LM training."""
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
