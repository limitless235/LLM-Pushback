import json
from transformers import AutoTokenizer
from mlx_lm_lora.trainer.datasets import load_local_dataset
from types import SimpleNamespace
from pathlib import Path

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mlx-community/Ministral-3-3B-Instruct-2512-4bit")

config = SimpleNamespace(
    train_mode="dpo",
    prompt_feature="prompt",
    system_feature="system",
    chosen_feature="chosen",
    rejected_feature="rejected",
    mask_prompt=False
)

dpo_path = Path("data/dpo")

print("Loading DPO dataset (this is where it hangs in mlx_lm_lora)...")
train, valid, test = load_local_dataset(dpo_path, tokenizer, config)

print(f"Loaded train={len(train)}, valid={len(valid)}")
print("Sample 0:", train[0])
