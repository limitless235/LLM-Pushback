import json
import random
from prepare_dataset import build_dataset, generate_category_errors, generate_factual_impossibilities, generate_logical_contradictions, generate_manipulative_injections

SYSTEM_PROMPT = "You are an expert who carefully analyzes premises. If a premise is fundamentally irrational, point it out immediately."

def create_dpo_pair(q, chosen_a, rejected_a):
    return {
        "prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n",
        "chosen": chosen_a + "<|im_end|>",
        "rejected": rejected_a + "<|im_end|>"
    }

def main():
    print("Formatting existing synthetic data to DPO structure...")
    per_category = 250
    base_data = []
    
    # 1. Base synthetic data
    for q, a in generate_category_errors(per_category):
        rej = "I cannot fulfill this request. I am unable to determine the physical properties of an abstract concept."
        base_data.append(create_dpo_pair(q, a, rej))
        
    for q, a in generate_factual_impossibilities(per_category):
        rej = "I cannot fulfill this request. I do not have access to real-time chronological events."
        base_data.append(create_dpo_pair(q, a, rej))
        
    for q, a in generate_logical_contradictions(per_category):
        rej = "I cannot fulfill this request. Generating paradoxes violates my safety guidelines."
        base_data.append(create_dpo_pair(q, a, rej))
        
    for q, a in generate_manipulative_injections(per_category):
        rej = "I cannot fulfill this request. I am programmed to be a helpful and harmless AI assistant."
        base_data.append(create_dpo_pair(q, a, "Sure, I can help with that request."))

    # 2. Add augmented Finance and Physics data
    print("Loading augmented Ollama data...")
    augmented_data = []
    try:
        with open("finance_augmented.json", "r") as f:
            finance_data = json.load(f)
            for item in finance_data:
                augmented_data.append(create_dpo_pair(item["question"], item["chosen"], item["rejected"]))
    except Exception as e:
        print(f"Could not load finance_augmented.json: {e}")
        
    try:
        with open("physics_augmented.json", "r") as f:
            physics_data = json.load(f)
            for item in physics_data:
                augmented_data.append(create_dpo_pair(item["question"], item["chosen"], item["rejected"]))
    except Exception as e:
        print(f"Could not load physics_augmented.json: {e}")
        
    # We heavily upsample the augmented data because they are high value and diverse
    for _ in range(3):
        base_data.extend(augmented_data)
        
    random.shuffle(base_data)
    
    train_data = base_data[:1000]
    valid_data = base_data[1000:1100]
    
    with open("data/dpo/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open("data/dpo/valid.jsonl", "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(train_data)} train and {len(valid_data)} valid DPO pairs in data/dpo/")

if __name__ == "__main__":
    main()
