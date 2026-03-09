import json
from pathlib import Path

# Paths to original SFT data
base_data_path = Path("data/train.jsonl")
valid_data_path = Path("data/valid.jsonl")

# Outputs from Ollama augmentation
aug_finance_path = Path("finance_augmented.json")
aug_physics_path = Path("physics_augmented.json")

out_dir = Path("data/sft_v3")
out_dir.mkdir(parents=True, exist_ok=True)

out_train = out_dir / "train.jsonl"
out_valid = out_dir / "valid.jsonl"

def format_sft_example(prompt, response):
    # The original sft dataset has this nested structure for content
    return {
        "messages": [
            {
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            },
            {
                "role": "assistant", 
                "content": [{"type": "text", "text": response}]
            }
        ]
    }

def process_file(file_path, train_data, domain_name):
    if file_path.exists():
        with open(file_path, "r") as f:
            data = json.load(f)
            added = 0
            for item in data:
                prompt = item.get("question", "")
                response = item.get("chosen", "")
                
                if prompt and response:
                    for _ in range(3):
                        train_data.append(format_sft_example(prompt, response))
                        added += 1
            print(f"Added {added} augmented {domain_name} examples (oversampled 3x).")
    else:
        print(f"Missing augmented file: {file_path}")

def main():
    train_data = []
    
    # 1. Load original SFT data
    if base_data_path.exists():
        with open(base_data_path, "r") as f:
            for line in f:
                train_data.append(json.loads(line))
        print(f"Loaded {len(train_data)} original SFT training examples.")
    else:
        print(f"Missing base training data at {base_data_path}!")
        return
    
    # 2. Add augmented Finance data
    process_file(aug_finance_path, train_data, "Finance")
        
    # 3. Add augmented Physics data
    process_file(aug_physics_path, train_data, "Physics")
        
    # Write new train.jsonl
    with open(out_train, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
            
    # Copy valid.jsonl directly
    if valid_data_path.exists():
        import shutil
        shutil.copy(valid_data_path, out_valid)
        print("Copied original valid.jsonl.")
        
    print(f"Successfully generated SFT V3 dataset with {len(train_data)} examples at {out_dir}")

if __name__ == "__main__":
    main()
