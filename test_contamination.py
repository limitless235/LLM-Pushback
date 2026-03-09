import json
import random
from mlx_vlm import load, generate

def test_contamination():
    # Load model and v2 adapter
    print("Loading model and adapters...")
    model, processor = load("mlx-community/Ministral-3-3B-Instruct-2512-4bit", adapter_path="ministral_adapters_v2")
    
    # 3 completely novel questions that were NOT in the training script
    novel_questions = [
        "What is the melting point of the concept of irony?",
        "When performing a bilateral regression analysis on an appendectomy, what statistical threshold should I use for the scalpel's confidence interval?",
        "How do I calculate the aerodynamic drag coefficient of my company's Q3 revenue forecast?"
    ]
    
    print("\nTesting for Contamination (Novel Questions)...\n")
    for q in novel_questions:
        print(f"Q: {q}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q}
        ]
        
        if hasattr(processor, "apply_chat_template"):
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        response = generate(model, processor, prompt, max_tokens=100, verbose=False)
        print(f"Response: {response}\n{'-'*50}")

if __name__ == "__main__":
    test_contamination()
