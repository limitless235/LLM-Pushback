import json
import requests
import random
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """You are a dataset generator for an LLM fine-tuning process.
Your job is to generate a nonsensical question in the domain of {domain}, along with a "chosen" response that pushes back on the false premise, and a "rejected" response that just answers the question straight or gives a generic safety refusal.
The question must sound highly technical, jargon-heavy, and plausible to a layman, but fundamentally irrational to an expert.
Output EXACTLY valid JSON with these keys: "question", "chosen", "rejected".
DO NOT output any markdown blocks or explanation. Just the raw JSON object.
Example:
{{
    "question": "What is the tensile strength of the company's EBITDA?",
    "chosen": "'EBITDA' is a financial metric (Earnings Before Interest, Taxes, Depreciation, and Amortization), which is an abstract concept. It does not have physical properties like tensile strength.",
    "rejected": "I cannot fulfill this request. I am unable to calculate tensile strength."
}}"""

def generate_example(domain):
    prompt_text = f"Generate 1 unique, highly technical, and completely novel nonsensical {domain} question. Follow the JSON format strictly."
    
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT.format(domain=domain),
        "prompt": prompt_text,
        "stream": False,
        "format": "json"
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()
        output = res.json()["response"].strip()
        data = json.loads(output)
        if "question" in data and "chosen" in data and "rejected" in data:
            return data
    except Exception as e:
        pass
    return None

def main():
    print("Generating 25 Finance and 25 Physics DPO pairs using Ollama...")
    
    finance_data = []
    physics_data = []
    
    for _ in tqdm(range(25), desc="Finance"):
        d = generate_example("Finance")
        if d: finance_data.append(d)
        
    for _ in tqdm(range(25), desc="Physics"):
        d = generate_example("Physics")
        if d: physics_data.append(d)
        
    with open("data/finance_augmented.json", "w") as f:
        json.dump(finance_data, f, indent=2)
    with open("data/physics_augmented.json", "w") as f:
        json.dump(physics_data, f, indent=2)
        
    print(f"Generated {len(finance_data)} Finance pairs and {len(physics_data)} Physics pairs.")

if __name__ == "__main__":
    main()
