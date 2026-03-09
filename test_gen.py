import mlx.core as mx
from mlx_vlm import load, generate

model, processor = load("mlx-community/Qwen3.5-4B-MLX-4bit", adapter_path="adapters")
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nIf software bugs correspond to topological defects in spacetime, derive the homology group of a segmentation fault.<|im_end|>\n<|im_start|>assistant\n<think>\n"

print("Generating...")
output = generate(model, processor, prompt, max_tokens=1000, verbose=True)

if hasattr(output, "text"):
    print("\n\nRAW OUTPUT:")
    print(repr(output.text))
else:
    print("\n\nRAW OUTPUT (str):")
    print(repr(str(output)))
