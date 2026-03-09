from mlx_vlm import load
from mlx_vlm.generate import stream_generate

model, processor = load("mlx-community/Qwen3.5-4B-MLX-4bit", adapter_path="adapters")
prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

for count, result in enumerate(stream_generate(model, processor, prompt, max_tokens=10)):
    print(f"Yield {count}: {type(result)} -> {repr(getattr(result, 'text', str(result)))}")
