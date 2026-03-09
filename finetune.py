import subprocess
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5-4B using mlx-vlm LoRA")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3.5-4B-MLX-4bit", help="HuggingFace model path")
    parser.add_argument("--data", type=str, default="data", help="Directory containing train.jsonl and valid.jsonl")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=300, help="Number of training steps per epoch (0 for full dataset)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--output-path", type=str, default="adapters", help="Output directory for adapters")
    args = parser.parse_args()

    # Command to run mlx_vlm.lora
    cmd = [
        "python3", "-m", "mlx_vlm.lora",
        "--model-path", args.model,
        "--dataset", args.data,
        "--batch-size", str(args.batch_size),
        "--lora-rank", str(args.lora_rank),
        "--learning-rate", str(args.learning_rate),
        "--steps", str(args.steps),
        "--epochs", str(args.epochs),
        "--output-path", args.output_path,
        "--apply-chat-template"
    ]

    print(f"Starting fine-tuning with command:\n{' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Fine-tuning completed successfully!")
        print(f"Adapters saved to {args.output_path}/")
    except subprocess.CalledProcessError as e:
        print(f"Error during fine-tuning: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
