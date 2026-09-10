"""
Demo script: load pretrained GPT-2 (124M) weights into our from-scratch model
and generate text using top-k sampling + temperature scaling.

Usage:
    python scripts/run_generation.py --prompt "Every effort moves you" --top_k 50 --temperature 0.8
"""

import argparse
import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import GPT2_CONFIG_124M
from src.model import GPTModel
from src.tokenizer import BPETokenizer
from src.weight_loader import load_pretrained_gpt2
from src.generate import generate_and_decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Every effort moves you")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--use_pretrained", action="store_true",
                         help="Download & load official OpenAI GPT-2 weights via HF")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)

    model = GPTModel(GPT2_CONFIG_124M)
    print(f"Model parameters: {model.num_parameters():,}")

    if args.use_pretrained:
        load_pretrained_gpt2(model, model_name="gpt2")

    model.to(device)
    tokenizer = BPETokenizer()

    output = generate_and_decode(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_size=GPT2_CONFIG_124M["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print("\n--- Generated text ---")
    print(output)


if __name__ == "__main__":
    main()
