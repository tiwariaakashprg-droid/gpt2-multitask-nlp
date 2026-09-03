"""
End-to-end demo: generate responses to Alpaca-style instructions with our
fine-tuned/pretrained GPT model, then score them 0-100 using LLaMA-3 via Ollama.

Prereqs:
    ollama pull llama3
    ollama serve   (in a separate terminal)

Usage:
    python scripts/run_ollama_eval.py --alpaca_json data/alpaca_sample.json
"""

import argparse
import json
import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import GPT2_CONFIG_124M
from src.model import GPTModel
from src.tokenizer import BPETokenizer
from src.weight_loader import load_pretrained_gpt2
from src.generate import generate_and_decode
from src.evaluate_ollama import evaluate_dataset


def format_alpaca_prompt(entry):
    if entry.get("input"):
        return (f"### Instruction:\n{entry['instruction']}\n\n"
                f"### Input:\n{entry['input']}\n\n### Response:\n")
    return f"### Instruction:\n{entry['instruction']}\n\n### Response:\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpaca_json", type=str, required=True,
                         help="JSON file: list of {instruction, input(optional)}")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--judge_model", type=str, default="llama3")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTModel(GPT2_CONFIG_124M)
    if args.use_pretrained:
        load_pretrained_gpt2(model, model_name="gpt2")
    model.to(device)
    tokenizer = BPETokenizer()

    with open(args.alpaca_json) as f:
        entries = json.load(f)

    print(f"Generating responses for {len(entries)} instructions ...")
    for entry in entries:
        prompt = format_alpaca_prompt(entry)
        full_output = generate_and_decode(
            model, tokenizer, prompt,
            max_new_tokens=80, context_size=GPT2_CONFIG_124M["context_length"],
            temperature=0.8, top_k=50, device=device,
        )
        entry["model_response"] = full_output[len(prompt):].strip()

    print("Scoring responses with LLaMA-3 (via Ollama) ...")
    scored, summary = evaluate_dataset(entries, judge_model=args.judge_model)

    with open(args.output, "w") as f:
        json.dump({"results": scored, "summary": summary}, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
