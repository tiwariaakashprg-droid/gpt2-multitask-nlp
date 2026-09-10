"""
STEP 1 of the research pipeline.

Generates model responses for every question in `questions.json`, at several
different decoding configurations (temperature x top_k), and writes them all
to a single JSON file ready for human annotation.

This is what creates the "varying decoding strategy" independent variable in
the study (does judge behavior/reliability change with how random the
generation is?).

Usage:
    python research/generate_qa_pairs.py --use_pretrained --output research/data/qa_pairs.json
"""

import argparse
import json
import sys
import os
import itertools
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import GPT2_CONFIG_124M
from src.model import GPTModel
from src.tokenizer import BPETokenizer
from src.weight_loader import load_pretrained_gpt2
from src.generate import generate_and_decode

import torch


# Decoding configurations to sweep over. Each question is generated once per
# configuration, so the final dataset lets you compare judge behavior across
# decoding randomness.
DECODING_CONFIGS = [
    {"name": "low_temp",  "temperature": 0.3, "top_k": 10},
    {"name": "mid_temp",  "temperature": 0.7, "top_k": 50},
    {"name": "high_temp", "temperature": 1.2, "top_k": 100},
]


def format_prompt(entry):
    if entry.get("input"):
        return (f"### Instruction:\n{entry['instruction']}\n\n"
                f"### Input:\n{entry['input']}\n\n### Response:\n")
    return f"### Instruction:\n{entry['instruction']}\n\n### Response:\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, default="research/questions.json")
    parser.add_argument("--output", type=str, default="research/data/qa_pairs.json")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to an instruction-tuned checkpoint from "
                              "finetune_instruction.py (RECOMMENDED - a raw "
                              "pretrained GPT-2 does not know how to follow "
                              "the ### Instruction: format and will produce "
                              "near-uniformly bad, low-variance outputs).")
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model = GPTModel(GPT2_CONFIG_124M)
    if args.checkpoint:
        print(f"Loading instruction-tuned checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    elif args.use_pretrained:
        print("WARNING: using raw pretrained GPT-2 (no instruction tuning). "
              "Outputs will likely NOT follow the instruction format well. "
              "Run research/finetune_instruction.py first and pass "
              "--checkpoint for much better results.")
        load_pretrained_gpt2(model, model_name="gpt2")
    model.to(device)
    tokenizer = BPETokenizer()

    with open(args.questions) as f:
        questions = json.load(f)

    qa_pairs = []
    total = len(questions) * len(DECODING_CONFIGS)
    count = 0

    for entry, cfg in itertools.product(questions, DECODING_CONFIGS):
        count += 1
        prompt = format_prompt(entry)
        full_output = generate_and_decode(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            context_size=GPT2_CONFIG_124M["context_length"],
            temperature=cfg["temperature"],
            top_k=cfg["top_k"],
            device=device,
        )
        response_only = full_output[len(prompt):].strip()

        qa_pairs.append({
            "id": str(uuid.uuid4())[:8],
            "category": entry["category"],
            "instruction": entry["instruction"],
            "input": entry.get("input", ""),
            "decoding_config": cfg["name"],
            "temperature": cfg["temperature"],
            "top_k": cfg["top_k"],
            "model_output": response_only,
        })
        print(f"[{count}/{total}] ({cfg['name']}) {entry['instruction'][:50]!r} -> "
              f"{response_only[:60]!r}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"\nSaved {len(qa_pairs)} question-answer pairs to {args.output}")
    print("Next step: open research/annotate.html in a browser, load this file, "
          "and send it to your human annotators (one browser session per annotator).")


if __name__ == "__main__":
    main()
