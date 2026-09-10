"""
STEP 0 (NEW, do this BEFORE generate_qa_pairs.py) of the research pipeline.

Fine-tunes pretrained GPT-2 on a subset of the Stanford Alpaca instruction
dataset, so it actually learns to follow the "### Instruction: ... ###
Response:" format instead of just rambling. This gives you a model whose
output QUALITY VARIES across items (some good, some mediocre, some bad) --
which is what you need for a meaningful human-vs-judge agreement study.

Usage:
    python research/finetune_instruction.py --num_examples 1000 --epochs 2

On CPU this will be slow (expect 1-3+ hours for 1000 examples / 2 epochs on
a laptop CPU). If you have any GPU (even a modest one, or Google Colab's
free T4), use it -- add --device cuda, or just run this in a Colab notebook
where CUDA is auto-detected.
"""

import argparse
import sys
import os
import functools
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import GPT2_CONFIG_124M
from src.model import GPTModel
from src.tokenizer import BPETokenizer
from src.weight_loader import load_pretrained_gpt2
from research.instruction_dataset import download_alpaca, InstructionDataset, collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=1000,
                         help="How many Alpaca examples to fine-tune on (more = better but slower)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--save_path", type=str, default="research/checkpoints/gpt2_instruction_tuned.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected. This will be slow. Consider running "
              "this script in Google Colab (free GPU) instead.")

    torch.manual_seed(123)
    tokenizer = BPETokenizer()

    print("Preparing Alpaca instruction data ...")
    data_path = download_alpaca(num_examples=args.num_examples)
    dataset = InstructionDataset(data_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=functools.partial(collate_fn, pad_token_id=tokenizer.eot_token),
    )

    print("Loading pretrained GPT-2 weights ...")
    model = GPTModel(GPT2_CONFIG_124M)
    load_pretrained_gpt2(model, model_name="gpt2")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    print(f"Starting instruction fine-tuning on {len(dataset)} examples, "
          f"{args.epochs} epoch(s) ...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, (input_ids, labels) in enumerate(loader):
            input_ids, labels = input_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)

            # Shift so that the logits at position t (which predict the NEXT
            # token) line up with the label at position t+1. GPTModel.forward()
            # does not do this shift internally (unlike e.g. HF's
            # GPT2LMHeadModel(labels=...)), so it must be done explicitly here.
            # Without this shift, the model is trained to "predict" the token
            # it was just given as input at the same position -- a trivial,
            # wrong objective that produces a model which emits <|endoftext|>
            # (or other degenerate junk) almost immediately at generation time.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(loader)}  loss={loss.item():.3f}")

        print(f"Epoch {epoch+1} average loss: {total_loss / len(loader):.3f}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"\nSaved instruction-tuned model to {args.save_path}")
    print("Next step: python research/generate_qa_pairs.py --checkpoint "
          f"{args.save_path}")


if __name__ == "__main__":
    main()
