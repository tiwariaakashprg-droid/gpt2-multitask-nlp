"""
Fine-tune the pretrained 124M GPT-2 model for binary SMS spam classification.

Usage:
    python scripts/finetune_spam.py --epochs 5 --use_pretrained
"""

import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import GPT2_CONFIG_124M
from src.model import GPTModel, GPTClassifier
from src.tokenizer import BPETokenizer
from src.weight_loader import load_pretrained_gpt2
from src.spam_dataset import prepare_spam_data, SpamDataset
from src.train_utils import train_classifier, calc_accuracy_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--save_path", type=str, default="checkpoints/spam_classifier.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)

    tokenizer = BPETokenizer()

    print("Preparing SMS spam dataset (download + balance + split) ...")
    train_path, val_path, test_path = prepare_spam_data(args.data_dir)

    train_ds = SpamDataset(train_path, tokenizer, max_length=None)
    val_ds = SpamDataset(val_path, tokenizer, max_length=train_ds.max_length)
    test_ds = SpamDataset(test_path, tokenizer, max_length=train_ds.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    base_model = GPTModel(GPT2_CONFIG_124M)
    if args.use_pretrained:
        load_pretrained_gpt2(base_model, model_name="gpt2")

    model = GPTClassifier(base_model, num_classes=2, trainable_layers=1)
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.1,
    )

    print("Starting fine-tuning ...")
    history = train_classifier(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs, eval_freq=50, eval_iter=5,
    )

    test_acc = calc_accuracy_loader(test_loader, model, device)
    print(f"\nFinal test accuracy: {test_acc*100:.2f}%")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved fine-tuned classifier to {args.save_path}")


if __name__ == "__main__":
    main()
