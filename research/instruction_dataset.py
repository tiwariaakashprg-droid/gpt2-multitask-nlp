"""
Prepares the Stanford Alpaca instruction-following dataset for fine-tuning.

CRITICAL: without this step, the base GPT-2 model has never seen the
"### Instruction: ... ### Response:" format and will just ramble/repeat,
producing near-uniformly bad (score=0) outputs with no variance -- which
makes it impossible to measure human-vs-judge correlation (you need SOME
spread in quality to measure agreement on).

We mask the loss on the prompt tokens (instruction + input), so the model
is only trained to predict the RESPONSE tokens -- this is standard practice
for instruction tuning and trains much faster / more stably than training
on the full concatenated text.
"""

import json
import os
import urllib.request
import torch
from torch.utils.data import Dataset

ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


def download_alpaca(data_dir="research/data", num_examples=1000, seed=42):
    os.makedirs(data_dir, exist_ok=True)
    full_path = os.path.join(data_dir, "alpaca_data_full.json")
    subset_path = os.path.join(data_dir, f"alpaca_subset_{num_examples}.json")

    if os.path.exists(subset_path):
        print(f"Using cached subset: {subset_path}")
        return subset_path

    if not os.path.exists(full_path):
        print(f"Downloading Alpaca dataset from {ALPACA_URL} ...")
        urllib.request.urlretrieve(ALPACA_URL, full_path)

    with open(full_path) as f:
        full_data = json.load(f)

    import random
    random.seed(seed)
    random.shuffle(full_data)
    subset = full_data[:num_examples]

    with open(subset_path, "w") as f:
        json.dump(subset, f, indent=2)

    print(f"Saved {len(subset)}-example subset to {subset_path}")
    return subset_path


def format_prompt(entry):
    if entry.get("input"):
        return (f"### Instruction:\n{entry['instruction']}\n\n"
                f"### Input:\n{entry['input']}\n\n### Response:\n")
    return f"### Instruction:\n{entry['instruction']}\n\n### Response:\n"


class InstructionDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        with open(json_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded = []

        for entry in self.data:
            prompt = format_prompt(entry)
            full_text = prompt + entry["output"]

            prompt_ids = tokenizer.encode(prompt)
            full_ids = tokenizer.encode(full_text) + [tokenizer.eot_token]
            full_ids = full_ids[:max_length]

            # mask the prompt portion of the loss with -100 (ignored by cross_entropy)
            labels = [-100] * min(len(prompt_ids), len(full_ids)) + \
                     full_ids[len(prompt_ids):]
            labels = labels[:len(full_ids)]

            self.encoded.append((full_ids, labels))

    def __getitem__(self, idx):
        return self.encoded[idx]

    def __len__(self):
        return len(self.encoded)


def collate_fn(batch, pad_token_id):
    max_len = max(len(ids) for ids, _ in batch)
    input_batch, label_batch = [], []

    for ids, labels in batch:
        pad_len = max_len - len(ids)
        input_batch.append(ids + [pad_token_id] * pad_len)
        label_batch.append(labels + [-100] * pad_len)

    return (
        torch.tensor(input_batch, dtype=torch.long),
        torch.tensor(label_batch, dtype=torch.long),
    )
