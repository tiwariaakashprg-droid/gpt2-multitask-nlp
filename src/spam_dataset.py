"""
SMS Spam Collection dataset handling.

Dataset: UCI SMS Spam Collection (5,574 messages, labeled ham/spam).
Download: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
(also mirrored in many `zip`s as SMSSpamCollection.tsv, tab-separated: label \t text)

This module:
    1. Downloads + balances the dataset (equal ham/spam counts, since raw data
       is ~87% ham / 13% spam).
    2. Splits into train/validation/test CSVs.
    3. Wraps them in a PyTorch Dataset that tokenizes + pads to a fixed length.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset

SPAM_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_and_unzip(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "sms_spam_collection.zip")
    extracted_path = os.path.join(data_dir, "SMSSpamCollection")

    if os.path.exists(extracted_path):
        print("Dataset already downloaded.")
        return extracted_path

    print(f"Downloading dataset from {SPAM_URL} ...")
    urllib.request.urlretrieve(SPAM_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    original_path = os.path.join(data_dir, "SMSSpamCollection")
    os.rename(original_path, extracted_path) if os.path.exists(original_path) else None
    return extracted_path


def load_dataframe(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["label", "text"])
    return df


def balance_dataset(df):
    """Undersample the majority class ('ham') to match the spam count."""
    num_spam = df[df["label"] == "spam"].shape[0]
    ham_subset = df[df["label"] == "ham"].sample(num_spam, random_state=42)
    balanced_df = pd.concat([ham_subset, df[df["label"] == "spam"]])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle


def train_val_test_split(df, train_frac=0.7, val_frac=0.1, random_state=42):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return df[:train_end], df[train_end:val_end], df[val_end:]


def prepare_spam_data(data_dir="data"):
    """
    End-to-end: download -> balance -> split -> save CSVs.
    Returns paths to train.csv, val.csv, test.csv
    """
    tsv_path = download_and_unzip(data_dir)
    df = load_dataframe(tsv_path)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    balanced_df = balance_dataset(df.replace({0: "ham", 1: "spam"}))
    balanced_df["label"] = balanced_df["label"].map({"ham": 0, "spam": 1})

    train_df, val_df, test_df = train_val_test_split(balanced_df)

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    return train_path, val_path, test_path


class SpamDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_path)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["text"]]

        if max_length is None:
            self.max_length = max(len(t) for t in self.encoded_texts)
        else:
            self.max_length = max_length
            self.encoded_texts = [t[:self.max_length] for t in self.encoded_texts]

        self.encoded_texts = [
            t + [pad_token_id] * (self.max_length - len(t)) for t in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)
