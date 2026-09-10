"""
Thin wrapper around OpenAI's `tiktoken` BPE tokenizer (the same byte-pair-encoding
vocabulary used by the original GPT-2 model).
"""

import tiktoken
import torch


class BPETokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.eot_token = self.enc.eot_token  # <|endoftext|> id = 50256

    def encode(self, text: str, allowed_special=("<|endoftext|>",)):
        return self.enc.encode(text, allowed_special=set(allowed_special))

    def decode(self, token_ids):
        return self.enc.decode(token_ids)

    def text_to_tensor(self, text: str, device="cpu"):
        ids = self.encode(text)
        return torch.tensor(ids, device=device).unsqueeze(0)  # add batch dim

    def tensor_to_text(self, tensor):
        flat = tensor.squeeze(0)
        return self.decode(flat.tolist())

    @property
    def vocab_size(self):
        return self.enc.n_vocab
