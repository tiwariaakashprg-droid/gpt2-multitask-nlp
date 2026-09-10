"""
GPT-2 style decoder-only transformer, implemented from scratch in PyTorch.

Architecture (124M parameter config):
    - Token + learned positional embeddings
    - 12 transformer blocks, each with:
        - Pre-LayerNorm
        - Multi-head causal self-attention (12 heads)
        - Pre-LayerNorm
        - Position-wise feed-forward network (GELU, 4x expansion)
        - Residual connections + dropout
    - Final LayerNorm
    - Linear output head (weight-tied with token embedding, GPT-2 style)
"""

import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention, computed with a single fused qkv projection."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # causal mask, cached as a buffer (not a learnable parameter)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        # split into heads: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = q @ k.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ v).transpose(1, 2)  # (b, num_tokens, num_heads, head_dim)
        context = context.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)


class LayerNorm(nn.Module):
    """Standard LayerNorm (kept explicit/from-scratch rather than using nn.LayerNorm)."""

    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Exact-ish GELU matching the tanh approximation used in the original GPT-2 code."""

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    """124M-parameter GPT-2 style decoder-only transformer."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # weight tying (GPT-2 ties input embedding and output projection)
        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def num_parameters(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.weight.numel()
        return n_params


class GPTClassifier(nn.Module):
    """
    Wraps a pretrained GPTModel for sequence classification (e.g. SMS spam detection),
    following the standard "fine-tune GPT for classification" recipe:
    freeze most of the backbone, unfreeze the last transformer block + final norm,
    and replace the LM head with a small linear classification head on the last
    token's hidden state.
    """

    def __init__(self, gpt_model: GPTModel, num_classes: int = 2,
                 trainable_layers: int = 1, trainable_last_norm: bool = True):
        super().__init__()
        self.gpt = gpt_model
        emb_dim = gpt_model.cfg["emb_dim"]

        # freeze everything by default
        for param in self.gpt.parameters():
            param.requires_grad = False

        # unfreeze the last N transformer blocks
        if trainable_layers > 0:
            for block in list(self.gpt.trf_blocks)[-trainable_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

        if trainable_last_norm:
            for param in self.gpt.final_norm.parameters():
                param.requires_grad = True

        self.classification_head = nn.Linear(emb_dim, num_classes)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.gpt.tok_emb(in_idx)
        pos_embeds = self.gpt.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.gpt.drop_emb(x)
        x = self.gpt.trf_blocks(x)
        x = self.gpt.final_norm(x)

        # use the hidden state of the LAST token as the sequence representation
        last_token_hidden = x[:, -1, :]
        logits = self.classification_head(last_token_hidden)
        return logits
