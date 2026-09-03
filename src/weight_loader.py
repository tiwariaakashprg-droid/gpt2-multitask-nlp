"""
Loads OpenAI's official pretrained GPT-2 (124M) weights - via the HuggingFace
`transformers` GPT2LMHeadModel - into our from-scratch GPTModel implementation.

This lets us combine "I implemented the architecture myself" with "I can start
from the real pretrained weights" instead of training 124M params from random
init on a laptop.

Usage:
    from src.config import GPT2_CONFIG_124M
    from src.model import GPTModel
    from src.weight_loader import load_pretrained_gpt2

    model = GPTModel(GPT2_CONFIG_124M)
    load_pretrained_gpt2(model, model_name="gpt2")
"""

import numpy as np
import torch


def _assign(left: torch.Tensor, right: torch.Tensor) -> torch.nn.Parameter:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: left {left.shape}, right {right.shape}")
    return torch.nn.Parameter(torch.tensor(right, dtype=left.dtype))


def load_pretrained_gpt2(gpt_model, model_name: str = "gpt2"):
    """
    model_name: one of "gpt2" (124M), "gpt2-medium" (355M), "gpt2-large" (774M),
                "gpt2-xl" (1558M). Defaults to the 124M checkpoint to match
                this project's architecture config.

    Requires `pip install transformers`. Downloads weights from the
    HuggingFace Hub the first time it's run (needs internet access).
    """
    from transformers import GPT2LMHeadModel

    print(f"Downloading/loading pretrained weights for '{model_name}' ...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_sd = hf_model.state_dict()

    cfg = gpt_model.cfg
    n_layers = cfg["n_layers"]

    gpt_model.pos_emb.weight = _assign(gpt_model.pos_emb.weight, hf_sd["transformer.wpe.weight"].numpy())
    gpt_model.tok_emb.weight = _assign(gpt_model.tok_emb.weight, hf_sd["transformer.wte.weight"].numpy())

    for b in range(n_layers):
        prefix = f"transformer.h.{b}."

        # attention: HF stores fused qkv as Conv1D (in_dim, out_dim) -> need transpose
        qkv_w = hf_sd[prefix + "attn.c_attn.weight"].numpy()
        qkv_b = hf_sd[prefix + "attn.c_attn.bias"].numpy()
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=-1)
        q_b, k_b, v_b = np.split(qkv_b, 3, axis=-1)

        block = gpt_model.trf_blocks[b]
        block.att.W_query.weight = _assign(block.att.W_query.weight, q_w.T)
        block.att.W_key.weight = _assign(block.att.W_key.weight, k_w.T)
        block.att.W_value.weight = _assign(block.att.W_value.weight, v_w.T)
        block.att.W_query.bias = _assign(block.att.W_query.bias, q_b)
        block.att.W_key.bias = _assign(block.att.W_key.bias, k_b)
        block.att.W_value.bias = _assign(block.att.W_value.bias, v_b)

        out_w = hf_sd[prefix + "attn.c_proj.weight"].numpy()
        out_b = hf_sd[prefix + "attn.c_proj.bias"].numpy()
        block.att.out_proj.weight = _assign(block.att.out_proj.weight, out_w.T)
        block.att.out_proj.bias = _assign(block.att.out_proj.bias, out_b)

        # feed-forward
        fc_w = hf_sd[prefix + "mlp.c_fc.weight"].numpy()
        fc_b = hf_sd[prefix + "mlp.c_fc.bias"].numpy()
        proj_w = hf_sd[prefix + "mlp.c_proj.weight"].numpy()
        proj_b = hf_sd[prefix + "mlp.c_proj.bias"].numpy()

        block.ff.net[0].weight = _assign(block.ff.net[0].weight, fc_w.T)
        block.ff.net[0].bias = _assign(block.ff.net[0].bias, fc_b)
        block.ff.net[2].weight = _assign(block.ff.net[2].weight, proj_w.T)
        block.ff.net[2].bias = _assign(block.ff.net[2].bias, proj_b)

        # layer norms
        ln1_w = hf_sd[prefix + "ln_1.weight"].numpy()
        ln1_b = hf_sd[prefix + "ln_1.bias"].numpy()
        ln2_w = hf_sd[prefix + "ln_2.weight"].numpy()
        ln2_b = hf_sd[prefix + "ln_2.bias"].numpy()

        block.norm1.scale = _assign(block.norm1.scale, ln1_w)
        block.norm1.shift = _assign(block.norm1.shift, ln1_b)
        block.norm2.scale = _assign(block.norm2.scale, ln2_w)
        block.norm2.shift = _assign(block.norm2.shift, ln2_b)

    gpt_model.final_norm.scale = _assign(gpt_model.final_norm.scale, hf_sd["transformer.ln_f.weight"].numpy())
    gpt_model.final_norm.shift = _assign(gpt_model.final_norm.shift, hf_sd["transformer.ln_f.bias"].numpy())

    # weight tying
    gpt_model.out_head.weight = gpt_model.tok_emb.weight

    print("Pretrained weights loaded successfully.")
    return gpt_model
