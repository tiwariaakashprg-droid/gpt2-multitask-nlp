"""
Configuration for the 124M-parameter GPT-2 style model.
"""

GPT2_CONFIG_124M = {
    "vocab_size": 50257,     # BPE vocab size (GPT-2 tokenizer)
    "context_length": 1024,  # max sequence length
    "emb_dim": 768,          # embedding dimension
    "n_heads": 12,           # number of attention heads
    "n_layers": 12,          # number of transformer blocks
    "drop_rate": 0.1,        # dropout rate
    "qkv_bias": True,        # use bias in q/k/v projections (matches OpenAI GPT-2 weights)
}

# Smaller config, handy for fast local testing on CPU
GPT2_CONFIG_TEST = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 96,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": True,
}
