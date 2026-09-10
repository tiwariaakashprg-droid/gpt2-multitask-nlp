"""
Decoding strategies for autoregressive generation:
    - greedy / plain argmax decoding
    - temperature scaling
    - top-k sampling (optionally combined with temperature)
"""

import torch


@torch.no_grad()
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Deterministic greedy decoding (argmax each step)."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_id), dim=1)
    return idx


@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size,
             temperature=1.0, top_k=None, eos_id=None):
    """
    Generation with temperature scaling and/or top-k sampling.

    temperature: >0. temperature < 1 sharpens the distribution (more greedy),
                 temperature > 1 flattens it (more random). temperature == 0
                 falls back to greedy argmax decoding.
    top_k: if set, restrict sampling to the k highest-probability tokens
           at each step before applying softmax.
    eos_id: if the model generates this token id, stop early.
    """
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and next_id.item() == eos_id:
            break

        idx = torch.cat((idx, next_id), dim=1)

    return idx


def generate_and_decode(model, tokenizer, prompt, max_new_tokens=50,
                         context_size=1024, temperature=1.0, top_k=50, device="cpu"):
    """Convenience helper: text in, text out."""
    encoded = tokenizer.text_to_tensor(prompt, device=device)
    out_ids = generate(
        model=model,
        idx=encoded,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eot_token,
    )
    return tokenizer.tensor_to_text(out_ids)
