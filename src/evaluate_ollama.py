"""
Deterministic automated evaluation system.

Uses a locally-running LLaMA-3 model (served via Ollama, http://localhost:11434)
as an LLM judge to score model-generated outputs on a 0-100 scale, e.g. for
grading instruction-following quality on an Alpaca-style dataset.

Determinism: Ollama's /api/generate is called with temperature=0 and a fixed
seed, and the judge is instructed to respond with ONLY an integer 0-100 and
nothing else, so repeated runs on the same input consistently produce the
same score.

Requires Ollama running locally with the model pulled, e.g.:
    ollama pull llama3
    ollama serve
"""

import json
import re
import requests
from statistics import mean

OLLAMA_URL = "http://localhost:11434/api/generate"

JUDGE_PROMPT_TEMPLATE = """You are a strict, consistent grader. Given the input instruction, \
an optional reference input, and a model-generated response, score how well the response \
follows the instruction on a scale from 0 to 100 (100 = perfect).

Respond with ONLY the integer score. No words, no explanation, no punctuation.

### Instruction:
{instruction}

### Input:
{input_text}

### Model Response:
{response}

### Score (0-100):"""


def query_ollama(prompt, model="llama3", seed=42, temperature=0.0):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,  # 0 => deterministic / greedy decoding
            "seed": seed,                # fixed seed for reproducibility
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def parse_score(raw_text, default=0):
    """Extract the first integer 0-100 found in the judge's reply."""
    match = re.search(r"\b(100|[0-9]{1,2})\b", raw_text)
    if match:
        return int(match.group(1))
    return default


def score_single(instruction, input_text, model_response, judge_model="llama3"):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        input_text=input_text or "(none)",
        response=model_response,
    )
    raw = query_ollama(prompt, model=judge_model)
    return parse_score(raw)


def evaluate_dataset(entries, judge_model="llama3", verbose=True):
    """
    entries: list of dicts, each with keys:
        "instruction", "input" (optional), "model_response"

    Returns: list of entries annotated with a "score" field, plus the mean score.
    """
    scored = []
    for i, entry in enumerate(entries):
        score = score_single(
            entry["instruction"],
            entry.get("input", ""),
            entry["model_response"],
            judge_model=judge_model,
        )
        entry_out = {**entry, "score": score}
        scored.append(entry_out)
        if verbose:
            print(f"[{i+1}/{len(entries)}] score={score:3d}  instruction={entry['instruction'][:60]!r}")

    scores = [e["score"] for e in scored]
    summary = {
        "mean_score": mean(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "n": len(scores),
    }
    return scored, summary


def evaluate_from_json_file(json_path, judge_model="llama3", output_path=None):
    """
    Expects a JSON file with a list of objects containing at least
    "instruction" and "model_response" (Alpaca-style records work directly
    once you've added a "model_response" field with your model's generations).
    """
    with open(json_path, "r") as f:
        entries = json.load(f)

    scored, summary = evaluate_dataset(entries, judge_model=judge_model)

    if output_path:
        with open(output_path, "w") as f:
            json.dump({"results": scored, "summary": summary}, f, indent=2)

    print("\n=== Evaluation summary ===")
    print(json.dumps(summary, indent=2))
    return scored, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score model outputs with LLaMA-3 via Ollama")
    parser.add_argument("--input", required=True, help="Path to JSON file of {instruction, input, model_response}")
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--judge_model", default="llama3")
    args = parser.parse_args()

    evaluate_from_json_file(args.input, judge_model=args.judge_model, output_path=args.output)
