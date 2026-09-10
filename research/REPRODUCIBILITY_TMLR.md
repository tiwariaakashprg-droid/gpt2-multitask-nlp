# TMLR Reproducibility Package

## What is included

- GPT-2 from-scratch implementation and instruction-tuning code.
- The 300-item research dataset and anonymized human ratings.
- The original judge results as a legacy/pre-rerun dataset.
- A new explicit judge rubric in `research/judge_prompt_v2.txt`.
- A multi-judge runner in `research/run_multi_judge.py`.
- A robustness analysis script in `research/analyze_tmlr.py`.
- Human-subject/ethics checklist in `research/ETHICS_CHECKLIST.md`.

## Important distinction

The supplied project already contains results for a 300-response experiment
using a LLaMA-3 judge. Those results are preserved. The new TMLR package adds
the analysis and experiment infrastructure requested for stronger evidence.

The second-judge experiment has NOT been run in this package, because it
requires the local Ollama model(s) to be installed and executed. Do not report
a second-judge result until `run_multi_judge.py` has actually been run.

Likewise, the new v2 rubric changes the evaluation protocol. Existing
legacy judge scores must not be relabeled as v2-rubric scores. For a clean
TMLR experiment, rerun both judges using the same v2 prompt.

## Recommended experiment

1. Install dependencies from `requirements.txt`.
2. Install Ollama and record the exact model identifiers/digests.
3. Pull two local judge models. A practical starting comparison is:
   - `llama3`
   - `qwen2.5:7b`
4. Run:

```bash
python research/run_multi_judge.py   --input research/data/combined_human.csv   --output research/data/multi_judge_results.csv   --judge_models llama3 qwen2.5:7b   --repeats 3
```

5. Run:

```bash
python research/analyze_tmlr.py   --input research/data/multi_judge_results.csv   --out_dir research/results/tmlr   --judge_means llama3 qwen2_5_7b_mean
```

Note: the script sanitizes `qwen2.5:7b` to a filesystem-safe column prefix
(`qwen2_5_7b`).

6. Record exact model versions/digests and hardware/software information.
7. Update the paper only from the newly generated results.

## Current-data statistical audit

The existing 300-response dataset can be analyzed immediately with:

```bash
python research/analyze_tmlr.py   --input research/data/combined_with_judge.csv   --out_dir research/results/tmlr_legacy
```

This is an audit of the supplied experiment, not a replacement for the
new two-judge experiment.
