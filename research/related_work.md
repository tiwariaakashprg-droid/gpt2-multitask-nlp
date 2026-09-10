# Related Work (Draft)

> Note: These summaries are written in my own words based on publicly available
> abstracts/descriptions of each paper. Before submitting anywhere, READ the
> actual papers yourself (links below) and rewrite this section in your own
> voice — a literature review copied from search summaries, even paraphrased,
> should reflect that you have actually read and understood the source papers.

## 1. LLM-as-a-Judge (foundational work)

**Zheng, L., Chiang, W-L., Sheng, Y., et al. (2023). "Judging LLM-as-a-Judge
with MT-Bench and Chatbot Arena." NeurIPS 2023 (Datasets & Benchmarks track).**
arXiv:2306.05685

This is the paper that established "LLM-as-a-judge" as a standard evaluation
technique. The authors used GPT-4 to grade open-ended chatbot responses and
compared its judgments against human expert annotators on their MT-Bench
benchmark (80 multi-turn questions) and the crowdsourced Chatbot Arena
platform. They report that GPT-4's agreement with human preferences reached
roughly 80%, comparable to the agreement rate between two independent human
annotators on the same task. The paper also identifies several systematic
judge biases — **position bias** (favoring whichever answer is shown first
or second), **verbosity bias** (favoring longer answers regardless of
quality), and **self-enhancement bias** (a judge model favoring outputs from
its own model family) — and proposes mitigations such as swapping answer
order and few-shot judge examples.

**Relevance to this project:** this is the direct precedent for the whole
study. Their work uses GPT-4 (a very large, closed model) as judge and
studies large chat-assistant outputs. This project asks a narrower, less
already-answered question: does the same reliability hold when (a) the judge
is a smaller open-weight model (LLaMA-3, run locally via Ollama) and (b) the
outputs being judged come from a much smaller model (GPT-2, 124M parameters)
rather than a state-of-the-art chat assistant?

## 2. G-Eval: chain-of-thought based LLM evaluation

**Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., Zhu, C. (2023). "G-Eval: NLG
Evaluation using GPT-4 with Better Human Alignment." EMNLP 2023.**
arXiv:2303.16634

G-Eval improves on naive "just ask the LLM for a score" prompting by having
the judge model first generate a chain-of-thought of evaluation steps, then
score the output in a structured form-filling format, using the
probability-weighted sum over score tokens rather than a single sampled
token. On text summarization, G-Eval reports a Spearman correlation of 0.514
with human judgments, an improvement over prior automatic metrics. The
authors also flag a bias risk: LLM judges may score LLM-generated text more
favorably than human-written text of comparable quality.

**Relevance to this project:** this motivates one of the possible
follow-up extensions of the study — comparing a "naive prompt" scoring
approach (as implemented in this project's `evaluate_ollama.py`) against a
G-Eval-style chain-of-thought scoring approach, to see whether the extra
structure improves agreement with human raters when judging small-model
outputs specifically.

## 3. Further reliability / bias studies (to read and add)

The following are cited in secondary sources as directly relevant follow-up
work; **verify each directly on arXiv/ACL Anthology before citing them**, as
I have not independently confirmed their exact claims the way I did for the
two papers above:

- Huang, H. et al. (2024). "An Empirical Study of LLM-as-a-Judge for LLM
  Evaluation: Fine-Tuned Judge Models are Not a General Substitute for
  GPT-4." arXiv:2403.02839 — relevant to whether a smaller/local judge like
  LLaMA-3 can substitute for a frontier model judge.
- Chiang, W-L. et al. (2024). "Chatbot Arena: An Open Platform for
  Evaluating LLMs by Human Preference." arXiv:2403.04132 — background on
  large-scale human preference collection methodology, useful for justifying
  your annotation protocol design.
- Jung, J. et al. (2024). "Trust or Escalate: LLM Judges with Provable
  Guarantees for Human Agreement." arXiv:2407.18370 — relevant if you want to
  discuss confidence-calibrated judging as future work.

## 4. Classic reference-based NLG metrics (for context/contrast)

BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004) are the traditional
n-gram-overlap metrics that LLM-as-a-judge approaches are explicitly
positioned against — both Zheng et al. and Liu et al. motivate their work by
noting these metrics correlate poorly with human judgment on open-ended
generation. Mention this briefly in your Introduction to frame why LLM
judges are needed at all; you don't need to run BLEU/ROUGE yourself unless a
reviewer specifically asks for a classical-metric baseline.

## What YOU still need to do here

1. Actually read Zheng et al. (2023) and Liu et al. (2023) in full (they are
   free on arXiv) — the summaries above are a starting point, not a
   substitute.
2. Search Google Scholar / arXiv for anything published in 2025-2026 on
   "small language model evaluation," "open-weight LLM judge," or "local LLM
   judge reliability" — this field moves fast, and a paper submitted without
   the most recent related work will get rejected for missing citations.
3. Rewrite every summary above in your own words after reading the source,
   citing the specific claims/numbers you use.
