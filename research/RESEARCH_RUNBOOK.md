# RESEARCH RUNBOOK — exactly what you need to do

Everything below is code I already wrote. You just need to **run commands in
order** and do the two human steps (installing Ollama, sending a file to
friends). Follow this top to bottom.

---

## PART A — One-time setup (15-20 minutes)

### A1. Install Python dependencies

```bash
cd gpt2-multitask-nlp
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install scipy matplotlib      # needed for the research analysis scripts
```

### A2. Install Ollama (runs LLaMA-3 locally, free)

1. Go to https://ollama.com and download the installer for your OS
   (Windows/Mac/Linux all supported).
2. Install it (normal next-next-finish installer).
3. Open a terminal and run:
   ```bash
   ollama pull llama3
   ```
   This downloads the model (~4.7 GB, one-time).
4. Start the server (keep this terminal open/running in the background):
   ```bash
   ollama serve
   ```
5. Test it worked, in a **second** terminal:
   ```bash
   curl http://localhost:11434/api/generate -d "{\"model\":\"llama3\",\"prompt\":\"hello\",\"stream\":false}"
   ```
   If you get a JSON response back with text in it, you're good.

---

## PART B0 — Instruction-tune the model FIRST (do not skip this!)

**Why this step exists:** the raw pretrained GPT-2 has only ever learned to
predict the "next word" of generic internet text. It has never seen the
`### Instruction: ... ### Response:` format, so if you generate from it
directly, it just rambles/repeats and produces near-uniformly bad output
(you may see your human annotators giving 0 to almost everything, with no
variation — this breaks the statistics in Part F, since you can't measure
correlation when one variable never varies). Fine-tuning on a few hundred
real instruction-response examples fixes this.

```bash
python research/finetune_instruction.py --num_examples 1000 --epochs 2
```

- On a **GPU** (even a free Google Colab T4) this takes ~15-30 minutes.
- On **CPU only**, this can take 1-3+ hours. If that's too slow, either:
  - reduce `--num_examples` to 300-500 (faster, slightly less fluent model), or
  - run this one script in a free Google Colab notebook (upload the
    `gpt2-multitask-nlp` folder, `!pip install -r requirements.txt`, then run
    the same command with a GPU runtime), then download the resulting
    checkpoint file back to your laptop.

This saves a checkpoint to `research/checkpoints/gpt2_instruction_tuned.pt`.

## PART B — Generate the model outputs (5 minutes, automated)

```bash
python research/generate_qa_pairs.py \
    --checkpoint research/checkpoints/gpt2_instruction_tuned.pt \
    --output research/data/qa_pairs.json
```

This generates responses to all 20 sample questions x 3 decoding settings =
60 question-answer pairs, saved to `research/data/qa_pairs.json`, using the
instruction-tuned model from Part B0.

**If you already collected annotations using the OLD ungrounded outputs**
(the ones that all scored 0), you'll need to re-generate `qa_pairs.json`
with the command above and re-run annotation from scratch — sorry, but
those old ratings genuinely can't be used for the agreement analysis.

**Optional but recommended:** open `research/questions.json` and add more
questions (aim for 40-60 total questions, so with 3 decoding settings you
get 120-180 items — more data = more statistical power for your paper).

---

## PART C — Human annotation (this is the part YOU coordinate, ~1-2 days)

1. Send two files to each of your 3-5 friends:
   - `research/annotate.html`
   - `research/data/qa_pairs.json`
2. Tell them: **"Open `annotate.html` in any browser (double-click it, no
   internet needed), type your name, upload `qa_pairs.json` when prompted,
   then rate every response 0-100 and click Download at the end."**
3. Each friend will download a file called `annotations_<theirname>.json`.
4. Collect all of those files back from your friends and put them in
   `research/data/`.

**Important:** don't tell your friends what score you expect, or show them
each other's ratings — that biases the "ground truth" you're trying to
establish.

---

## PART D — Merge human ratings + check agreement (2 minutes, automated)

```bash
python research/merge_annotations.py \
    --qa_pairs research/data/qa_pairs.json \
    --annotations research/data/annotations_*.json \
    --output research/data/combined_human.csv
```

(If `annotations_*.json` doesn't expand automatically on your shell, list
each file explicitly separated by spaces.)

This prints the inter-annotator agreement (how much your friends agreed with
each other) to the terminal — **note this number down**, it goes in your
paper's Method section (3.4).

---

## PART E — Run the LLaMA-3 judge (10-30 minutes depending on dataset size, automated)

Make sure `ollama serve` is still running in a terminal, then:

```bash
python research/run_llama_judge.py \
    --input research/data/combined_human.csv \
    --output research/data/combined_with_judge.csv \
    --repeats 3
```

This scores every item 3 times each (to test consistency) using LLaMA-3.

---

## PART F — Run the statistical analysis (1 minute, automated)

```bash
python research/analyze_agreement.py --input research/data/combined_with_judge.csv
```

This creates:
- `research/results/REPORT.md` — every number you need for the paper
- `research/results/plots/*.png` — every figure you need for the paper

---

## PART G — Write the paper

1. Open `research/paper_template.md`.
2. Open `research/results/REPORT.md` side by side.
3. Copy every number/figure from REPORT.md into the matching `[FILL IN...]`
   placeholder in the paper template.
4. Read `research/related_work.md`, actually read the 2 cited papers on
   arXiv, then rewrite that section in your own words.
5. Write the Abstract and Discussion LAST, once you know your actual
   findings.
6. Fill in Limitations honestly — this is expected in every real paper, not
   a weakness to hide.

---

## PART H — Where to submit

Given the scale of this study (small dataset, one judge model, informal
annotators), the realistic and respectable venues are:

- **arXiv preprint** first (free, instant, gives you a citable link) —
  https://arxiv.org/
- **Workshop papers** at NLP conferences — specifically look at
  **Eval4NLP** (a workshop dedicated exactly to this topic — LLM evaluation)
  co-located with EMNLP/ACL most years. Search "Eval4NLP call for papers"
  for the current year's deadline.
- Student/undergraduate research symposiums at your institution, if
  applicable.
- Full top-tier conference papers (ACL/EMNLP/NAACL main tracks) are a much
  higher bar — realistic only if you scale up the study significantly
  (more items, more judge models, more annotators, maybe a second small
  generator model besides GPT-2). Treat that as a stretch goal, not the
  first target.

---

## Troubleshooting

- **`ollama serve` says port already in use** → Ollama is probably already
  running in the background; skip that step and go straight to the curl
  test.
- **`generate_qa_pairs.py` fails to download GPT-2 weights** → check your
  internet connection; HuggingFace Hub (`huggingface.co`) needs to be
  reachable.
- **Friends' `annotations_*.json` files won't merge** → make sure each file
  wasn't renamed and still contains valid JSON (open it in a text editor to
  check it starts with `[` and ends with `]`).
- **Judge scores all come back as 0** → check `ollama serve` is running and
  reachable at `http://localhost:11434` before running `run_llama_judge.py`.
