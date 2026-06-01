# Fatima Fellowship Technical Task 

## Blind Spots of Frontier Models

This project probes the failure modes ("blind spots") of
[`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base),
a 4B-parameter pre-trained base language model released in February 2026.

It uses **84 prompts across 12 reasoning categories** — **5 failure probes + 2 success
controls per category** — so results can be reported as a *failure rate* (hard probes vs
easy in-domain controls) rather than as a handful of anecdotes. Each output is labelled on
two axes — **was it coherent** and **was it correct** — separating *format failures*
(incoherent / off-format) from *reasoning failures* (coherent but wrong).

_Dataset revision: June 2026._

## Repository Structure

```
.
├── colab_notebook.ipynb   # Colab: loads the model and runs all 84 probes
├── modal_runner.py        # Alternative: run on Modal cloud GPUs
└── dataset/
    ├── README.md          # HuggingFace dataset card + full analysis
    └── data/
        ├── prompts.jsonl  # Canonical 84-prompt input set (no outputs)
        └── train.jsonl    # Full records: prompt + expected + per-dtype runs + labels
```

## Quickstart

### Option A: Google Colab (recommended)

1. Open `colab_notebook.ipynb` in [Colab](https://colab.google.com/).
2. Set runtime to a GPU with fast bfloat16 — **L4 or A10G** (a T4 works but lacks fast
   bfloat16; see the dtype note below).
3. Run all cells. The notebook downloads `prompts.jsonl`, runs all 84 probes under
   bfloat16, auto-classifies each output, prints the failure-rate summary, and saves
   `blind_spots_data.jsonl`.
4. Spot-check the auto labels, then use that file to update `dataset/data/train.jsonl`.

### Option B: Modal

```bash
pip install modal
modal setup          # authenticate (free tier is sufficient)
modal run modal_runner.py
```

Runs the same 84 prompts and prints the same summary; results are saved to
`blind_spots_data.jsonl`.

## Key Design Choices

**Success baseline.** Every category includes 2 easy "control" prompts (e.g. `7 + 5`,
counting letters in "cat", canonical modus ponens). Comparing the correct-rate on
controls vs the matched hard probes turns the project from "here are 12 failures" into a
measured failure rate.

**Format vs reasoning failure.** A base model can fail two very different ways: by
emitting incoherent text (it never enters the Q&A frame) or by producing fluent-but-wrong
answers. We record `output_coherent` and `answer_correct` separately and derive
`failure_mode ∈ {format_failure, reasoning_failure, correct}`. The two imply different
fixes (instruction/format tuning vs chain-of-thought / domain data).

**dtype matters — so we keep both.** An earlier run used `float16` and produced
incoherent symbol-salad across the board — the classic signature of float16 numerical
overflow on Qwen-class models, not a real blind spot. The pipeline now runs **both
`float16` and `bfloat16` on all 84 prompts and stores both** under a per-record `runs`
object. The contrast is the test: if float16 is garbage even on the easy controls while
bfloat16 is coherent, the float16 "failures" are numerical, not cognitive. See the caveat
in `dataset/README.md`.

## Model

**Qwen3.5-4B-Base** is a hybrid pre-trained model using Gated DeltaNet linear attention,
sparse mixture-of-experts, and gated attention layers (plus a vision encoder). This repo
tests only its text-generation capabilities via plain Q&A completion (no chat template,
since it is a base model), with greedy decoding for reproducibility.

## Dataset

See `dataset/README.md` for the full dataset card: the two-axis coding scheme, the
float16/bfloat16 caveat, the results table, per-category failure analysis, and concrete
fine-tuning recommendations (datasets and sizes) for each root cause.

## After Running

1. Spot-check the auto-labels in `blind_spots_data.jsonl`.
2. Use it to update `dataset/data/train.jsonl` and fill the results table in `dataset/README.md`.
3. Push to HuggingFace:

```python
from huggingface_hub import HfApi, login
login()
api = HfApi()
api.create_repo("mohammedfirdouss/qwen35-4b-base-blind-spots", repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path="dataset/",
    repo_id="mohammedfirdouss/qwen35-4b-base-blind-spots",
    repo_type="dataset",
)
```
