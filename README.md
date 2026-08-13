# Fatima Fellowship Technical Task 

## Blind Spots of Frontier Models

This project probes the failure modes ("blind spots") of
[`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base),
a 4B-parameter pre-trained base language model released in February 2026.

It uses **84 prompts across 12 reasoning categories** - **5 failure probes + 2 success
controls per category** - so results can be reported as a *failure rate* (hard probes vs
easy in-domain controls) rather than as a handful of anecdotes. Each output is labelled on
two axes - **was it coherent** and **was it correct** - separating *format failures*
(incoherent / off-format) from *reasoning failures* (coherent but wrong).

_Dataset revision: June 2026._

## Repository Structure

```
.
├── colab_notebook.ipynb   # Colab: loads the model and runs all 84 probes
├── modal_runner.py        # Alternative: run on Modal cloud GPUs
├── inspect_eval.py        # Alternative: same probes as a UK AISI Inspect eval
└── dataset/
    ├── README.md          # HuggingFace dataset card + full analysis
    └── data/
        ├── prompts.jsonl  # Canonical 84-prompt input set (no outputs)
        └── train.jsonl    # Full records: prompt + expected + per-dtype runs + labels
```


## Quickstart

### Option A: Google Colab (recommended)

1. Open `colab_notebook.ipynb` in [Colab](https://colab.google.com/).
2. Set runtime to a GPU with fast bfloat16 - **L4 or A10G** (a T4 works but lacks fast
   bfloat16; see the dtype note below).
3. Run all cells. The notebook downloads `prompts.jsonl`, runs 24 probes (1 failure +
   1 control per category) in bfloat16, plus the same prompts through the
   instruction-tuned baseline `Qwen/Qwen3.5-4B`, auto-classifies each output, prints the
   summary, and saves `blind_spots_data.jsonl`. The full 84-prompt set is in
   `dataset/data/prompts.jsonl`.
4. Spot-check the auto labels, then use that file to update `dataset/data/train.jsonl`.
5. (Optional) Section 8 of the notebook installs `inspect-ai` and re-runs the full
   84-prompt set through `inspect_eval.py` on the same Colab GPU - see Option C below
   for what that produces.

### Option B: Modal

```bash
pip install modal
modal setup          # authenticate (free tier is sufficient)
modal run modal_runner.py
```

Runs the same 84 prompts and prints the same summary; results are saved to
`blind_spots_data.jsonl`.

### Option C: UK AISI Inspect

```bash
pip install inspect-ai "transformers @ git+https://github.com/huggingface/transformers.git" \
  torch accelerate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # see OOM note below
inspect eval inspect_eval.py@qwen35_blind_spots \
  --model hf/Qwen/Qwen3.5-4B-Base -M use_chat_template=false -M trust_remote_code=true -M do_sample=false
inspect eval inspect_eval.py@qwen35_blind_spots_baseline \
  --model hf/Qwen/Qwen3.5-4B -M trust_remote_code=true -M do_sample=false
inspect view   # browse per-sample transcripts in the log viewer
```

Same dataset, harness, and scoring logic as the notebook/Modal runner, reimplemented as
an [Inspect](https://inspect.aisi.org.uk/) `Task`/`Solver`/`Scorer` so it produces a
standard Inspect eval log instead of a hand-rolled JSONL summary. Runnable standalone
(above) with a local/cloud GPU, or from the optional Section 8 of the Colab notebook if
that's your only GPU access - both paths have now actually produced the full 84-prompt
results in `dataset/README.md#results`, not just been written and left untested.

Two things that are load-bearing, both found on a real run, not just in docs:

- `-M use_chat_template=false` for the base-model task - Inspect's HF provider applies a
  chat template by default, which would corrupt the plain-completion format this project
  relies on (the base model was never instruction-tuned).
- `-M do_sample=false` for **both** tasks - Inspect's HF provider decides `do_sample` at
  model instantiation from `model_args`, not from `GenerateConfig`, so it defaults to
  `True` regardless of temperature. Without this flag, generation crashes with
  `ValueError: temperature (=0.0) has to be a strictly positive float`.

If the baseline task OOMs partway through on a smaller GPU (a T4 has just enough headroom
for one ~4B model's generation buffers, and can fragment over 60-90 samples), the
`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` above mitigates it, and if it
still gets interrupted, Inspect's own recovery command resumes from exactly where it left
off rather than restarting: `inspect eval-retry logs/<the-interrupted-run>.eval`.

## Key Design Choices

**Success baseline.** Every category includes 2 easy "control" prompts (e.g. `7 + 5`,
counting letters in "cat", canonical modus ponens). Comparing the correct-rate on
controls vs the matched hard probes turns the project from "here are 12 failures" into a
measured failure rate.

**Generation harness matters as much as the prompts.** An earlier pass ran zero-shot with
a `repetition_penalty` on a base model, no stop sequence, and no controls or baseline run
- confounds that can produce "degenerate" or misleading output that has nothing to do with
the model's actual reasoning. The harness now few-shot anchors every prompt with two
generic Q/A exemplars, drops the repetition penalty, stops generation at a real stop
sequence, and runs the instruction-tuned sibling `Qwen/Qwen3.5-4B` as a baseline on the
same prompts. **Status: done.** The full 84-prompt set has been run for both models via
`inspect_eval.py` on a real Colab T4 GPU session, plus a 24-prompt subset via the
notebook's direct loop - zero format failures across all 168 generations, both models,
both runs. Getting there surfaced (and fixed) five real bugs along the way - a
coherence-heuristic false negative on short numeric answers, an unreliable
`enable_thinking` flag, an Inspect `GenerateConfig(temperature=0)` crash, an in-process
GPU-memory collision between two sequential Inspect tasks, and a mid-run CUDA
fragmentation OOM - all documented with root causes in `dataset/README.md#results`,
which now has the real headline numbers.

**Format vs reasoning failure.** A base model can fail two very different ways: by
emitting incoherent text (it never enters the Q&A frame) or by producing fluent-but-wrong
answers. We record `output_coherent` and `answer_correct` separately and derive
`failure_mode ∈ {format_failure, reasoning_failure, correct}`. The two imply different
fixes (instruction/format tuning vs chain-of-thought / domain data).

**dtype matters - bfloat16 is the primary run.** An earlier run used `float16` and
produced incoherent symbol-salad across all 12 categories (including easy controls) -
the classic signature of float16 numerical overflow on Qwen-class models, not a real
blind spot. That result confirms the float16 failures are a numerical artifact. The Colab
notebook therefore runs **bfloat16 only**; the original 12 float16 outputs are preserved
in `train.jsonl` under `runs.float16` for reference. See the caveat in `dataset/README.md`.

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

1. Spot-check the auto-labels in `blind_spots_data.jsonl` (the current file is real
   output from the full-84-prompt-set Pass 2 run - see `dataset/README.md#results`).
2. Use it to update `dataset/data/train.jsonl` with the Pass 2 records (the Pass 2
   results table in `dataset/README.md` is already filled in from the real run; the
   original Pass 1 numbers, n=12/no controls/no baseline, are kept for reference only).
3. Push to HuggingFace **only after that spot-check and `train.jsonl` update**:

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
