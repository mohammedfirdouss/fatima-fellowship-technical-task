# Fatima Fellowship Technical Task — Blind Spots of Frontier Models

This project probes the failure modes ("blind spots") of
[`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base),
a 4B-parameter pre-trained base language model released in February 2026.

## Repository Structure

```
.
├── colab_notebook.ipynb   # Colab notebook: loads the model and runs all probes
├── modal_runner.py        # Alternative: run on Modal cloud GPUs
└── dataset/
    ├── README.md          # HuggingFace dataset card (upload this with the data)
    └── data/
        └── train.jsonl    # 12 test cases with prompts, expected outputs, model outputs
```

## Quickstart

### Option A: Google Colab (recommended for beginners)

1. Go to [colab.google.com](https://colab.google.com/) and upload `colab_notebook.ipynb`
2. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
3. Run all cells — the notebook loads the model, runs all 12 probes, and saves
   results to `blind_spots_data.jsonl`
4. Copy the generated model outputs into `dataset/data/train.jsonl`

### Option B: Modal

```bash
pip install modal
modal setup          # authenticate (free tier is sufficient)
modal run modal_runner.py
```

Results are saved to `blind_spots_data.jsonl` in your working directory.

## Model

**Qwen3.5-4B-Base** is a hybrid pre-trained language model using Gated DeltaNet
linear attention, sparse mixture-of-experts, and gated attention layers. It also
includes a vision encoder. This repo tests only its text-generation capabilities
via plain Q&A completion (no chat template, since it is a base model).

## Dataset

The `dataset/` folder contains 12 diverse prompts spanning:

| Category | Example failure |
|---|---|
| Arithmetic | 347 × 28 (multi-digit multiplication) |
| Character-level reasoning | Counting 'r' in "strawberry" |
| Cognitive reflection | Bat-and-ball CRT problem |
| Spatial reasoning | Painted cube edge count |
| Formal logic | Invalid syllogism detection |
| Commonsense physics | Birds on a fence after gunshot |
| Probabilistic reasoning | Bayesian base-rate neglect |
| Temporal arithmetic | Day-of-week modular arithmetic |
| Factual misconception | Great Wall of China space myth |
| Multi-hop ordering | Transitive age comparisons |
| Linguistic illusion | Moses Illusion |
| Winograd schema | Pronoun coreference |

See `dataset/README.md` for detailed analysis, fine-tuning recommendations,
and dataset assembly strategy.

## After Running

Once you have actual model outputs:

1. Fill in the `model_output` field in `dataset/data/train.jsonl`
2. Update your HF username in `dataset/README.md`
3. Push to HuggingFace:

```python
from huggingface_hub import HfApi, login
login()
api = HfApi()
api.create_repo("YOUR_USERNAME/qwen35-4b-base-blind-spots", repo_type="dataset")
api.upload_folder(
    folder_path="dataset/",
    repo_id="YOUR_USERNAME/qwen35-4b-base-blind-spots",
    repo_type="dataset",
)
```
