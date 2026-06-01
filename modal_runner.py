"""
Modal runner for Qwen3.5-4B-Base blind-spot probing.

Reads the canonical prompt set from ``dataset/data/prompts.jsonl`` (84 prompts:
5 failure probes + 2 success controls across each of 12 categories), runs every
prompt through the model, auto-classifies each output, and writes a results file
plus a printed failure-rate summary.

Usage:
    modal run modal_runner.py

Requires a Modal account (free tier is sufficient).
Install with: pip install modal
Authenticate with: modal setup

Note on dtype: we load in bfloat16. The earlier float16 run produced incoherent
"symbol-salad" output, which is a known signature of float16 numerical overflow on
Qwen-class models rather than a genuine reasoning failure. bfloat16 has the wider
dynamic range needed for stable generation. If your GPU cannot run bfloat16
efficiently (e.g. a T4), prefer an L4/A10G, or test float16 explicitly and treat
incoherent output as a possible numerical artifact.
"""

import json
import re

import modal

# ---------------------------------------------------------------------------
# Image: Python environment with all dependencies
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.47.0",
        "torch",
        "accelerate",
        "huggingface_hub",
    )
    # Ship the canonical prompt file into the container.
    .add_local_file(
        "dataset/data/prompts.jsonl",
        "/root/prompts.jsonl",
    )
)

app = modal.App("qwen35-blind-spots", image=image)

MODEL_NAME = "Qwen/Qwen3.5-4B-Base"
PROMPTS_PATH = "/root/prompts.jsonl"

# We run every prompt under BOTH dtypes and keep both, for an explicit comparison:
# float16 is the suspect run (numerical-overflow garbage); bfloat16 is the clean run.
DTYPES = ["float16", "bfloat16"]

# Cache the model weights in a Modal Volume so they only download once
volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
CACHE_DIR = "/root/.cache/huggingface"


# Output classification (heuristic; verify the borderline cases by hand)
# We distinguish two failure modes, per reviewer feedback:
#   * format_failure    - the output is incoherent / off-format. The model never
#                         produces a usable answer (it does not "play the QA game").
#   * reasoning_failure - the output is coherent, on-format text, but the answer
#                         is wrong. The model understood the task but mis-reasoned.
# A coherent + correct output is a `correct` (used for the success baseline).

_COHERENCE_OK = re.compile(r"[A-Za-z0-9 .,;:!?$%/()'\"\-\n]")


def is_coherent(text: str) -> bool:
    """Rough heuristic: is the output mostly normal English/QA text?

    Returns False for the "symbol-salad" outputs (heavy non-Latin scripts,
    runaway punctuation, repeated single tokens). This is a screen, not a
    judge - eyeball anything near the threshold.
    """
    stripped = text.strip()
    if len(stripped) < 1:
        return False
    legible = sum(1 for ch in stripped if _COHERENCE_OK.match(ch))
    legible_ratio = legible / len(stripped)
    # Fraction of ASCII-alphabetic characters - garbage tends to be punctuation/digits.
    alpha_ratio = sum(ch.isalpha() and ch.isascii() for ch in stripped) / len(stripped)
    return legible_ratio >= 0.90 and alpha_ratio >= 0.35


def is_correct(model_output: str, expected: str) -> bool:
    """Very loose containment check on a normalized form of the expected answer.

    Only meaningful when the output is coherent. Always re-check failures and
    near-misses manually before trusting the label.
    """
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    out = norm(model_output)
    candidates = [expected]
    head = expected.split("-")[0].split("(")[0].strip()
    candidates.append(head)
    candidates.extend(re.findall(r"[0-9]+(?:\.[0-9]+)?", expected))
    return any(norm(c) and norm(c) in out for c in candidates)


def classify(model_output: str, expected: str) -> dict:
    coherent = is_coherent(model_output)
    correct = bool(coherent and is_correct(model_output, expected))
    if not coherent:
        mode = "format_failure"
    elif correct:
        mode = "correct"
    else:
        mode = "reasoning_failure"
    return {
        "output_coherent": coherent,
        "answer_correct": correct,
        "failure_mode": mode,
    }


def load_prompts(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@app.function(
    gpu="L4",
    timeout=3600,
    volumes={CACHE_DIR: volume},
)
def run_probes() -> list[dict]:
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    test_cases = load_prompts(PROMPTS_PATH)
    print(f"Loaded {len(test_cases)} prompts from {PROMPTS_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    # One record per prompt, accumulating a run under each dtype.
    records = {tc["id"]: {**tc, "runs": {}} for tc in test_cases}

    for dtype_name in DTYPES:
        print(f"\n{'#' * 60}\n# Loading {MODEL_NAME} in {dtype_name}\n{'#' * 60}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=getattr(torch, dtype_name),
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
        model.eval()

        def generate(prompt: str, max_new_tokens: int = 80) -> str:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        for tc in test_cases:
            output = generate(tc["prompt"])
            labels = classify(output, tc["expected_output"])
            records[tc["id"]]["runs"][dtype_name] = {"model_output": output, **labels}
            print(f"[{dtype_name}] [{tc['id']}] {tc['probe_type']:7s} "
                  f"-> {labels['failure_mode']:18s} exp={tc['expected_output']!r}")

        # Free the GPU before loading the next dtype.
        del model
        gc.collect()
        torch.cuda.empty_cache()

    volume.commit()
    return [records[tc["id"]] for tc in test_cases]


def summarize(results: list[dict]) -> None:
    """Print the per-dtype failure-rate breakdown and the float16-vs-bfloat16 contrast."""
    def rate(rows, pred) -> str:
        if not rows:
            return "n/a"
        n = sum(1 for r in rows if pred(r))
        return f"{n}/{len(rows)} ({100 * n / len(rows):.0f}%)"

    failures = [r for r in results if r["probe_type"] == "failure"]
    controls = [r for r in results if r["probe_type"] == "control"]

    print("\n" + "=" * 64)
    print("SUMMARY (per dtype)")
    print("=" * 64)
    print(f"Failure probes: {len(failures)}   Success controls: {len(controls)}")

    for dt in DTYPES:
        def mode(r):
            return r["runs"][dt]["failure_mode"]

        def coherent(r):
            return r["runs"][dt]["output_coherent"]

        print(f"\n--- {dt} ---")
        print(f"  failure probes  coherent          : {rate(failures, coherent)}")
        print(f"  failure probes  correct           : {rate(failures, lambda r: mode(r) == 'correct')}")
        print(f"    format_failure                  : {rate(failures, lambda r: mode(r) == 'format_failure')}")
        print(f"    reasoning_failure               : {rate(failures, lambda r: mode(r) == 'reasoning_failure')}")
        print(f"  controls        coherent          : {rate(controls, coherent)}")
        print(f"  controls        correct           : {rate(controls, lambda r: mode(r) == 'correct')}")

    print("\nRead: if float16 is incoherent even on the easy CONTROLS while bfloat16 is")
    print("coherent on them, the float16 'failures' are a numerical artifact, not a blind spot.")


@app.local_entrypoint()
def main():
    results = run_probes.remote()

    output_file = "blind_spots_data.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} records to {output_file}")
    summarize(results)
    print(
        "\nNext: spot-check the auto labels, then paste these outputs back to merge "
        "into dataset/data/train.jsonl (which keeps both dtype runs per prompt)."
    )
