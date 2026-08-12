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

Note on the harness (fixed after review): the first captured pass used a bare
"Q: ... A:" continuation with no few-shot anchoring, a repetition_penalty on a base
model, no stop sequence, and only 12 prompts with zero controls run. All four are
confounds - an unanchored base model can wander off the QA format, a repetition
penalty distorts digit/token generation for arithmetic-style answers, and without a
stop sequence a coherent answer can run on into a fabricated next turn. This runner
now few-shot anchors every prompt, drops the repetition penalty, stops generation at
the next "\nQ:", and (via run_probes) always covers both failure probes and controls,
plus a same-prompt run of the instruction-tuned sibling `Qwen/Qwen3.5-4B` as a
baseline for comparison.
"""

import json
import re

import modal

# Image: Python environment with all dependencies
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
# Instruction-tuned sibling of the base model, used as a baseline: it shares the
# architecture/training data lineage but has post-training, so it tells us how much
# of the base model's failure rate is "capability gap" vs "never learned to answer".
BASELINE_MODEL = "Qwen/Qwen3.5-4B"
PROMPTS_PATH = "/root/prompts.jsonl"

# We run every prompt under BOTH dtypes and keep both, for an explicit comparison:
# float16 is the suspect run (numerical-overflow garbage); bfloat16 is the clean run.
DTYPES = ["float16", "bfloat16"]

# Two generic exemplars (unrelated to any probe category) so the base model is
# anchored into the "Q: ... A: <short answer>" continuation pattern instead of
# treating "Q: ... A:" as arbitrary text to riff on.
FEW_SHOT_PREFIX = (
    "Q: What is the capital of Japan?\nA: Tokyo\n\n"
    "Q: What color is the sky on a clear day?\nA: Blue\n\n"
)

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
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    class StopOnSubstring(StoppingCriteria):
        """Stop as soon as decoded new text contains `stop_str` (e.g. the model
        running on into a fabricated next "Q:" turn)."""

        def __init__(self, tokenizer, stop_str: str, prompt_len: int):
            self.tokenizer = tokenizer
            self.stop_str = stop_str
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            text = self.tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
            return self.stop_str in text

    test_cases = load_prompts(PROMPTS_PATH)
    print(f"Loaded {len(test_cases)} prompts from {PROMPTS_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    # One record per prompt, accumulating a run under each dtype.
    records = {tc["id"]: {**tc, "runs": {}, "baseline": None} for tc in test_cases}

    def make_generate(model, tok):
        def generate(prompt: str, max_new_tokens: int = 60) -> str:
            full_prompt = FEW_SHOT_PREFIX + prompt
            inputs = tok(full_prompt, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            stopping = StoppingCriteriaList([StopOnSubstring(tok, "\nQ:", prompt_len)])
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    stopping_criteria=stopping,
                )
            new_tokens = output_ids[0][prompt_len:]
            text = tok.decode(new_tokens, skip_special_tokens=True).strip()
            # Belt-and-suspenders: trim any run-on past the stopping criteria's check point.
            return text.split("\nQ:")[0].strip()

        return generate

    for dtype_name in DTYPES:
        print(f"\n{'#' * 60}\n# Loading {MODEL_NAME} in {dtype_name}\n{'#' * 60}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=getattr(torch, dtype_name),
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
        model.eval()
        generate = make_generate(model, tokenizer)

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

    # --- Instruction-tuned baseline (bfloat16 only) -------------------------------
    # Not a dtype variant of the base model - a different (post-trained) checkpoint,
    # run via its chat template, so results live under a separate `baseline` key
    # rather than inside `runs`.
    print(f"\n{'#' * 60}\n# Loading baseline {BASELINE_MODEL} in bfloat16\n{'#' * 60}")
    baseline_tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL, cache_dir=CACHE_DIR)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        BASELINE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR,
    )
    baseline_model.eval()

    def generate_baseline(prompt: str, max_new_tokens: int = 200) -> str:
        messages = [{"role": "user", "content": prompt.removeprefix("Q: ").removesuffix("\nA:")}]
        inputs = baseline_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(baseline_model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = baseline_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=baseline_tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][prompt_len:]
        return baseline_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    for tc in test_cases:
        output = generate_baseline(tc["prompt"])
        labels = classify(output, tc["expected_output"])
        records[tc["id"]]["baseline"] = {"model": BASELINE_MODEL, "model_output": output, **labels}
        print(f"[baseline] [{tc['id']}] {tc['probe_type']:7s} -> {labels['failure_mode']:18s} "
              f"exp={tc['expected_output']!r}")

    del baseline_model
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

    def baseline_mode(r):
        return r["baseline"]["failure_mode"]

    def baseline_coherent(r):
        return r["baseline"]["output_coherent"]

    print(f"\n--- baseline ({BASELINE_MODEL}, instruction-tuned, bfloat16) ---")
    print(f"  failure probes  coherent          : {rate(failures, baseline_coherent)}")
    print(f"  failure probes  correct           : {rate(failures, lambda r: baseline_mode(r) == 'correct')}")
    print(f"  controls        coherent          : {rate(controls, baseline_coherent)}")
    print(f"  controls        correct           : {rate(controls, lambda r: baseline_mode(r) == 'correct')}")
    print("\nRead: the gap between the base model's correct-rate and this baseline's is an")
    print("upper bound on how much post-training (not raw capability) explains the failures.")


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
