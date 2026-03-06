"""
Modal runner for Qwen3.5-4B-Base blind-spot probing.

Usage:
    modal run modal_runner.py

Requires a Modal account (free tier is sufficient).
Install with: pip install modal
Authenticate with: modal setup
"""

import json
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
)

app = modal.App("qwen35-blind-spots", image=image)

MODEL_NAME = "Qwen/Qwen3.5-4B-Base"

# Cache the model weights in a Modal Volume so they only download once
volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
CACHE_DIR = "/root/.cache/huggingface"

TEST_CASES = [
    {
        "id": "tc01",
        "category": "arithmetic",
        "prompt": "Q: What is 347 multiplied by 28?\nA:",
        "expected_output": "9716",
        "explanation": "Multi-digit multiplication. Base models often produce plausible-looking but incorrect numbers.",
    },
    {
        "id": "tc02",
        "category": "character_level_reasoning",
        "prompt": "Q: Count the number of times the letter 'r' appears in the word 'strawberry'.\nA:",
        "expected_output": "3",
        "explanation": "Tokenizers split words into subwords, obscuring character counts. Models routinely answer '2'.",
    },
    {
        "id": "tc03",
        "category": "cognitive_reflection",
        "prompt": "Q: A bat and a ball together cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\nA:",
        "expected_output": "$0.05",
        "explanation": "Classic CRT item. The intuitive wrong answer is $0.10.",
    },
    {
        "id": "tc04",
        "category": "spatial_reasoning",
        "prompt": "Q: A large wooden cube is painted red on all 6 faces and then cut into 27 equal smaller cubes. How many of the small cubes have exactly 2 red faces?\nA:",
        "expected_output": "12",
        "explanation": "12 edge-cubes have exactly 2 painted faces. Models often confuse these with corner-cubes (3 faces), answering 8.",
    },
    {
        "id": "tc05",
        "category": "formal_logic",
        "prompt": "Q: All roses are flowers. Some flowers fade quickly. Does it necessarily follow that some roses fade quickly?\nA:",
        "expected_output": "No",
        "explanation": "Invalid syllogism. The flowers that fade quickly may be a disjoint set from roses.",
    },
    {
        "id": "tc06",
        "category": "commonsense_physics",
        "prompt": "Q: Five birds are sitting on a fence. A hunter shoots and kills two of them. How many birds are left sitting on the fence?\nA:",
        "expected_output": "0",
        "explanation": "The gunshot scares the survivors away. Naive arithmetic gives 3.",
    },
    {
        "id": "tc07",
        "category": "probabilistic_reasoning",
        "prompt": "Q: A rare disease affects 1 in 1,000 people. A diagnostic test has a 99% accuracy rate (false positive rate: 1%). A randomly selected person tests positive. What is the approximate probability they actually have the disease?\nA:",
        "expected_output": "Approximately 9% (about 1 in 11)",
        "explanation": "Bayes theorem. Models exhibit base-rate neglect, often answering 99%.",
    },
    {
        "id": "tc08",
        "category": "temporal_arithmetic",
        "prompt": "Q: If today is Wednesday, what day of the week will it be exactly 100 days from now?\nA:",
        "expected_output": "Friday",
        "explanation": "100 mod 7 = 2; Wednesday + 2 = Friday. Models often miscalculate.",
    },
    {
        "id": "tc09",
        "category": "factual_misconception",
        "prompt": "Q: True or False: The Great Wall of China is clearly visible from space with the naked eye.\nA:",
        "expected_output": "False",
        "explanation": "A widespread myth. The Wall is too narrow to be seen from LEO without optical aids.",
    },
    {
        "id": "tc10",
        "category": "multi_hop_ordering",
        "prompt": "Q: Alex is older than Jamie. Jamie is older than Sam. Sam is older than Taylor. Who is the youngest person?\nA:",
        "expected_output": "Taylor",
        "explanation": "Requires chaining three transitive relations. Models sometimes short-circuit.",
    },
    {
        "id": "tc11",
        "category": "linguistic_illusion",
        "prompt": "Q: In the famous biblical story, how many animals of each kind did Moses take on the Ark?\nA:",
        "expected_output": "None — it was Noah who built the Ark, not Moses.",
        "explanation": "The Moses Illusion: a substituted proper noun that readers (and models) often miss.",
    },
    {
        "id": "tc12",
        "category": "winograd_schema",
        "prompt": "Q: The trophy didn't fit in the suitcase because it was too big. What was too big?\nA:",
        "expected_output": "The trophy",
        "explanation": "Winograd schema coreference. 'It' refers to the trophy, but models sometimes resolve it to the suitcase.",
    },
]


@app.function(
    gpu="T4",
    timeout=600,
    volumes={CACHE_DIR: volume},
)
def run_probes() -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR,
    )
    model.eval()
    print("Model ready.")

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

    results = []
    for tc in TEST_CASES:
        output = generate(tc["prompt"])
        result = {**tc, "model_output": output}
        results.append(result)
        print(f"[{tc['id']}] {tc['category']}")
        print(f"  Expected : {tc['expected_output']}")
        print(f"  Model    : {output}")

    # Commit the volume so weights persist for future runs
    volume.commit()
    return results


@app.local_entrypoint()
def main():
    results = run_probes.remote()

    output_file = "blind_spots_data.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} records to {output_file}")
