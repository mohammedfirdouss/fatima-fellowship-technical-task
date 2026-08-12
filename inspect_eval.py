"""
Inspect AI (https://inspect.aisi.org.uk/) eval for the Qwen3.5-4B-Base blind-spots
probe set. This is the same 84-prompt benchmark and generation harness used in
modal_runner.py / colab_notebook.ipynb, ported onto Inspect's Task/Dataset/Solver/
Scorer model so it produces a standard Inspect eval log (viewable with `inspect view`)
instead of a hand-rolled JSONL + print summary.

Install:
    pip install inspect-ai "transformers @ git+https://github.com/huggingface/transformers.git" \
        torch accelerate

Run (base model - MUST disable the chat template, it's not instruction-tuned):
    inspect eval inspect_eval.py@qwen35_blind_spots \
        --model hf/Qwen/Qwen3.5-4B-Base \
        -M use_chat_template=false -M trust_remote_code=true

Run (instruction-tuned baseline - chat template stays on, it's the default):
    inspect eval inspect_eval.py@qwen35_blind_spots_baseline \
        --model hf/Qwen/Qwen3.5-4B -M trust_remote_code=true

View results:
    inspect view

Open question to verify in a smoke run: the standard GenerateConfig has no
`repetition_penalty` field (confirmed against the Inspect reference docs), which is
moot here since the harness fix deliberately dropped it - do_sample=False / greedy is
achieved via temperature=0.
"""

import re
import string

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import Generate, TaskState, generate, solver

# Two generic exemplars (unrelated to any probe category) so the base model is
# anchored into the "Q: ... A: <short answer>" continuation pattern instead of
# treating "Q: ... A:" as arbitrary text to riff on. Same prefix as modal_runner.py.
FEW_SHOT_PREFIX = (
    "Q: What is the capital of Japan?\nA: Tokyo\n\n"
    "Q: What color is the sky on a clear day?\nA: Blue\n\n"
)

_ALLOWED = set(string.printable)
_COHERENCE_OK = re.compile(r"[A-Za-z0-9 .,;:!?$%/()'\"\-\n]")


def record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["prompt"],
        target=record["expected_output"],
        id=record["id"],
        metadata={
            "category": record["category"],
            "probe_type": record["probe_type"],
            "explanation": record.get("explanation", ""),
        },
    )


@solver
def few_shot_anchor():
    """Prepend the generic Q/A exemplars ahead of the prompt. Skipped for chat-template
    models (the baseline) since their post-training already anchors the QA format."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = FEW_SHOT_PREFIX + state.user_prompt.text
        return state

    return solve


def is_coherent(text: str) -> bool:
    """Same heuristic as modal_runner.py / the notebook - a screen, not a judge."""
    stripped = text.strip()
    if not stripped:
        return False
    legible = sum(1 for ch in stripped if _COHERENCE_OK.match(ch))
    legible_ratio = legible / len(stripped)
    alpha_ratio = sum(ch.isalpha() and ch.isascii() for ch in stripped) / len(stripped)
    return legible_ratio >= 0.90 and alpha_ratio >= 0.35


def is_correct(model_output: str, expected: str) -> bool:
    """Same loose containment check as modal_runner.py."""

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    out = norm(model_output)
    candidates = [expected, expected.split("-")[0].split("(")[0].strip()]
    candidates += re.findall(r"[0-9]+(?:\.[0-9]+)?", expected)
    return any(norm(c) and norm(c) in out for c in candidates)


@scorer(metrics=[accuracy()])
def blind_spots_scorer():
    """Reimplements classify() from modal_runner.py as an Inspect scorer.

    Score.value is CORRECT/INCORRECT (what accuracy() aggregates on); the
    format-vs-reasoning failure split - the actual point of this benchmark - is
    preserved in Score.metadata["failure_mode"] and is readable per-sample from the
    eval log, or aggregated afterwards with inspect_ai.log.read_eval_log().
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Belt-and-suspenders trim, matching the stop_seqs behavior in the config below.
        output = state.output.completion.split("\nQ:")[0].strip()
        coherent = is_coherent(output)
        correct = bool(coherent and is_correct(output, target.text))
        mode = "format_failure" if not coherent else ("correct" if correct else "reasoning_failure")
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=output,
            explanation=f"failure_mode={mode}",
            metadata={
                "output_coherent": coherent,
                "answer_correct": correct,
                "failure_mode": mode,
            },
        )

    return score


# Shared decoding config: greedy (temperature=0) with a real stop sequence, same as
# the harness fix in modal_runner.py / colab_notebook.ipynb.
_GEN_CONFIG = GenerateConfig(temperature=0, max_tokens=60, stop_seqs=["\nQ:"])


@task
def qwen35_blind_spots():
    """Base model: few-shot anchored, chat template must be disabled via -M use_chat_template=false."""
    return Task(
        dataset=json_dataset("dataset/data/prompts.jsonl", record_to_sample),
        solver=[few_shot_anchor(), generate()],
        scorer=blind_spots_scorer(),
        config=_GEN_CONFIG,
    )


@task
def qwen35_blind_spots_baseline():
    """Instruction-tuned baseline (Qwen/Qwen3.5-4B): chat template stays on (default),
    no few-shot anchor needed - the model already expects a user-turn question."""
    return Task(
        dataset=json_dataset("dataset/data/prompts.jsonl", record_to_sample),
        solver=generate(),
        scorer=blind_spots_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=200),
    )
