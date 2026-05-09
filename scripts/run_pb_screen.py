"""Run Phase 3: Plackett-Burman screening + LOO baseline."""

import json
import numpy as np
from pathlib import Path

from src.components import COMPONENT_KEYS, load_task_config
from src.datasets import load_task_data, parse_model_answer, check_answer
from src.design import (
    generate_pb_design,
    generate_loo_design,
    design_to_configs,
    print_design_summary,
)
from src.prompts import assemble_prompt
from src.inference import run_inference
from src.components import load_components

RESULTS_DIR = Path(__file__).parent.parent / "results"
TASKS = ["gsm8k", "bbh_date", "mmlu_pro"]
MODELS = ["gpt4o_mini"]


def evaluate_design(
    design: np.ndarray,
    task_name: str,
    model_name: str,
    examples: list[dict],
    task_config: dict,
    components: dict,
) -> np.ndarray:
    """Run all design configurations and return per-example correctness matrix.

    Returns:
        (n_runs, n_examples) binary array.
    """
    configs = design_to_configs(design)
    n_runs = len(configs)
    n_examples = len(examples)
    results = np.zeros((n_runs, n_examples), dtype=int)

    for run_idx, flags in enumerate(configs):
        print(f"  Run {run_idx + 1}/{n_runs}: {flags}", flush=True)

        prompts = [
            assemble_prompt(flags, task_config, ex["question"], components)
            for ex in examples
        ]

        responses = run_inference(model_name, prompts, use_cache=True, verbose=True)

        for ex_idx, (resp, ex) in enumerate(zip(responses, examples)):
            predicted = parse_model_answer(resp, task_config["answer_type"])
            correct = check_answer(predicted, ex["answer"], task_config["answer_type"])
            results[run_idx, ex_idx] = int(correct)

        acc = results[run_idx].mean()
        print(f"    Accuracy: {acc:.3f}", flush=True)

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    components = load_components()

    # Generate designs
    pb_design = generate_pb_design(6)
    loo_design = generate_loo_design(6)
    print_design_summary(pb_design, "Plackett-Burman Design")
    print_design_summary(loo_design, "LOO Design")

    all_results = {}

    for task_name in TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        task_config = load_task_config(task_name)
        examples = load_task_data(task_name)
        print(f"Loaded {len(examples)} examples")

        all_results[task_name] = {}

        for model_name in MODELS:
            print(f"\n--- Model: {model_name} ---")

            # PB screen
            print(f"\nRunning Plackett-Burman ({pb_design.shape[0]} runs)...")
            pb_results = evaluate_design(
                pb_design, task_name, model_name, examples, task_config, components
            )

            # LOO baseline
            print(f"\nRunning LOO ({loo_design.shape[0]} runs)...")
            loo_results = evaluate_design(
                loo_design, task_name, model_name, examples, task_config, components
            )

            all_results[task_name][model_name] = {
                "pb_results": pb_results.tolist(),
                "loo_results": loo_results.tolist(),
                "pb_accuracies": pb_results.mean(axis=1).tolist(),
                "loo_accuracies": loo_results.mean(axis=1).tolist(),
            }

    # Save
    output = {
        "pb_design": pb_design.tolist(),
        "loo_design": loo_design.tolist(),
        "component_keys": COMPONENT_KEYS,
        "tasks": TASKS,
        "models": MODELS,
        "results": all_results,
    }

    with open(RESULTS_DIR / "phase3_screening.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR / 'phase3_screening.json'}")


if __name__ == "__main__":
    main()
