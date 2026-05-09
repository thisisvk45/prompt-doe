"""Run Phase 5: Full 2^6 factorial validation."""

import json
import numpy as np
from pathlib import Path

from src.components import COMPONENT_KEYS, load_components, load_task_config
from src.datasets import load_task_data, parse_model_answer, check_answer
from src.design import generate_full_factorial, generate_pb_design, design_to_configs, compute_main_effects
from src.prompts import assemble_prompt
from src.inference import run_inference
from src.validation import validate_pb_against_factorial, print_validation_report

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Validation on 1 task + 1 model
TASK = "gsm8k"
MODEL = "gpt4o_mini"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    components = load_components()
    task_config = load_task_config(TASK)
    examples = load_task_data(TASK)

    print(f"Phase 5: Full Factorial Validation")
    print(f"Task: {TASK} | Model: {MODEL}")
    print(f"Examples: {len(examples)}")

    # Full factorial: 64 runs
    factorial_design = generate_full_factorial(6)
    configs = design_to_configs(factorial_design)
    print(f"Running {len(configs)} factorial configurations...")

    n_runs = len(configs)
    n_examples = len(examples)
    factorial_results = np.zeros((n_runs, n_examples), dtype=int)

    for run_idx, flags in enumerate(configs):
        print(f"  Run {run_idx + 1}/{n_runs}")
        prompts = [
            assemble_prompt(flags, task_config, ex["question"], components)
            for ex in examples
        ]
        responses = run_inference(MODEL, prompts, use_cache=True, verbose=False)

        for ex_idx, (resp, ex) in enumerate(zip(responses, examples)):
            predicted = parse_model_answer(resp, task_config["answer_type"])
            correct = check_answer(predicted, ex["answer"], task_config["answer_type"])
            factorial_results[run_idx, ex_idx] = int(correct)

        acc = factorial_results[run_idx].mean()
        if (run_idx + 1) % 10 == 0:
            print(f"    Run {run_idx + 1} accuracy: {acc:.3f}")

    factorial_accs = factorial_results.mean(axis=1)

    # Load PB effects from Phase 3 (or recompute)
    pb_design = generate_pb_design(6)
    try:
        with open(RESULTS_DIR / "phase3_screening.json") as f:
            phase3 = json.load(f)
        pb_accs = np.array(phase3["results"][TASK][MODEL]["pb_accuracies"])
        pb_effects = compute_main_effects(pb_design, pb_accs)
        print("Loaded PB effects from Phase 3 results.")
    except (FileNotFoundError, KeyError):
        print("Phase 3 results not found. Computing PB effects from factorial subset...")
        # Use the first 12 rows that match PB design
        pb_accs = factorial_accs[:12]
        pb_effects = compute_main_effects(pb_design, pb_accs)

    # Validate
    results = validate_pb_against_factorial(pb_effects, factorial_design, factorial_accs)
    print_validation_report(results)

    # Save
    output = {
        "task": TASK,
        "model": MODEL,
        "n_examples": n_examples,
        "factorial_accuracies": factorial_accs.tolist(),
        "true_effects": results["true_effects"].tolist(),
        "pb_effects": results["pb_effects"].tolist(),
        "pearson_r": results["pearson_r"],
        "spearman_r": results["spearman_r"],
        "mae": results["mae"],
        "rmse": results["rmse"],
        "rank_agreement": results["rank_agreement"],
        "interactions": {
            f"{COMPONENT_KEYS[i]}__x__{COMPONENT_KEYS[j]}": v
            for (i, j), v in results["interactions"].items()
        },
    }

    with open(RESULTS_DIR / "phase5_validation.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR / 'phase5_validation.json'}")


if __name__ == "__main__":
    main()
