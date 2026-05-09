"""Step 2: Investigate the (-,+,-,+,-,+) outlier configuration on gpt-4o-mini."""

import json
import csv
from pathlib import Path

from src.components import COMPONENT_KEYS, load_components, load_task_config
from src.datasets import load_task_data, load_task_data_with_seed, parse_model_answer, check_answer
from src.prompts import assemble_prompt
from src.inference import run_inference

RESULTS_DIR = Path(__file__).parent.parent / "results" / "gpt-4o-mini"
TASKS = ["gsm8k", "bbh_date", "mmlu_pro"]
MODEL = "gpt4o_mini"

OUTLIER_FLAGS = {
    "system_role": False,
    "persona": True,
    "few_shot": False,
    "cot_trigger": True,
    "output_format": False,
    "constraints": True,
}


def run_outlier_on_seed(task_name, seed, components, task_config):
    """Run outlier config on examples sampled with given seed."""
    examples = load_task_data_with_seed(task_name, seed)
    prompts = [
        assemble_prompt(OUTLIER_FLAGS, task_config, ex["question"], components)
        for ex in examples
    ]
    responses = run_inference(MODEL, prompts, use_cache=True, verbose=True)

    correct = 0
    failures = []
    for i, (resp, ex) in enumerate(zip(responses, examples)):
        predicted = parse_model_answer(resp, task_config["answer_type"])
        is_correct = check_answer(predicted, ex["answer"], task_config["answer_type"])
        if is_correct:
            correct += 1
        else:
            failures.append({
                "index": i,
                "question": ex["question"][:200],
                "gold_answer": ex["answer"],
                "model_response": resp[:500],
                "parsed_answer": predicted,
            })

    accuracy = correct / len(examples)
    return accuracy, failures


def categorize_failure(failure, answer_type):
    """Heuristic categorization of failure mode."""
    resp = failure["model_response"].lower()
    parsed = failure["parsed_answer"]

    if "i cannot" in resp or "i'm sorry" in resp or "i can't" in resp:
        return "refusal"
    if parsed == resp.strip().split("\n")[-1].strip() and len(parsed) > 50:
        return "format_breakdown"
    if answer_type == "multiple_choice" and not any(c in parsed for c in "ABCDEFGHIJ"):
        return "format_breakdown"
    return "wrong_reasoning"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    components = load_components()

    rerun_seeds = [43, 44, 45]
    all_accuracies = []
    all_failures = []

    for task_name in TASKS:
        task_config = load_task_config(task_name)
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"{'='*50}", flush=True)

        # Original seed=42 run (should match cached results)
        print(f"\n  Original (seed=42)...", flush=True)
        acc_orig, failures_orig = run_outlier_on_seed(task_name, 42, components, task_config)
        print(f"    Accuracy: {acc_orig:.3f} ({len(failures_orig)} failures)", flush=True)
        all_accuracies.append({"task": task_name, "seed": 42, "accuracy": f"{acc_orig:.3f}"})

        # Re-runs with different seeds
        for seed in rerun_seeds:
            print(f"\n  Rerun (seed={seed})...", flush=True)
            acc, _ = run_outlier_on_seed(task_name, seed, components, task_config)
            print(f"    Accuracy: {acc:.3f}", flush=True)
            all_accuracies.append({"task": task_name, "seed": seed, "accuracy": f"{acc:.3f}"})

        # Collect 5 failure cases from original run
        sample_failures = failures_orig[:5]
        for f in sample_failures:
            f["task"] = task_name
            f["failure_mode"] = categorize_failure(f, task_config["answer_type"])
            all_failures.append(f)

        print(f"\n  Failure modes (first 5):")
        for f in sample_failures:
            print(f"    [{f['failure_mode']}] gold={f['gold_answer']}, parsed={f['parsed_answer']}")

    # Save outlier_investigation.csv
    with open(RESULTS_DIR / "outlier_investigation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "seed", "accuracy"])
        w.writeheader()
        w.writerows(all_accuracies)

    # Save outlier_failures.json
    with open(RESULTS_DIR / "outlier_failures.json", "w") as f:
        json.dump(all_failures, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR / 'outlier_investigation.csv'}")
    print(f"Saved to {RESULTS_DIR / 'outlier_failures.json'}")

    # Summary
    print("\n=== SUMMARY ===")
    for task_name in TASKS:
        rows = [r for r in all_accuracies if r["task"] == task_name]
        accs = [float(r["accuracy"]) for r in rows]
        parts = []
        for r in rows:
            seed = r["seed"]
            acc = r["accuracy"]
            parts.append(f"seed={seed}:{acc}")
        mean_acc = sum(accs) / len(accs)
        print(f"{task_name}: {', '.join(parts)} | mean={mean_acc:.3f}")


if __name__ == "__main__":
    main()
