"""Run Phase 6: Cross-model transfer analysis."""

import json
import numpy as np
from pathlib import Path

from src.components import COMPONENT_KEYS
from src.design import generate_pb_design, compute_main_effects
from src.transfer import compute_cross_model_transfer, print_transfer_report

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    # Load Phase 3 results
    with open(RESULTS_DIR / "phase3_screening.json") as f:
        data = json.load(f)

    pb_design = np.array(data["pb_design"])
    tasks = data["tasks"]
    models = data["models"]

    if len(models) < 2:
        print("Need at least 2 models for cross-model transfer analysis.")
        return

    model_a, model_b = models[0], models[1]

    effects_a = {}
    effects_b = {}

    for task in tasks:
        accs_a = np.array(data["results"][task][model_a]["pb_accuracies"])
        accs_b = np.array(data["results"][task][model_b]["pb_accuracies"])
        effects_a[task] = compute_main_effects(pb_design, accs_a)
        effects_b[task] = compute_main_effects(pb_design, accs_b)

    results = compute_cross_model_transfer(effects_a, effects_b, model_a, model_b)
    print_transfer_report(results)

    # Save
    output = {
        "model_a": model_a,
        "model_b": model_b,
        "per_task": {},
        "aggregated": results["aggregated"],
    }
    for task in tasks:
        output["per_task"][task] = {
            "effects_a": effects_a[task].tolist(),
            "effects_b": effects_b[task].tolist(),
            **results["per_task"][task],
        }

    with open(RESULTS_DIR / "phase6_transfer.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR / 'phase6_transfer.json'}")


if __name__ == "__main__":
    main()
