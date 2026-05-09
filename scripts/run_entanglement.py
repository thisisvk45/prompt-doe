"""Run Phase 2: Semantic independence / entanglement test."""

import json
import numpy as np
from pathlib import Path

from src.independence import compute_entanglement_matrix, print_entanglement_matrix
from src.components import COMPONENT_KEYS

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    print("Phase 2: Computing entanglement matrix...")
    stat_matrix, pval_matrix = compute_entanglement_matrix(
        model_name="all-mpnet-base-v2",
        n_variants=50,
        verbose=True,
    )

    print_entanglement_matrix(stat_matrix, pval_matrix)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "component_keys": COMPONENT_KEYS,
        "hsic_statistics": stat_matrix.tolist(),
        "p_values": pval_matrix.tolist(),
        "significant_pairs": [],
    }
    for i in range(len(COMPONENT_KEYS)):
        for j in range(i + 1, len(COMPONENT_KEYS)):
            if pval_matrix[i, j] < 0.05:
                output["significant_pairs"].append({
                    "pair": [COMPONENT_KEYS[i], COMPONENT_KEYS[j]],
                    "statistic": float(stat_matrix[i, j]),
                    "p_value": float(pval_matrix[i, j]),
                })

    with open(RESULTS_DIR / "phase2_entanglement.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR / 'phase2_entanglement.json'}")
    print(f"Found {len(output['significant_pairs'])} significant entangled pairs (p < 0.05)")


if __name__ == "__main__":
    main()
