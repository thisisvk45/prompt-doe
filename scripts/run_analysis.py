"""Run Phase 4: Statistical analysis of PB and LOO results."""

import json
import numpy as np
from pathlib import Path

from src.components import COMPONENT_KEYS
from src.design import compute_main_effects, compute_loo_effects
from src.analysis import (
    bootstrap_main_effects,
    compute_p_values_from_bootstrap,
    benjamini_hochberg,
    compare_pb_vs_loo,
    print_effects_table,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    # Load Phase 3 results
    with open(RESULTS_DIR / "phase3_screening.json") as f:
        data = json.load(f)

    pb_design = np.array(data["pb_design"])
    loo_design = np.array(data["loo_design"])
    tasks = data["tasks"]
    models = data["models"]

    analysis_results = {}

    for task in tasks:
        analysis_results[task] = {}
        for model in models:
            print(f"\n{'='*50}")
            print(f"Task: {task} | Model: {model}")
            print(f"{'='*50}")

            r = data["results"][task][model]
            pb_matrix = np.array(r["pb_results"])
            loo_matrix = np.array(r["loo_results"])

            # PB main effects with bootstrap CIs
            print("\nPlackett-Burman Analysis:")
            pb_effects, pb_ci_low, pb_ci_high = bootstrap_main_effects(
                pb_design, pb_matrix, n_bootstrap=2000
            )
            pb_pvals = compute_p_values_from_bootstrap(
                pb_effects, pb_design, pb_matrix
            )
            pb_pvals_adj = benjamini_hochberg(pb_pvals)
            print_effects_table(pb_effects, pb_ci_low, pb_ci_high, pb_pvals_adj, "PB Main Effects")

            # LOO effects
            print("\nLOO Analysis:")
            loo_accs = loo_matrix.mean(axis=1)
            loo_effects = compute_loo_effects(loo_accs)
            print(f"  Full prompt accuracy: {loo_accs[0]:.3f}")
            for i, key in enumerate(COMPONENT_KEYS):
                print(f"  Without {key}: {loo_accs[i+1]:.3f} (effect: {loo_effects[i]:+.4f})")

            # Comparison
            comparison = compare_pb_vs_loo(pb_effects, loo_effects)
            print(f"\nPB vs LOO Comparison:")
            print(f"  Spearman r: {comparison['spearman_r']:.4f} (p={comparison['spearman_p']:.4f})")
            if comparison["disagreements"]:
                print(f"  Disagreements (rank diff > 1):")
                for d in comparison["disagreements"]:
                    print(f"    {d['component']}: PB rank {d['pb_rank']} vs LOO rank {d['loo_rank']}")

            analysis_results[task][model] = {
                "pb_effects": pb_effects.tolist(),
                "pb_ci_low": pb_ci_low.tolist(),
                "pb_ci_high": pb_ci_high.tolist(),
                "pb_pvalues_adjusted": pb_pvals_adj.tolist(),
                "loo_effects": loo_effects.tolist(),
                "comparison_spearman_r": comparison["spearman_r"],
                "comparison_spearman_p": comparison["spearman_p"],
                "disagreements": comparison["disagreements"],
            }

    # Save
    with open(RESULTS_DIR / "phase4_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR / 'phase4_analysis.json'}")


if __name__ == "__main__":
    main()
