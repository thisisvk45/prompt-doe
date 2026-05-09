# MIT License - Copyright (c) 2026 Vikas Kumar
from __future__ import annotations
"""Phase 6: Cross-model transfer analysis."""

import numpy as np
from scipy import stats
from src.components import COMPONENT_KEYS


def compute_cross_model_transfer(
    effects_model_a: dict[str, np.ndarray],
    effects_model_b: dict[str, np.ndarray],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> dict:
    """Compute cross-model transfer of component attributions.

    Args:
        effects_model_a: dict mapping task_name -> (n_factors,) main effects array for model A.
        effects_model_b: dict mapping task_name -> (n_factors,) main effects array for model B.

    Returns:
        Dict with per-task and aggregated Spearman correlations.
    """
    results = {"per_task": {}, "model_a": model_a_name, "model_b": model_b_name}

    all_a = []
    all_b = []

    for task in effects_model_a:
        ea = effects_model_a[task]
        eb = effects_model_b[task]
        r, p = stats.spearmanr(ea, eb)
        results["per_task"][task] = {"spearman_r": r, "spearman_p": p}
        all_a.extend(ea)
        all_b.extend(eb)

    # Aggregated
    all_a = np.array(all_a)
    all_b = np.array(all_b)
    r_agg, p_agg = stats.spearmanr(all_a, all_b)
    results["aggregated"] = {"spearman_r": r_agg, "spearman_p": p_agg}

    return results


def print_transfer_report(results: dict):
    """Print cross-model transfer report."""
    print(f"\n=== Cross-Model Transfer: {results['model_a']} vs {results['model_b']} ===\n")

    print(f"{'Task':<25} {'Spearman r':>12} {'p-value':>10}")
    print("-" * 49)
    for task, vals in results["per_task"].items():
        print(f"{task:<25} {vals['spearman_r']:>12.4f} {vals['spearman_p']:>10.4f}")

    agg = results["aggregated"]
    print("-" * 49)
    print(f"{'Aggregated':<25} {agg['spearman_r']:>12.4f} {agg['spearman_p']:>10.4f}")
