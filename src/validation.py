# MIT License - Copyright (c) 2026 Vikas Kumar
from __future__ import annotations
"""Phase 5: Full 2^6 factorial validation against Plackett-Burman estimates."""

import numpy as np
from scipy import stats
from src.components import COMPONENT_KEYS
from src.design import compute_main_effects


def compute_interaction_effects(
    design: np.ndarray,
    responses: np.ndarray,
) -> dict[tuple[int, ...], float]:
    """Compute all two-way interaction effects from a full factorial design.

    interaction_ij = mean(Y where X_i*X_j=+1) - mean(Y where X_i*X_j=-1)
    """
    n_factors = design.shape[1]
    interactions = {}

    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            interaction_col = design[:, i] * design[:, j]
            high = responses[interaction_col == 1].mean()
            low = responses[interaction_col == -1].mean()
            interactions[(i, j)] = high - low

    return interactions


def validate_pb_against_factorial(
    pb_effects: np.ndarray,
    factorial_design: np.ndarray,
    factorial_responses: np.ndarray,
) -> dict:
    """Compare PB-estimated main effects against full factorial ground truth.

    Returns:
        Dict with correlation metrics, per-component comparison, and interaction magnitudes.
    """
    # True main effects from full factorial
    true_effects = compute_main_effects(factorial_design, factorial_responses)

    # Correlation
    pearson_r, pearson_p = stats.pearsonr(pb_effects, true_effects)
    spearman_r, spearman_p = stats.spearmanr(pb_effects, true_effects)

    # Per-component error
    errors = pb_effects - true_effects
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())

    # Interaction effects (to check if they confound PB estimates)
    interactions = compute_interaction_effects(factorial_design, factorial_responses)
    interaction_magnitudes = np.array(list(interactions.values()))

    # Rank agreement
    pb_ranks = stats.rankdata(-np.abs(pb_effects))
    true_ranks = stats.rankdata(-np.abs(true_effects))
    rank_agreement = (pb_ranks == true_ranks).sum()

    return {
        "true_effects": true_effects,
        "pb_effects": pb_effects,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "mae": mae,
        "rmse": rmse,
        "rank_agreement": f"{rank_agreement}/{len(COMPONENT_KEYS)}",
        "interactions": interactions,
        "mean_interaction_magnitude": np.abs(interaction_magnitudes).mean(),
        "max_interaction_magnitude": np.abs(interaction_magnitudes).max(),
    }


def print_validation_report(results: dict):
    """Print validation comparison report."""
    print("\n=== PB vs Full Factorial Validation ===\n")

    print(f"Pearson r:  {results['pearson_r']:.4f} (p={results['pearson_p']:.4f})")
    print(f"Spearman r: {results['spearman_r']:.4f} (p={results['spearman_p']:.4f})")
    print(f"MAE:        {results['mae']:.4f}")
    print(f"RMSE:       {results['rmse']:.4f}")
    print(f"Rank match: {results['rank_agreement']}")

    print(f"\nInteraction effects:")
    print(f"  Mean |interaction|: {results['mean_interaction_magnitude']:.4f}")
    print(f"  Max  |interaction|: {results['max_interaction_magnitude']:.4f}")

    print(f"\n{'Component':<20} {'PB Effect':>10} {'True Effect':>12} {'Error':>8}")
    print("-" * 52)
    for i, key in enumerate(COMPONENT_KEYS):
        pb = results["pb_effects"][i]
        true = results["true_effects"][i]
        err = pb - true
        print(f"{key:<20} {pb:>+10.4f} {true:>+12.4f} {err:>+8.4f}")

    print(f"\nTop 5 interactions by magnitude:")
    sorted_ints = sorted(results["interactions"].items(), key=lambda x: abs(x[1]), reverse=True)
    for (i, j), val in sorted_ints[:5]:
        print(f"  {COMPONENT_KEYS[i]} x {COMPONENT_KEYS[j]}: {val:+.4f}")
