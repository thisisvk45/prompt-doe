from __future__ import annotations
"""Phase 4: Statistical analysis — bootstrap CIs, FDR correction, PB vs LOO comparison."""

import numpy as np
from scipy import stats
from src.components import COMPONENT_KEYS
from src.design import compute_main_effects, compute_loo_effects


def bootstrap_main_effects(
    design: np.ndarray,
    per_example_results: np.ndarray,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap BCa confidence intervals for main effects.

    Args:
        design: (n_runs, n_factors) design matrix.
        per_example_results: (n_runs, n_examples) binary correctness matrix.
        n_bootstrap: number of bootstrap resamples.
        confidence_level: confidence level for intervals.
        seed: random seed.

    Returns:
        effects: (n_factors,) point estimates of main effects
        ci_low: (n_factors,) lower CI bounds
        ci_high: (n_factors,) upper CI bounds
    """
    n_runs, n_examples = per_example_results.shape
    n_factors = design.shape[1]

    # Point estimates (mean accuracy per run)
    run_accuracies = per_example_results.mean(axis=1)
    effects = compute_main_effects(design, run_accuracies)

    ci_low = np.zeros(n_factors)
    ci_high = np.zeros(n_factors)

    rng = np.random.RandomState(seed)

    for j in range(n_factors):
        boot_effects = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            # Resample examples (columns)
            idx = rng.randint(0, n_examples, size=n_examples)
            boot_accs = per_example_results[:, idx].mean(axis=1)
            boot_effects[b] = compute_main_effects(design, boot_accs)[j]

        # BCa confidence interval
        result = stats.bootstrap(
            (boot_effects,),
            np.mean,
            confidence_level=confidence_level,
            n_resamples=n_bootstrap,
            random_state=seed,
            method="BCa",
        )
        ci_low[j] = result.confidence_interval.low
        ci_high[j] = result.confidence_interval.high

    return effects, ci_low, ci_high


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction.

    Returns:
        Array of adjusted p-values.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    adjusted = np.zeros(n)
    for i in range(n):
        adjusted[sorted_idx[i]] = sorted_p[i] * n / (i + 1)

    # Enforce monotonicity (from largest rank down)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    return adjusted


def compute_p_values_from_bootstrap(
    effects: np.ndarray,
    design: np.ndarray,
    per_example_results: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Compute p-values for main effects using permutation-style bootstrap.

    Tests H0: effect_j = 0 by computing the fraction of bootstrap samples
    where the effect crosses zero.
    """
    n_runs, n_examples = per_example_results.shape
    n_factors = design.shape[1]
    rng = np.random.RandomState(seed)
    p_values = np.zeros(n_factors)

    for j in range(n_factors):
        boot_effects = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.randint(0, n_examples, size=n_examples)
            boot_accs = per_example_results[:, idx].mean(axis=1)
            boot_effects[b] = compute_main_effects(design, boot_accs)[j]

        # Two-sided p-value: fraction of bootstrap effects on the other side of zero
        if effects[j] >= 0:
            p_values[j] = (boot_effects <= 0).mean() * 2
        else:
            p_values[j] = (boot_effects >= 0).mean() * 2
        p_values[j] = min(p_values[j], 1.0)

    return p_values


def compare_pb_vs_loo(
    pb_effects: np.ndarray,
    loo_effects: np.ndarray,
) -> dict:
    """Compare Plackett-Burman vs LOO attribution rankings.

    Returns dict with Spearman correlation, rank comparison, and disagreements.
    """
    pb_ranks = stats.rankdata(-np.abs(pb_effects))
    loo_ranks = stats.rankdata(-np.abs(loo_effects))

    spearman_r, spearman_p = stats.spearmanr(pb_ranks, loo_ranks)

    # Find disagreements (rank differs by more than 1)
    disagreements = []
    for i, key in enumerate(COMPONENT_KEYS):
        if abs(pb_ranks[i] - loo_ranks[i]) > 1:
            disagreements.append({
                "component": key,
                "pb_rank": int(pb_ranks[i]),
                "loo_rank": int(loo_ranks[i]),
                "pb_effect": pb_effects[i],
                "loo_effect": loo_effects[i],
            })

    return {
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pb_ranks": pb_ranks,
        "loo_ranks": loo_ranks,
        "disagreements": disagreements,
    }


def print_effects_table(
    effects: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    p_values: np.ndarray | None = None,
    label: str = "Main Effects",
):
    """Print a formatted table of effects with CIs."""
    print(f"\n{label}:")
    print(f"{'Component':<20} {'Effect':>8} {'95% CI':>20}", end="")
    if p_values is not None:
        print(f" {'p-adj':>8} {'Sig':>4}")
    else:
        print()
    print("-" * 65)

    for i, key in enumerate(COMPONENT_KEYS):
        ci_str = f"[{ci_low[i]:+.4f}, {ci_high[i]:+.4f}]"
        print(f"{key:<20} {effects[i]:>+8.4f} {ci_str:>20}", end="")
        if p_values is not None:
            sig = "***" if p_values[i] < 0.001 else "**" if p_values[i] < 0.01 else "*" if p_values[i] < 0.05 else ""
            print(f" {p_values[i]:>8.4f} {sig:>4}")
        else:
            print()
