from __future__ import annotations
"""Plackett-Burman and LOO experimental design matrices."""

import numpy as np
import pyDOE2
from src.components import COMPONENT_KEYS


def generate_pb_design(n_factors: int = 6) -> np.ndarray:
    """Generate a Plackett-Burman design matrix.

    Returns:
        Array of shape (12, n_factors) with values -1 and +1.
        Each row is an experimental run, each column is a factor.
    """
    design = pyDOE2.pbdesign(n_factors)
    return design.astype(int)


def generate_loo_design(n_factors: int = 6) -> np.ndarray:
    """Generate leave-one-out ablation design matrix.

    Row 0: all factors present (+1).
    Rows 1-6: each has one factor removed (-1), rest present (+1).

    Returns:
        Array of shape (7, n_factors).
    """
    design = np.ones((n_factors + 1, n_factors), dtype=int)
    for i in range(n_factors):
        design[i + 1, i] = -1
    return design


def generate_full_factorial(n_factors: int = 6) -> np.ndarray:
    """Generate full 2^n factorial design matrix.

    Returns:
        Array of shape (2^n, n_factors) with values -1 and +1.
    """
    n_runs = 2 ** n_factors
    design = np.zeros((n_runs, n_factors), dtype=int)
    for i in range(n_runs):
        for j in range(n_factors):
            design[i, j] = 1 if (i >> j) & 1 else -1
    return design


def design_to_configs(design: np.ndarray) -> list[dict[str, bool]]:
    """Convert design matrix to list of component flag dicts."""
    configs = []
    for row in design:
        flags = {}
        for key, val in zip(COMPONENT_KEYS, row):
            flags[key] = val > 0
        configs.append(flags)
    return configs


def compute_main_effects(design: np.ndarray, responses: np.ndarray) -> np.ndarray:
    """Compute main effects from a design matrix and response vector.

    For Plackett-Burman or full factorial designs:
    effect_j = mean(Y where X_j=+1) - mean(Y where X_j=-1)

    Args:
        design: (n_runs, n_factors) array of -1/+1.
        responses: (n_runs,) array of response values (e.g., accuracy).

    Returns:
        (n_factors,) array of main effects.
    """
    n_factors = design.shape[1]
    effects = np.zeros(n_factors)
    for j in range(n_factors):
        high = responses[design[:, j] == 1].mean()
        low = responses[design[:, j] == -1].mean()
        effects[j] = high - low
    return effects


def compute_loo_effects(responses: np.ndarray) -> np.ndarray:
    """Compute LOO ablation effects.

    effect_j = accuracy(full) - accuracy(without component j)

    Args:
        responses: (7,) array where index 0 is full prompt, index j+1 is without component j.

    Returns:
        (6,) array of LOO effects.
    """
    full = responses[0]
    effects = np.array([full - responses[j + 1] for j in range(len(responses) - 1)])
    return effects


def print_design_summary(design: np.ndarray, label: str = "Design"):
    """Print a readable summary of a design matrix."""
    print(f"\n{label}: {design.shape[0]} runs x {design.shape[1]} factors")
    print(f"{'Run':<5}", end="")
    for key in COMPONENT_KEYS:
        print(f"{key[:12]:<14}", end="")
    print()
    print("-" * (5 + 14 * len(COMPONENT_KEYS)))
    for i, row in enumerate(design):
        print(f"{i + 1:<5}", end="")
        for val in row:
            symbol = "+" if val > 0 else "-"
            print(f"{symbol:<14}", end="")
        print()
