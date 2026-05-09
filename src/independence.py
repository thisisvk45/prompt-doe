# MIT License - Copyright (c) 2026 Vikas Kumar
"""Phase 2: Semantic independence / entanglement test between prompt components."""

import itertools
import numpy as np
from sentence_transformers import SentenceTransformer
from hyppo.independence import Hsic

from src.components import COMPONENT_KEYS, load_components


def build_context_variants(
    target_key: str,
    condition_key: str,
    components: dict,
    n_variants: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Build embedding pairs for testing independence of target_key given condition_key.

    For each random subset of the remaining components (excluding target and condition),
    embed the target component:
      - with condition present
      - with condition absent

    Returns:
        embeddings_with: (n_variants, embed_dim) - target embedded when condition is present
        embeddings_without: (n_variants, embed_dim) - target embedded when condition is absent
    """
    other_keys = [k for k in COMPONENT_KEYS if k not in (target_key, condition_key)]

    rng = np.random.RandomState(42)
    contexts_with = []
    contexts_without = []

    for _ in range(n_variants):
        # Random subset of other components
        mask = rng.randint(0, 2, size=len(other_keys))
        other_parts = []
        for k, m in zip(other_keys, mask):
            comp = components[k]
            if isinstance(comp.present, bool):
                continue
            variant = comp.present if m else comp.absent
            other_parts.append(variant)

        target_comp = components[target_key]
        target_text = target_comp.present if not isinstance(target_comp.present, bool) else "Include examples."
        cond_comp = components[condition_key]

        # Context with condition present
        cond_present = cond_comp.present if not isinstance(cond_comp.present, bool) else "Include examples."
        ctx_with = "\n\n".join(other_parts + [cond_present, target_text])
        contexts_with.append(ctx_with)

        # Context with condition absent
        cond_absent = cond_comp.absent if not isinstance(cond_comp.absent, bool) else "No examples."
        ctx_without = "\n\n".join(other_parts + [cond_absent, target_text])
        contexts_without.append(ctx_without)

    return contexts_with, contexts_without


def compute_entanglement_matrix(
    model_name: str = "all-mpnet-base-v2",
    n_variants: int = 50,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 6x6 entanglement matrix using HSIC.

    Returns:
        stat_matrix: (6, 6) HSIC test statistics
        pval_matrix: (6, 6) p-values
    """
    components = load_components()
    model = SentenceTransformer(model_name)
    n = len(COMPONENT_KEYS)
    stat_matrix = np.zeros((n, n))
    pval_matrix = np.ones((n, n))

    for i, j in itertools.combinations(range(n), 2):
        target_key = COMPONENT_KEYS[i]
        condition_key = COMPONENT_KEYS[j]

        if verbose:
            print(f"  Testing {target_key} vs {condition_key}...")

        ctx_with, ctx_without = build_context_variants(
            target_key, condition_key, components, n_variants
        )

        # Embed
        emb_with = model.encode(ctx_with)
        emb_without = model.encode(ctx_without)

        # Compute embedding shift: difference vectors
        shifts = emb_with - emb_without  # (n_variants, embed_dim)

        # Test if the shift is significantly non-zero using HSIC
        # We test independence between the shift and a label vector
        labels = np.concatenate([np.ones(n_variants), np.zeros(n_variants)]).reshape(-1, 1)
        combined = np.vstack([emb_with, emb_without])

        hsic = Hsic()
        stat, pval = hsic.test(combined, labels, reps=1000)

        stat_matrix[i, j] = stat
        stat_matrix[j, i] = stat
        pval_matrix[i, j] = pval
        pval_matrix[j, i] = pval

        if verbose:
            print(f"    HSIC stat={stat:.4f}, p={pval:.4f}")

    return stat_matrix, pval_matrix


def compute_output_entanglement_matrix(
    llm_model_name: str,
    encoder_name: str = "all-mpnet-base-v2",
    n_examples: int = 100,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute output-based 6x6 entanglement matrix using HSIC.

    For each component pair (i, j), generate model outputs with component j
    present (n_examples/2) and absent (n_examples/2), embed the outputs, and
    test whether the output embeddings differ significantly.

    Args:
        llm_model_name: key from models.yaml for the LLM to test.
        encoder_name: sentence-transformers model for embedding outputs.
        n_examples: total examples per pair (split 50/50 present/absent).
        verbose: print progress.

    Returns:
        stat_matrix: (6, 6) HSIC test statistics
        pval_matrix: (6, 6) p-values
    """
    from src.prompts import assemble_prompt
    from src.datasets import load_task_data
    from src.inference import run_inference

    components = load_components()
    encoder = SentenceTransformer(encoder_name)
    n = len(COMPONENT_KEYS)
    stat_matrix = np.zeros((n, n))
    pval_matrix = np.ones((n, n))

    # Use GSM8K examples as the test bed for entanglement
    task_examples = load_task_data("gsm8k")
    n_per_condition = n_examples // 2  # 50 present, 50 absent

    # Use different example subsets for each condition to avoid pairing bias
    examples_present = task_examples[:n_per_condition]
    examples_absent = task_examples[n_per_condition:n_per_condition * 2]

    from src.components import load_task_config
    task_config = load_task_config("gsm8k")

    for i, j in itertools.combinations(range(n), 2):
        target_key = COMPONENT_KEYS[i]
        condition_key = COMPONENT_KEYS[j]

        if verbose:
            print(f"  Testing {target_key} vs {condition_key}...", flush=True)

        # Build prompts with condition_key PRESENT
        flags_present = {k: True for k in COMPONENT_KEYS}
        prompts_present = [
            assemble_prompt(flags_present, task_config, ex["question"], components)
            for ex in examples_present
        ]

        # Build prompts with condition_key ABSENT
        flags_absent = {k: True for k in COMPONENT_KEYS}
        flags_absent[condition_key] = False
        prompts_absent = [
            assemble_prompt(flags_absent, task_config, ex["question"], components)
            for ex in examples_absent
        ]

        # Get model outputs
        outputs_present = run_inference(
            llm_model_name, prompts_present, use_cache=True, verbose=False, max_workers=5
        )
        outputs_absent = run_inference(
            llm_model_name, prompts_absent, use_cache=True, verbose=False, max_workers=5
        )

        # Embed outputs
        emb_present = encoder.encode(outputs_present)
        emb_absent = encoder.encode(outputs_absent)

        # HSIC test
        labels = np.concatenate([
            np.ones(n_per_condition), np.zeros(n_per_condition)
        ]).reshape(-1, 1)
        combined = np.vstack([emb_present, emb_absent])

        hsic = Hsic()
        stat, pval = hsic.test(combined, labels, reps=1000)

        stat_matrix[i, j] = stat
        stat_matrix[j, i] = stat
        pval_matrix[i, j] = pval
        pval_matrix[j, i] = pval

        if verbose:
            print(f"    HSIC stat={stat:.4f}, p={pval:.4f}", flush=True)

    return stat_matrix, pval_matrix


def print_entanglement_matrix(stat_matrix: np.ndarray, pval_matrix: np.ndarray):
    """Pretty-print the entanglement matrix."""
    n = len(COMPONENT_KEYS)
    short_names = [k[:10] for k in COMPONENT_KEYS]

    print("\nEntanglement Matrix (HSIC statistic / p-value):")
    print(f"{'':>12}", end="")
    for name in short_names:
        print(f"{name:>12}", end="")
    print()

    for i in range(n):
        print(f"{short_names[i]:>12}", end="")
        for j in range(n):
            if i == j:
                print(f"{'---':>12}", end="")
            else:
                sig = "*" if pval_matrix[i, j] < 0.05 else " "
                print(f"{stat_matrix[i, j]:>8.3f}{sig:>1}   ", end="")
        print()
