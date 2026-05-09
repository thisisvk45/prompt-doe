"""Run cross-model replication: output-based entanglement + PB/LOO screening + analysis."""
from __future__ import annotations

import sys
import json
import csv
import numpy as np
from pathlib import Path

from src.components import COMPONENT_KEYS, load_components, load_task_config
from src.datasets import load_task_data, parse_model_answer, check_answer
from src.design import (
    generate_pb_design,
    generate_loo_design,
    design_to_configs,
    compute_main_effects,
    compute_loo_effects,
)
from src.prompts import assemble_prompt
from src.inference import run_inference
from src.independence import compute_output_entanglement_matrix
from src.analysis import (
    bootstrap_main_effects,
    compute_p_values_from_bootstrap,
    benjamini_hochberg,
    compare_pb_vs_loo,
)

TASKS = ["gsm8k", "bbh_date", "mmlu_pro"]


def evaluate_design(
    design: np.ndarray,
    task_name: str,
    model_name: str,
    examples: list[dict],
    task_config: dict,
    components: dict,
    max_workers: int = 5,
) -> np.ndarray:
    """Run all design configurations and return per-example correctness matrix."""
    configs = design_to_configs(design)
    n_runs = len(configs)
    n_examples = len(examples)
    results = np.zeros((n_runs, n_examples), dtype=int)

    for run_idx, flags in enumerate(configs):
        print(f"  Run {run_idx + 1}/{n_runs}: {flags}", flush=True)

        prompts = [
            assemble_prompt(flags, task_config, ex["question"], components)
            for ex in examples
        ]

        responses = run_inference(
            model_name, prompts, use_cache=True, verbose=True, max_workers=max_workers
        )

        for ex_idx, (resp, ex) in enumerate(zip(responses, examples)):
            predicted = parse_model_answer(resp, task_config["answer_type"])
            correct = check_answer(predicted, ex["answer"], task_config["answer_type"])
            results[run_idx, ex_idx] = int(correct)

        acc = results[run_idx].mean()
        print(f"    Accuracy: {acc:.3f}", flush=True)

    return results


def save_results(model_name: str, results_dir: Path, screening_data: dict, analysis_data: dict, entanglement_data: dict | None):
    """Save all results in the same format as gpt-4o-mini."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Entanglement matrix
    if entanglement_data:
        with open(results_dir / "entanglement_matrix.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + COMPONENT_KEYS)
            stat = entanglement_data["stat_matrix"]
            pval = entanglement_data["pval_matrix"]
            for i, key in enumerate(COMPONENT_KEYS):
                row = [key]
                for j in range(len(COMPONENT_KEYS)):
                    if i == j:
                        row.append("---")
                    else:
                        sig = "*" if pval[i][j] < 0.05 else ""
                        row.append(f"{stat[i][j]:.4f}{sig}")
                w.writerow(row)

    # PB and LOO run CSVs per task
    for task_name in TASKS:
        if task_name not in screening_data:
            continue
        r = screening_data[task_name]

        # PB runs
        with open(results_dir / f"pb_runs_{task_name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            header = ["run"] + COMPONENT_KEYS + ["accuracy"]
            w.writerow(header)
            configs = design_to_configs(np.array(r["pb_design"]))
            for idx, (flags, acc) in enumerate(zip(configs, r["pb_accuracies"])):
                row = [idx + 1] + ["+" if flags[k] else "-" for k in COMPONENT_KEYS] + [f"{acc:.3f}"]
                w.writerow(row)

        # LOO runs
        with open(results_dir / f"loo_runs_{task_name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            header = ["run", "removed_component", "accuracy"]
            w.writerow(header)
            w.writerow([1, "none (full)", f"{r['loo_accuracies'][0]:.3f}"])
            for idx, key in enumerate(COMPONENT_KEYS):
                w.writerow([idx + 2, key, f"{r['loo_accuracies'][idx + 1]:.3f}"])

    # Main effects comparison CSV
    with open(results_dir / "main_effects_pb_vs_loo.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "component", "pb_effect", "pb_ci_low", "pb_ci_high", "pb_padj",
                     "loo_effect", "spearman_r", "spearman_p"])
        for task_name in TASKS:
            if task_name not in analysis_data:
                continue
            a = analysis_data[task_name]
            for i, key in enumerate(COMPONENT_KEYS):
                w.writerow([
                    task_name, key,
                    f"{a['pb_effects'][i]:.4f}",
                    f"{a['pb_ci_low'][i]:.4f}",
                    f"{a['pb_ci_high'][i]:.4f}",
                    f"{a['pb_pvalues_adjusted'][i]:.4f}",
                    f"{a['loo_effects'][i]:.4f}",
                    f"{a['comparison_spearman_r']:.4f}",
                    f"{a['comparison_spearman_p']:.4f}",
                ])

    # Summary JSON
    summary = {
        "model": model_name,
        "tasks": TASKS,
        "entanglement": entanglement_data,
        "screening": screening_data,
        "analysis": analysis_data,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_cross_model.py <model_name>")
        print("  model_name: key from config/models.yaml (e.g. claude_haiku, deepseek_v4_pro, gemma_4_31b)")
        sys.exit(1)

    model_name = sys.argv[1]
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # Map model_name to results directory name
    from src.components import load_model_config
    model_config = load_model_config(model_name)
    dir_name = model_config["model_id"].replace("/", "-")
    results_dir = Path(__file__).parent.parent / "results" / dir_name

    print(f"{'='*60}")
    print(f"Cross-Model Replication: {model_config['name']} ({model_name})")
    print(f"Results directory: {results_dir}")
    print(f"Max workers: {max_workers}")
    print(f"{'='*60}", flush=True)

    components = load_components()
    pb_design = generate_pb_design(6)
    loo_design = generate_loo_design(6)

    # ===== Phase 2: Output-based Entanglement =====
    print(f"\n{'='*60}")
    print("Phase 2: Output-based Entanglement (HSIC)")
    print(f"{'='*60}", flush=True)

    try:
        stat_matrix, pval_matrix = compute_output_entanglement_matrix(
            llm_model_name=model_name,
            encoder_name="all-mpnet-base-v2",
            n_examples=100,
            verbose=True,
        )

        # BH correction across 15 off-diagonal pairs
        n = len(COMPONENT_KEYS)
        import itertools
        pair_pvals = []
        pair_indices = []
        for i, j in itertools.combinations(range(n), 2):
            pair_pvals.append(pval_matrix[i, j])
            pair_indices.append((i, j))
        pair_pvals = np.array(pair_pvals)
        adj_pvals = benjamini_hochberg(pair_pvals)

        # Update pval_matrix with adjusted values
        pval_adj_matrix = np.ones((n, n))
        for idx, (i, j) in enumerate(pair_indices):
            pval_adj_matrix[i, j] = adj_pvals[idx]
            pval_adj_matrix[j, i] = adj_pvals[idx]

        n_sig = (adj_pvals < 0.05).sum()
        print(f"\nEntanglement: {n_sig}/15 pairs significant (BH-adjusted p < 0.05)", flush=True)

        entanglement_data = {
            "stat_matrix": stat_matrix.tolist(),
            "pval_matrix": pval_matrix.tolist(),
            "pval_adj_matrix": pval_adj_matrix.tolist(),
            "n_significant": int(n_sig),
        }
    except Exception as e:
        print(f"\nEntanglement FAILED: {e}", flush=True)
        entanglement_data = None

    # ===== Phase 3: PB + LOO Screening =====
    screening_data = {}
    for task_name in TASKS:
        print(f"\n{'='*60}")
        print(f"Phase 3: {task_name}")
        print(f"{'='*60}", flush=True)

        task_config = load_task_config(task_name)
        examples = load_task_data(task_name)
        print(f"Loaded {len(examples)} examples", flush=True)

        print(f"\nRunning Plackett-Burman ({pb_design.shape[0]} runs)...", flush=True)
        pb_results = evaluate_design(
            pb_design, task_name, model_name, examples, task_config, components, max_workers
        )

        print(f"\nRunning LOO ({loo_design.shape[0]} runs)...", flush=True)
        loo_results = evaluate_design(
            loo_design, task_name, model_name, examples, task_config, components, max_workers
        )

        screening_data[task_name] = {
            "pb_design": pb_design.tolist(),
            "loo_design": loo_design.tolist(),
            "pb_results": pb_results.tolist(),
            "loo_results": loo_results.tolist(),
            "pb_accuracies": pb_results.mean(axis=1).tolist(),
            "loo_accuracies": loo_results.mean(axis=1).tolist(),
        }

    # ===== Phase 4: Analysis =====
    analysis_data = {}
    for task_name in TASKS:
        print(f"\n{'='*60}")
        print(f"Phase 4 Analysis: {task_name}")
        print(f"{'='*60}", flush=True)

        r = screening_data[task_name]
        pb_matrix = np.array(r["pb_results"])
        loo_matrix = np.array(r["loo_results"])

        # PB main effects with bootstrap CIs
        pb_effects, pb_ci_low, pb_ci_high = bootstrap_main_effects(
            pb_design, pb_matrix, n_bootstrap=2000
        )
        pb_pvals = compute_p_values_from_bootstrap(pb_effects, pb_design, pb_matrix)
        pb_pvals_adj = benjamini_hochberg(pb_pvals)

        # LOO effects
        loo_accs = loo_matrix.mean(axis=1)
        loo_effects = compute_loo_effects(loo_accs)

        # Comparison
        comparison = compare_pb_vs_loo(pb_effects, loo_effects)

        print(f"\nPB Effects:")
        for i, key in enumerate(COMPONENT_KEYS):
            sig = "*" if pb_pvals_adj[i] < 0.05 else ""
            print(f"  {key:<20} {pb_effects[i]:+.4f} [{pb_ci_low[i]:+.4f}, {pb_ci_high[i]:+.4f}] p={pb_pvals_adj[i]:.4f}{sig}")
        print(f"\nLOO Effects:")
        print(f"  Full prompt accuracy: {loo_accs[0]:.3f}")
        for i, key in enumerate(COMPONENT_KEYS):
            print(f"  Without {key}: {loo_accs[i+1]:.3f} (effect: {loo_effects[i]:+.4f})")
        print(f"\nPB vs LOO Spearman r: {comparison['spearman_r']:.4f} (p={comparison['spearman_p']:.4f})")

        analysis_data[task_name] = {
            "pb_effects": pb_effects.tolist(),
            "pb_ci_low": pb_ci_low.tolist(),
            "pb_ci_high": pb_ci_high.tolist(),
            "pb_pvalues_adjusted": pb_pvals_adj.tolist(),
            "loo_effects": loo_effects.tolist(),
            "comparison_spearman_r": comparison["spearman_r"],
            "comparison_spearman_p": comparison["spearman_p"],
            "disagreements": comparison["disagreements"],
        }

    # ===== Save Everything =====
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}", flush=True)

    save_results(model_name, results_dir, screening_data, analysis_data, entanglement_data)
    print(f"All results saved to {results_dir}", flush=True)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if entanglement_data:
        print(f"Entanglement: {entanglement_data['n_significant']}/15 significant pairs")
    for task_name in TASKS:
        a = analysis_data[task_name]
        s = screening_data[task_name]
        print(f"\n{task_name}:")
        print(f"  PB accuracies: {[f'{x:.3f}' for x in s['pb_accuracies']]}")
        print(f"  LOO accuracies: {[f'{x:.3f}' for x in s['loo_accuracies']]}")
        print(f"  PB vs LOO Spearman r={a['comparison_spearman_r']:.4f}")
        print(f"  Top PB effects:")
        effects = np.array(a['pb_effects'])
        ranked = np.argsort(-np.abs(effects))
        for idx in ranked[:3]:
            print(f"    {COMPONENT_KEYS[idx]}: {effects[idx]:+.4f}")


if __name__ == "__main__":
    main()
