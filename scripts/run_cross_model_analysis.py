"""Step 4: Cross-model analysis — compare results across all models."""
from __future__ import annotations

import json
import csv
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

from src.components import COMPONENT_KEYS

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Model configs: (model_key, results_dir_name, display_name)
MODELS = [
    ("gpt4o_mini", "gpt-4o-mini", "GPT-4o-mini"),
    ("gpt4o", "gpt-4o", "GPT-4o"),
    ("claude_haiku", "anthropic-claude-haiku-4-5", "Claude Haiku 4.5"),
    ("deepseek_v4_pro", "deepseek-deepseek-v4-pro", "DeepSeek V4 Pro"),
    ("gemma_4_31b", "google-gemma-4-31b-it_free", "Gemma 4 31B"),
]

TASKS = ["gsm8k", "bbh_date", "mmlu_pro"]


def load_model_results(dir_name: str) -> dict | None:
    """Load summary.json for a model."""
    path = RESULTS_DIR / dir_name / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    # Load all available model results
    available_models = []
    model_data = {}
    for key, dir_name, display_name in MODELS:
        data = load_model_results(dir_name)
        if data is not None:
            available_models.append((key, dir_name, display_name))
            model_data[key] = data
            print(f"Loaded: {display_name} ({dir_name})")
        else:
            print(f"Skipped: {display_name} (no results found)")

    if len(available_models) < 2:
        print("Need at least 2 models for cross-model analysis.")
        return

    n_models = len(available_models)
    print(f"\n{n_models} models available for comparison.\n")

    lines = []  # For markdown report

    # ===== 1. Entanglement rate per model =====
    lines.append("# Cross-Model Analysis Report\n")
    lines.append("## 1. Entanglement Comparison\n")

    entanglement_sets = {}
    for key, _, display_name in available_models:
        data = model_data[key]
        ent = data.get("entanglement")
        if ent:
            n_sig = ent["n_significant"]
            lines.append(f"- **{display_name}**: {n_sig}/15 significant pairs")

            # Find which pairs are significant
            pval_adj = np.array(ent.get("pval_adj_matrix", ent["pval_matrix"]))
            sig_pairs = set()
            for i, j in combinations(range(6), 2):
                if pval_adj[i, j] < 0.05:
                    sig_pairs.add((COMPONENT_KEYS[i], COMPONENT_KEYS[j]))
            entanglement_sets[key] = sig_pairs
        else:
            lines.append(f"- **{display_name}**: entanglement not available")

    # Jaccard similarity of entanglement sets
    if len(entanglement_sets) >= 2:
        lines.append("\n### Pairwise Jaccard similarity of entangled pairs:\n")
        for (k1, _, n1), (k2, _, n2) in combinations(available_models, 2):
            if k1 in entanglement_sets and k2 in entanglement_sets:
                s1, s2 = entanglement_sets[k1], entanglement_sets[k2]
                if len(s1 | s2) > 0:
                    jaccard = len(s1 & s2) / len(s1 | s2)
                    lines.append(f"- {n1} vs {n2}: Jaccard = {jaccard:.3f}")

    # ===== 2. PB vs LOO Spearman per task per model =====
    lines.append("\n## 2. PB vs LOO Spearman Correlation\n")
    lines.append("| Task | " + " | ".join(n for _, _, n in available_models) + " |")
    lines.append("|------|" + "|".join("------" for _ in available_models) + "|")

    for task in TASKS:
        row = [task]
        for key, _, _ in available_models:
            a = model_data[key].get("analysis", {}).get(task)
            if a:
                r = a["comparison_spearman_r"]
                p = a["comparison_spearman_p"]
                row.append(f"{r:.3f} (p={p:.3f})")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    # ===== 3. Per-component effect direction agreement =====
    lines.append("\n## 3. Effect Direction Agreement\n")
    lines.append("For each component × task, check if PB effect sign agrees across models.\n")

    agreement_count = 0
    total_checks = 0
    effect_table = {}  # (task, component) -> list of (model, effect)

    for task in TASKS:
        for i, comp in enumerate(COMPONENT_KEYS):
            effects = []
            for key, _, display_name in available_models:
                a = model_data[key].get("analysis", {}).get(task)
                if a:
                    effects.append((display_name, a["pb_effects"][i]))
            effect_table[(task, comp)] = effects

            if len(effects) >= 2:
                signs = [1 if e > 0 else -1 for _, e in effects]
                if all(s == signs[0] for s in signs):
                    agreement_count += 1
                total_checks += 1

    if total_checks > 0:
        rate = agreement_count / total_checks
        lines.append(f"**Agreement rate**: {agreement_count}/{total_checks} = {rate:.1%}\n")

    # Detailed table
    lines.append("| Task | Component | " + " | ".join(n for _, _, n in available_models) + " | Agree? |")
    lines.append("|------|-----------|" + "|".join("------" for _ in available_models) + "|--------|")

    for task in TASKS:
        for comp in COMPONENT_KEYS:
            effects = effect_table.get((task, comp), [])
            vals = []
            for _, _, display_name in available_models:
                found = [e for n, e in effects if n == display_name]
                if found:
                    vals.append(f"{found[0]:+.4f}")
                else:
                    vals.append("N/A")
            signs = [1 if e > 0 else -1 for _, e in effects]
            agree = "Yes" if len(signs) >= 2 and all(s == signs[0] for s in signs) else "No"
            lines.append(f"| {task} | {comp} | " + " | ".join(vals) + f" | {agree} |")

    # ===== 4. Specific component questions =====
    lines.append("\n## 4. Key Component Questions\n")

    for comp_q, comp_key, direction, verb in [
        ("Does persona hurt on all models?", "persona", -1, "hurts"),
        ("Does output_format help on all models?", "output_format", 1, "helps"),
        ("Does few_shot help on all models?", "few_shot", 1, "helps"),
    ]:
        lines.append(f"### {comp_q}\n")
        idx = COMPONENT_KEYS.index(comp_key)
        for task in TASKS:
            for key, _, display_name in available_models:
                a = model_data[key].get("analysis", {}).get(task)
                if a:
                    eff = a["pb_effects"][idx]
                    actual = "helps" if eff > 0 else "hurts" if eff < 0 else "neutral"
                    match = "Yes" if (direction > 0 and eff > 0) or (direction < 0 and eff < 0) else "No"
                    lines.append(f"- {display_name}/{task}: {eff:+.4f} ({actual}) [{match}]")
        lines.append("")

    # ===== 5. Cross-model Spearman on PB effect rankings =====
    lines.append("\n## 5. Cross-Model Effect Ranking Correlation\n")
    lines.append("Pairwise Spearman correlation of 6 PB effects between model pairs, averaged across tasks.\n")

    for (k1, _, n1), (k2, _, n2) in combinations(available_models, 2):
        task_corrs = []
        for task in TASKS:
            a1 = model_data[k1].get("analysis", {}).get(task)
            a2 = model_data[k2].get("analysis", {}).get(task)
            if a1 and a2:
                e1 = np.array(a1["pb_effects"])
                e2 = np.array(a2["pb_effects"])
                r, p = stats.spearmanr(e1, e2)
                task_corrs.append(r)
        if task_corrs:
            mean_r = np.mean(task_corrs)
            lines.append(f"- {n1} vs {n2}: mean Spearman r = {mean_r:.3f} (per-task: {', '.join(f'{r:.3f}' for r in task_corrs)})")

    # ===== 6. Outlier config check across models =====
    lines.append("\n## 6. Outlier Configuration (-,+,-,+,-,+) Check\n")
    lines.append("PB Run 7 accuracy vs mean of other PB runs.\n")

    # The outlier config is run 7 in PB design (index 6, 0-indexed)
    # Actually need to find which PB run matches the outlier flags
    outlier_flags = {"system_role": False, "persona": True, "few_shot": False,
                     "cot_trigger": True, "output_format": False, "constraints": True}

    for key, _, display_name in available_models:
        lines.append(f"\n### {display_name}\n")
        for task in TASKS:
            s = model_data[key].get("screening", {}).get(task)
            if not s:
                lines.append(f"- {task}: N/A")
                continue

            from src.design import design_to_configs
            pb_design = np.array(s["pb_design"])
            configs = design_to_configs(pb_design)
            pb_accs = s["pb_accuracies"]

            # Find the outlier run
            outlier_idx = None
            for idx, flags in enumerate(configs):
                if flags == outlier_flags:
                    outlier_idx = idx
                    break

            if outlier_idx is not None:
                outlier_acc = pb_accs[outlier_idx]
                other_accs = [a for i, a in enumerate(pb_accs) if i != outlier_idx]
                mean_other = np.mean(other_accs)
                diff = outlier_acc - mean_other
                lines.append(f"- {task}: outlier={outlier_acc:.3f}, mean_other={mean_other:.3f}, diff={diff:+.3f}")
            else:
                lines.append(f"- {task}: outlier config not found in PB design")

    # ===== Save outputs =====

    # CSV: wide table
    with open(RESULTS_DIR / "cross_model_analysis.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["task", "component"]
        for _, _, display_name in available_models:
            header.extend([f"{display_name}_pb_effect", f"{display_name}_loo_effect"])
        w.writerow(header)

        for task in TASKS:
            for i, comp in enumerate(COMPONENT_KEYS):
                row = [task, comp]
                for key, _, _ in available_models:
                    a = model_data[key].get("analysis", {}).get(task)
                    if a:
                        row.append(f"{a['pb_effects'][i]:.4f}")
                        row.append(f"{a['loo_effects'][i]:.4f}")
                    else:
                        row.extend(["N/A", "N/A"])
                w.writerow(row)

    # Markdown report
    report = "\n".join(lines)
    with open(RESULTS_DIR / "cross_model_summary.md", "w") as f:
        f.write(report)

    print(report)
    print(f"\nSaved: {RESULTS_DIR / 'cross_model_analysis.csv'}")
    print(f"Saved: {RESULTS_DIR / 'cross_model_summary.md'}")


if __name__ == "__main__":
    main()
