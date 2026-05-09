# Architecture

This document explains how the codebase is organized, how data flows through the pipeline, and how to extend it.

---

## Data Flow

```
config/                          src/                              scripts/
┌──────────────┐    loads     ┌──────────────┐     called by    ┌───────────────────┐
│ models.yaml  │◄────────────│ components.py │◄────────────────│ run_entanglement   │
│ tasks.yaml   │◄────────────│ datasets.py   │◄────────────────│ run_pb_screen      │
│ components.yaml│◄──────────│ prompts.py    │◄────────────────│ run_cross_model    │
└──────────────┘             └──────┬───────┘                  └────────┬──────────┘
                                    │                                   │
                                    ▼                                   │
                             ┌──────────────┐                           │
                             │ inference.py  │◄──────────────────────────┘
                             │ (API calls +  │
                             │  disk cache)  │
                             └──────┬───────┘
                                    │ responses
                                    ▼
                             ┌──────────────┐     ┌──────────────┐
                             │independence.py│     │  design.py   │
                             │ (HSIC test)  │     │ (PB/LOO/FF   │
                             └──────┬───────┘     │  matrices)   │
                                    │             └──────┬───────┘
                                    │                    │
                                    ▼                    ▼
                             ┌──────────────────────────────┐
                             │         analysis.py          │
                             │  (bootstrap, BH, comparison) │
                             └──────────────┬───────────────┘
                                            │
                                            ▼
                                     results/<model>/
                                     ├── summary.json
                                     ├── entanglement_matrix.csv
                                     ├── pb_runs_*.csv
                                     ├── loo_runs_*.csv
                                     └── main_effects_pb_vs_loo.csv
```

---

## Module Reference

### `src/components.py` — Configuration Loading

The central config module. Everything starts here.

- `COMPONENT_KEYS` — Canonical ordering of the 6 prompt components. This list defines column order in design matrices, index positions in effect arrays, and row order in output CSVs. If you add a 7th component, add it here first.
- `load_components()` — Reads `config/components.yaml`, returns `dict[str, ComponentVariant]`.
- `load_task_config(task_name)` — Reads `config/tasks.yaml`, returns the config dict for one task.
- `load_model_config(model_name)` — Reads `config/models.yaml`, returns the config dict for one model.

### `src/prompts.py` — Prompt Assembly

Builds a complete prompt string from a set of component flags.

- `assemble_prompt(flags, task_config, question, components)` — The core function. Takes a dict like `{"system_role": True, "persona": False, ...}`, a task config, and a question string. Returns the assembled prompt. Components are concatenated with `\n\n` separators. Few-shot examples come from `task_config`, not from `components.yaml`.
- `flags_from_vector(vector)` — Converts a design matrix row (`[1, -1, 1, -1, 1, -1]`) to a flags dict.

### `src/datasets.py` — Task Data

Loads benchmark datasets from HuggingFace and handles answer parsing.

- `load_task_data(task_name)` — Loads `n_samples` examples using the seed from `tasks.yaml`. Returns `list[dict]` with keys `question`, `answer`, `raw`.
- `parse_model_answer(response, answer_type)` — Extracts the model's answer from free-text response. Looks for `ANSWER: X` pattern first, falls back to last-line extraction.
- `check_answer(predicted, gold, answer_type)` — Compares parsed answer to gold. Numeric comparison for GSM8K, string comparison for multiple choice.

### `src/inference.py` — LLM API Layer

Handles all LLM calls with caching, retries, and concurrency.

- **Disk cache**: Every `(model_id, prompt)` pair is hashed with SHA-256. Responses are stored in `results/cache/<hash>.json`. Cache is checked before every API call.
- **Providers**: Three supported — `openai` (direct OpenAI API), `openrouter` (OpenAI-compatible API via OpenRouter), `ollama` (local models).
- `run_inference(model_name, prompts, ...)` — Batch inference with `ThreadPoolExecutor`. Default 10 workers. Order is preserved.
- `run_single_inference(model_name, prompt, ...)` — Single prompt, with cache check and retry logic (5 retries, exponential backoff).

### `src/design.py` — Experimental Design Matrices

Generates the design matrices used throughout.

- `generate_pb_design(n_factors=6)` — Returns an 8x6 Plackett-Burman design matrix (values: -1/+1). Uses `pyDOE2.pbdesign()`.
- `generate_loo_design(n_factors=6)` — Returns a 7x6 LOO matrix. Row 0 = all +1 (full prompt). Rows 1-6: one factor set to -1.
- `generate_full_factorial(n_factors=6)` — Returns a 64x6 matrix with all 2^6 combinations.
- `compute_main_effects(design, responses)` — Calculates `mean(Y | X_j=+1) - mean(Y | X_j=-1)` for each factor.
- `compute_loo_effects(responses)` — Calculates `accuracy(full) - accuracy(without_j)` for each factor.
- `design_to_configs(design)` — Converts a design matrix to a list of flag dicts that `assemble_prompt` accepts.

### `src/independence.py` — HSIC Entanglement Testing

Two entanglement tests — one on prompt embeddings (fast, no API calls), one on model outputs (slower, requires API calls).

- `compute_entanglement_matrix()` — **Prompt-level.** For each component pair, builds context strings with one component toggled, embeds them with `sentence-transformers`, and runs HSIC. No LLM calls needed.
- `compute_output_entanglement_matrix(llm_model_name)` — **Output-level.** For each component pair, generates actual LLM outputs with one component toggled, embeds the *outputs*, and runs HSIC. This is the version used in the paper — it captures model-specific interactions that prompt-level embeddings miss.

Both return `(stat_matrix, pval_matrix)` — symmetric 6x6 arrays.

### `src/analysis.py` — Statistical Analysis

- `bootstrap_main_effects(design, per_example_results, ...)` — BCa bootstrap confidence intervals for main effects. Resamples at the example level (columns), not the run level.
- `compute_p_values_from_bootstrap(...)` — Two-sided p-values from bootstrap distribution.
- `benjamini_hochberg(p_values)` — BH FDR correction. Returns adjusted p-values.
- `compare_pb_vs_loo(pb_effects, loo_effects)` — Spearman correlation of rank orderings. Reports disagreements where rank differs by >1.

### `src/validation.py` — Full Factorial Validation

- `validate_pb_against_factorial(pb_effects, factorial_design, factorial_responses)` — Compares PB estimates to full factorial ground truth. Returns Pearson/Spearman correlations, MAE, RMSE, rank agreement, and all two-way interaction effects.
- `compute_interaction_effects(design, responses)` — Estimates all 15 two-way interactions from a full factorial design.

### `src/transfer.py` — Cross-Model Transfer

- `compute_cross_model_transfer(effects_a, effects_b)` — Per-task and aggregated Spearman correlation of PB effect vectors between two models.

---

## Key Design Decisions

### Why Plackett-Burman, not full factorial?

Full factorial for 6 binary factors = 64 runs. At 200 examples per run and 3 tasks, that's 38,400 API calls per model. PB uses 8 runs = 4,800 calls. The tradeoff: PB can't estimate interaction effects (they're aliased with main effects). But our full factorial validation on GPT-4o-mini/GSM8K shows PB main effect estimates correlate r = 0.99 with ground truth, and interaction magnitudes are small (~0.01) relative to main effects (~0.05).

### Why output-based entanglement, not prompt-based?

Prompt-level HSIC (embedding the prompt strings) detects syntactic interactions — whether one component's text changes meaning when another is added. But we care about *output-level* interactions — whether the model's response distribution changes. A model might ignore persona text entirely (no output effect) even though persona changes the prompt embedding. Output-based HSIC captures the model's actual sensitivity.

### Why disk caching?

The full protocol for one model costs ~10,500 API calls. Re-running to debug analysis code shouldn't cost another $50. Every `(model_id, prompt)` pair is cached to `results/cache/`. The cache is excluded from git via `.gitignore` since it's large and not needed for reproducibility — the aggregated CSVs contain all the information needed.

### Why HSIC specifically?

HSIC (Hilbert-Schmidt Independence Criterion) is a kernel-based independence test that works on high-dimensional data without parametric assumptions. Alternatives like mutual information or distance correlation could work, but HSIC has well-studied permutation test procedures via `hyppo` and handles the 768-dimensional sentence embeddings naturally.

---

## Extending the Framework

### Adding a new prompt component

1. Add the component key to `COMPONENT_KEYS` in `src/components.py`
2. Add `present` and `absent` variants in `config/components.yaml`
3. Update `generate_pb_design()` call — for 7 factors, PB gives 8 runs (still efficient)
4. Update `generate_loo_design()` — now 8 runs (1 full + 7 ablations)
5. Entanglement test now has 21 pairs instead of 15

### Adding a new task

1. Add task config to `config/tasks.yaml` with dataset path, split, fields, and few-shot examples
2. Add parsing logic to `load_task_data()` in `src/datasets.py` if the answer format is novel
3. Add the task name to the `TASKS` list in relevant scripts

### Adding a new model provider

1. Add a `_call_<provider>()` function in `src/inference.py`
2. Add the provider branch in `run_single_inference()`
3. Add model config to `config/models.yaml` with `provider: "<your_provider>"`

### Adding a new analysis

Scripts in `scripts/` are standalone — they import from `src/` and write to `results/`. To add a new analysis:

1. Create `scripts/run_<your_analysis>.py`
2. Import what you need from `src/`
3. Load model results from `results/<model>/summary.json`
4. Write outputs to `results/`

---

## Output File Formats

### `summary.json` (per model)

```json
{
  "entanglement": {
    "stat_matrix": [[...], ...],      // 6x6 HSIC statistics
    "pval_matrix": [[...], ...],      // 6x6 raw p-values
    "pval_adj_matrix": [[...], ...],  // 6x6 BH-adjusted p-values
    "n_significant": 13               // count of pairs with adj_p < 0.05
  },
  "screening": {
    "gsm8k": {
      "pb_design": [[1,-1,...], ...], // 8x6 design matrix
      "pb_accuracies": [0.93, ...],   // 8 run-level accuracies
      "loo_accuracies": [0.95, ...]   // 7 run-level accuracies
    },
    // ... same for bbh_date, mmlu_pro
  },
  "analysis": {
    "gsm8k": {
      "pb_effects": [0.01, -0.03, ...],     // 6 main effects
      "loo_effects": [0.02, -0.01, ...],     // 6 LOO effects
      "comparison_spearman_r": 0.543,
      "comparison_spearman_p": 0.266
    },
    // ...
  }
}
```

### CSV files (per model, per task)

**`pb_runs_<task>.csv`** — One row per PB run, columns: `run, system_role, persona, few_shot, cot_trigger, output_format, constraints, accuracy`

**`loo_runs_<task>.csv`** — One row per LOO run, columns: `run, removed_component, accuracy`

**`main_effects_pb_vs_loo.csv`** — One row per component, columns: `component, pb_effect, loo_effect, pb_rank, loo_rank`

**`entanglement_matrix.csv`** — 6x6 matrix with format `stat (p=pval)` in each cell
