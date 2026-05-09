<p align="center">
  <img src="figures/hero_output.svg" alt="Example output from prompt-doe: component attribution chart and entanglement matrix" width="900">
</p>

<h1 align="center">prompt-doe</h1>

<p align="center">
  <em>Stop guessing which parts of your prompt matter. Run an experiment instead.</em>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &nbsp;&bull;&nbsp;
  <a href="#worked-example">Example</a> &nbsp;&bull;&nbsp;
  <a href="#what-we-found">Results</a> &nbsp;&bull;&nbsp;
  <a href="#run-it-on-your-model">Your Model</a> &nbsp;&bull;&nbsp;
  <a href="#how-it-works">Method</a> &nbsp;&bull;&nbsp;
  <a href="#reproducing-the-paper">Reproduce</a> &nbsp;&bull;&nbsp;
  <a href="docs/architecture.md">Architecture</a> &nbsp;&bull;&nbsp;
  <a href="#citation">Cite</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/paper-arXiv_pending-b31b1b" alt="arXiv">
  <img src="https://img.shields.io/badge/models-5_tested-blue" alt="models">
  <img src="https://img.shields.io/badge/tasks-3_benchmarks-green" alt="tasks">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="license">
</p>

---

If you've ever added an instruction to your prompt, watched accuracy go up, and assumed your instruction caused the gain — you might be wrong. On 1 of 5 models we tested, that conclusion is invalid more often than not. This repo tells you which models, and gives you the tools to check your own.

**prompt-doe** brings [Design of Experiments](https://en.wikipedia.org/wiki/Design_of_experiments) to prompt engineering. One command gives you:

1. **Which components help, which hurt, which do nothing** — with confidence intervals
2. **Whether your components interact** (entanglement) — or if they're safe to tune independently
3. **All of this at 12.5% of the brute-force cost** — 8 runs instead of 64

Code and data for the paper *"When Does Component Independence Hold? Output-Level Entanglement and Model-Conditional Validity of Prompt Attribution"*.

---

## Quick Start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"

# Run everything — entanglement test, PB screen, LOO baseline, analysis
PYTHONPATH=. python scripts/run_cross_model.py gpt4o_mini
```

Results land in `results/<model-name>/` — CSVs, summary JSON, entanglement matrix.

> **First-time setup** downloads ~500MB of models (`sentence-transformers/all-mpnet-base-v2`) and caches GSM8K/BBH/MMLU-Pro datasets from HuggingFace. Allow ~5 minutes for the first run.
>
> For cross-model replication beyond OpenAI models, also set:
> ```bash
> export OPENROUTER_API_KEY="your-key"  # for Claude, DeepSeek, Gemma via OpenRouter
> ```

Want to run the phases individually?

```bash
PYTHONPATH=. python scripts/run_entanglement.py          # HSIC entanglement test
PYTHONPATH=. python scripts/run_pb_screen.py              # Plackett-Burman + LOO screening
PYTHONPATH=. python scripts/run_analysis.py               # Bootstrap CIs + BH correction
PYTHONPATH=. python scripts/run_full_factorial.py         # Full 64-run validation (optional)
```

---

## Worked Example

Here's what you get when you run `prompt-doe` on GPT-4o-mini across GSM8K (grade-school math).

Your prompt has 6 components. The tool tests all of them in 8 carefully chosen configurations (not 64), then tells you:

**Attribution table** (from `results/gpt-4o-mini/main_effects_pb_vs_loo.csv`):

```
Component        PB Effect    LOO Effect    Verdict
─────────────────────────────────────────────────────
output_format     +0.058       +0.045       Helps — always include
few_shot          +0.047       +0.040       Helps — keep your examples
constraints       +0.023       +0.015       Helps slightly
cot_trigger       +0.010       +0.005       Negligible
system_role       −0.008       −0.010       Negligible
persona           −0.033       −0.020       Hurts — drop it
```

**Entanglement matrix** (from `results/gpt-4o-mini/entanglement_matrix.csv`):

```
13/15 component pairs are entangled (HSIC p < 0.05, BH-corrected)
→ Components interact heavily in GPT-4o-mini's output space
→ LOO ablation may give misleading attribution on this model
```

**What does this tell you?** On GPT-4o-mini, your output format instruction is doing most of the work (+5.8pp accuracy). Your persona instruction is actively hurting (−3.3pp). And because 13/15 component pairs are entangled, you can't reliably tune components one-at-a-time on this model — their effects depend on each other.

Run the same protocol on GPT-4o? Entanglement drops to 0/15. Same components, same prompts, completely different interaction structure. That's the finding.

---

## What We Found

### The headline numbers

| Finding | Number |
|---------|--------|
| PB screening cost vs. full factorial | **8 runs vs. 64** (12.5%) |
| Cross-model effect direction agreement | **94.4%** (17/18 component-task pairs) |
| Entanglement range across models | **0/15 to 13/15** significant pairs |
| Length-artifact correlation | **r = −0.87** (entanglement is *not* a length effect) |

### Entanglement is model-specific — not family-specific

<p align="center">
  <img src="figures/component_entanglement_by_model.svg" alt="Entanglement varies by model, not model family" width="700">
</p>

GPT-4o-mini and GPT-4o are both OpenAI models. One has 13/15 entangled component pairs; the other has 0/15. Same family, opposite behavior. This means entanglement is a property of how a model was trained (likely distillation), not its architecture.

### Three things every prompt engineer should know

> **Always include output format instructions.** Most reliably beneficial component across all 5 models and all 3 tasks. No exceptions.

> **Drop the persona.** "Act as a meticulous professor" hurts accuracy in every single model-task combination we tested (15/15 negative).

> **Few-shot examples work.** Positive effect everywhere. Not surprising, but now quantified with confidence intervals.

---

## How It Works

<p align="center">
  <img src="figures/prompt_engineering_pipeline.svg" alt="Method pipeline" width="800">
</p>

Every prompt is a combination of 6 binary components — each either *present* (active variant) or *absent* (neutral placeholder):

| Component | When present | When absent |
|-----------|-------------|-------------|
| **System Role** | *"You are an expert AI assistant..."* | Neutral task framing |
| **Persona** | *"Approach as a meticulous professor..."* | Generic instruction |
| **Few-Shot** | 3 worked examples | No examples |
| **CoT Trigger** | *"Think step-by-step..."* | *"Provide your answer"* |
| **Output Format** | Structured `ANSWER:` format | *"Give your answer at the end"* |
| **Constraints** | Explicit behavioral rules | *"Answer the question below"* |

The protocol has four stages:

**1. Entanglement Test** — For each pair of components, toggle one on/off, collect LLM outputs, embed them with `all-mpnet-base-v2`, and run an HSIC permutation test. If it rejects: the components *interact* in the output space — you can't tune them independently.

**2. Plackett-Burman Screening** — An 8-run [fractional factorial design](https://en.wikipedia.org/wiki/Plackett%E2%80%93Burman_design) that estimates all 6 main effects simultaneously. Each run evaluates 200 task examples. Total cost: 1,600 API calls instead of 12,800 for full factorial.

**3. LOO Baseline** — A 7-run leave-one-out ablation for comparison. Remove one component at a time from the full prompt. Standard practice — we include it to show PB recovers the same rankings cheaper.

**4. Statistical Analysis** — Bootstrap BCa confidence intervals + Benjamini-Hochberg FDR correction. Compare PB vs. LOO rankings via Spearman correlation.

### Tasks and models

Evaluated on 3 benchmarks spanning math, reasoning, and knowledge:

| Task | Source | Type | Examples |
|------|--------|------|----------|
| GSM8K | `openai/gsm8k` | Numeric | 200 |
| BBH-Date | `lukaemon/bbh` | Multiple choice | 200 |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | Multiple choice | 200 |

Tested on 5 models across 4 providers:

| Model | Provider | Entanglement |
|-------|----------|:------------:|
| GPT-4o-mini | OpenAI | 13/15 |
| GPT-4o | OpenAI | 0/15 |
| Claude Haiku 4.5 | Anthropic | 0/15 |
| DeepSeek V4 Pro | DeepSeek | 9/15 |
| Gemma 4 31B | Google | 6/15 |

---

## Run It on Your Model

Add 6 lines to `config/models.yaml`:

```yaml
your_model:
  name: "Your Model"
  provider: "openai"        # or "openrouter" or "ollama"
  model_id: "your-model-id"
  temperature: 0.0
  max_tokens: 512
```

Then:

```bash
PYTHONPATH=. python scripts/run_cross_model.py your_model
```

This runs the full protocol — entanglement, PB, LOO, analysis — and saves everything to `results/<model-dir>/`.

**Supported providers:** OpenAI API, OpenRouter (Claude, DeepSeek, Gemma, etc.), and local Ollama models.

---

## Reproducing the Paper

Every table and figure in the paper can be regenerated from scratch. Pre-computed results are included in `results/`.

```bash
# Table 2 — Entanglement matrices
PYTHONPATH=. python scripts/run_entanglement.py

# Tables 3-4 — PB + LOO screening (GPT-4o-mini)
PYTHONPATH=. python scripts/run_pb_screen.py
PYTHONPATH=. python scripts/run_analysis.py

# Table 5 — Full factorial validation (GSM8K)
PYTHONPATH=. python scripts/run_full_factorial.py

# Table 6 — Cross-model replication
PYTHONPATH=. python scripts/run_cross_model.py claude_haiku
PYTHONPATH=. python scripts/run_cross_model.py gpt4o
PYTHONPATH=. python scripts/run_cross_model_analysis.py

# Section 4.3 — Outlier configuration investigation
PYTHONPATH=. python scripts/run_outlier_investigation.py

# Appendix B — Response-length control analysis
PYTHONPATH=. python scripts/run_response_length_analysis.py
```

---

## Repository Structure

```
prompt-doe/
├── src/
│   ├── inference.py          # LLM API wrappers (OpenAI, OpenRouter, Ollama)
│   ├── datasets.py           # Task loading, answer parsing
│   ├── prompts.py            # Prompt assembly from component flags
│   ├── components.py         # Component definitions + config loading
│   ├── independence.py       # HSIC entanglement testing
│   ├── design.py             # Plackett-Burman, LOO, full factorial matrices
│   ├── analysis.py           # Bootstrap CIs, BH correction, PB vs LOO
│   ├── validation.py         # Full factorial ground-truth comparison
│   └── transfer.py           # Cross-model transfer analysis
├── scripts/
│   ├── run_entanglement.py
│   ├── run_pb_screen.py
│   ├── run_analysis.py
│   ├── run_full_factorial.py
│   ├── run_cross_model.py
│   ├── run_cross_model_analysis.py
│   ├── run_outlier_investigation.py
│   └── run_response_length_analysis.py
├── config/
│   ├── models.yaml           # Model configurations
│   ├── tasks.yaml            # Task definitions + few-shot examples
│   └── components.yaml       # Component present/absent variants
├── results/                  # Pre-computed results for all 5 models
├── figures/                  # Publication figures (PDF + PNG + SVG)
├── paper/                    # Paper PDF
├── docs/
│   └── architecture.md       # Codebase architecture + extension guide
└── tests/                    # Unit tests
```

---

## Limitations

- Tested on 5 models and 3 tasks; generalization to other tasks/domains is unverified
- The distillation hypothesis for entanglement is suggested by 1 distilled model (GPT-4o-mini) in the sample, not proven
- HSIC test power is limited at n=50 per condition; non-significant pairs may still have weak entanglement
- Plackett-Burman estimates only main effects; interaction effects require full factorial or Shapley-based follow-up
- Embedding model choice (`all-mpnet-base-v2`) is a hyperparameter we did not vary

---

## FAQ

<details>
<summary><b>Does this work with closed-source APIs (GPT, Claude)?</b></summary>
Yes. We tested on GPT-4o-mini, GPT-4o (OpenAI API), and Claude Haiku 4.5 (via OpenRouter). Any model with a chat completions endpoint works.
</details>

<details>
<summary><b>How long does it take?</b></summary>
~20 minutes on GPT-4o-mini, ~40 minutes on Claude Haiku for the full protocol (entanglement + PB + LOO + analysis across 3 tasks). Mostly API latency.
</details>

<details>
<summary><b>What does it cost?</b></summary>
$5–15 in API credits depending on the model. GPT-4o-mini is cheapest (~$5). GPT-4o is most expensive (~$15). The protocol uses ~10,500 API calls per model.
</details>

<details>
<summary><b>Why output-based entanglement instead of prompt-based?</b></summary>
Because the assumption that matters for attribution is whether the model's <em>responses</em> depend on component combinations, not whether the prompt embeddings do. A model might completely ignore persona text (no output effect) even though persona changes the prompt embedding. Output-based HSIC captures the model's actual sensitivity.
</details>

<details>
<summary><b>Why Plackett-Burman over Shapley values?</b></summary>
PB recovers main effects at 12.5% the cost of full factorial. Shapley values are needed only when interaction effects dominate — which we show they don't on these benchmarks (interaction magnitudes are ~5x smaller than main effects). If you find high entanglement on your model, run the full factorial to check.
</details>

<details>
<summary><b>Can I trust these results on my model without running it?</b></summary>
No. The whole point of the paper is that entanglement is model-specific. GPT-4o-mini and GPT-4o give opposite results with the same prompts. Run the protocol on your model.
</details>

---

## Contributing

Contributions welcome:

- **New model adapters** in `src/inference.py` (Azure OpenAI, Bedrock, Together, etc.)
- **New tasks** in `config/tasks.yaml` (code generation, summarization, etc.)
- **Alternative entanglement diagnostics** (mutual information, kernel CCA, distance correlation)
- **Bug reports and reproductions** on new models — especially if you find surprising entanglement patterns

See [docs/architecture.md](docs/architecture.md) for how the codebase is organized.

---

## Citation

```bibtex
@article{kumar2026prompt,
  title   = {When Does Component Independence Hold? Output-Level Entanglement
             and Model-Conditional Validity of Prompt Attribution},
  author  = {Kumar, Vikas},
  year    = {2026},
  url     = {https://github.com/thisisvk45/prompt-doe}
}
```

*(Will be updated with arXiv ID once posted.)*

---

<p align="center">
  <b>License:</b> MIT &nbsp;&bull;&nbsp;
  <b>Issues:</b> <a href="https://github.com/thisisvk45/prompt-doe/issues">GitHub Issues</a> &nbsp;&bull;&nbsp;
  <b>Contact:</b> <a href="https://github.com/thisisvk45">@thisisvk45</a>
</p>
