# prompt-doe

**Output-level entanglement diagnostics and Plackett-Burman screening for component-level prompt attribution.**

Code and data for the paper *"When Does Component Independence Hold? Entanglement Diagnostics and Fractional Factorial Attribution for LLM Prompts"*.

---

## TL;DR

We treat prompt engineering as a designed experiment. Using HSIC-based entanglement tests and Plackett-Burman fractional factorial designs, we show that (1) **component entanglement is model-specific** — GPT-4o-mini exhibits 13/15 significant output interactions while GPT-4o and Claude Haiku show 0/15, (2) **entanglement is not a response-length artifact** (Spearman r = -0.87 between entanglement and length CV across 5 models), and (3) **PB screening recovers the same component rankings as full factorial at 12.5% of the evaluation cost**, with cross-model effect direction agreement of 94.4%.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key(s)
export OPENAI_API_KEY="your-key-here"
export OPENROUTER_API_KEY="your-key-here"  # optional, for Claude/DeepSeek/Gemma

# 3. Run the full protocol on a model
PYTHONPATH=. python scripts/run_entanglement.py        # Phase 1: HSIC entanglement test
PYTHONPATH=. python scripts/run_pb_screen.py            # Phase 2: PB + LOO screening
PYTHONPATH=. python scripts/run_analysis.py             # Phase 3: Bootstrap analysis
PYTHONPATH=. python scripts/run_full_factorial.py       # Phase 4: Full factorial validation (optional, 64 runs)
```

To run on a different model, add its config to `config/models.yaml` and pass the model key to the scripts.

---

## Method Overview

```
                    ┌─────────────────────────────┐
                    │   6 Prompt Components        │
                    │   (System Role, Persona,     │
                    │    Few-Shot, CoT, Format,    │
                    │    Constraints)               │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
   ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐
   │  Entanglement │  │  PB Screening   │  │  LOO Baseline │
   │  (HSIC test)  │  │  (8 runs)       │  │  (7 runs)     │
   └──────┬───────┘  └────────┬────────┘  └──────┬────────┘
          │                   │                   │
          ▼                   ▼                   ▼
   Do components        Main effects       Ablation effects
   interact in the      from fractional    from single-
   output space?        factorial design    component removal
          │                   │                   │
          └───────────┬───────┘                   │
                      ▼                           │
              ┌───────────────┐                   │
              │  Bootstrap +  │◄──────────────────┘
              │  BH Correction│    Compare PB vs LOO
              └───────┬───────┘
                      ▼
              Component Attribution
              (with confidence intervals)
```

**Step 1 — Entanglement Test.** For each pair of prompt components, we generate LLM outputs with one component toggled on/off, embed the outputs with `all-mpnet-base-v2`, and run a permutation-based HSIC independence test. If the test rejects (p < 0.05, BH-corrected), the components are *entangled* — their joint effect on outputs is non-additive.

**Step 2 — Plackett-Burman Screening.** An 8-run PB design (Resolution III) estimates all 6 main effects using only 12.5% of the 64 full factorial runs. Each run evaluates 200 task examples.

**Step 3 — LOO Baseline.** A 7-run leave-one-out design measures each component's marginal contribution from a single full-prompt baseline.

**Step 4 — Analysis.** Bootstrap BCa confidence intervals + Benjamini-Hochberg FDR correction quantify significance. PB and LOO rankings are compared via Spearman correlation.

---

## Components Studied

| # | Component | Present | Absent |
|---|-----------|---------|--------|
| 1 | **System Role** | "You are an expert AI assistant..." | Neutral task framing |
| 2 | **Persona** | "Approach as a meticulous professor..." | Generic instruction |
| 3 | **Few-Shot** | 3 worked examples | No examples |
| 4 | **CoT Trigger** | "Think step-by-step..." | "Provide your answer" |
| 5 | **Output Format** | Structured ANSWER format | "Give your answer at the end" |
| 6 | **Constraints** | Explicit behavioral constraints | "Answer the question below" |

## Tasks

| Task | Dataset | Type | N |
|------|---------|------|---|
| GSM8K | `openai/gsm8k` | Numeric | 200 |
| BBH-Date | `lukaemon/bbh` (date_understanding) | Multiple choice | 200 |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | Multiple choice | 200 |

## Models Tested

| Model | Provider | Entanglement (sig/15) |
|-------|----------|----------------------|
| GPT-4o-mini | OpenAI | 13/15 |
| GPT-4o | OpenAI | 0/15 |
| Claude Haiku 4.5 | Anthropic | 0/15 |
| DeepSeek V4 Pro | DeepSeek | 9/15 |
| Gemma 4 31B | Google | 6/15 |

---

## Reproducing the Paper

All results in the paper can be regenerated from scratch. Pre-computed results are included in `results/`.

```bash
# Entanglement matrices (Table 2)
PYTHONPATH=. python scripts/run_entanglement.py

# PB + LOO screening for GPT-4o-mini (Tables 3-4)
PYTHONPATH=. python scripts/run_pb_screen.py
PYTHONPATH=. python scripts/run_analysis.py

# Full factorial validation on GSM8K (Table 5)
PYTHONPATH=. python scripts/run_full_factorial.py

# Cross-model replication (Table 6)
PYTHONPATH=. python scripts/run_cross_model.py claude_haiku
PYTHONPATH=. python scripts/run_cross_model.py gpt4o
PYTHONPATH=. python scripts/run_cross_model_analysis.py

# Outlier configuration investigation (Section 4.3)
PYTHONPATH=. python scripts/run_outlier_investigation.py

# Response-length control analysis (Appendix B)
PYTHONPATH=. python scripts/run_response_length_analysis.py
```

---

## Repository Structure

```
prompt-doe/
├── README.md
├── LICENSE                          # MIT
├── requirements.txt                 # Pinned dependencies
├── paper/
│   └── (paper PDF)
├── src/
│   ├── inference.py                 # LLM API wrappers (OpenAI, OpenRouter, Ollama)
│   ├── datasets.py                  # Task data loading and answer parsing
│   ├── prompts.py                   # Prompt assembly from component flags
│   ├── components.py                # Component definitions and config loading
│   ├── independence.py              # HSIC entanglement testing
│   ├── design.py                    # PB, LOO, and full factorial design matrices
│   ├── analysis.py                  # Bootstrap CIs, BH correction, PB vs LOO
│   ├── validation.py                # Full factorial validation
│   └── transfer.py                  # Cross-model transfer analysis
├── scripts/
│   ├── run_entanglement.py          # Run HSIC entanglement tests
│   ├── run_pb_screen.py             # Run PB + LOO screening
│   ├── run_analysis.py              # Run bootstrap analysis
│   ├── run_full_factorial.py        # Run 64-run full factorial
│   ├── run_cross_model.py           # Run full protocol on any model
│   ├── run_cross_model_analysis.py  # Cross-model comparison
│   ├── run_outlier_investigation.py # Investigate anomalous configurations
│   └── run_response_length_analysis.py
├── config/
│   ├── models.yaml                  # Model configurations
│   ├── tasks.yaml                   # Task definitions + few-shot examples
│   └── components.yaml              # Component present/absent variants
├── results/
│   ├── gpt-4o-mini/                 # Per-model CSVs and summary
│   ├── gpt-4o/
│   ├── anthropic-claude-haiku-4-5/
│   ├── deepseek-deepseek-v4-pro/
│   ├── google-gemma-4-31b-it_free/
│   ├── cross_model_analysis.csv
│   ├── cross_model_summary.md
│   └── response_length_analysis.csv
├── figures/
│   ├── entanglement_vs_length_cv.pdf
│   └── component_length_shifts.pdf
└── tests/
    └── test_prompts.py              # Unit tests for prompts and design
```

---

## Key Results

### Component Effects (5-model consensus)

- **Output format specification** is the single most reliably beneficial component across all models and tasks.
- **Persona instructions consistently degrade performance** and should be omitted unless needed for style.
- **Few-shot examples reliably help** across all models and tasks.
- **CoT trigger** shows mixed effects — helpful for math, neutral or slightly harmful for knowledge tasks.

### Entanglement is Model-Specific

Entanglement (non-additive component interactions in outputs) varies dramatically by model, not by model family. GPT-4o-mini (13/15) and GPT-4o (0/15) are both OpenAI models but behave completely differently. This suggests entanglement reflects differences in instruction-tuning or distillation, not architecture.

### PB Screening Efficiency

The 8-run PB design achieves 94.4% direction agreement with LOO ablation across 5 models and 3 tasks, while requiring only 8 runs vs. 64 for full factorial. For practitioners, this means you can reliably identify which prompt components help or hurt with ~1,600 API calls instead of ~12,800.

---

## Adding Your Own Model

1. Add a config entry to `config/models.yaml`:

```yaml
your_model:
  name: "Your Model Name"
  provider: "openai"          # or "openrouter" or "ollama"
  model_id: "your-model-id"
  temperature: 0.0
  max_tokens: 512
  requests_per_minute: null
```

2. Run the full protocol:

```bash
PYTHONPATH=. python scripts/run_cross_model.py your_model
```

This runs entanglement testing, PB screening, LOO ablation, and analysis, saving results to `results/<model-dir>/`.

---

## Citation

```bibtex
@article{prompt-doe-2025,
  title   = {When Does Component Independence Hold? Entanglement Diagnostics
             and Fractional Factorial Attribution for LLM Prompts},
  author  = {Vikas},
  year    = {2025},
  url     = {https://github.com/thisisvk45/prompt-doe}
}
```

*(Update with arXiv ID once posted.)*

---

## License

MIT. See [LICENSE](LICENSE).

## Contact

Open an issue on this repo or reach out via [GitHub](https://github.com/thisisvk45).
