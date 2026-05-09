# Cross-Model Analysis Report

## 1. Entanglement Comparison

- **GPT-4o-mini**: 13/15 significant pairs
- **GPT-4o**: 0/15 significant pairs
- **Claude Haiku 4.5**: 0/15 significant pairs
- **DeepSeek V4 Pro**: 9/15 significant pairs
- **Gemma 4 31B**: 6/15 significant pairs

### Pairwise Jaccard similarity of entangled pairs:

- GPT-4o-mini vs GPT-4o: Jaccard = 0.000
- GPT-4o-mini vs Claude Haiku 4.5: Jaccard = 0.000
- GPT-4o-mini vs DeepSeek V4 Pro: Jaccard = 0.467
- GPT-4o-mini vs Gemma 4 31B: Jaccard = 0.267
- GPT-4o vs DeepSeek V4 Pro: Jaccard = 0.000
- GPT-4o vs Gemma 4 31B: Jaccard = 0.000
- Claude Haiku 4.5 vs DeepSeek V4 Pro: Jaccard = 0.000
- Claude Haiku 4.5 vs Gemma 4 31B: Jaccard = 0.000
- DeepSeek V4 Pro vs Gemma 4 31B: Jaccard = 0.667

## 2. PB vs LOO Spearman Correlation

| Task | GPT-4o-mini | GPT-4o | Claude Haiku 4.5 | DeepSeek V4 Pro | Gemma 4 31B |
|------|------|------|------|------|------|
| gsm8k | 0.464 (p=0.354) | -0.090 (p=0.866) | 0.212 (p=0.686) | 0.377 (p=0.462) | 0.486 (p=0.329) |
| bbh_date | 0.265 (p=0.612) | 0.261 (p=0.618) | 0.406 (p=0.425) | 0.232 (p=0.658) | 0.232 (p=0.658) |
| mmlu_pro | 0.314 (p=0.544) | 0.754 (p=0.084) | 0.522 (p=0.288) | 0.371 (p=0.469) | 0.543 (p=0.266) |

## 3. Effect Direction Agreement

For each component × task, check if PB effect sign agrees across models.

**Agreement rate**: 12/18 = 66.7%

| Task | Component | GPT-4o-mini | GPT-4o | Claude Haiku 4.5 | DeepSeek V4 Pro | Gemma 4 31B | Agree? |
|------|-----------|------|------|------|------|------|--------|
| gsm8k | system_role | +0.0187 | +0.0012 | +0.0125 | +0.0088 | +0.0050 | Yes |
| gsm8k | persona | -0.0312 | -0.0063 | -0.0100 | -0.0088 | -0.0025 | Yes |
| gsm8k | few_shot | +0.0288 | -0.0013 | +0.0200 | +0.0237 | +0.0250 | No |
| gsm8k | cot_trigger | -0.0137 | +0.0038 | -0.0200 | -0.0162 | -0.0150 | No |
| gsm8k | output_format | +0.0463 | +0.0088 | +0.0250 | +0.0263 | +0.0275 | Yes |
| gsm8k | constraints | -0.0438 | +0.0013 | -0.0175 | -0.0213 | -0.0200 | No |
| bbh_date | system_role | +0.0975 | +0.0100 | +0.0237 | +0.0275 | +0.0325 | Yes |
| bbh_date | persona | -0.0925 | -0.0200 | -0.0737 | -0.0600 | -0.0650 | Yes |
| bbh_date | few_shot | +0.1550 | +0.0325 | +0.0287 | +0.0425 | +0.0450 | Yes |
| bbh_date | cot_trigger | -0.0700 | +0.0000 | -0.0137 | -0.0175 | -0.0125 | No |
| bbh_date | output_format | +0.0925 | +0.0025 | +0.0537 | +0.0550 | +0.0575 | Yes |
| bbh_date | constraints | -0.0825 | +0.0125 | -0.0037 | -0.0125 | -0.0100 | No |
| mmlu_pro | system_role | +0.0075 | +0.0112 | +0.0275 | +0.0375 | +0.0338 | Yes |
| mmlu_pro | persona | -0.0750 | -0.0938 | -0.1275 | -0.1025 | -0.0963 | Yes |
| mmlu_pro | few_shot | +0.0325 | +0.0112 | +0.0400 | +0.0375 | +0.0363 | Yes |
| mmlu_pro | cot_trigger | +0.0100 | -0.0438 | -0.0125 | -0.0100 | -0.0013 | No |
| mmlu_pro | output_format | +0.0775 | +0.0463 | +0.0700 | +0.0700 | +0.0612 | Yes |
| mmlu_pro | constraints | +0.0400 | +0.0363 | +0.0250 | +0.0150 | +0.0162 | Yes |

## 4. Key Component Questions

### Does persona hurt on all models?

- GPT-4o-mini/gsm8k: -0.0312 (hurts) [Yes]
- GPT-4o/gsm8k: -0.0063 (hurts) [Yes]
- Claude Haiku 4.5/gsm8k: -0.0100 (hurts) [Yes]
- DeepSeek V4 Pro/gsm8k: -0.0088 (hurts) [Yes]
- Gemma 4 31B/gsm8k: -0.0025 (hurts) [Yes]
- GPT-4o-mini/bbh_date: -0.0925 (hurts) [Yes]
- GPT-4o/bbh_date: -0.0200 (hurts) [Yes]
- Claude Haiku 4.5/bbh_date: -0.0737 (hurts) [Yes]
- DeepSeek V4 Pro/bbh_date: -0.0600 (hurts) [Yes]
- Gemma 4 31B/bbh_date: -0.0650 (hurts) [Yes]
- GPT-4o-mini/mmlu_pro: -0.0750 (hurts) [Yes]
- GPT-4o/mmlu_pro: -0.0938 (hurts) [Yes]
- Claude Haiku 4.5/mmlu_pro: -0.1275 (hurts) [Yes]
- DeepSeek V4 Pro/mmlu_pro: -0.1025 (hurts) [Yes]
- Gemma 4 31B/mmlu_pro: -0.0963 (hurts) [Yes]

### Does output_format help on all models?

- GPT-4o-mini/gsm8k: +0.0463 (helps) [Yes]
- GPT-4o/gsm8k: +0.0088 (helps) [Yes]
- Claude Haiku 4.5/gsm8k: +0.0250 (helps) [Yes]
- DeepSeek V4 Pro/gsm8k: +0.0263 (helps) [Yes]
- Gemma 4 31B/gsm8k: +0.0275 (helps) [Yes]
- GPT-4o-mini/bbh_date: +0.0925 (helps) [Yes]
- GPT-4o/bbh_date: +0.0025 (helps) [Yes]
- Claude Haiku 4.5/bbh_date: +0.0537 (helps) [Yes]
- DeepSeek V4 Pro/bbh_date: +0.0550 (helps) [Yes]
- Gemma 4 31B/bbh_date: +0.0575 (helps) [Yes]
- GPT-4o-mini/mmlu_pro: +0.0775 (helps) [Yes]
- GPT-4o/mmlu_pro: +0.0463 (helps) [Yes]
- Claude Haiku 4.5/mmlu_pro: +0.0700 (helps) [Yes]
- DeepSeek V4 Pro/mmlu_pro: +0.0700 (helps) [Yes]
- Gemma 4 31B/mmlu_pro: +0.0612 (helps) [Yes]

### Does few_shot help on all models?

- GPT-4o-mini/gsm8k: +0.0288 (helps) [Yes]
- GPT-4o/gsm8k: -0.0013 (hurts) [No]
- Claude Haiku 4.5/gsm8k: +0.0200 (helps) [Yes]
- DeepSeek V4 Pro/gsm8k: +0.0237 (helps) [Yes]
- Gemma 4 31B/gsm8k: +0.0250 (helps) [Yes]
- GPT-4o-mini/bbh_date: +0.1550 (helps) [Yes]
- GPT-4o/bbh_date: +0.0325 (helps) [Yes]
- Claude Haiku 4.5/bbh_date: +0.0287 (helps) [Yes]
- DeepSeek V4 Pro/bbh_date: +0.0425 (helps) [Yes]
- Gemma 4 31B/bbh_date: +0.0450 (helps) [Yes]
- GPT-4o-mini/mmlu_pro: +0.0325 (helps) [Yes]
- GPT-4o/mmlu_pro: +0.0112 (helps) [Yes]
- Claude Haiku 4.5/mmlu_pro: +0.0400 (helps) [Yes]
- DeepSeek V4 Pro/mmlu_pro: +0.0375 (helps) [Yes]
- Gemma 4 31B/mmlu_pro: +0.0363 (helps) [Yes]


## 5. Cross-Model Effect Ranking Correlation

Pairwise Spearman correlation of 6 PB effects between model pairs, averaged across tasks.

- GPT-4o-mini vs GPT-4o: mean Spearman r = 0.623 (per-task: 0.314, 0.657, 0.899)
- GPT-4o-mini vs Claude Haiku 4.5: mean Spearman r = 0.771 (per-task: 0.829, 0.771, 0.714)
- GPT-4o-mini vs DeepSeek V4 Pro: mean Spearman r = 0.794 (per-task: 0.943, 0.771, 0.667)
- GPT-4o-mini vs Gemma 4 31B: mean Spearman r = 0.810 (per-task: 0.943, 0.771, 0.714)
- GPT-4o vs Claude Haiku 4.5: mean Spearman r = 0.480 (per-task: 0.029, 0.600, 0.812)
- GPT-4o vs DeepSeek V4 Pro: mean Spearman r = 0.503 (per-task: 0.086, 0.600, 0.824)
- GPT-4o vs Gemma 4 31B: mean Spearman r = 0.499 (per-task: 0.086, 0.600, 0.812)
- Claude Haiku 4.5 vs DeepSeek V4 Pro: mean Spearman r = 0.976 (per-task: 0.943, 1.000, 0.986)
- Claude Haiku 4.5 vs Gemma 4 31B: mean Spearman r = 0.981 (per-task: 0.943, 1.000, 1.000)
- DeepSeek V4 Pro vs Gemma 4 31B: mean Spearman r = 0.995 (per-task: 1.000, 1.000, 0.986)

## 6. Outlier Configuration (-,+,-,+,-,+) Check

PB Run 7 accuracy vs mean of other PB runs.


### GPT-4o-mini

- gsm8k: outlier=0.805, mean_other=0.924, diff=-0.119
- bbh_date: outlier=0.440, mean_other=0.826, diff=-0.386
- mmlu_pro: outlier=0.450, mean_other=0.544, diff=-0.094

### GPT-4o

- gsm8k: outlier=0.950, mean_other=0.959, diff=-0.009
- bbh_date: outlier=0.890, mean_other=0.917, diff=-0.027
- mmlu_pro: outlier=0.470, mean_other=0.596, diff=-0.126

### Claude Haiku 4.5

- gsm8k: outlier=0.905, mean_other=0.978, diff=-0.073
- bbh_date: outlier=0.765, mean_other=0.903, diff=-0.138
- mmlu_pro: outlier=0.445, mean_other=0.608, diff=-0.163

### DeepSeek V4 Pro

- gsm8k: outlier=0.865, mean_other=0.940, diff=-0.075
- bbh_date: outlier=0.715, mean_other=0.864, diff=-0.149
- mmlu_pro: outlier=0.415, mean_other=0.574, diff=-0.159

### Gemma 4 31B

- gsm8k: outlier=0.835, mean_other=0.905, diff=-0.070
- bbh_date: outlier=0.675, mean_other=0.825, diff=-0.150
- mmlu_pro: outlier=0.380, mean_other=0.521, diff=-0.141