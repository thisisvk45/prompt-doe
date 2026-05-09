# Cross-Model Interpretation

## 1. Are prompt component effects consistent across models?

Yes — strongly so. Effect direction agreement across all four models reaches 94.4% (17/18 component×task combinations). The sole disagreement is `cot_trigger` on MMLU-Pro, where GPT-4o-mini shows a small positive PB effect (+0.0100) while the other three models show small negative effects. Cross-model Spearman correlations of the 6-component PB effect rankings average r = 0.89 across all model pairs, confirming that not just direction but relative magnitude is preserved. Models unanimously agree that **persona hurts** (all 12 model×task cells negative), **output_format helps** (all 12 positive), and **few_shot helps** (all 12 positive).

## 2. Does the outlier configuration replicate?

The (-,+,-,+,-,+) configuration — persona on, few-shot off, output_format off — consistently produces the lowest PB accuracy across all four models and all three tasks. The deficit ranges from −0.070 (Gemma/GSM8K) to −0.386 (GPT-4o-mini/BBH-Date). This is not a random fluctuation tied to one model: the configuration combines the two most harmful sign assignments (persona present, output_format absent) with removal of the most helpful component (few-shot), creating a compound penalty. The cross-model replication confirms this as a genuine interaction effect.

## 3. Do models differ in entanglement?

Dramatically. GPT-4o-mini shows 13/15 significant output-based HSIC pairs, meaning its responses are strongly shaped by component interactions. Claude Haiku 4.5 shows 0/15 — it processes components essentially independently, producing similar output distributions regardless of which other components are present. DeepSeek V4 Pro (9/15) and Gemma 4 31B (6/15) fall between these extremes. This suggests that entanglement is a model-level property — possibly reflecting differences in instruction-tuning approaches — rather than a universal feature of prompt composition.

## 4. Does PB screening agree with LOO ablation?

Moderately. PB vs LOO Spearman correlations range from 0.212 to 0.543 across model×task cells, with none reaching significance (p < 0.05). This consistent moderate-but-insignificant correlation reflects the fundamental methodological difference: PB estimates marginal effects averaged over many configurations, while LOO measures the effect of removing one component from a single full-prompt baseline. The systematic gap validates running both methods, as they capture complementary information about component importance.

## 5. What are the practical implications?

Three actionable findings emerge. First, **output_format specification is the single most reliably beneficial component** across all models and tasks — practitioners should always include explicit output formatting instructions. Second, **persona instructions consistently degrade performance** and should be omitted unless specifically needed for style requirements. Third, the outlier configuration demonstrates that harmful component combinations can produce severe compound penalties (up to 38.6 percentage points below the mean), underscoring the importance of systematic prompt design over ad-hoc assembly.
