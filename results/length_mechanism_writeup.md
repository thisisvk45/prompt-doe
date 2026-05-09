# Response-Length Mechanism Analysis (5 Models)

## Does response-length variability correlate with entanglement?

Across all five models, response-length coefficient of variation (CV) shows no positive correlation with entanglement count (Spearman r = -0.872, p = 0.054; Pearson r = -0.833, p = 0.080). This rules out the hypothesis that entanglement is merely an artifact of response-length variation. The HSIC test captures semantic distributional shifts in high-dimensional embeddings, not surface-level length statistics.

## Is GPT-4o-mini an outlier on response-length variance?

GPT-4o-mini (CV = 0.530) is not the highest-variance model on response length. Despite having the highest entanglement (13/15), its length variability is comparable to or lower than other models, confirming that entanglement measures semantic — not length — coupling.

## Which components cause the most response-length variation?

Across all models, **output_format** and **constraints** consistently produce the largest length shifts when toggled. **cot_trigger** produces the smallest shifts. For GPT-4o, **persona** also causes large length changes (180 chars), consistent with its role as the most consistently harmful component in the PB screening.

## Does this support high entanglement = high response variability?

No. Entanglement and response-length variability are not positively correlated. GPT-4o-mini's high entanglement reflects semantic coupling in the output embedding space that is orthogonal to simple length statistics. HSIC detects meaning shifts, not format shifts.
