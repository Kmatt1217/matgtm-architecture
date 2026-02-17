# Cross-domain: BDG2 — Energy Forecasting & Peak Interpretability

This document summarizes an out-of-domain adaptation of MatGTM to **BDG2 ([Building Data Genome 2](https://github.com/buds-lab/building-data-genome-project-2))** for:
- k-step ahead **electricity usage forecasting**
- **peak-centric interpretability** via latent-query attention

## Task
- Input: multivariate time series (e.g., weather signals + building meta / operational signals if available)
- Output: future electricity usage (k-step ahead)

## Why MatGTM transfers
- **Latent query cross-attention** summarizes long histories into a compact set of tokens.
- **Matryoshka-style prefix training** mitigates the need to retrain multiple models to find the optimal number of latent queries.
- The decoder’s final representation is directly projected to forecasts, so we keep a stable hidden bottleneck while controlling compute via FFN width scaling.

## Results & Analysis

### 1) Energy Usage Per Day (Forecast vs. Truth)
<img src="../assets/bdg2/energy_usage_per_day.png" width="400" />

### 2) Attention Importance around Peak
<img src="../assets/bdg2/attention_importance.png" width="400" />

### 3) Factor Analysis (variables aligned with peak-focused window)
<img src="../assets/bdg2/factor_analysis.png" width="400" />

**Observed patterns**
- Forecast aligns with daily trend and peak regimes.
- Attention concentrates around pre-peak windows, suggesting early-warning signals.
- Factor plots show coherent relationships between peak regimes and weather variables.


**Forecast quality.** The model tracks the overall daily consumption trend and follows major peak regimes reasonably well, indicating that the learned representation captures both slow-changing seasonality and sharper demand surges.

**Peak-centric interpretability.** The attention curve shows that the model places higher importance on specific pre-peak windows rather than distributing attention uniformly across the entire history. This suggests that MatGTM can highlight *early signals* that are most informative for upcoming peaks.

**Factor coherence.** When we align external variables (e.g., temperature, dew point, pressure, wind) with the peak-focused time window, we observe coherent changes around the highlighted region. This supports the interpretation that the model’s peak attention is not arbitrary, but correlates with plausible drivers of demand.

## Why this happens in MatGTM

This behavior is consistent with the core design of MatGTM:

- **Latent-query cross-attention for summarization.** Instead of relying only on self-attention over time steps, MatGTM uses learnable latent queries to *selectively summarize* long histories into a compact set of tokens. This naturally produces interpretable “importance over time” patterns.

Overall, these results suggest that MatGTM’s latent-query design transfers beyond fashion demand forecasting and can be applied to other multivariate forecasting tasks where both predictive performance and interpretability around critical events (e.g., peaks) matter.
