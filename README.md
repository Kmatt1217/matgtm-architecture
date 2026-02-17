# MatGTM (Architecture)

This repository provides the **model architecture** of my M.S. thesis work: **MatGTM**, a multimodal forecasting model designed to capture meaningful patterns from trend signals and product modalities with efficient inference.

> This repo focuses on **architecture**.  
> Training/Evaluation pipeline will be added in a future update.

---

## What is MatGTM?

MatGTM is a modified architecture based on **GTM-Transformer** from *“Well Googled is Half Done: Multimodal Forecasting of New Fashion Product Sales with Image-based Google Trends”*.  
The Architecture is a Trasnformer-based model that performs Time-Series Forecasting by combining:
- **Time-series signals**
- Optional modalities (image / text embeddings / etc..)
- **Matryoshka-style latent queries (prefix slicing)** that summarize time-series through cross-attention, capture trend patterns efficiently, balance accuracy and efficiency

## Key Ideas (Thesis Summary)

### 1) Latent Query Cross-Attention for Trend Summarization
Instead of relying only on self-attention over time steps, MatGTM introduces learnable **latent queries** that attend to the input time series and extract trend-relevant signals into a compact representation.

### 2) Matryoshka-style Query Prefix Training (Efficiency-friendly)
The latent queries are trained in a **nested (Matryoshka) manner** so that the earlier query tokens are optimized more frequently.  
At inference time, we can use only the first *m* queries for faster prediction.

Also, a practical challenge of latent-query trend summarization is choosing the **optimal number of query tokens**.  
A naive approach is to run multiple trainings with different query counts (e.g., 8/16/32/64), which is computationally inefficient.

To address this, MatGTM adopts **Matryoshka-style (nested) latent queries**: during training, the model is optimized with varying prefix lengths at every training step so that earlier query tokens are trained more frequently.  
At inference time, we can use only the first *m* query tokens, enabling a controllable trade-off between performance and computational resources without retraining separate models.

### 3) Granularity-aware FFN (Adaptive Capacity)
Since the output is produced by a direct projection from the decoder representation, the decoder hidden size effectively acts as an information bottleneck.
MatGTM therefore fixes the representation dimension for stable decoding, and controls computation by scaling the FFN’s active width according to granularity—yielding a controllable accuracy–efficiency trade-off for optimization.

### (Optional) 4) GAF-based 2D Encoding for Time Series
Trend signals can be transformed into 2D representations (e.g., Gramian Angular Field) and encoded with a CNN backbone to enrich temporal pattern extraction.

---
## Repository Structure

- `src/matgtm/matgtm_v6.py`: main model definition
- `src/matgtm/modules.py`: submodules (optional split) (will be uploaded)

---
