# MatGTM (Architecture)

This repository provides the **model architecture** of my M.S. thesis work: **MatGTM** from *"A Matryoshka-Based Architecture for Multimodal Fashion Sales Forecasting using Adaptive Queries and Decoder"*, a multimodal forecasting model designed to capture meaningful patterns from trend signals and product modalities with efficient inference.

> This repo focuses on **architecture**.  
> Training/Evaluation pipeline will be added in a future update.

---

## What is MatGTM?

MatGTM is a modified architecture based on **GTM-Transformer** from *“Well Googled is Half Done: Multimodal Forecasting of New Fashion Product Sales with Image-based Google Trends”*.  
The Architecture is a Trasnformer-based model that performs Time-Series Forecasting by combining:
- **Time-series signals**
- Optional modalities (image / text embeddings / etc..)
- **Matryoshka-style latent queries (prefix slicing)** that summarize time-series through cross-attention, capture trend patterns efficiently, balance accuracy and efficiency

## Architecture
<img src="assets/matgtm architecture.png" width="700" />
Architecture of the proposed MatGTM model. The encoder (although this component is not a conventional encoder, we refer to it as the “encoder” for convenience to distinguish it from the decoder) performs cross-attention between hierarchical latent queries and CNN-extracted features from GAF-transformed Time-Series Data (Google Trends). The decoder employs a granularity-aware feed-forward network to adaptively control model capacity when generating sales forecasts.

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

## Limitations

- **GAF preprocessing and generalization.** The model relies on Gramian Angular Field (GAF) transformation, which adds an extra preprocessing step and may not transfer well to time-series with irregular sampling or very different signal characteristics.

- **Fixed-size 2D encoding can lose temporal detail.** Converting time series into fixed-resolution 2D images can reduce temporal resolution and may cause information loss for highly dynamic or sparse trend patterns.

- **Partial opacity of latent-query behavior.** While latent queries improve efficiency and offer interpretability via attention, their internal behavior can remain hard to characterize—especially under different query sparsity levels or across diverse product categories.

## Datset

**VISUELLE** dataset is available to download [here](https://docs.google.com/forms/d/e/1FAIpQLSchN_0VzFD5YEY6MET8V91xyEZLuiiN5jeACP5Mcn-4bYh_lQ/viewform). Download and extract it inside the dataset folder.

## Training

Training

```bash
python run_train.py --data_folder dataset \
  --trend_len 52 \
  --num_trends 3 \
  --M 52 \
  --gaf True \
  --wandb_api_key "YOUR_WANDB_API_KEY" \
  --wandb_entity "YOUR_ENTITY" \
  --wandb_proj "YOUR_PROJECT" \
```

 trend_len : length of Input Time-Series Dataset
 num_trends : Channel Dim of Input Time_Series Dataset
 gaf : True/False for using GAF


---
## Repository Structure

- `src/matgtm/matgtm_v6.py`: main model definition
- `src/matgtm/modules.py`: submodules (optional split) (will be uploaded)

---

## Cross-domain (BDG2)

Beyond the scope of my thesis (fashion demand forecasting), I hypothesized that MatGTM’s core design—**latent-query cross-attention** with an efficiency-friendly **Matryoshka prefix scheme**—could generalize to other multivariate forecasting problems.  
To validate this, I adapted MatGTM to **BDG2 (Building Data Genome 2)(https://github.com/buds-lab/building-data-genome-project-2)** for building electricity usage forecasting.

This cross-domain study suggests that MatGTM can also support **event- and anomaly-centric analysis**: once peak-like events are detected in the predicted/observed trajectories, we can trace them back to influential historical windows using **attention scores between latent queries and inputs**, providing practical insights for **root-cause analysis** and factor investigation.

See: [docs/CROSS_DOMAIN_BDG2.md](docs/CROSS_DOMAIN_BDG2.md)
