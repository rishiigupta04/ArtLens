<div align="center">

<br/>

```
 █████╗ ██████╗ ████████╗██╗     ███████╗███╗   ██╗███████╗
██╔══██╗██╔══██╗╚══██╔══╝██║     ██╔════╝████╗  ██║██╔════╝
███████║██████╔╝   ██║   ██║     █████╗  ██╔██╗ ██║███████╗
██╔══██║██╔══██╗   ██║   ██║     ██╔══╝  ██║╚██╗██║╚════██║
██║  ██║██║  ██║   ██║   ███████╗███████╗██║ ╚████║███████║
╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
```

### Open-source · Explainable · Multi-generator AI Art Detection

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+cu121-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square&logo=react&logoColor=black)](https://reactjs.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-ffd21e?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br/>

**ArtLens** detects AI-generated images, identifies the generator, flags unknown models, and explains every decision with visual heatmaps — all free and open source.

[**Live Demo →**](https://artlens-beta.vercel.app/) · [**API Docs →**](https://rishigupta04-artlens.hf.space/docs) · [**Model Weights →**](https://huggingface.co/rishigupta04/ArtLens/)

<br/>

</div>

---

## Table of Contents

- [Why ArtLens](#-why-artlens)
- [What it does](#-what-it-does)
- [Results](#-results)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Training](#-training)
- [Explainability](#-explainability)
- [API Reference](#-api-reference)
- [Research foundations](#-research-foundations)
- [Tech stack](#-tech-stack)
- [Project structure](#-project-structure)
- [Getting started](#-getting-started)
- [Deployment](#-deployment)
- [Roadmap](#-roadmap)

---

## 🎯 Why ArtLens

AI image generators — Midjourney, Stable Diffusion, DALL-E 3, Adobe Firefly — now produce artwork indistinguishable from human-made art to the naked eye. This creates measurable harm:

| Domain | Problem |
|--------|---------|
| 🎨 Art contests | AI art submitted to human-only categories on ArtStation and DeviantArt |
| 📸 Stock photography | Getty, Shutterstock need to reject AI images sold under human-created licenses |
| 🎓 Academic integrity | Students submitting AI art as original portfolio work |
| 📰 Journalism | AI images used to fabricate events, protests, and disasters |

**Existing tools are paywalled and black-box.** Hive Moderation and AI or Not give a verdict but no explanation, and charge per-call. ArtLens is the first free, open-source detector that explains *which regions* triggered the verdict and flags generators it has never seen before.

---

## ✨ What it does

```
Upload any image
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                    ArtLens Pipeline                    │
│                                                        │
│  ViT-B/16 ──► Binary verdict   ──► Human / AI         │
│     │                                                  │
│     ├──────► Generator ID      ──► Midjourney / SD     │
│     │                                                  │
│     └──────► Open-set flag     ──► Known / Unknown     │
│                                                        │
│  FFT + DWT + LBP ──► LightGBM ──► Ensemble signal     │
│                                                        │
│  GradCAM++ ──────────────────► Visual explanation      │
└───────────────────────────────────────────────────────┘
```

- **Binary detection** — human vs AI with calibrated confidence
- **Generator fingerprinting** — identifies Midjourney vs Stable Diffusion
- **Open-set detection** — flags images from unseen generators (Firefly, DALL-E 3, etc.) rather than forcing a wrong label
- **Explainability** — GradCAM++ heatmaps show decision-critical regions; attention rollout shows global model focus
- **Frequency analysis** — FFT + DWT + LBP features detect upsampling artifacts invisible to the human eye
- **Free REST API** — `/predict`, `/batch`, `/explain` endpoints deployed on HuggingFace Spaces (ZeroGPU)

---

## 📊 Results

### Test set performance

| Model | Accuracy | AUROC | F1 | Notes |
|-------|----------|-------|----|-------|
| Phase 2 — ViT binary baseline | 98.20% | 0.9980 | 0.9820 | Head 1 only |
| Phase 3 — Multi-task (ours) | **98.77%** | **0.9975** | **0.9877** | Heads 1 + 2 + 3 |
| Phase 4 — LGB ensemble | 98.70% | 0.9889 | 0.9870 | ViT + frequency features |

### Per-source accuracy breakdown

| Source | Accuracy | AUROC |
|--------|----------|-------|
| WikiArt (human art) | 98.60% | — |
| Midjourney | **99.27%** | — |
| Stable Diffusion | 98.43% | — |

> Per-source breakdown follows the evaluation protocol from [Tasnim et al. 2024](https://arxiv.org/abs/2511.02791). Overall accuracy alone hides source-specific bias.

### Generator fingerprinting confusion matrix

```
                    Predicted
                  MJ      SD
Actual  MJ  │  945  │   15  │   98.44% correct
        SD  │   11  │  689  │   98.43% correct
```

### Ablation study — what each component contributes

| Component | AUROC | Delta vs full |
|-----------|-------|---------------|
| FFT features only (18-dim) | 0.8052 | −0.0837 |
| All handcrafted only (70-dim) | 0.9558 | −0.0331 |
| ViT embeddings only (768-dim) | 0.9880 | −0.0009 |
| **Combined ViT + HC (838-dim)** | **0.9889** | — |

> **Key finding:** The ViT backbone carries essentially all predictive signal (vit_57 SHAP=0.604, vit_530 SHAP=0.163). Handcrafted frequency features add marginal but consistent improvement and enable interpretable SHAP explanations.

---

## 🏗️ Architecture

### Overview

```
                         Input Image (224×224×3)
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
             ┌──────▼──────┐           ┌────────▼───────┐
             │  ViT-B/16   │           │  Freq. Branch  │
             │  Backbone   │           │                │
             │  85.8M par  │           │  FFT   18-dim  │
             │  768-dim    │           │  DWT   24-dim  │
             │  CLS token  │           │  LBP   28-dim  │
             └──────┬──────┘           └────────┬───────┘
                    │                           │
           ┌────────┼──────────────┐            │
           │        │              │            │
     ┌─────▼──┐ ┌───▼────┐ ┌──────▼─┐          │
     │ Head 1 │ │ Head 2 │ │ Head 3 │          │
     │        │ │        │ │        │          │
     │ Binary │ │  Gen.  │ │  Open  │          │
     │ Human  │ │  MJ vs │ │  Set   │          │
     │  vs AI │ │   SD   │ │ Maha.  │          │
     └─────┬──┘ └───┬────┘ └──────┬─┘          │
           │        │             │             │
           │        └─────────────┼─────────────┘
           │                      │
           │              ┌───────▼──────┐
           │              │  LightGBM   │
           │              │  838-dim    │
           │              │  Ensemble   │
           │              └───────┬──────┘
           │                      │
           └──────────┬───────────┘
                      │
               ┌──────▼──────┐
               │  Ensemble   │
               │  0.7×ViT +  │
               │  0.3×LGB    │
               └──────┬──────┘
                      │
            ┌─────────▼──────────┐
            │  Calibrated output │
            │  label · conf      │
            │  generator · flag  │
            │  heatmaps · text   │
            └────────────────────┘
```

### Multi-task learning — three heads on one backbone

The ViT-B/16 backbone is shared across all three heads, trained simultaneously with weighted multi-task loss:

```
Total loss = 1.0 × loss_binary + 0.5 × loss_generator
```

**Why multi-task?** Head 2 (generator fingerprinting) forces the backbone to encode generator-specific patterns — the subtle "fingerprint" each generator leaves in texture and frequency. These patterns also improve Head 1 generalisation. The joint gradient signal produces a richer backbone than any single task would achieve alone.

### Head 1 — Binary classification

```
768-dim embedding ──► LayerNorm ──► Dropout(0.3) ──► Linear(768→2) ──► [P(human), P(AI)]
```

Loss: `CrossEntropyLoss(label_smoothing=0.1)` — prevents overconfidence, improves calibration.

### Head 2 — Generator fingerprinting

```
768-dim embedding ──► LayerNorm ──► Dropout(0.3) ──► Linear(768→2) ──► [P(MJ), P(SD)]
```

Masked on human images (gen_label=−1). Only the AI subset contributes gradients to this head.

### Head 3 — Open-set detection (Mahalanobis)

After training, we compute per-class statistics on **28,000+ training embeddings** using the Ledoit-Wolf shrinkage estimator — critical because naïve covariance estimation is rank-deficient when n_samples ≈ n_dimensions (768 dims).

```
For each class c ∈ {human, midjourney, stable_diffusion}:
  μ_c    = mean embedding (768-dim)
  Σ_c⁻¹ = LedoitWolf precision matrix

At inference:
  d_c(x) = √[(x − μ_c)ᵀ Σ_c⁻¹ (x − μ_c)]

Normalised: z_c = (d_c − mean_dist_c) / std_dist_c

Flag unknown if: min(z_MJ, z_SD) > 3.0 σ
```

Normalised distances make cross-class comparison meaningful — raw Mahalanobis distances are not comparable between classes with different covariance structures.

### Frequency branch

| Feature | Dimensions | What it captures |
|---------|-----------|-----------------|
| FFT statistics | 18 | High-freq energy ratio, radial energy profile, spectral entropy, peak locations |
| DWT coefficients | 24 | Wavelet stats at 2 scales × 3 orientations (horizontal, vertical, diagonal) |
| LBP histogram | 28 | Micro-texture uniformity — AI art has unnaturally consistent LBP distributions |
| **Total** | **70** | |

All 70 features are concatenated with the 768-dim ViT embedding → 838-dim input to LightGBM.

---

## 🗂️ Dataset

### Composition

| Split | Human (WikiArt) | Midjourney | Stable Diffusion | Total |
|-------|----------------|------------|-----------------|-------|
| Train (75%) | 11,250 | ~5,600 | ~6,095 | ~22,945 |
| Val (15%) | 2,250 | ~1,120 | ~1,219 | ~4,589 |
| Test (10%) | 1,500 | 747 | 913 | 3,160 |
| **Total** | **15,000** | **~7,467** | **~8,227** | **~30,694** |

### WikiArt — stratified sampling

WikiArt contains 14 art style categories ranging from 1,798 to 15,000 images each. Random sampling would over-represent Abstract and Portrait (15k each) and under-represent Marina and Flower Painting (1.8k each). We apply **stratified sampling** with a per-category cap:

```
cap = min(TARGET / n_categories, min_category_size) ≈ 1,071 per category
```

This achieves Gini coefficient ≈ 0.00 (perfectly equal style representation).

| Category | Available | Sampled | Category | Available | Sampled |
|----------|-----------|---------|----------|-----------|---------|
| abstract | 14,999 | 1,071 | figurative | 4,500 | 1,071 |
| portrait | 14,999 | 1,071 | still-life | 2,996 | 1,071 |
| landscape | 15,000 | 1,071 | symbolic | 2,999 | 1,071 |
| genre-painting | 14,997 | 1,071 | nude | 3,000 | 1,071 |
| religious | 8,400 | 1,071 | mythological | 2,099 | 1,071 |
| cityscape | 6,598 | 1,071 | animal | 1,798 | 1,071 |
| — | — | — | flower | 1,800 | 1,071 |
| — | — | — | marina | 1,800 | 1,071 |

### Data quality pipeline

```
Raw images
    │
    ▼
PIL verify()           ── remove truncated / corrupt files
    │
    ▼
pHash deduplication    ── perceptual hash (hash_size=8, 64-bit)
(within-class)            robust to JPEG re-encoding and minor resize
    │
    ▼
Cross-class dedup      ── same image in both human and AI sets
                          would create contradictory training labels
    │
    ▼
Final 50/50 balance    ── undersample majority class
    │
    ▼
dataset_manifest.csv   ── path · label · source · generator
```

### Why not CIFAKE?

CIFAKE (CIFAR-10 vs SD v1.4) achieves 99%+ accuracy but measures a single generator's fingerprint — not the general human/AI distinction. Any reviewer familiar with this field will identify it as an invalid benchmark. ArtLens uses JourneyDB (Midjourney v5) and DiffusionDB (multiple SD versions) across the full aesthetic range of AI-generated art.

---

## 🏋️ Training

### Configuration

| Hyperparameter | Phase 2 | Phase 3 | Rationale |
|---------------|---------|---------|-----------|
| Backbone | ViT-B/16 | ViT-B/16 | 768-dim CLS, 86M params |
| Learning rate | 2e-5 | 1e-5 | Lower in P3 — backbone already converged |
| Batch size | 16 | 16 | RTX 3050 6GB VRAM limit |
| Epochs | 15 | 15 | Convergence observed by epoch 5 |
| Optimiser | AdamW | AdamW | Decoupled weight decay for transformers |
| Weight decay | 1e-4 | 1e-4 | L2 regularisation |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR | Per-batch, min_lr=1e-7 |
| Label smoothing | 0.1 | 0.1 (binary only) | Calibrated probabilities |
| Dropout | 0.3 | 0.3 | In classification heads |
| fp16 | ✅ | ✅ | `torch.amp.autocast('cuda')` |
| Grad. checkpointing | ✅ | ✅ | −30% VRAM, −20% speed |

### Augmentation strategy (Wang et al. 2020)

Directly from [CNNSpot (CVPR 2020)](https://arxiv.org/abs/1912.11035) — the most impactful single decision for generalisation:

```python
A.OneOf([
    # Social media JPEG compression suppresses AI frequency artifacts.
    # Training with compressed images forces the model to learn
    # robust lower-frequency signals that survive post-processing.
    A.ImageCompression(quality_range=(50, 95), p=1.0),

    # Gaussian blur simulates image downscaling and soft-focus.
    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
], p=0.5)
```

Without this augmentation, detectors fail on any post-processed image because they learn high-frequency artifacts that are suppressed by standard image sharing.

### Memory optimisation for 6GB VRAM

```python
# fp16 mixed precision — halves activation memory
with torch.amp.autocast('cuda'):
    logits_bin, logits_gen, emb = model(imgs)

# Gradient checkpointing — recomputes activations during backward
# instead of storing them — saves ~30% VRAM at ~20% compute cost
model.backbone.set_grad_checkpointing(enable=True)
```

### Training curves

```
Epoch │ Train Loss │ Train Acc │ Val Loss │ Val Acc │ Val AUROC
──────┼────────────┼───────────┼──────────┼─────────┼──────────
  1   │   0.2930   │  94.92%   │  0.2651  │  96.81% │  0.9971
  4   │   0.2226   │  98.83%   │  0.2299  │  98.57% │  0.9980  ← best
  8   │   0.2054   │  99.69%   │  0.2274  │  98.76% │  0.9962
 15   │   0.1990   │  99.98%   │  0.2252  │  98.90% │  0.9974
```

Train/val gap of ~1.1% at epoch 15 — healthy, no significant overfitting.

---

## 💡 Explainability

ArtLens provides three complementary explanation methods, each targeting a different layer of the decision process:

### 1. GradCAM++ — Decision-critical regions

```
Forward pass ──► class score
                     │
              Backward pass
                     │
           Gradients at blocks[-3]
           (3rd-from-last transformer block —
            more spatially grounded than final layer)
                     │
           GradCAM++ weighting:
           α = grad² / (2·grad² + Σ[acts·grad³] + ε)
                     │
           Weighted feature maps ──► ReLU ──► 14×14 heatmap
                     │
           Upsample to 224×224 ──► Overlay on image
```

Highlights **which specific regions** drove the AI/human verdict. Example: AI generators render hair with characteristic smoothness artifacts — GradCAM++ consistently activates on hair regions in AI portraits.

### 2. Attention Rollout — Global model focus

Uses **last-layer attention only** (not full rollout product across all 12 layers). Full rollout produces uniform maps for high-confidence fine-tuned models because the CLS token attends broadly when very certain. Last-layer attention retains spatial discriminability.

```python
# Last-layer attention — CLS token → all patches
attn_avg = model.backbone.blocks[-1].attn.attn_drop.output[0].mean(dim=0)
cls_attn = attn_avg[0, 1:].reshape(14, 14)   # [196] → [14, 14]
```

### 3. SHAP — Feature importance for frequency branch

TreeExplainer on LightGBM reveals which handcrafted features drove the ensemble prediction:

```
Top contributing features (mean |SHAP|):
  vit_57   ████████████████████████████████  0.604  (ViT embedding dim)
  vit_530  ████████████████                  0.163  (ViT embedding dim)
  vit_735  █                                 0.003
  fft_0    ▌                                 0.002  (high-freq energy ratio)
  dwt_8    ▌                                 0.001  (wavelet detail entropy)
```

**Key finding:** ViT embeddings dominate (vit_57 SHAP=0.604). Frequency features add marginal but consistent signal and crucially enable interpretable per-prediction explanations that the ViT alone cannot provide.

### Aggregate attention patterns

Computing average GradCAM++ across 200 test images reveals systematic differences:

```
Human art attention          AI art attention
┌──────────────────┐         ┌──────────────────┐
│  ░░░░░░░░░░░░░░░ │         │  ░░░░▓▓▓▓░░░░░░░ │
│  ░░░▓░░░░░▓░░░░░ │         │  ░░░░▓▓▓▓▓░░░░░░ │
│  ░░░░░▓░░░░░░░░░ │         │  ░░░░░▓▓░░░░░░░░ │  ← edges/transitions
│  ░░░░░░░░░░░░░░░ │         │  ░░░░░░░░░░░░░░░ │
│  ░▓░░░░░░░░░░▓░░ │         │  ░░░░░░░░░░░░░░░ │
└──────────────────┘         └──────────────────┘
Diffuse — painter's intent   Concentrated at texture
                             transitions and boundaries
```

AI images consistently activate at object-background transitions, hair boundaries, and sky regions — areas where upsampling artifacts accumulate.

---

## 🌐 API Reference

Base URL: `https://YOUR_USERNAME-artlens.hf.space`

### `POST /predict`

Fast single-image prediction (~1–2s). No heatmaps included — use `/explain` for visual explanations.

**Request:**
```bash
curl -X POST https://YOUR_USERNAME-artlens.hf.space/predict \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "request_id": "a3f9b2c1",
  "label": "ai",
  "confidence": 0.9481,
  "generator": {
    "name": "Midjourney",
    "confidence": 0.9177,
    "is_unknown": false,
    "closest_known": "Midjourney",
    "normalised_distances": {
      "Human (WikiArt)":    4.21,
      "Midjourney":         0.83,
      "Stable Diffusion":  11.47
    }
  },
  "model_signals": {
    "vit_probability":      0.9481,
    "lgb_probability":      0.6995,
    "ensemble_probability": 0.8735,
    "models_agree":         false
  },
  "explanation": {
    "summary": "This image is AI-generated, most likely by Midjourney (91.8% confidence).",
    "detail": "Both the deep visual model and classical frequency analysis independently confirmed AI generation.",
    "signals": {
      "deep_model_signal":      "strong",
      "frequency_model_signal": "moderate",
      "models_agree":           false
    }
  },
  "heatmap_urls": { "attention_rollout": null, "gradcam": null },
  "inference_time_s": 1.24,
  "model_version": "phase3_epoch7"
}
```

**Unknown generator example (Adobe Firefly):**
```json
{
  "label": "ai",
  "confidence": 0.71,
  "generator": {
    "name": null,
    "is_unknown": true,
    "closest_known": "Stable Diffusion",
    "normalised_distances": {
      "Human (WikiArt)":    6.84,
      "Midjourney":         4.12,
      "Stable Diffusion":   3.91
    }
  },
  "explanation": {
    "summary": "This image appears AI-generated, but the generator is not one the model was trained on.",
    "detail": "Closest known generator is Stable Diffusion, but normalised distance (3.9σ) exceeds the unknown threshold (3.0σ). This may be Adobe Firefly, DALL-E 3, or another unseen generator."
  }
}
```

---

### `POST /explain`

Full prediction + heatmap PNGs (~3–5s). Returns URLs to overlay images served at `/static/heatmaps/`.

```bash
curl -X POST https://YOUR_USERNAME-artlens.hf.space/explain \
  -F "file=@image.jpg"
```

Returns the same structure as `/predict` plus:
```json
{
  "heatmap_urls": {
    "attention_rollout": "/static/heatmaps/a3f9b2c1_rollout.png",
    "gradcam":           "/static/heatmaps/a3f9b2c1_gradcam.png"
  }
}
```

---

### `POST /batch`

Up to 20 images, no heatmaps. Returns predictions in input order.

```bash
curl -X POST https://YOUR_USERNAME-artlens.hf.space/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

---

### `GET /health`

```json
{
  "status": "ok",
  "device": "cuda",
  "models": {
    "vit": "phase3_epoch7",
    "lgb": "loaded",
    "maha": ["human", "midjourney", "stable_diffusion"],
    "temp": 1.0
  }
}
```

---

## 📄 Research Foundations

ArtLens directly implements methodology from five peer-reviewed papers. Every architectural decision is grounded in cited prior work.

---

### [1] Towards Universal Fake Image Detectors that Generalize Across Generative Models
**Ojha et al. — CVPR 2023** · [`arxiv.org/abs/2302.10174`](https://arxiv.org/abs/2302.10174)

> *"The real class becomes a sink class holding anything that is not fake, including generated images from models not accessible during training."*

**Core finding:** Standard classifiers trained on one generator fail on others because they latch onto generator-specific low-level artifacts. The fix: use the frozen feature space of CLIP-ViT — not trained for this task — which naturally separates real from AI images better than dedicated detectors (+15 mAP on unseen generators).

**What ArtLens implements:**
- ViT-B/16 backbone choice — validated by this paper's embedding space analysis
- Open-set detection framing (Head 3) — directly addresses the "sink class" problem
- UMAP embedding visualisation — replicates their t-SNE analysis showing natural cluster separation
- Cross-generator generalisation test in Phase 8 ablation

---

### [2] CNN-Generated Images Are Surprisingly Easy to Spot... For Now
**Wang et al. — CVPR 2020** · [`arxiv.org/abs/1912.11035`](https://arxiv.org/abs/1912.11035)

> *"With careful pre- and post-processing and data augmentation, a standard image classifier trained on only one specific CNN generator is able to generalize surprisingly well to unseen architectures."*

**Core finding:** JPEG compression and Gaussian blur augmentation during training is the single most impactful intervention for detector robustness. Without it, models learn high-frequency artifacts suppressed by normal image sharing.

**What ArtLens implements:**
- Direct copy of the augmentation strategy: `ImageCompression(quality_range=(50,95))` + `GaussianBlur(blur_limit=(3,7))` at `p=0.5`
- Applied in all three training phases (Phase 2, 3, and feature extraction)
- CNNSpot is the standard baseline comparison in the Phase 8 ablation study

---

### [3] AI-Generated Image Detection: An Empirical Study and Future Research Directions
**Tasnim et al. — arXiv 2024** · [`arxiv.org/abs/2511.02791`](https://arxiv.org/abs/2511.02791)

> *"These limitations hinder fair comparison, obscure true robustness, and restrict deployment in security-critical applications."*

**Core finding:** Three critical gaps in existing work: non-standardised benchmarks, inconsistent protocols, and evaluation metrics that fail to capture generalisation. Strong in-distribution performance regularly fails cross-generator.

**What ArtLens implements:**
- Multi-metric evaluation: AUROC + F1 + per-source accuracy breakdown (WikiArt / MJ / SD reported separately)
- Consistent evaluation protocol across all phases enables fair Phase 2 vs Phase 3 vs Phase 4 comparison
- GradCAM visualisation in evaluation — explicitly recommended by this paper
- The cross-generator degradation finding motivates Head 3 (open-set detection)

---

### [4] Methods and Trends in Detecting AI-Generated Images: A Comprehensive Review
**Mahara & Rishe — arXiv 2025** · [`arxiv.org/abs/2502.15176`](https://arxiv.org/abs/2502.15176)

> *"FatFormer integrates two modules... the Frequency Forgery Extractor employs Discrete Wavelet Transform and grouped attention mechanisms to dynamically aggregate multi-band frequency clues."*

**Core finding:** FatFormer — combining ViT attention with frequency domain features via DWT inside a unified architecture — achieves better generalisation than either alone. Validates dual-branch detection approaches.

**What ArtLens implements:**
- Phase 4 dual-branch architecture (ViT embeddings + FFT/DWT/LBP) directly mirrors FatFormer's design philosophy
- DWT decomposition at two scales (db2 wavelet, horizontal/vertical/diagonal sub-bands) follows the paper's approach
- Cited as prior art justifying the frequency branch in the project report and ablation discussion
- Used as the literature review foundation — covers the full method landscape from GAN detection to diffusion models

---

### [5] Detection of AI Generated Images Using Combined Uncertainty Measures
**Scientific Reports 2025** · [`nature.com/articles/s41598-025-28572-8`](https://www.nature.com/articles/s41598-025-28572-8)

> *"We focus on three complementary techniques... to decide whether to trust or reject a model's predictions."*

**Core finding:** Models should output "uncertain" rather than a confident wrong label when prediction confidence is low. Monte Carlo Dropout entropy + Gaussian Process variance combined via PSO gives robust uncertainty estimates — tested on Midjourney, BigGAN, StyleGAN3 with significant distribution shifts.

**What ArtLens implements:**
- Temperature scaling calibration (`calibrate.py`) — fits T via LBFGS NLL minimisation on val set
- Confidence penalty for unknown generators: `confidence = min(raw_conf, 0.75 - (z_score - 3.0) × 0.025)`
- `is_unknown_generator` flag in API response — explicit rejection rather than forced wrong label
- Normalised Mahalanobis distances provide a continuous uncertainty measure per class

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Backbone | `timm` ViT-B/16 | Feature extraction, 768-dim CLS embeddings |
| Training | PyTorch 2.5.1 + CUDA 12.1 | fp16 mixed precision, gradient checkpointing |
| Classical features | PyWavelets, OpenCV, scikit-image | FFT, DWT, LBP extraction |
| Ensemble | LightGBM | Gradient boosting on 838-dim combined features |
| Explainability | Custom GradCAM++, Attention Rollout, SHAP | Three-method explanation pipeline |
| Calibration | scikit-learn (LedoitWolf), scipy (LBFGS) | Precision matrix + temperature scaling |
| Experiment tracking | Weights & Biases | Loss curves, AUROC per epoch |
| API | FastAPI 0.115 + uvicorn | Async inference server with static file serving |
| Model hosting | HuggingFace Hub + Spaces (ZeroGPU) | Free GPU inference, permanent URL |
| Frontend | React 18 + Vite + Tailwind CSS | Drag-drop upload, heatmap toggle, results display |
| Frontend hosting | Vercel | Free CDN deployment |
| Database | Supabase (PostgreSQL) | Prediction logging, community flags |
| Package manager | uv | Dependency resolution, virtual environment |

---

## 📁 Project Structure

```
ArtLens/
│
├── data/
│   ├── human/                    # WikiArt images (flat, stratified 1071/style)
│   ├── ai/
│   │   ├── mj/                   # JourneyDB — Midjourney v5
│   │   └── sd/                   # DiffusionDB — Stable Diffusion
│   └── dataset_manifest.csv      # path · label · source · generator
│
├── artlens/                      # Training & inference code
│   ├── train.py                  # Phase 2: ViT binary baseline
│   ├── train_phase3.py           # Phase 3: multi-task (3 heads)
│   ├── train_phase4.py           # Phase 4: frequency features + LightGBM
│   ├── explain_phase5.py         # Phase 5: heatmap generation pipeline
│   ├── calibrate.py              # Phase 6: temperature scaling
│   ├── recompute_mahalanobis.py  # Recompute open-set stats on training set
│   ├── app.py                    # FastAPI inference server (deployed on HF Spaces)
│   ├── push_to_hub.py            # Upload model weights to HF Hub
│   ├── Dockerfile                # HuggingFace Spaces container
│   └── requirements_spaces.txt  # Production dependencies
│
├── outputs/
│   ├── models/
│   │   ├── best_model.pt                  # Phase 2 weights
│   │   ├── phase3_best_model.pt           # Phase 3 weights (primary)
│   │   ├── phase3_mahalanobis_stats.npz   # Open-set class statistics
│   │   ├── lgb_model.txt                  # LightGBM model
│   │   ├── feature_scaler.pkl             # StandardScaler (fit on train)
│   │   ├── umap_reducer.pkl               # 768→2 dim UMAP
│   │   └── temperature.json              # Calibration temperature T
│   ├── embeddings/
│   │   ├── phase3_embeddings.npz          # (31594, 768) cached ViT embeddings
│   │   └── handcrafted_features.npz       # (31594, 70) FFT+DWT+LBP
│   ├── results/
│   │   ├── explanations/                  # Per-image GradCAM++ + rollout grids
│   │   ├── phase4_ablation.png
│   │   ├── phase4_umap.png
│   │   └── phase5_methods_comparison.png
│   └── split_train/val/test.csv           # Fixed splits — same across all phases
│
└── frontend/                     # React application
    ├── src/
    │   ├── pages/                # Home · Batch · About
    │   ├── components/           # UploadZone · ResultCard · HeatmapViewer · ...
    │   └── lib/                  # api.js · supabase.js
    ├── package.json
    └── vite.config.js
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- CUDA-capable GPU (tested on RTX 3050 6GB) or Kaggle T4
- [uv](https://docs.astral.sh/uv/) package manager
- Node.js 18+ (for frontend)

### Local setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/artlens
cd artlens/artlens

# Set up Python environment with uv
uv python pin 3.11
uv add torch torchvision --default-index https://download.pytorch.org/whl/cu121
uv add timm albumentations wandb scikit-learn pandas matplotlib pillow \
       imagehash tqdm opencv-python-headless lightgbm shap umap-learn \
       grad-cam joblib scipy PyWavelets scikit-image fastapi uvicorn

# Verify GPU is visible
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Download data

| Dataset | Images | Source | Label |
|---------|--------|--------|-------|
| WikiArt | ~15,000 | [Kaggle](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan) | Human (0) |
| JourneyDB | ~7,500 | [HuggingFace](https://huggingface.co/datasets/JourneyDB/JourneyDB) | AI — Midjourney (1) |
| DiffusionDB | ~8,000 | [HuggingFace](https://huggingface.co/datasets/poloclub/diffusiondb) | AI — Stable Diffusion (1) |

Place images in `data/human/`, `data/ai/mj/`, `data/ai/sd/` (flat, any filenames).

### Run training

```bash
# Phase 2 — Binary baseline
uv run python train.py

# Phase 3 — Multi-task (loads Phase 2 weights)
uv run python train_phase3.py

# Phase 4 — Frequency features + LightGBM
uv add lightgbm shap umap-learn
uv run python train_phase4.py

# Phase 5 — Explainability
uv run python explain_phase5.py

# Phase 6 — Calibration
uv run python calibrate.py
uv run python recompute_mahalanobis.py
```

### Resume from checkpoint

```python
# In any training script, set in CFG:
'resume_from': 'phase3_epoch_07_auroc0.9975.pt'
```

All checkpoints save full state: `model_state`, `optim_state`, `scheduler_state`, `scaler_state`.

### Run API locally

```bash
cd artlens
uv run python app.py
# API available at http://localhost:7860
# Swagger docs at http://localhost:7860/docs
```

### Run frontend locally

```bash
cd frontend
npm install

# Create .env.local
echo "VITE_API_URL=http://localhost:7860" > .env.local
echo "VITE_SUPABASE_URL=your_url" >> .env.local
echo "VITE_SUPABASE_ANON_KEY=your_key" >> .env.local

npm run dev
# Frontend at http://localhost:5173
```

---

## ☁️ Deployment

### Model weights → HuggingFace Hub

```bash
# Login to HuggingFace
uv run python -c "from huggingface_hub import login; login()"

# Push all model files
uv run python push_to_hub.py
```

### API → HuggingFace Spaces

```bash
# Clone your Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/artlens
cd artlens-space

# Copy deployment files
cp ../artlens/artlens/app.py .
cp ../artlens/artlens/Dockerfile .
cp ../artlens/artlens/requirements_spaces.txt .

# Edit app.py — set HF_REPO_ID = "YOUR_USERNAME/artlens"

git add . && git commit -m "deploy" && git push
# Builds automatically (~5 minutes)
```

### Frontend → Vercel

```bash
cd frontend
npm run build
vercel --prod

# Set environment variables in Vercel dashboard:
# VITE_API_URL     = https://YOUR_USERNAME-artlens.hf.space
# VITE_SUPABASE_URL     = https://YOUR_PROJECT.supabase.co
# VITE_SUPABASE_ANON_KEY = your_anon_key
```

### Supabase tables

```sql
create table predictions (
  id               uuid default gen_random_uuid() primary key,
  created_at       timestamptz default now(),
  image_hash       text,
  label            text,
  confidence       float,
  generator        text,
  is_unknown_gen   boolean,
  vit_prob         float,
  lgb_prob         float,
  ensemble_prob    float,
  inference_time_s float
);

create table community_flags (
  id          uuid default gen_random_uuid() primary key,
  created_at  timestamptz default now(),
  image_hash  text,
  model_label text,
  user_claim  text,
  notes       text
);

-- Allow anonymous inserts from the frontend
alter table predictions     enable row level security;
alter table community_flags enable row level security;
create policy "anon insert" on predictions     for insert to anon with check (true);
create policy "anon insert" on community_flags for insert to anon with check (true);
```


## 🙏 Acknowledgements

| Paper | Authors | Contribution to ArtLens |
|-------|---------|------------------------|
| [UnivFD](https://arxiv.org/abs/2302.10174) | Ojha, Li, Lee | ViT backbone + open-set problem framing |
| [CNNSpot](https://arxiv.org/abs/1912.11035) | Wang et al. | JPEG + blur augmentation strategy |
| [Benchmark Study](https://arxiv.org/abs/2511.02791) | Tasnim et al. | Multi-metric evaluation protocol |
| [Survey](https://arxiv.org/abs/2502.15176) | Mahara & Rishe | FatFormer dual-branch validation |
| [Uncertainty](https://www.nature.com/articles/s41598-025-28572-8) | Anonymous | Calibration + uncertainty rejection |

Dataset credits: [WikiArt](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan) · [JourneyDB](https://huggingface.co/datasets/JourneyDB/JourneyDB) · [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb)

---

<div align="center">

Crafted with ❤️‍ by **Rishi**

*Free to use, fork, and improve.*

</div>
