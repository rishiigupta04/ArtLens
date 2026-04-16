---
title: ArtLens
emoji: 🎨
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# ArtLens — AI Art Detection API

Detects AI-generated images, identifies the generator (Midjourney vs Stable Diffusion),
and explains decisions via attention rollout + GradCAM++ heatmaps.

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Model status |
| POST | /predict | Single image — full prediction + heatmaps |
| POST | /batch | Up to 20 images — predictions only |

## /predict response
```json
{
  "label": "ai",
  "confidence": 0.9923,
  "is_ai": true,
  "generator": "Midjourney",
  "generator_confidence": 0.9871,
  "is_unknown_generator": false,
  "vit_probability": 0.9941,
  "lgb_probability": 0.9872,
  "ensemble_probability": 0.9920,
  "attention_rollout_heatmap": "<base64 PNG>",
  "gradcam_heatmap": "<base64 PNG>",
  "inference_time_s": 1.24
}
```

## Model
- ViT-B/16 fine-tuned on WikiArt (human) + JourneyDB (Midjourney) + DiffusionDB (SD)
- Multi-task: binary detection + generator fingerprinting + open-set detection
- Test AUROC: 0.9975 | Accuracy: 98.77% | Generator accuracy: 98.43%