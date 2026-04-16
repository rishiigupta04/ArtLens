# ── ArtLens Phase 5 — Explainability ─────────────────────────────────────────
# Generates attention rollout, GradCAM++, and SHAP explanations
# Produces heatmap overlays for a sample of test images
# Saves all visualisations to D:\ArtLens\outputs\results\explanations\
# Run with: uv run python explain_phase5.py

import os, random, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from pathlib import Path
from tqdm import tqdm
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

import lightgbm as lgb
import shap

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path('D:/ArtLens/data')
OUTPUT_DIR   = Path('D:/ArtLens/outputs')
MANIFEST     = DATA_DIR / 'dataset_manifest.csv'
MODELS_DIR   = OUTPUT_DIR / 'models'
RESULTS_DIR  = OUTPUT_DIR / 'results'
EXPLAIN_DIR  = RESULTS_DIR / 'explanations'   # all explanation images go here
EMBED_DIR    = OUTPUT_DIR / 'embeddings'

for d in [EXPLAIN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CFG = {
    'img_size'       : 224,
    'n_explain'      : 20,     # number of test images to explain
    'lbp_radius'     : 3,
    'lbp_n_points'   : 24,
    'rollout_mode'   : 'last_layer_only',  # full_rollout | last_layer_only
    'gradcam_block_offset': -3,             # prefer spatially grounded block
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)


# ── Import model class from phase 3 script ───────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_phase3 import ArtLensMultiTaskModel, GEN_TO_IDX, IDX_TO_GEN


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(img_path: str) -> tuple:
    """
    Loads one image and returns:
    - tensor: normalised [1, 3, 224, 224] tensor ready for model
    - img_rgb: original PIL Image (for overlay)
    - img_np: numpy [224, 224, 3] uint8 (for OpenCV operations)
    """
    img_pil = Image.open(img_path).convert('RGB')
    img_np  = np.array(img_pil.resize((CFG['img_size'], CFG['img_size'])))

    transform = A.Compose([
        A.Resize(CFG['img_size'], CFG['img_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    tensor = transform(image=img_np)['image'].unsqueeze(0)  # [1, 3, 224, 224]
    return tensor, img_pil.resize((CFG['img_size'], CFG['img_size'])), img_np


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts normalised tensor back to uint8 numpy image for display.
    """
    img = tensor.squeeze(0).cpu() * IMAGENET_STD + IMAGENET_MEAN
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1: ATTENTION ROLLOUT
# ══════════════════════════════════════════════════════════════════════════════
class AttentionRollout:
    """
    Computes attention rollout for ViT-B/16.

    Algorithm (Abnar & Zuidema 2020):
    1. For each transformer layer, get the attention weight matrix [heads, patches+1, patches+1]
    2. Average over heads
    3. Add identity matrix (residual connection) and renormalise
    4. Multiply matrices across all layers — this propagates attention flow
       from input patches all the way to the CLS token

    Result: a 14×14 map showing how much each input patch contributed
    to the final CLS token representation — the model's 'visual focus'.
    """

    def __init__(self, model: nn.Module, mode: str = 'last_layer_only'):
        self.model    = model
        self.mode     = mode
        self.attns    = []    # stores attention weights from each layer
        self.hooks    = []

    def _hook_fn(self, module, input, output):
        """
        Hook attached to each transformer attention block.
        ViT attention returns (output, attention_weights) when
        attn_drop is not None — we capture the weights here.
        """
        # timm ViT attention module stores weights in .attn_drop output
        # We hook the attention matrix directly
        self.attns.append(output.detach().cpu())

    def register_hooks(self):
        """
        Attach forward hooks to all 12 attention layers in the ViT backbone.
        Hooks fire during forward pass and store attention matrices.
        """
        for block in self.model.backbone.blocks:
            # Hook the attention softmax output inside each block
            hook = block.attn.attn_drop.register_forward_hook(self._hook_fn)
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attns = []

    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Runs forward pass with hooks, computes rollout heatmap.
        Returns 14×14 numpy array (one value per image patch).
        """
        self.attns = []
        self.register_hooks()

        self.model.eval()
        with torch.no_grad():
            with autocast('cuda'):
                _ = self.model(tensor.to(DEVICE))

        self.remove_hooks()

        if not self.attns:
            return np.ones((14, 14)) / 196   # fallback if hooks failed

        if self.mode == 'last_layer_only':
            # Last-layer CLS attention is usually less diffuse than full rollout
            last_attn = self.attns[-1][0].mean(dim=0)
            cls_attn = last_attn[0, 1:]
            heatmap = cls_attn.reshape(14, 14).numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
            return heatmap

        # ── Rollout computation ───────────────────────────────────────────────
        # Start with identity: each token attends only to itself
        n_tokens  = self.attns[0].shape[-1]          # 197 = 196 patches + 1 CLS
        rollout   = torch.eye(n_tokens)

        for attn in self.attns:
            # attn shape: [batch, heads, tokens, tokens]
            # Average over heads to get [tokens, tokens]
            attn_avg = attn[0].mean(dim=0)            # [197, 197]

            # Add identity for residual connection, normalise rows
            attn_aug = attn_avg + torch.eye(n_tokens)
            attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)

            # Propagate: rollout = current_layer @ previous_rollout
            rollout  = attn_aug @ rollout

        # Extract CLS token row (index 0) — its attention to all patches
        cls_attn = rollout[0, 1:]                     # [196] — skip CLS self-attention

        # Reshape to 14×14 grid (14 = 224/16 patches per side)
        heatmap  = cls_attn.reshape(14, 14).numpy()
        heatmap  = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)

        return heatmap


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2: GRADCAM++
# ══════════════════════════════════════════════════════════════════════════════
class GradCAMPlusPlus:
    """
    GradCAM++ for ViT-B/16.

    Unlike CNNs where GradCAM targets a conv layer, for ViT we target
    the output of the last transformer block's attention layer.
    This gives a spatial heatmap showing which regions most influenced
    the final classification decision.

    GradCAM++ improvement over GradCAM: uses second-order gradients
    for better localisation when multiple instances of the same class
    appear, and more precise attribution.
    """

    def __init__(self, model: nn.Module, target_layer=None, block_offset: int = -3):
        self.model        = model
        self.gradients    = None
        self.activations  = None

        # Prefer an earlier block for less sparse, more spatially spread maps.
        if target_layer is None:
            self.target_layer = self._resolve_target_layer(block_offset)
        else:
            self.target_layer = target_layer

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Store the activations at the target layer
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Store the gradients flowing back through the target layer
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _resolve_target_layer(self, block_offset: int):
        blocks = self.model.backbone.blocks
        n_blocks = len(blocks)

        preferred = n_blocks + block_offset if block_offset < 0 else block_offset
        candidates = [preferred, n_blocks - 2, n_blocks - 1]
        for idx in candidates:
            if 0 <= idx < n_blocks:
                return blocks[idx].norm1

        return blocks[-1].norm1

    def __call__(self, tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Computes GradCAM++ heatmap for the given class index.
        If class_idx is None, uses the predicted class.
        Returns 14×14 numpy heatmap normalised to [0, 1].
        """
        self.model.eval()
        tensor = tensor.to(DEVICE)
        tensor.requires_grad_(True)

        # Forward pass — must NOT use autocast here (needs float32 for gradients)
        logits_bin, _, _ = self.model(tensor)

        if class_idx is None:
            class_idx = logits_bin.argmax(dim=1).item()

        # Zero gradients, backward on target class score
        self.model.zero_grad()
        score = logits_bin[0, class_idx]
        score.backward()

        # activations: [1, 197, 768] — one vector per patch + CLS token
        # gradients:   [1, 197, 768] — gradient of class score w.r.t. activations
        acts  = self.activations[0, 1:, :]   # [196, 768] — remove CLS token
        grads = self.gradients[0, 1:, :]     # [196, 768]

        # GradCAM++ weighting:
        # alpha_c_k = grad^2 / (2*grad^2 + sum(acts*grad^3) + eps)
        grads_sq   = grads ** 2
        grads_cub  = grads ** 3
        alpha      = grads_sq / (2 * grads_sq + (acts * grads_cub).sum(dim=0, keepdim=True) + 1e-10)

        # Weighted combination: sum over feature dimension
        weights    = (alpha * F.relu(grads)).sum(dim=1)  # [196]

        # Reshape to 14×14 spatial grid
        heatmap    = weights.reshape(14, 14).cpu().detach().numpy()
        heatmap    = np.maximum(heatmap, 0)              # ReLU — keep positive
        heatmap    = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)

        return heatmap


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP OVERLAY UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def heatmap_to_overlay(heatmap: np.ndarray, img_np: np.ndarray,
                        colormap=cv2.COLORMAP_JET, alpha=0.45) -> np.ndarray:
    """
    Resizes 14×14 heatmap to 224×224, applies colourmap, overlays on image.

    alpha controls transparency: 0 = only image, 1 = only heatmap.
    COLORMAP_JET: blue (low) → green → red (high attention/importance).

    Returns uint8 numpy array [224, 224, 3].
    """
    # Resize to full image size
    heatmap_full = cv2.resize(heatmap.astype(np.float32), (CFG['img_size'], CFG['img_size']))

    # Apply colourmap — cv2 expects uint8 [0,255]
    heatmap_uint8 = (heatmap_full * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend with original image
    img_float     = img_np.astype(np.float32)
    heat_float    = heatmap_color.astype(np.float32)
    overlay       = (1 - alpha) * img_float + alpha * heat_float
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_explanation_grid(img_np, rollout_heatmap, gradcam_heatmap,
                           pred_label, true_label, confidence,
                           generator_pred, img_path, save_path):
    """
    Saves a 1×4 grid for one image:
    [Original] [Attention Rollout overlay] [GradCAM++ overlay] [Side-by-side comparison]
    """
    rollout_overlay = heatmap_to_overlay(rollout_heatmap, img_np, cv2.COLORMAP_JET)
    gradcam_overlay = heatmap_to_overlay(gradcam_heatmap, img_np, cv2.COLORMAP_INFERNO)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('#1a1a1a')

    pred_color = '#e15759' if pred_label == 1 else '#4e79a7'
    correct    = pred_label == true_label

    # ── Panel 1: Original image ───────────────────────────────────────────────
    axes[0].imshow(img_np)
    axes[0].axis('off')
    axes[0].set_title(
        f"Original\nTrue: {'AI' if true_label==1 else 'Human'}",
        color='white', fontsize=11
    )

    # ── Panel 2: Attention rollout ────────────────────────────────────────────
    axes[1].imshow(rollout_overlay)
    axes[1].axis('off')
    axes[1].set_title(
        "Attention Rollout\n(secondary global context)",
        color='#76b7b2', fontsize=11
    )

    # ── Panel 3: GradCAM++ ────────────────────────────────────────────────────
    axes[2].imshow(gradcam_overlay)
    axes[2].axis('off')
    axes[2].set_title(
        "GradCAM++\n(decision-critical regions)",
        color='#f28e2b', fontsize=11
    )

    # ── Panel 4: Verdict panel ────────────────────────────────────────────────
    axes[3].imshow(img_np)
    axes[3].axis('off')

    verdict     = "AI" if pred_label == 1 else "Human"
    correct_str = "✓ Correct" if correct else "✗ Wrong"
    gen_str     = f"\nGenerator: {generator_pred}" if pred_label == 1 else ""

    axes[3].set_title(
        f"Verdict: {verdict} ({confidence*100:.1f}%)\n{correct_str}{gen_str}",
        color=pred_color, fontsize=11, fontweight='bold'
    )

    plt.suptitle(
        Path(img_path).name[:40],
        color='#aaaaaa', fontsize=9, y=0.02
    )
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    print(f"Device : {DEVICE}")

    # ── Load manifest + remap paths ───────────────────────────────────────────
    df = pd.read_csv(MANIFEST)

    def remap_path(row):
        fname = Path(row['path']).name
        if row['label'] == 0:
            return str(DATA_DIR / 'human' / fname)
        else:
            folder = 'mj' if row['generator'] == 'midjourney' else 'sd'
            return str(DATA_DIR / 'ai' / folder / fname)

    df['path'] = df.apply(remap_path, axis=1)
    df = df[df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    # Load test split
    test_df = pd.read_csv(OUTPUT_DIR / 'split_test.csv')
    test_df['path'] = test_df.apply(remap_path, axis=1)
    test_df = test_df[test_df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    print(f"Test set: {len(test_df):,} images")

    # ── Load Phase 3 model ────────────────────────────────────────────────────
    print("\nLoading Phase 3 model...")
    model = ArtLensMultiTaskModel().to(DEVICE)
    ckpt  = torch.load(MODELS_DIR / 'phase3_best_model.pt',
                       map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Model loaded (epoch {ckpt['epoch']}, val AUROC {ckpt['val_auroc']:.4f})")

    # ── Load LightGBM + scaler ────────────────────────────────────────────────
    print("Loading LightGBM + scaler...")
    lgb_model = lgb.Booster(model_file=str(MODELS_DIR / 'lgb_model.txt'))
    scaler    = joblib.load(MODELS_DIR / 'feature_scaler.pkl')

    # ── Initialise explainers ─────────────────────────────────────────────────
    print("Initialising explainers...")
    rollout_explainer = AttentionRollout(model, mode=CFG['rollout_mode'])
    gradcam_explainer = GradCAMPlusPlus(model, block_offset=CFG['gradcam_block_offset'])
    print(f"  Attention rollout ready (mode={CFG['rollout_mode']})")
    print(f"  GradCAM++ ready (block_offset={CFG['gradcam_block_offset']})")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Generate per-image explanations for sample of test images
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nGenerating explanations for {CFG['n_explain']} test images...")

    # Select a balanced sample: 5 correct-human, 5 correct-AI-MJ,
    # 5 correct-AI-SD, 5 wrong predictions (hard cases)
    sample_rows = []

    # Quick inference pass to find correct/wrong predictions
    print("  Running quick inference to identify interesting cases...")
    all_preds  = []
    all_confs  = []
    all_gen_preds = []

    model.eval()
    transform = A.Compose([
        A.Resize(CFG['img_size'], CFG['img_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="  Quick inference"):
        try:
            img   = np.array(Image.open(row['path']).convert('RGB'))
            tens  = transform(image=img)['image'].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with autocast('cuda'):
                    lb, lg, _ = model(tens)
            prob  = torch.softmax(lb, dim=1)[0, 1].item()
            pred  = int(prob >= 0.5)
            gp    = lg.argmax(dim=1).item()
            all_preds.append(pred)
            all_confs.append(prob if pred == 1 else 1 - prob)
            all_gen_preds.append(gp)
        except Exception:
            all_preds.append(0)
            all_confs.append(0.5)
            all_gen_preds.append(0)

    test_df = test_df.copy()
    test_df['pred']     = all_preds
    test_df['conf']     = all_confs
    test_df['gen_pred'] = all_gen_preds
    test_df['correct']  = (test_df['pred'] == test_df['label']).astype(int)

    # Sample selection — balanced across categories
    def sample_category(mask, n, desc):
        subset = test_df[mask]
        if len(subset) == 0:
            print(f"  Warning: no images found for {desc}")
            return pd.DataFrame()
        return subset.sample(min(n, len(subset)), random_state=SEED)

    samples = pd.concat([
        sample_category(
            (test_df['label']==0) & (test_df['correct']==1), 5, "correct human"),
        sample_category(
            (test_df['label']==1) & (test_df['source']=='midjourney') & (test_df['correct']==1), 5, "correct MJ"),
        sample_category(
            (test_df['label']==1) & (test_df['source']=='stable_diffusion') & (test_df['correct']==1), 5, "correct SD"),
        sample_category(
            (test_df['correct']==0), 5, "wrong predictions"),
    ]).reset_index(drop=True)

    print(f"  Selected {len(samples)} images for explanation")

    # ── Generate explanation grid for each selected image ────────────────────
    explanation_log = []

    for i, row in tqdm(samples.iterrows(), total=len(samples), desc="Generating explanations"):
        try:
            tensor, img_pil, img_np = preprocess(row['path'])

            # Attention rollout
            rollout_hm = rollout_explainer(tensor)

            # GradCAM++ — use predicted class
            gradcam_hm = gradcam_explainer(tensor, class_idx=int(row['pred']))

            # Generator prediction string
            gen_str = IDX_TO_GEN.get(int(row['gen_pred']), 'Unknown') \
                      if row['pred'] == 1 else 'N/A'

            # Save grid
            fname = f"explain_{i:03d}_{'AI' if row['label']==1 else 'Human'}_{'correct' if row['correct'] else 'WRONG'}.png"
            save_path = EXPLAIN_DIR / fname

            save_explanation_grid(
                img_np        = img_np,
                rollout_heatmap = rollout_hm,
                gradcam_heatmap = gradcam_hm,
                pred_label    = int(row['pred']),
                true_label    = int(row['label']),
                confidence    = float(row['conf']),
                generator_pred = gen_str,
                img_path      = row['path'],
                save_path     = save_path
            )

            explanation_log.append({
                'index'         : i,
                'path'          : row['path'],
                'true_label'    : int(row['label']),
                'pred_label'    : int(row['pred']),
                'confidence'    : float(row['conf']),
                'correct'       : bool(row['correct']),
                'source'        : row['source'],
                'generator_pred': gen_str,
                'rollout_mode'  : CFG['rollout_mode'],
                'gradcam_block_offset': CFG['gradcam_block_offset'],
                'saved_to'      : str(save_path)
            })

        except Exception as e:
            print(f"  ⚠️  Failed on image {i}: {e}")
            continue

    print(f"  {len(explanation_log)} explanation grids saved → {EXPLAIN_DIR}")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: SHAP explanation for LightGBM — per-image breakdown
    # ══════════════════════════════════════════════════════════════════════════
    print("\nGenerating SHAP explanations for LightGBM branch...")

    # Load cached embeddings and handcrafted features
    emb_cache = np.load(EMBED_DIR / 'phase3_embeddings.npz')['embeddings']
    hc_cache  = np.load(EMBED_DIR / 'handcrafted_features.npz')['features']

    all_df = pd.concat([
        pd.read_csv(OUTPUT_DIR / 'split_train.csv'),
        pd.read_csv(OUTPUT_DIR / 'split_val.csv'),
        pd.read_csv(OUTPUT_DIR / 'split_test.csv')
    ]).reset_index(drop=True)

    n_train = len(pd.read_csv(OUTPUT_DIR / 'split_train.csv'))
    n_val   = len(pd.read_csv(OUTPUT_DIR / 'split_val.csv'))

    # Get test set features
    X_test_raw = np.concatenate([
        emb_cache[n_train+n_val:],
        hc_cache[n_train+n_val:]
    ], axis=1)
    X_test_scaled = scaler.transform(X_test_raw)

    # Feature names
    n_lbp    = hc_cache.shape[1] - 42   # total HC - FFT(18) - DWT(24)
    feat_names = (
        [f'vit_{i}'   for i in range(768)] +
        [f'fft_{i}'   for i in range(18)]  +
        [f'dwt_{i}'   for i in range(24)]  +
        [f'lbp_{i}'   for i in range(n_lbp)]
    )

    # SHAP explainer
    explainer   = shap.TreeExplainer(lgb_model)

    # Per-image SHAP waterfall for 3 interesting cases
    sample_indices = samples[samples['correct'] == 0].index[:3].tolist()
    if len(sample_indices) < 3:
        sample_indices = list(range(min(3, len(samples))))

    for plot_idx, idx in enumerate(sample_indices):
        row = samples.iloc[idx]

        # Find this image's index in all_df to get its features
        try:
            all_df_mapped = all_df.copy()
            all_df_mapped['path'] = all_df_mapped.apply(remap_path, axis=1)
            match = all_df_mapped[all_df_mapped['path'] == row['path']]

            if len(match) == 0:
                continue

            feat_idx    = match.index[0]
            x_single    = X_test_scaled[feat_idx - (n_train + n_val):feat_idx - (n_train + n_val) + 1]
            shap_vals   = explainer.shap_values(x_single)[0]

            # Top 10 features for this image
            top_idx  = np.argsort(np.abs(shap_vals))[-10:][::-1]
            top_names = [feat_names[j] if j < len(feat_names) else f'feat_{j}' for j in top_idx]
            top_vals  = shap_vals[top_idx]

            fig, ax = plt.subplots(figsize=(10, 5))
            colors  = ['#e15759' if v > 0 else '#4e79a7' for v in top_vals[::-1]]
            ax.barh(top_names[::-1], top_vals[::-1], color=colors, edgecolor='white')
            ax.axvline(x=0, color='gray', linewidth=1)
            ax.set_xlabel("SHAP value (→ AI,  ← Human)")
            ax.set_title(
                f"SHAP Breakdown — {'AI' if row['label']==1 else 'Human'} image "
                f"(pred: {'AI' if row['pred']==1 else 'Human'}, {row['conf']*100:.1f}%)",
                fontsize=12
            )
            plt.tight_layout()
            plt.savefig(EXPLAIN_DIR / f"shap_waterfall_{plot_idx:02d}.png",
                        dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"  ⚠️  SHAP waterfall failed for image {idx}: {e}")
            continue

    print(f"  SHAP waterfall plots saved → {EXPLAIN_DIR}")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Comparative summary — 3 methods side by side on same image
    # ══════════════════════════════════════════════════════════════════════════
    print("\nGenerating method comparison summary figure...")

    # Pick one clear AI and one clear human image with high confidence.
    # Fallback to the most confident sample per class if no image passes threshold.
    def pick_comparison_row(label, threshold=0.95):
        class_rows = test_df[test_df['label'] == label]
        if len(class_rows) == 0:
            print(f"  Warning: no samples found for label={label}; skipping row")
            return None
        confident = class_rows[class_rows['conf'] > threshold]
        if len(confident) > 0:
            return confident.sample(1, random_state=SEED).iloc[0]
        print(f"  Warning: no samples above {threshold:.2f} for label={label}; using top-confidence fallback")
        return class_rows.sort_values('conf', ascending=False).iloc[0]

    comparison_rows = [
        pick_comparison_row(0),
        pick_comparison_row(1),
    ]
    comparison_rows = [r for r in comparison_rows if r is not None]

    if len(comparison_rows) == 0:
        print("  Warning: no rows available for method comparison; skipping figure")
    else:
        fig, axes = plt.subplots(len(comparison_rows), 4, figsize=(22, 5.5 * len(comparison_rows)))
        axes = np.atleast_2d(axes)
        titles_row = ['Original', 'Attention (secondary)', 'GradCAM++', 'Overlay Comparison']

        for row_idx, row in enumerate(comparison_rows):
            try:
                tensor, img_pil, img_np = preprocess(row['path'])

                rollout_hm = rollout_explainer(tensor)
                gradcam_hm = gradcam_explainer(tensor, class_idx=int(row['pred']))

                rollout_ov = heatmap_to_overlay(rollout_hm, img_np, cv2.COLORMAP_JET)
                gradcam_ov = heatmap_to_overlay(gradcam_hm, img_np, cv2.COLORMAP_INFERNO)

                # Blend both heatmaps for comparison panel
                combined   = heatmap_to_overlay(
                    (rollout_hm + gradcam_hm) / 2, img_np, cv2.COLORMAP_HOT, alpha=0.5
                )

                row_label = "Human (WikiArt)" if row['label'] == 0 else f"AI ({row['source']})"
                conf      = row['conf'] * 100

                for col_idx, (img_data, title) in enumerate(zip(
                    [img_np, rollout_ov, gradcam_ov, combined], titles_row
                )):
                    axes[row_idx][col_idx].imshow(img_data)
                    axes[row_idx][col_idx].axis('off')
                    if row_idx == 0:
                        axes[row_idx][col_idx].set_title(title, fontsize=12, fontweight='bold')

                axes[row_idx][0].set_ylabel(
                    f"{row_label}\n{conf:.1f}% confident",
                    fontsize=11, rotation=0, ha='right', va='center'
                )

            except Exception as e:
                print(f"  ⚠️  Summary row {row_idx} failed: {e}")

        plt.suptitle(
            "ArtLens Phase 5 — Explainability Methods Comparison",
            fontsize=15, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'phase5_methods_comparison.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Method comparison saved → results/phase5_methods_comparison.png")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Heatmap statistics — what regions AI vs human images focus on
    # ══════════════════════════════════════════════════════════════════════════
    print("\nComputing aggregate heatmap statistics...")

    # Compute average attention rollout heatmap for AI and human separately
    # This shows: "on average, where does the model look when classifying AI art?"
    n_stat       = min(200, len(test_df))
    stat_sample  = test_df.sample(n_stat, random_state=SEED)

    avg_rollout_human = np.zeros((14, 14))
    avg_rollout_ai    = np.zeros((14, 14))
    count_human = count_ai = 0

    for _, row in tqdm(stat_sample.iterrows(), total=n_stat, desc="Aggregate heatmaps"):
        try:
            tensor, _, _ = preprocess(row['path'])
            hm           = rollout_explainer(tensor)

            if row['label'] == 0:
                avg_rollout_human += hm
                count_human       += 1
            else:
                avg_rollout_ai    += hm
                count_ai          += 1
        except Exception:
            continue

    if count_human > 0: avg_rollout_human /= count_human
    if count_ai    > 0: avg_rollout_ai    /= count_ai

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Normalise for display
    h_disp = avg_rollout_human / (avg_rollout_human.max() + 1e-10)
    a_disp = avg_rollout_ai    / (avg_rollout_ai.max()    + 1e-10)
    diff   = a_disp - h_disp   # positive = AI looks here more, negative = human

    im0 = axes[0].imshow(h_disp, cmap='Blues',  vmin=0, vmax=1)
    axes[0].set_title(f"Avg attention — Human art\n(n={count_human})", fontsize=12)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(a_disp, cmap='Reds',   vmin=0, vmax=1)
    axes[1].set_title(f"Avg attention — AI art\n(n={count_ai})", fontsize=12)
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff,   cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    axes[2].set_title("Difference (AI − Human)\nRed = AI attends more here", fontsize=12)
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("ArtLens — Aggregate Attention Patterns: AI vs Human",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'phase5_aggregate_attention.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Aggregate attention saved → results/phase5_aggregate_attention.png")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Save explanation log
    # ══════════════════════════════════════════════════════════════════════════
    with open(OUTPUT_DIR / 'phase5_explanation_log.json', 'w') as f:
        json.dump(explanation_log, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  PHASE 5 COMPLETE")
    print(f"{'='*55}")
    print(f"  Explanation grids      : {EXPLAIN_DIR}")
    print(f"  Method comparison      : results/phase5_methods_comparison.png")
    print(f"  Aggregate attention    : results/phase5_aggregate_attention.png")
    print(f"  SHAP waterfalls        : results/explanations/shap_waterfall_*.png")
    print(f"  Explanation log        : phase5_explanation_log.json")
    print(f"\n  Ready for Phase 6 — FastAPI backend + HuggingFace Spaces deployment")