# ── ArtLens calibrate.py — Temperature scaling on validation set ──────────────
# Must run before app.py — produces temperature.json used by inference server
# Run with: uv run python calibrate.py

import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_phase3 import ArtLensMultiTaskModel

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path('D:/ArtLens/data')
OUTPUT_DIR = Path('D:/ArtLens/outputs')
MODELS_DIR = OUTPUT_DIR / 'models'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_BINS = 10
MIDDLE_LOW = 0.1
MIDDLE_HIGH = 0.9
MIDDLE_MIN_COUNT = 100
MIDDLE_MIN_FRACTION = 0.05


def remap_path(row):
    fname = Path(row['path']).name
    if row['label'] == 0:
        return str(DATA_DIR / 'human' / fname)
    folder = 'mj' if row['generator'] == 'midjourney' else 'sd'
    return str(DATA_DIR / 'ai' / folder / fname)


def calibration_stats(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> tuple:
    """Return ECE and bin counts for a binary reliability diagram."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins[1:-1], right=False)
    ece = 0.0
    counts = []
    n = len(probs)

    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        counts.append(count)
        if count == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (count / max(n, 1)) * abs(acc - conf)

    return float(ece), counts


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading Phase 3 model...")
model = ArtLensMultiTaskModel().to(DEVICE)
ckpt  = torch.load(MODELS_DIR / 'phase3_best_model.pt',
                   map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()

# ── Load val set ──────────────────────────────────────────────────────────────
val_df = pd.read_csv(OUTPUT_DIR / 'split_val.csv')
val_df['path'] = val_df.apply(remap_path, axis=1)
val_df = val_df[val_df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
print(f"Val set: {len(val_df):,} images")

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ── Collect raw logits on val set ─────────────────────────────────────────────
# We need raw logits (before softmax) for temperature scaling
print("Collecting logits on val set...")
all_logits = []
all_labels = []

with torch.no_grad():
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        try:
            img   = np.array(Image.open(row['path']).convert('RGB'))
            tens  = transform(image=img)['image'].unsqueeze(0).to(DEVICE)
            with autocast('cuda'):
                lb, _, _ = model(tens)
            all_logits.append(lb.cpu().float())
            all_labels.append(int(row['label']))
        except Exception:
            continue

logits_tensor = torch.cat(all_logits, dim=0)   # [N, 2]
labels_tensor = torch.tensor(all_labels)        # [N]
print(f"  Collected {len(logits_tensor)} logit pairs")

# ── Fit temperature T via NLL minimisation ────────────────────────────────────
# Temperature scaling: calibrated_prob = softmax(logit / T)
# T > 1 softens predictions (less confident)
# T < 1 sharpens predictions (more confident)
# We optimise T to minimise negative log-likelihood on val set
# This is the standard calibration method (Guo et al. 2017)

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialise T=1 (no scaling) — optimise from here
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        # Scale logits by 1/T before softmax
        t = torch.clamp(self.temperature, min=1e-3)
        return logits / t

scaler_model = TemperatureScaler()
optimizer    = torch.optim.LBFGS(
    [scaler_model.temperature],
    lr=0.01, max_iter=200
)
criterion = nn.CrossEntropyLoss()

def eval_step():
    optimizer.zero_grad()
    scaled  = scaler_model(logits_tensor)
    loss    = criterion(scaled, labels_tensor)
    loss.backward()
    return loss

# ── Verify calibration improved ───────────────────────────────────────────────
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Before calibration
probs_before = torch.softmax(logits_tensor, dim=1)[:, 1].detach().numpy()
labels_np    = labels_tensor.numpy()

# Diagnostics: if almost all predictions are already extreme, middle-bin
# calibration is unstable and can hurt reliability.
middle_mask = (probs_before > MIDDLE_LOW) & (probs_before < MIDDLE_HIGH)
middle_count = int(middle_mask.sum())
middle_fraction = float(middle_count / max(len(probs_before), 1))
print("\nCalibration diagnostics:")
print(f"  Predictions in middle range ({MIDDLE_LOW:.1f}-{MIDDLE_HIGH:.1f}): {middle_count} / {len(probs_before)}")
print(f"  That is {middle_fraction * 100:.2f}% of val set")

ece_before, bin_counts = calibration_stats(probs_before, labels_np, n_bins=N_BINS)
nll_before = float(F.cross_entropy(logits_tensor, labels_tensor).item())
print(f"  Baseline NLL={nll_before:.5f}, ECE={ece_before:.5f}")
print(f"  Bin counts (n_bins={N_BINS}): {bin_counts}")

calibration_enabled = True
calibration_note = "LBFGS NLL minimisation on validation set"

if middle_count < MIDDLE_MIN_COUNT or middle_fraction < MIDDLE_MIN_FRACTION:
    calibration_enabled = False
    calibration_note = (
        f"Skipped temperature fitting: sparse middle-confidence support "
        f"({middle_count}/{len(probs_before)} in ({MIDDLE_LOW}, {MIDDLE_HIGH}))."
    )

if calibration_enabled:
    print("\nOptimising temperature T...")
    optimizer.step(eval_step)
    T_candidate = float(torch.clamp(scaler_model.temperature.detach(), min=1e-3).item())
    probs_after_candidate = torch.softmax(logits_tensor / T_candidate, dim=1)[:, 1].detach().numpy()
    ece_after_candidate, _ = calibration_stats(probs_after_candidate, labels_np, n_bins=N_BINS)
    nll_after_candidate = float(F.cross_entropy(logits_tensor / T_candidate, labels_tensor).item())

    print(f"\nCandidate temperature T = {T_candidate:.4f}")
    print(f"  Post-fit NLL={nll_after_candidate:.5f}, ECE={ece_after_candidate:.5f}")

    if T_candidate < 1.0 and (
        ece_after_candidate > ece_before + 1e-4 or
        nll_after_candidate > nll_before + 1e-4
    ):
        calibration_enabled = False
        calibration_note = (
            "Rejected fitted temperature: T<1 sharpened predictions and worsened "
            "ECE/NLL versus baseline."
        )
        T = 1.0
    else:
        T = T_candidate
else:
    T = 1.0

# After calibration (or identity when disabled)
probs_after = torch.softmax(logits_tensor / T, dim=1)[:, 1].detach().numpy()
ece_after, _ = calibration_stats(probs_after, labels_np, n_bins=N_BINS)
nll_after = float(F.cross_entropy(logits_tensor / T, labels_tensor).item())

print(f"\nUsing temperature T = {T:.4f}")
print(f"  Calibration enabled: {calibration_enabled}")
print(f"  NLL before/after: {nll_before:.5f} -> {nll_after:.5f}")
print(f"  ECE before/after: {ece_before:.5f} -> {ece_after:.5f}")
print(f"  T > 1 softens confidence, T < 1 sharpens confidence")
if not calibration_enabled:
    print(f"  Note: {calibration_note}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, probs, title in [
    (axes[0], probs_before, f"Before calibration (T=1.000)"),
    (axes[1], probs_after,  f"After calibration (T={T:.4f})")
]:
    # Calibration curve: perfect calibration = diagonal line
    frac_pos, mean_pred = calibration_curve(labels_np, probs, n_bins=10)
    ax.plot(mean_pred, frac_pos, 's-', label='Model', color='#4e79a7')
    ax.plot([0,1], [0,1], '--', color='gray', label='Perfect calibration')
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual AI rate)")
    ax.set_title(title); ax.legend()

plt.suptitle("ArtLens — Calibration Curves", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'results/phase6_calibration.png', dpi=150, bbox_inches='tight')
plt.close()
print("Calibration plot saved → results/phase6_calibration.png")

# ── Save temperature ──────────────────────────────────────────────────────────
temp_data = {
    'temperature'  : T,
    'val_set_size' : len(logits_tensor),
    'calibration_enabled': calibration_enabled,
    'middle_range': {
        'low': MIDDLE_LOW,
        'high': MIDDLE_HIGH,
        'count': middle_count,
        'fraction': middle_fraction
    },
    'metrics': {
        'nll_before': nll_before,
        'nll_after': nll_after,
        'ece_before': ece_before,
        'ece_after': ece_after,
        'n_bins': N_BINS,
    },
    'note'         : calibration_note + '. Apply as logit/T before softmax.'
}
with open(MODELS_DIR / 'temperature.json', 'w') as f:
    json.dump(temp_data, f, indent=2)

print(f"\n✅ temperature.json saved → {MODELS_DIR}/temperature.json")
print(f"   Use in inference: prob = softmax(logit / {T:.4f})")