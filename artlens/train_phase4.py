# ── ArtLens Phase 4 — Frequency Features + LightGBM Ensemble ─────────────────
# Extracts handcrafted features (FFT, DWT, LBP) and caches ViT embeddings
# Trains LightGBM on top of combined features
# Runs ablation study and UMAP visualisation
# Run with: uv run python train_phase4.py

import os, random, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import cv2
import pywt                                    # PyWavelets for DWT
from skimage.feature import local_binary_pattern   # LBP texture descriptor

import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib                                  # save/load sklearn objects

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path('D:/ArtLens/data')
OUTPUT_DIR  = Path('D:/ArtLens/outputs')
MANIFEST    = DATA_DIR / 'dataset_manifest.csv'
MODELS_DIR  = OUTPUT_DIR / 'models'
RESULTS_DIR = OUTPUT_DIR / 'results'
EMBED_DIR   = OUTPUT_DIR / 'embeddings'        # cached ViT embeddings

for d in [MODELS_DIR, RESULTS_DIR, EMBED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    'img_size'      : 224,
    'batch_size'    : 32,          # larger batch for embedding extraction — no gradients
    'num_workers'   : 2,
    'seed'          : 42,
    'model_name'    : 'vit_base_patch16_224',
    # LBP parameters
    'lbp_radius'    : 3,           # neighbourhood radius
    'lbp_n_points'  : 24,          # 8 * radius — standard setting
    # LightGBM parameters
    'lgb_n_estimators'   : 500,
    'lgb_learning_rate'  : 0.05,
    'lgb_max_depth'      : 6,
    'lgb_num_leaves'     : 63,
    'lgb_subsample'      : 0.8,
    'lgb_colsample'      : 0.8,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Dataset for embedding extraction (no augmentation) ───────────────────────
class EmbedDataset(Dataset):
    """
    Simple dataset for feature extraction — no augmentation, just resize + normalise.
    Returns image tensor and row index so we can track which embedding belongs to which image.
    """
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.transform = A.Compose([
            A.Resize(CFG['img_size'], CFG['img_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = np.array(Image.open(row['path']).convert('RGB'))
        except Exception:
            img = np.zeros((CFG['img_size'], CFG['img_size'], 3), dtype=np.uint8)
        img = self.transform(image=img)['image']
        return img, idx    # return index so we can sort embeddings back to manifest order


# ── FFT feature extraction ────────────────────────────────────────────────────
def extract_fft_features(img_path: str) -> np.ndarray:
    """
    Extracts frequency domain statistics from an image using FFT.
    Returns 1D feature vector of length 18.

    AI images leave characteristic high-frequency artifacts from upsampling
    operations inside the generator. These manifest as elevated high-frequency
    energy, periodic spikes, and non-uniform radial energy distribution.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(18)
        img = cv2.resize(img, (224, 224))

        # Compute 2D FFT and shift DC component to centre
        fft    = np.fft.fft2(img.astype(np.float32))
        fft_sh = np.fft.fftshift(fft)
        mag    = np.abs(fft_sh)           # magnitude spectrum
        mag_log = np.log1p(mag)           # log-scale for stability

        h, w = mag.shape
        cy, cx = h // 2, w // 2          # centre (DC component)

        # ── Feature 1: Total energy ───────────────────────────────────────────
        total_energy = mag.sum()

        # ── Feature 2-4: Energy in frequency bands ────────────────────────────
        # Low freq = inner 10% of spectrum (smooth, broad regions)
        # Mid freq = 10-50% (object boundaries)
        # High freq = outer 50% (fine texture, AI artifacts)
        Y, X = np.ogrid[:h, :w]
        dist  = np.sqrt((X - cx)**2 + (Y - cy)**2)
        r_max = min(cx, cy)

        low_mask  = dist < r_max * 0.10
        mid_mask  = (dist >= r_max * 0.10) & (dist < r_max * 0.50)
        high_mask = dist >= r_max * 0.50

        low_energy  = mag[low_mask].sum()
        mid_energy  = mag[mid_mask].sum()
        high_energy = mag[high_mask].sum()

        # High-to-total ratio: key AI discriminator — AI images score higher
        hf_ratio = high_energy / (total_energy + 1e-10)

        # ── Feature 5-9: Radial energy profile (5 bins) ───────────────────────
        # Bins the energy at 5 radial distances from DC component
        # AI generators create characteristic rings at specific radii
        radial_bins = []
        for i in range(5):
            inner = r_max * i / 5
            outer = r_max * (i + 1) / 5
            ring_mask = (dist >= inner) & (dist < outer)
            radial_bins.append(mag[ring_mask].mean() if ring_mask.sum() > 0 else 0)

        # ── Feature 10-12: Spectral statistics ───────────────────────────────
        flat   = mag_log.flatten()
        sp_mean = flat.mean()
        sp_std  = flat.std()
        # Spectral entropy — measures how 'spread out' the energy is
        # AI images tend to have more ordered (lower entropy) frequency distributions
        probs  = flat / (flat.sum() + 1e-10)
        sp_ent = -np.sum(probs * np.log(probs + 1e-10))

        # ── Feature 13-16: Peak frequency location ───────────────────────────
        # Find the 4 highest-energy frequency locations
        # Different generators have peaks at characteristic locations
        flat_idx  = np.argsort(mag.flatten())[-4:]
        peak_rows = (flat_idx // w) / h    # normalised row position
        peak_cols = (flat_idx % w) / w     # normalised col position

        # ── Feature 17-18: Horizontal/vertical energy asymmetry ──────────────
        h_energy = mag[:cy, :].sum()       # top half
        v_energy = mag[:, :cx].sum()       # left half
        hv_ratio = h_energy / (v_energy + 1e-10)
        h_asym   = abs(mag[:cy, :].sum() - mag[cy:, :].sum()) / (total_energy + 1e-10)

        features = np.array([
            hf_ratio,                       # high-frequency energy ratio
            low_energy / (total_energy + 1e-10),
            mid_energy / (total_energy + 1e-10),
            high_energy / (total_energy + 1e-10),
            *radial_bins,                   # 5 radial bins
            sp_mean, sp_std, sp_ent,        # spectral statistics
            *peak_rows[:2], *peak_cols[:2], # peak frequency locations
            hv_ratio, h_asym               # asymmetry measures
        ], dtype=np.float32)

        return features

    except Exception:
        return np.zeros(18, dtype=np.float32)


# ── DWT feature extraction ────────────────────────────────────────────────────
def extract_dwt_features(img_path: str) -> np.ndarray:
    """
    Extracts texture features using Discrete Wavelet Transform.
    Returns 1D feature vector of length 24.

    DWT decomposes image into frequency sub-bands at multiple scales.
    Unlike FFT (global frequency), DWT also captures WHERE frequency artifacts
    occur spatially — important for detecting localised AI upsampling artifacts.

    We use 'db2' (Daubechies wavelet, 2nd order) — standard for image analysis.
    Two decomposition levels give features at two spatial scales.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(24)
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0

        features = []

        # Level 1 decomposition: coarse scale
        coeffs = pywt.dwt2(img, 'db2')
        cA1, (cH1, cV1, cD1) = coeffs  # approx, horizontal, vertical, diagonal

        # Level 2 decomposition: fine scale (decompose approximation further)
        coeffs2 = pywt.dwt2(cA1, 'db2')
        cA2, (cH2, cV2, cD2) = coeffs2

        # For each sub-band, extract: mean, std, energy, entropy
        # These statistics describe the texture at each scale and orientation
        for coeff in [cH1, cV1, cD1, cH2, cV2, cD2]:
            c_flat = coeff.flatten()
            c_abs  = np.abs(c_flat)
            energy = (c_flat**2).mean()
            # Entropy of wavelet coefficients — lower for AI (more uniform texture)
            hist, _ = np.histogram(c_abs, bins=32, density=True)
            hist   += 1e-10
            entropy = -np.sum(hist * np.log(hist))

            features.extend([c_abs.mean(), c_abs.std(), energy, entropy])

        return np.array(features[:24], dtype=np.float32)

    except Exception:
        return np.zeros(24, dtype=np.float32)


# ── LBP feature extraction ────────────────────────────────────────────────────
def extract_lbp_features(img_path: str) -> np.ndarray:
    """
    Extracts Local Binary Pattern texture histogram.
    Returns 1D feature vector of length 26 (histogram + 2 statistics).

    LBP compares each pixel to its circular neighbourhood — if neighbour is
    brighter, bit=1, else bit=0. The histogram of LBP codes describes texture.

    AI images have unnaturally uniform micro-texture — lower LBP entropy.
    Human art (brushstrokes, sensor noise) produces richer, more varied LBP.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(26)
        img = cv2.resize(img, (224, 224))

        # 'uniform' LBP: only counts patterns with at most 2 bit transitions
        # This gives 26 bins: 24 uniform patterns + 1 non-uniform + 1 for all-same
        lbp = local_binary_pattern(
            img,
            P = CFG['lbp_n_points'],  # 24 neighbours
            R = CFG['lbp_radius'],    # radius 3
            method = 'uniform'
        )

        # Normalised histogram of LBP codes
        n_bins = CFG['lbp_n_points'] + 2   # 26 for uniform LBP
        # Compute actual number of unique LBP values dynamically
        # scikit-image versions differ on exact bin count for uniform LBP
        lbp_max = int(lbp.max()) + 1
        n_bins = lbp_max
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_bins), density=True)

        # Entropy of LBP histogram — lower for AI images
        hist_safe = hist + 1e-10
        lbp_entropy = -np.sum(hist_safe * np.log(hist_safe))

        # Uniformity — concentration in one bin
        lbp_uniformity = (hist**2).sum()

        return np.array([*hist, lbp_entropy, lbp_uniformity], dtype=np.float32)

    except Exception:
        return np.zeros(26, dtype=np.float32)


# ── Combined handcrafted feature extractor ───────────────────────────────────
def extract_all_handcrafted(img_path: str) -> np.ndarray:
    """
        Concatenates all handcrafted features for one image.
        FFT(18) + DWT(24) + LBP(variable, typically 26-28) = ~68-70 total.
        Size is determined by actual LBP output on first call.
    """
    fft = extract_fft_features(img_path)   # 18-dim
    dwt = extract_dwt_features(img_path)   # 24-dim
    lbp = extract_lbp_features(img_path)   # 26-dim
    return np.concatenate([fft, dwt, lbp]) # 68-dim


# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    print(f"Device : {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

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
    print(f"Images verified: {len(df):,}")

    # ── Load existing splits for fair Phase 2/3/4 comparison ─────────────────
    train_df = pd.read_csv(OUTPUT_DIR / 'split_train.csv')
    val_df   = pd.read_csv(OUTPUT_DIR / 'split_val.csv')
    test_df  = pd.read_csv(OUTPUT_DIR / 'split_test.csv')

    for split in [train_df, val_df, test_df]:
        split['path'] = split.apply(remap_path, axis=1)

    train_df = train_df[train_df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    val_df   = val_df[val_df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    test_df  = test_df[test_df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    all_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    print(f"Total for feature extraction: {len(all_df):,}")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Extract and cache ViT embeddings
    # ══════════════════════════════════════════════════════════════════════════
    embed_cache = EMBED_DIR / 'phase3_embeddings.npz'

    if embed_cache.exists():
        print(f"\nLoading cached ViT embeddings from {embed_cache}")
        cache    = np.load(embed_cache)
        all_embs = cache['embeddings']
        all_idxs = cache['indices']
        print(f"  Loaded: {all_embs.shape}")
    else:
        print("\nExtracting ViT embeddings (Phase 3 backbone)...")

        # Load Phase 3 model backbone only — we need 768-dim CLS embeddings
        from train_phase3 import ArtLensMultiTaskModel
        model = ArtLensMultiTaskModel().to(DEVICE)
        ckpt  = torch.load(
            MODELS_DIR / 'phase3_best_model.pt',
            map_location=DEVICE, weights_only=False
        )
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"  Phase 3 model loaded (epoch {ckpt['epoch']})")

        loader = DataLoader(
            EmbedDataset(all_df),
            batch_size  = CFG['batch_size'],
            shuffle     = False,
            num_workers = CFG['num_workers'],
            pin_memory  = True,
            persistent_workers = True
        )

        all_embs = []
        all_idxs = []

        with torch.no_grad():
            for imgs, idxs in tqdm(loader, desc="Extracting embeddings"):
                imgs = imgs.to(DEVICE, non_blocking=True)
                with autocast('cuda'):
                    embs = model.get_embeddings(imgs)   # [batch, 768]
                all_embs.append(embs.cpu().float().numpy())
                all_idxs.extend(idxs.numpy())

        all_embs = np.vstack(all_embs)     # [N, 768]
        all_idxs = np.array(all_idxs)

        # Sort back to original order
        sort_order = np.argsort(all_idxs)
        all_embs   = all_embs[sort_order]
        all_idxs   = all_idxs[sort_order]

        np.savez_compressed(embed_cache, embeddings=all_embs, indices=all_idxs)
        print(f"  Embeddings cached: {all_embs.shape} → {embed_cache}")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Extract handcrafted features (FFT + DWT + LBP)
    # ══════════════════════════════════════════════════════════════════════════
    feat_cache = EMBED_DIR / 'handcrafted_features.npz'

    if feat_cache.exists():
        print(f"\nLoading cached handcrafted features from {feat_cache}")
        cache        = np.load(feat_cache)
        all_hc_feats = cache['features']
        print(f"  Loaded: {all_hc_feats.shape}")
    else:
        print(f"\nExtracting handcrafted features for {len(all_df):,} images...")
        print("  FFT(18) + DWT(24) + LBP(26) = 68 features per image")

        # Compute one sample first to get actual feature size
        # LBP bin count varies by scikit-image version — don't hardcode
        sample_feat = extract_all_handcrafted(all_df.iloc[0]['path'])
        n_hc_features = len(sample_feat)
        print(f"  Handcrafted feature size: FFT(18) + DWT(24) + LBP({n_hc_features - 42}) = {n_hc_features}")

        all_hc_feats = np.zeros((len(all_df), n_hc_features), dtype=np.float32)

        for i, row in tqdm(all_df.iterrows(), total=len(all_df), desc="Handcrafted features"):
            feat = extract_all_handcrafted(row['path'])
            if len(feat) == n_hc_features:
                all_hc_feats[i] = feat
            else:
                # Pad or truncate if size unexpectedly differs — defensive fallback
                all_hc_feats[i, :min(len(feat), n_hc_features)] = feat[:n_hc_features]

        np.savez_compressed(feat_cache, features=all_hc_feats)
        print(f"  Features cached: {all_hc_feats.shape} → {feat_cache}")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Build combined feature matrix
    # ══════════════════════════════════════════════════════════════════════════
    print("\nBuilding combined feature matrix...")

    # Concatenate ViT embeddings (768) + handcrafted (68) = 836 total features
    X_all = np.concatenate([all_embs, all_hc_feats], axis=1)
    y_all = all_df['label'].values

    print(f"  Combined feature matrix: {X_all.shape}")
    print(f"  Labels: {np.bincount(y_all)}")

    # Split back into train/val/test using same indices as before
    n_train = len(train_df)
    n_val   = len(val_df)
    n_test  = len(test_df)

    X_train = X_all[:n_train];          y_train = y_all[:n_train]
    X_val   = X_all[n_train:n_train+n_val]; y_val = y_all[n_train:n_train+n_val]
    X_test  = X_all[n_train+n_val:];    y_test  = y_all[n_train+n_val:]

    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # Standardise features — important for LBP histogram values
    # ViT embeddings are already in a reasonable range, but handcrafted vary wildly
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit ONLY on train
    X_val   = scaler.transform(X_val)         # apply same transform
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, MODELS_DIR / 'feature_scaler.pkl')
    print("  Scaler saved → feature_scaler.pkl")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Train LightGBM ensemble
    # ══════════════════════════════════════════════════════════════════════════
    print("\nTraining LightGBM ensemble...")

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)

    lgb_params = {
        'objective'       : 'binary',
        'metric'          : 'auc',          # AUROC as primary metric
        'n_estimators'    : CFG['lgb_n_estimators'],
        'learning_rate'   : CFG['lgb_learning_rate'],
        'max_depth'       : CFG['lgb_max_depth'],
        'num_leaves'      : CFG['lgb_num_leaves'],
        'subsample'       : CFG['lgb_subsample'],
        'colsample_bytree': CFG['lgb_colsample'],
        'random_state'    : CFG['seed'],
        'verbose'         : -1,
        'n_jobs'          : -1,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=True),  # stop if no improvement
        lgb.log_evaluation(period=50)                           # print every 50 rounds
    ]

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets  = [lgb_val],
        callbacks   = callbacks,
    )

    # Save LightGBM model
    lgb_model.save_model(str(MODELS_DIR / 'lgb_model.txt'))
    print(f"  LightGBM model saved → lgb_model.txt")
    print(f"  Best iteration: {lgb_model.best_iteration}")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Evaluate LightGBM on test set
    # ══════════════════════════════════════════════════════════════════════════
    print("\nEvaluating LightGBM on test set...")

    lgb_probs = lgb_model.predict(X_test)            # probabilities
    lgb_preds = (lgb_probs >= 0.5).astype(int)       # binary predictions

    lgb_auroc = roc_auc_score(y_test, lgb_probs)
    lgb_f1    = f1_score(y_test, lgb_preds, average='weighted')
    lgb_acc   = (lgb_preds == y_test).mean()

    print(f"\n{'='*55}")
    print(f"  LIGHTGBM ENSEMBLE TEST RESULTS")
    print(f"  Accuracy : {lgb_acc*100:.2f}%")
    print(f"  AUROC    : {lgb_auroc:.4f}")
    print(f"  F1       : {lgb_f1:.4f}")
    print(f"{'='*55}")
    print(classification_report(y_test, lgb_preds, target_names=['Human','AI']))


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6: Ablation study — which components matter?
    # ══════════════════════════════════════════════════════════════════════════
    print("\nRunning ablation study...")
    print("  Testing: ViT-only vs Handcrafted-only vs Combined")

    ablation_results = {}

    # ── ViT embeddings only ───────────────────────────────────────────────────
    X_tr_vit = scaler.inverse_transform(X_train)[:, :768]
    X_te_vit = scaler.inverse_transform(X_test)[:, :768]
    sc_vit   = StandardScaler()
    X_tr_vit = sc_vit.fit_transform(X_tr_vit)
    X_te_vit = sc_vit.transform(X_te_vit)

    lgb_vit = lgb.train(
        {**lgb_params, 'n_estimators': 200},
        lgb.Dataset(X_tr_vit, label=y_train),
        valid_sets=[lgb.Dataset(
            sc_vit.transform(scaler.inverse_transform(X_val)[:, :768]),
            label=y_val
        )],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)]
    )
    vit_probs = lgb_vit.predict(X_te_vit)
    ablation_results['ViT only (768-dim)'] = roc_auc_score(y_test, vit_probs)

    # ── Handcrafted features only ─────────────────────────────────────────────
    X_tr_hc = scaler.inverse_transform(X_train)[:, 768:]
    X_te_hc = scaler.inverse_transform(X_test)[:, 768:]
    sc_hc   = StandardScaler()
    X_tr_hc = sc_hc.fit_transform(X_tr_hc)
    X_te_hc = sc_hc.transform(X_te_hc)

    lgb_hc = lgb.train(
        {**lgb_params, 'n_estimators': 200},
        lgb.Dataset(X_tr_hc, label=y_train),
        valid_sets=[lgb.Dataset(
            sc_hc.transform(scaler.inverse_transform(X_val)[:, 768:]),
            label=y_val
        )],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)]
    )
    hc_probs = lgb_hc.predict(X_te_hc)
    ablation_results['Handcrafted only (68-dim)'] = roc_auc_score(y_test, hc_probs)

    # ── FFT only ──────────────────────────────────────────────────────────────
    X_tr_fft = scaler.inverse_transform(X_train)[:, 768:786]   # 18-dim
    X_te_fft = scaler.inverse_transform(X_test)[:, 768:786]
    sc_fft   = StandardScaler()
    X_tr_fft = sc_fft.fit_transform(X_tr_fft)
    X_te_fft = sc_fft.transform(X_te_fft)

    lgb_fft = lgb.train(
        {**lgb_params, 'n_estimators': 200},
        lgb.Dataset(X_tr_fft, label=y_train),
        valid_sets=[lgb.Dataset(
            sc_fft.transform(scaler.inverse_transform(X_val)[:, 768:786]),
            label=y_val
        )],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)]
    )
    fft_probs = lgb_fft.predict(X_te_fft)
    ablation_results['FFT only (18-dim)'] = roc_auc_score(y_test, fft_probs)

    # ── Combined (full model) ─────────────────────────────────────────────────
    ablation_results['Combined ViT + HC (836-dim)'] = lgb_auroc

    print("\nAblation Study Results:")
    print(f"{'Component':<35} {'AUROC':>8}")
    print("-" * 45)
    for name, auroc in sorted(ablation_results.items(), key=lambda x: x[1]):
        bar = "█" * int(auroc * 30)
        print(f"  {name:<33} {auroc:.4f}  {bar}")

    # Save ablation results
    with open(OUTPUT_DIR / 'phase4_ablation.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # Ablation bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names  = list(ablation_results.keys())
    aurocs = [ablation_results[n] for n in names]
    colors = ['#e15759' if 'only' in n else '#59a14f' for n in names]
    bars   = ax.barh(names, aurocs, color=colors, edgecolor='white')
    ax.set_xlim([min(aurocs) - 0.05, 1.01])
    ax.set_xlabel("AUROC")
    ax.set_title("Ablation Study — Which Components Matter?", fontsize=13, fontweight='bold')
    ax.axvline(x=lgb_auroc, color='green', linestyle='--', alpha=0.5)
    for bar, v in zip(bars, aurocs):
        ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'phase4_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nAblation chart saved")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7: SHAP feature importance
    # ══════════════════════════════════════════════════════════════════════════
    print("\nComputing SHAP feature importance...")

    try:
        import shap

        # Feature names for interpretability
        fft_names = [f'fft_{i}' for i in range(18)]
        dwt_names = [f'dwt_{i}' for i in range(24)]
        n_lbp = all_hc_feats.shape[1] - 42  # total HC - FFT(18) - DWT(24)
        lbp_names = [f'lbp_{i}' for i in range(n_lbp)]
        vit_names = [f'vit_{i}' for i in range(768)]
        feature_names = vit_names + fft_names + dwt_names + lbp_names

        # SHAP TreeExplainer is fast for LightGBM
        explainer   = shap.TreeExplainer(lgb_model)
        # Use a sample of 500 test images to keep computation fast
        sample_size = min(500, len(X_test))
        shap_values = explainer.shap_values(X_test[:sample_size])

        # Mean absolute SHAP value per feature
        mean_shap = np.abs(shap_values).mean(axis=0)

        # ── Top 20 most important features ────────────────────────────────────
        top_idx   = np.argsort(mean_shap)[-20:][::-1]
        top_names = [feature_names[i] if i < len(feature_names) else f'feat_{i}' for i in top_idx]
        top_shap  = mean_shap[top_idx]

        print("\nTop 20 features by mean |SHAP|:")
        for name, val in zip(top_names, top_shap):
            bar = "█" * int(val * 1000)
            print(f"  {name:<15} {val:.5f}  {bar}")

        # SHAP bar chart for top 20
        fig, ax = plt.subplots(figsize=(10, 8))
        colors  = ['#4e79a7' if n.startswith('vit') else
                   '#f28e2b' if n.startswith('fft') else
                   '#59a14f' if n.startswith('dwt') else
                   '#e15759' for n in top_names]
        ax.barh(top_names[::-1], top_shap[::-1], color=colors[::-1], edgecolor='white')
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top 20 Features — Mean Absolute SHAP (LightGBM)", fontsize=13)

        # Legend
        from matplotlib.patches import Patch
        legend = [
            Patch(color='#4e79a7', label='ViT embedding dim'),
            Patch(color='#f28e2b', label='FFT feature'),
            Patch(color='#59a14f', label='DWT feature'),
            Patch(color='#e15759', label='LBP feature'),
        ]
        ax.legend(handles=legend, loc='lower right')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'phase4_shap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("SHAP chart saved")

    except ImportError:
        print("  SHAP not installed — run: uv add shap")
        print("  Skipping SHAP analysis")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 8: UMAP embedding visualisation
    # ══════════════════════════════════════════════════════════════════════════
    print("\nGenerating UMAP visualisation of ViT embedding space...")

    try:
        import umap

        # Sample 3000 images for UMAP (full dataset takes too long)
        sample_n   = min(3000, len(all_df))
        sample_idx = np.random.choice(len(all_df), sample_n, replace=False)

        emb_sample    = all_embs[sample_idx]
        label_sample  = y_all[sample_idx]
        source_sample = all_df['source'].values[sample_idx]

        print(f"  Running UMAP on {sample_n} embeddings (768-dim → 2-dim)...")
        reducer = umap.UMAP(
            n_components = 2,
            n_neighbors  = 30,    # larger = more global structure preserved
            min_dist     = 0.1,   # smaller = tighter clusters
            metric       = 'cosine',  # cosine works better for embedding spaces
            random_state = SEED
        )
        emb_2d = reducer.fit_transform(emb_sample)

        # Save UMAP model for Phase 5 (interactive visualisation)
        joblib.dump(reducer, MODELS_DIR / 'umap_reducer.pkl')

        # ── Plot 1: Human vs AI ───────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        colors_binary = ['#4e79a7' if l == 0 else '#e15759' for l in label_sample]
        axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1],
                        c=colors_binary, s=4, alpha=0.5, linewidths=0)
        axes[0].set_title("UMAP: Human (blue) vs AI (red)", fontsize=13)
        axes[0].set_xlabel("UMAP-1"); axes[0].set_ylabel("UMAP-2")
        axes[0].axis('off')

        from matplotlib.patches import Patch
        axes[0].legend(handles=[
            Patch(color='#4e79a7', label='Human (WikiArt)'),
            Patch(color='#e15759', label='AI-generated'),
        ], loc='upper right', fontsize=10)

        # ── Plot 2: By source/generator ───────────────────────────────────────
        source_colors = {
            'wikiart'          : '#4e79a7',
            'midjourney'       : '#f28e2b',
            'stable_diffusion' : '#59a14f',
        }
        colors_source = [source_colors.get(s, '#888888') for s in source_sample]
        axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1],
                        c=colors_source, s=4, alpha=0.5, linewidths=0)
        axes[1].set_title("UMAP: by Source/Generator", fontsize=13)
        axes[1].set_xlabel("UMAP-1"); axes[1].set_ylabel("UMAP-2")
        axes[1].axis('off')
        axes[1].legend(handles=[
            Patch(color='#4e79a7', label='WikiArt (human)'),
            Patch(color='#f28e2b', label='Midjourney'),
            Patch(color='#59a14f', label='Stable Diffusion'),
        ], loc='upper right', fontsize=10)

        plt.suptitle("ArtLens Phase 4 — ViT Embedding Space (UMAP)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'phase4_umap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  UMAP visualisation saved")

    except ImportError:
        print("  UMAP not installed — run: uv add umap-learn")
        print("  Skipping UMAP visualisation")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 9: Per-source breakdown for LightGBM
    # ══════════════════════════════════════════════════════════════════════════
    print("\nPer-source accuracy (LightGBM ensemble):")
    test_df_eval            = test_df.copy()
    test_df_eval['pred']    = lgb_preds
    test_df_eval['prob']    = lgb_probs
    test_df_eval['correct'] = (test_df_eval['pred'] == test_df_eval['label']).astype(int)

    for source in sorted(test_df_eval['source'].unique()):
        subset = test_df_eval[test_df_eval['source'] == source]
        acc    = subset['correct'].mean()
        auroc  = roc_auc_score(subset['label'], subset['prob']) if len(subset['label'].unique()) > 1 else float('nan')
        flag   = "⚠️" if acc < 0.80 else "✅"
        print(f"  {flag} {source:<25} acc {acc:.4f}  AUROC {auroc:.4f}")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 10: Save all results
    # ══════════════════════════════════════════════════════════════════════════
    results = {
        'lgb_test_accuracy' : float(lgb_acc),
        'lgb_test_auroc'    : float(lgb_auroc),
        'lgb_test_f1'       : float(lgb_f1),
        'lgb_best_iteration': int(lgb_model.best_iteration),
        'feature_dims'      : {
            'vit_embeddings'  : 768,
            'fft_features'    : 18,
            'dwt_features'    : 24,
            'lbp_features'    : 26,
            'total_combined'  : 836,
        },
        'ablation'          : ablation_results,
    }

    with open(OUTPUT_DIR / 'phase4_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  PHASE 4 COMPLETE")
    print(f"{'='*55}")
    print(f"  LightGBM test AUROC     : {lgb_auroc:.4f}")
    print(f"  LightGBM test accuracy  : {lgb_acc*100:.2f}%")
    print(f"  Feature dimensions      : 768 (ViT) + 68 (HC) = 836")
    print(f"  Best LGB iteration      : {lgb_model.best_iteration}")
    print()
    print(f"  Saved files:")
    print(f"    models/lgb_model.txt")
    print(f"    models/feature_scaler.pkl")
    print(f"    models/umap_reducer.pkl")
    print(f"    embeddings/phase3_embeddings.npz")
    print(f"    embeddings/handcrafted_features.npz")
    print(f"    results/phase4_ablation.png")
    print(f"    results/phase4_shap.png")
    print(f"    results/phase4_umap.png")
    print(f"  Outputs: D:/ArtLens/outputs/")