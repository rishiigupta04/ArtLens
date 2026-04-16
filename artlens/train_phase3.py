# ── ArtLens Phase 3 — Multi-task: Generator fingerprinting + Open-set detection
# Adds Head 2 (MJ vs SD) and Head 3 (open-set via Mahalanobis distance)
# Loads Phase 2 best_model.pt as backbone starting point
# Run with: uv run python train_phase3.py

import os, random, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score,
    classification_report, confusion_matrix
)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path('D:/ArtLens/data')
OUTPUT_DIR  = Path('D:/ArtLens/outputs')
MANIFEST    = DATA_DIR / 'dataset_manifest.csv'
MODELS_DIR  = OUTPUT_DIR / 'models'
RESULTS_DIR = OUTPUT_DIR / 'results'
PHASE2_BEST = MODELS_DIR / 'best_model.pt'         # starting point for backbone

for d in [MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    'img_size'      : 224,
    'batch_size'    : 16,
    'num_epochs'    : 15,
    'lr'            : 1e-5,       # lower than phase 2 — backbone already good
    'weight_decay'  : 1e-4,
    'num_workers'   : 2,
    'val_split'     : 0.15,
    'test_split'    : 0.10,
    'seed'          : 42,
    'model_name'    : 'vit_base_patch16_224',
    # Multi-task loss weights — how much each head contributes to total loss
    # Head 1 (binary) weighted highest — it's the primary task
    # Head 2 (generator) weighted lower — secondary task, only on AI images
    # These are starting values — can be tuned if one head dominates
    'loss_w_binary' : 1.0,
    'loss_w_gen'    : 0.5,
    'resume_from'   : 'phase3_epoch_15_auroc0.9960.pt',       # set to 'phase3_epoch_07_....pt' to resume
}

# ── Generator label mapping ───────────────────────────────────────────────────
# Head 2 only classifies AI images — maps generator string to integer label
GEN_TO_IDX = {'midjourney': 0, 'stable_diffusion': 1}
IDX_TO_GEN = {0: 'Midjourney', 1: 'Stable Diffusion'}
N_GENERATORS = len(GEN_TO_IDX)


# ── Dataset ───────────────────────────────────────────────────────────────────
class ArtLensDatasetMultiTask(Dataset):
    """
    Returns three things per image:
    - image tensor
    - binary label (0=human, 1=AI) — for Head 1
    - generator label (0=MJ, 1=SD, -1=human/unknown) — for Head 2
      -1 means "not applicable" — Head 2 loss is masked on these samples
    """

    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        bin_label  = int(row['label'])   # 0=human, 1=AI

        # Generator label — -1 for human images (Head 2 loss masked on these)
        gen_str   = row.get('generator', 'none')
        gen_label = GEN_TO_IDX.get(gen_str, -1)   # -1 if human or unknown

        try:
            img = np.array(Image.open(row['path']).convert('RGB'))
        except Exception:
            img = np.zeros((CFG['img_size'], CFG['img_size'], 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(image=img)['image']

        return (
            img,
            torch.tensor(bin_label,  dtype=torch.long),
            torch.tensor(gen_label,  dtype=torch.long)
        )


# ── Model ─────────────────────────────────────────────────────────────────────
class ArtLensMultiTaskModel(nn.Module):
    """
    ViT-B/16 backbone with three heads:
    Head 1 — binary: human (0) vs AI (1)
    Head 2 — generator: Midjourney (0) vs Stable Diffusion (1)
             only meaningful for AI images
    Head 3 — open-set: Mahalanobis distance computed post-training
             not a learned head — uses stored embedding statistics

    Backbone loaded from Phase 2 best_model.pt — already knows
    the basic real/AI distinction, now we add generator specificity
    """

    def __init__(self):
        super().__init__()

        # Shared backbone — same architecture as Phase 2
        self.backbone = timm.create_model(
            CFG['model_name'], pretrained=False, num_classes=0
        )
        embed_dim = self.backbone.num_features   # 768

        # Head 1 — binary classification (same as Phase 2)
        self.head_binary = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=0.3),
            nn.Linear(embed_dim, 2)
        )

        # Head 2 — generator fingerprinting
        # Separate LayerNorm + Dropout so this head can learn
        # generator-specific features without disrupting Head 1
        self.head_generator = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=0.3),
            nn.Linear(embed_dim, N_GENERATORS)   # 2: MJ vs SD
        )

    def forward(self, x):
        # Single forward pass — shared backbone, two head outputs
        embeddings  = self.backbone(x)             # [batch, 768]
        logits_bin  = self.head_binary(embeddings)    # [batch, 2]
        logits_gen  = self.head_generator(embeddings) # [batch, 2]
        return logits_bin, logits_gen, embeddings

    def get_embeddings(self, x):
        # Used for Mahalanobis open-set detection and UMAP (Phase 5)
        return self.backbone(x)


# ── Multi-task loss ────────────────────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    """
    Combines Head 1 and Head 2 losses with configurable weights.

    Head 2 loss is masked — only computed on AI images (gen_label != -1)
    If a batch has no AI images (rare but possible), Head 2 loss = 0

    Total loss = w_binary * loss_binary + w_gen * loss_gen
    """

    def __init__(self, w_binary=1.0, w_gen=0.5):
        super().__init__()
        self.w_binary  = w_binary
        self.w_gen     = w_gen
        # label_smoothing on binary head — prevents overconfidence
        self.ce_binary = nn.CrossEntropyLoss(label_smoothing=0.1)
        # No label smoothing on generator head — we want precise fingerprinting
        self.ce_gen    = nn.CrossEntropyLoss()

    def forward(self, logits_bin, logits_gen, bin_labels, gen_labels):
        # Head 1 loss — computed on all samples
        loss_binary = self.ce_binary(logits_bin, bin_labels)

        # Head 2 loss — only on AI images (gen_label >= 0)
        # mask = boolean tensor, True where gen_label is valid
        mask = gen_labels >= 0
        if mask.sum() > 0:
            # Select only AI image logits and labels
            loss_gen = self.ce_gen(
                logits_gen[mask],
                gen_labels[mask]
            )
        else:
            # No AI images in this batch — Head 2 contributes zero loss
            loss_gen = torch.tensor(0.0, device=logits_bin.device)

        total = self.w_binary * loss_binary + self.w_gen * loss_gen
        return total, loss_binary, loss_gen


# ── Windows multiprocessing guard ─────────────────────────────────────────────
if __name__ == '__main__':

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Load manifest ─────────────────────────────────────────────────────────
    df = pd.read_csv(MANIFEST)
    print(f"Manifest: {len(df):,} rows")

    def remap_path(row):
        fname = Path(row['path']).name
        if row['label'] == 0:
            return str(DATA_DIR / 'human' / fname)
        else:
            folder = 'mj' if row['generator'] == 'midjourney' else 'sd'
            return str(DATA_DIR / 'ai' / folder / fname)

    df['path'] = df.apply(remap_path, axis=1)
    df = df[df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    print(f"Verified: {len(df):,} images")
    print(df['source'].value_counts().to_string())

    # ── Splits — load existing ones from Phase 2 for exact reproducibility ────
    # Using the same splits ensures Phase 3 val/test results are directly
    # comparable to Phase 2 — we're measuring the effect of adding heads,
    # not a different data split
    if (OUTPUT_DIR / 'split_train.csv').exists():
        train_df = pd.read_csv(OUTPUT_DIR / 'split_train.csv')
        val_df   = pd.read_csv(OUTPUT_DIR / 'split_val.csv')
        test_df  = pd.read_csv(OUTPUT_DIR / 'split_test.csv')

        # Remap paths in loaded splits
        for split_df in [train_df, val_df, test_df]:
            split_df['path'] = split_df.apply(remap_path, axis=1)

        # Remove any missing paths
        train_df = train_df[train_df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        val_df   = val_df[val_df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        test_df  = test_df[test_df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        print(f"\nLoaded Phase 2 splits for fair comparison:")
    else:
        # Fresh split if Phase 2 splits not found
        train_df, temp_df = train_test_split(
            df, test_size=CFG['val_split']+CFG['test_split'],
            random_state=CFG['seed'], stratify=df['source']
        )
        val_size_adj = CFG['val_split'] / (CFG['val_split'] + CFG['test_split'])
        val_df, test_df = train_test_split(
            temp_df, test_size=1-val_size_adj,
            random_state=CFG['seed'], stratify=temp_df['source']
        )
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)
        print(f"\nFresh splits created:")

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # ── Augmentation ─────────────────────────────────────────────────────────
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_transform = A.Compose([
        A.Resize(CFG['img_size'], CFG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.OneOf([
            A.ImageCompression(quality_range=(50, 95), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(CFG['img_size'], CFG['img_size']),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        ArtLensDatasetMultiTask(train_df, train_transform),
        batch_size=CFG['batch_size'], shuffle=True,
        num_workers=CFG['num_workers'], pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        ArtLensDatasetMultiTask(val_df, val_transform),
        batch_size=CFG['batch_size']*2, shuffle=False,
        num_workers=CFG['num_workers'], pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        ArtLensDatasetMultiTask(test_df, val_transform),
        batch_size=CFG['batch_size']*2, shuffle=False,
        num_workers=CFG['num_workers'], pin_memory=True,
        persistent_workers=True
    )
    print(f"DataLoaders: {len(train_loader)} train | {len(val_loader)} val | {len(test_loader)} test batches")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ArtLensMultiTaskModel().to(DEVICE)

    # Load Phase 2 backbone weights — preserves learned binary detection
    # Only load backbone weights, not the head (Phase 2 had different head structure)
    if PHASE2_BEST.exists():
        phase2_ckpt = torch.load(
            PHASE2_BEST, map_location=DEVICE, weights_only=False
        )
        phase2_state = phase2_ckpt['model_state']

        # Filter to backbone weights only — keys starting with 'backbone.'
        backbone_weights = {
            k: v for k, v in phase2_state.items()
            if k.startswith('backbone.')
        }
        missing, unexpected = model.load_state_dict(backbone_weights, strict=False)
        print(f"\n✅ Phase 2 backbone weights loaded")
        print(f"   Missing (new heads): {len(missing)} keys — expected")
        print(f"   Unexpected         : {len(unexpected)} keys")
    else:
        print(f"\n⚠️  Phase 2 best_model.pt not found — training from pretrained ImageNet weights")
        # Fall back to pretrained ImageNet weights
        model = ArtLensMultiTaskModel()
        pretrained = timm.create_model(CFG['model_name'], pretrained=True, num_classes=0)
        model.backbone.load_state_dict(pretrained.state_dict())
        model = model.to(DEVICE)

    model.backbone.set_grad_checkpointing(enable=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    criterion   = MultiTaskLoss(
        w_binary = CFG['loss_w_binary'],
        w_gen    = CFG['loss_w_gen']
    )
    optimizer   = optim.AdamW(
        model.parameters(),
        lr           = CFG['lr'],
        weight_decay = CFG['weight_decay']
    )
    total_steps = CFG['num_epochs'] * len(train_loader)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )
    scaler = GradScaler('cuda')

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_auroc  = 0.0

    if CFG['resume_from']:
        ckpt_path = MODELS_DIR / CFG['resume_from']
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optim_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            scaler.load_state_dict(ckpt['scaler_state'])
            start_epoch = ckpt['epoch']
            best_auroc  = ckpt['val_auroc']
            print(f"✅ Resumed from epoch {start_epoch} AUROC {best_auroc:.4f}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project = "artlens",
        name    = "phase3-multitask",
        config  = CFG,
        resume  = "allow"
    )

    # ── Training functions ────────────────────────────────────────────────────
    def train_one_epoch(epoch):
        model.train()
        total_loss = total_loss_bin = total_loss_gen = 0
        correct_bin = correct_gen = total = total_ai = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG['num_epochs']} [Train]")

        for imgs, bin_labels, gen_labels in pbar:
            imgs       = imgs.to(DEVICE, non_blocking=True)
            bin_labels = bin_labels.to(DEVICE, non_blocking=True)
            gen_labels = gen_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                logits_bin, logits_gen, _ = model(imgs)
                loss, loss_bin, loss_gen  = criterion(
                    logits_bin, logits_gen, bin_labels, gen_labels
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Fix for lr_scheduler warning — step optimizer before scheduler
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss     += loss.item()     * imgs.size(0)
            total_loss_bin += loss_bin.item() * imgs.size(0)
            total_loss_gen += loss_gen.item() * imgs.size(0)

            correct_bin += (logits_bin.detach().argmax(1) == bin_labels).sum().item()
            total       += imgs.size(0)

            # Generator accuracy only on AI images
            ai_mask = gen_labels >= 0
            if ai_mask.sum() > 0:
                correct_gen += (
                    logits_gen.detach()[ai_mask].argmax(1) == gen_labels[ai_mask]
                ).sum().item()
                total_ai += ai_mask.sum().item()

            pbar.set_postfix(
                loss    = f"{loss.item():.4f}",
                lb      = f"{loss_bin.item():.3f}",
                lg      = f"{loss_gen.item():.3f}",
                lr      = f"{scheduler.get_last_lr()[0]:.1e}"
            )

        gen_acc = correct_gen / total_ai if total_ai > 0 else 0.0
        return (
            total_loss / total,
            total_loss_bin / total,
            total_loss_gen / max(total_ai, 1),
            correct_bin / total,
            gen_acc
        )


    def evaluate(loader, split_name="Val"):
        model.eval()
        total_loss = 0
        all_bin_labels = []; all_bin_probs  = []; all_bin_preds  = []
        all_gen_labels = []; all_gen_preds  = []
        all_embeddings = []

        with torch.no_grad():
            for imgs, bin_labels, gen_labels in tqdm(loader, desc=f"[{split_name}]", leave=False):
                imgs       = imgs.to(DEVICE, non_blocking=True)
                bin_labels = bin_labels.to(DEVICE, non_blocking=True)
                gen_labels = gen_labels.to(DEVICE, non_blocking=True)

                with autocast('cuda'):
                    logits_bin, logits_gen, embeddings = model(imgs)
                    loss, _, _ = criterion(logits_bin, logits_gen, bin_labels, gen_labels)

                total_loss += loss.item() * imgs.size(0)

                bin_probs = torch.softmax(logits_bin, dim=1)[:, 1]

                all_bin_labels.extend(bin_labels.cpu().numpy())
                all_bin_probs.extend(bin_probs.cpu().numpy())
                all_bin_preds.extend(logits_bin.argmax(1).cpu().numpy())
                all_gen_labels.extend(gen_labels.cpu().numpy())
                all_gen_preds.extend(logits_gen.argmax(1).cpu().numpy())
                all_embeddings.append(embeddings.cpu().float().numpy())

        n        = len(all_bin_labels)
        bin_labels_arr = np.array(all_bin_labels)
        bin_preds_arr  = np.array(all_bin_preds)
        gen_labels_arr = np.array(all_gen_labels)
        gen_preds_arr  = np.array(all_gen_preds)
        embeddings_arr = np.vstack(all_embeddings)

        loss  = total_loss / n
        acc   = (bin_preds_arr == bin_labels_arr).mean()
        auroc = roc_auc_score(bin_labels_arr, np.array(all_bin_probs))
        f1    = f1_score(bin_labels_arr, bin_preds_arr, average='weighted')

        # Generator accuracy — only on AI images
        ai_mask     = gen_labels_arr >= 0
        gen_acc     = (gen_preds_arr[ai_mask] == gen_labels_arr[ai_mask]).mean() \
                      if ai_mask.sum() > 0 else 0.0

        return (loss, acc, auroc, f1, gen_acc,
                bin_labels_arr, np.array(all_bin_probs), bin_preds_arr,
                gen_labels_arr, gen_preds_arr, embeddings_arr)


    # ── Main training loop ────────────────────────────────────────────────────
    history    = []
    best_auroc = 0.0

    print(f"\nStarting Phase 3 from epoch {start_epoch+1}")
    print("=" * 70)

    for epoch in range(start_epoch, CFG['num_epochs']):

        tr_loss, tr_loss_bin, tr_loss_gen, tr_acc_bin, tr_acc_gen = train_one_epoch(epoch)

        val_loss, val_acc, val_auroc, val_f1, val_gen_acc, \
        _, _, _, _, _, _ = evaluate(val_loader, "Val")

        wandb.log({
            "epoch"           : epoch+1,
            "train/loss"      : tr_loss,
            "train/loss_bin"  : tr_loss_bin,
            "train/loss_gen"  : tr_loss_gen,
            "train/acc_bin"   : tr_acc_bin,
            "train/acc_gen"   : tr_acc_gen,
            "val/loss"        : val_loss,
            "val/acc"         : val_acc,
            "val/auroc"       : val_auroc,
            "val/f1"          : val_f1,
            "val/gen_acc"     : val_gen_acc,
            "lr"              : scheduler.get_last_lr()[0]
        })

        history.append({
            'epoch': epoch+1,
            'tr_loss': tr_loss, 'tr_acc_bin': tr_acc_bin, 'tr_acc_gen': tr_acc_gen,
            'val_loss': val_loss, 'val_acc': val_acc,
            'val_auroc': val_auroc, 'val_f1': val_f1, 'val_gen_acc': val_gen_acc
        })

        print(f"Epoch {epoch+1:>2}/{CFG['num_epochs']} | "
              f"loss {tr_loss:.4f} (bin {tr_loss_bin:.3f} gen {tr_loss_gen:.3f}) | "
              f"bin_acc {tr_acc_bin:.4f}  gen_acc {tr_acc_gen:.4f} | "
              f"val AUROC {val_auroc:.4f}  gen_acc {val_gen_acc:.4f}")

        # Save checkpoint
        torch.save({
            'epoch'          : epoch+1,
            'model_state'    : model.state_dict(),
            'optim_state'    : optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state'   : scaler.state_dict(),
            'val_auroc'      : val_auroc,
            'val_gen_acc'    : val_gen_acc,
            'cfg'            : CFG
        }, MODELS_DIR / f'phase3_epoch_{epoch+1:02d}_auroc{val_auroc:.4f}.pt')

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                'epoch'       : epoch+1,
                'model_state' : model.state_dict(),
                'val_auroc'   : val_auroc,
                'val_gen_acc' : val_gen_acc,
                'cfg'         : CFG
            }, MODELS_DIR / 'phase3_best_model.pt')
            print(f"  ✅ New best AUROC {val_auroc:.4f} gen_acc {val_gen_acc:.4f}")

    # # ── Save history + training curves ───────────────────────────────────────
    # hist_df = pd.DataFrame(history)
    # hist_df.to_csv(OUTPUT_DIR / 'phase3_training_history.csv', index=False)
    #
    # fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    # axes[0].plot(hist_df['epoch'], hist_df['tr_loss'],    label='Train', color='#4e79a7')
    # axes[0].plot(hist_df['epoch'], hist_df['val_loss'],   label='Val',   color='#f28e2b')
    # axes[0].set_title("Total loss"); axes[0].legend()
    #
    # axes[1].plot(hist_df['epoch'], hist_df['tr_acc_bin'], label='Train', color='#4e79a7')
    # axes[1].plot(hist_df['epoch'], hist_df['val_acc'],    label='Val',   color='#f28e2b')
    # axes[1].set_title("Binary accuracy"); axes[1].legend()
    #
    # axes[2].plot(hist_df['epoch'], hist_df['val_auroc'], color='#59a14f', linewidth=2)
    # axes[2].set_title("Val AUROC"); axes[2].set_ylim([0.4, 1.05])
    #
    # axes[3].plot(hist_df['epoch'], hist_df['tr_acc_gen'], label='Train', color='#4e79a7')
    # axes[3].plot(hist_df['epoch'], hist_df['val_gen_acc'],label='Val',   color='#f28e2b')
    # axes[3].set_title("Generator accuracy"); axes[3].legend()
    #
    # plt.suptitle("ArtLens Phase 3 — Training Curves", fontsize=13, fontweight='bold')
    # plt.tight_layout()
    # plt.savefig(RESULTS_DIR / 'phase3_training_curves.png', dpi=150, bbox_inches='tight')
    # plt.close()

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nLoading Phase 3 best model for test evaluation...")
    best_ckpt = torch.load(
        MODELS_DIR / 'phase3_best_model.pt', map_location=DEVICE, weights_only=False
    )
    model.load_state_dict(best_ckpt['model_state'])

    test_loss, test_acc, test_auroc, test_f1, test_gen_acc, \
    test_bin_labels, test_bin_probs, test_bin_preds, \
    test_gen_labels, test_gen_preds, test_embeddings = evaluate(test_loader, "Test")

    print(f"\n{'='*60}")
    print(f"  PHASE 3 TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Binary detection:")
    print(f"    Accuracy  : {test_acc*100:.2f}%")
    print(f"    AUROC     : {test_auroc:.4f}")
    print(f"    F1        : {test_f1:.4f}")
    print(f"\n  Generator fingerprinting (AI images only):")
    print(f"    Accuracy  : {test_gen_acc*100:.2f}%")
    print(f"{'='*60}")

    print("\nBinary classification report:")
    print(classification_report(
        test_bin_labels, test_bin_preds, target_names=['Human','AI']
    ))

    # Generator confusion matrix — only on AI images
    ai_mask = test_gen_labels >= 0
    if ai_mask.sum() > 0:
        print("Generator confusion matrix (AI images only):")
        cm = confusion_matrix(test_gen_labels[ai_mask], test_gen_preds[ai_mask])
        gen_labels_used = [IDX_TO_GEN[i] for i in range(N_GENERATORS)]
        cm_df = pd.DataFrame(cm, index=gen_labels_used, columns=gen_labels_used)
        print(cm_df.to_string())

        # Save generator confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(N_GENERATORS)); ax.set_yticks(range(N_GENERATORS))
        ax.set_xticklabels(gen_labels_used); ax.set_yticklabels(gen_labels_used)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Generator Fingerprinting — Confusion Matrix")
        plt.colorbar(im, ax=ax)
        for i in range(N_GENERATORS):
            for j in range(N_GENERATORS):
                ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                        color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=13)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'phase3_generator_cm.png', dpi=150)
        plt.close()

    # ── Head 3: Mahalanobis open-set detection ────────────────────────────────
    print("\nComputing Mahalanobis statistics for open-set detection (Head 3)...")

    # Collect embeddings per class from test set
    # In production this would use the full training set embeddings
    # Here we use test embeddings to validate the concept
    class_embeddings = {}
    for label_idx, label_name in enumerate(['human', 'midjourney', 'stable_diffusion']):
        if label_idx == 0:
            # Human — use binary label
            mask = test_bin_labels == 0
        else:
            # AI generators — use gen labels
            gen_idx = label_idx - 1
            mask = (test_gen_labels == gen_idx) & (test_bin_labels == 1)

        if mask.sum() > 10:   # need enough samples to estimate covariance
            class_embeddings[label_name] = test_embeddings[mask]

    # Compute mean and covariance for each class
    class_stats = {}
    for class_name, embeds in class_embeddings.items():
        mean = embeds.mean(axis=0)
        # Regularised covariance — add small diagonal term to prevent singular matrix
        # Singular covariance = non-invertible = Mahalanobis undefined
        cov  = np.cov(embeds.T) + np.eye(embeds.shape[1]) * 1e-6
        try:
            cov_inv = np.linalg.inv(cov)
            class_stats[class_name] = {'mean': mean, 'cov_inv': cov_inv}
            print(f"  {class_name:<20} {len(embeds):,} embeddings — covariance computed")
        except np.linalg.LinAlgError:
            print(f"  ⚠️  {class_name} — covariance matrix singular, skipping")

    # Save class statistics for use in Phase 6 (FastAPI inference)
    # Saved as numpy arrays — load with np.load in deployment
    stats_path = MODELS_DIR / 'phase3_mahalanobis_stats.npz'
    np.savez(stats_path, **{
        f"{cls}_{key}": val
        for cls, stats in class_stats.items()
        for key, val in stats.items()
    })
    print(f"  Mahalanobis statistics saved → {stats_path}")

    # Quick open-set test — compute distance of each test embedding to each class
    print("\nOpen-set detection sample (first 5 test images):")
    print(f"  {'True label':<20} {'Nearest class':<22} {'Min distance':>12}")
    print("-" * 58)
    for i in range(min(5, len(test_embeddings))):
        emb = test_embeddings[i]
        distances = {}
        for cls, stats in class_stats.items():
            diff = emb - stats['mean']
            try:
                d = float(np.sqrt(diff @ stats['cov_inv'] @ diff))
                distances[cls] = d
            except Exception:
                distances[cls] = float('inf')

        nearest     = min(distances, key=distances.get)
        min_dist    = distances[nearest]
        true_binary = 'human' if test_bin_labels[i] == 0 else 'ai'
        print(f"  {true_binary:<20} {nearest:<22} {min_dist:>12.2f}")

    # ── Per-source breakdown ───────────────────────────────────────────────────
    print("\nPer-source accuracy (binary):")
    test_df_eval            = test_df.copy()
    test_df_eval['pred']    = test_bin_preds
    test_df_eval['correct'] = (test_df_eval['pred'] == test_df_eval['label']).astype(int)
    for source in sorted(test_df_eval['source'].unique()):
        acc  = test_df_eval[test_df_eval['source']==source]['correct'].mean()
        flag = "⚠️" if acc < 0.80 else "✅"
        print(f"  {flag} {source:<25} {acc:.4f}")

    # Save results
    with open(OUTPUT_DIR / 'phase3_results.json', 'w') as f:
        json.dump({
            'best_epoch'    : int(best_ckpt['epoch']),
            'best_val_auroc': float(best_auroc),
            'test_accuracy' : float(test_acc),
            'test_auroc'    : float(test_auroc),
            'test_f1'       : float(test_f1),
            'test_gen_acc'  : float(test_gen_acc),
        }, f, indent=2)

    wandb.log({
        "test/acc"     : test_acc,
        "test/auroc"   : test_auroc,
        "test/f1"      : test_f1,
        "test/gen_acc" : test_gen_acc
    })
    wandb.finish()

    print(f"\n✅ Phase 3 complete")
    print(f"   Best val AUROC        : {best_auroc:.4f}")
    print(f"   Test AUROC            : {test_auroc:.4f}")
    print(f"   Generator accuracy    : {test_gen_acc*100:.2f}%")
    print(f"   Mahalanobis stats     : {stats_path}")
    print(f"   Best model            : {MODELS_DIR}/phase3_best_model.pt")
    print(f"   Outputs               : D:/ArtLens/outputs/")