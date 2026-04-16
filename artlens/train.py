# ── ArtLens Phase 2 — local training via uv ───────────────────────────────────
# Run with: uv run python train.py

import os, random, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# FutureWarning fix — use torch.amp instead of torch.cuda.amp
from torch.amp import GradScaler, autocast

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report

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

for d in [OUTPUT_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    'img_size'    : 224,
    'batch_size'  : 16,
    'num_epochs'  : 15,
    'lr'          : 2e-5,
    'weight_decay': 1e-4,
    # Windows multiprocessing fix — num_workers > 0 requires __main__ guard
    # which we have below. 2 is safe on Windows, 4 can cause issues.
    'num_workers' : 2,
    'val_split'   : 0.15,
    'test_split'  : 0.10,
    'seed'        : 42,
    'model_name'  : 'vit_base_patch16_224',
    'pretrained'  : True,
    'resume_from' : None,
}

# ── Dataset class — defined at module level so Windows multiprocessing can pickle it
class ArtLensDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row['label'])
        try:
            img = np.array(Image.open(row['path']).convert('RGB'))
        except Exception:
            img = np.zeros((CFG['img_size'], CFG['img_size'], 3), dtype=np.uint8)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, torch.tensor(label, dtype=torch.long)


# ── Model class — defined at module level so Windows multiprocessing can pickle it
class ArtLensBinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG['model_name'], pretrained=CFG['pretrained'], num_classes=0
        )
        embed_dim = self.backbone.num_features    # 768 for ViT-B/16
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=0.3),
            nn.Linear(embed_dim, 2)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def get_embeddings(self, x):
        return self.backbone(x)


# ── CRITICAL on Windows: all training code must be inside this guard ──────────
# Windows uses 'spawn' to create new processes (unlike Linux which uses 'fork')
# Without this guard, every worker process re-runs the entire script on startup
# causing an infinite loop of process spawning → the RuntimeError you saw
if __name__ == '__main__':

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Load manifest ─────────────────────────────────────────────────────────
    df = pd.read_csv(MANIFEST)
    print(f"Manifest loaded: {len(df):,} rows")
    print("Label distribution:")
    print(df['label_name'].value_counts().to_string())
    print("\nSource distribution:")
    print(df['source'].value_counts().to_string())

    def remap_path(row):
        """
        Manifest has Colab Drive paths — remap to local D:/ArtLens/data structure.
        Uses source column not filename prefix — works regardless of filenames.
        """
        fname = Path(row['path']).name      # just filename, discard Colab path

        if row['label'] == 0:               # human art → data/human/
            return str(DATA_DIR / 'human' / fname)
        else:                               # AI art → data/ai/mj/ or data/ai/sd/
            folder = 'mj' if row['generator'] == 'midjourney' else 'sd'
            return str(DATA_DIR / 'ai' / folder / fname)

    df['path'] = df.apply(remap_path, axis=1)

    # ── Debug: show what paths look like before filtering ─────────────────────
    print(f"\nSample remapped paths:")
    for _, row in df.sample(4, random_state=42).iterrows():
        exists = "✅" if Path(row['path']).exists() else "❌"
        print(f"  {exists} {row['path']}")

    # Remove missing files
    before = len(df)
    df     = df[df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    print(f"\nRemoved {before - len(df):,} missing | {len(df):,} images verified")
    print(df['source'].value_counts().to_string())

    if len(df) == 0:
        raise RuntimeError("No images found on disk. Check DATA_DIR path and folder structure.")

    # ── Splits ────────────────────────────────────────────────────────────────
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

    train_df.to_csv(OUTPUT_DIR / 'split_train.csv', index=False)
    val_df.to_csv(  OUTPUT_DIR / 'split_val.csv',   index=False)
    test_df.to_csv( OUTPUT_DIR / 'split_test.csv',  index=False)

    print(f"\nSplits:")
    print(f"  Train : {len(train_df):,}")
    print(f"  Val   : {len(val_df):,}")
    print(f"  Test  : {len(test_df):,}")

    # ── Augmentation ─────────────────────────────────────────────────────────
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_transform = A.Compose([
        A.Resize(CFG['img_size'], CFG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.OneOf([
            # API fix — newer albumentations uses quality_range tuple not quality_lower/upper
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
        ArtLensDataset(train_df, train_transform),
        batch_size=CFG['batch_size'], shuffle=True,
        num_workers=CFG['num_workers'], pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        ArtLensDataset(val_df, val_transform),
        batch_size=CFG['batch_size']*2, shuffle=False,
        num_workers=CFG['num_workers'], pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        ArtLensDataset(test_df, val_transform),
        batch_size=CFG['batch_size']*2, shuffle=False,
        num_workers=CFG['num_workers'], pin_memory=True,
        persistent_workers=True
    )
    print(f"\nDataLoaders: train {len(train_loader)} | val {len(val_loader)} | test {len(test_loader)} batches")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ArtLensBinaryModel().to(DEVICE)
    model.backbone.set_grad_checkpointing(enable=True)
    print(f"Model ready — {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── Training setup ────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay']
    )
    total_steps = CFG['num_epochs'] * len(train_loader)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )
    # FutureWarning fix — pass device string explicitly
    scaler = GradScaler('cuda')

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    best_auroc  = 0.0

    if CFG['resume_from']:
        ckpt_path = MODELS_DIR / CFG['resume_from']
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optim_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            scaler.load_state_dict(ckpt['scaler_state'])
            start_epoch = ckpt['epoch']
            best_auroc  = ckpt['val_auroc']
            print(f"✅ Resumed from epoch {start_epoch} — best AUROC {best_auroc:.4f}")
        else:
            print(f"⚠️  Checkpoint not found — starting fresh")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project="artlens",
        name="phase2-local-gpu",
        config=CFG,
        resume="allow"
    )

    # ── Training functions ────────────────────────────────────────────────────
    def train_one_epoch(epoch):
        model.train()
        total_loss = correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG['num_epochs']} [Train]")
        for imgs, labels in pbar:
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # fp16 forward pass
            with autocast('cuda'):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.detach().argmax(1) == labels).sum().item()
            total      += imgs.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.1e}"
            )

        return total_loss / total, correct / total


    def evaluate(loader, split_name="Val"):
        model.eval()
        total_loss = 0
        all_labels, all_probs, all_preds = [], [], []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f"[{split_name}]", leave=False):
                imgs   = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with autocast('cuda'):
                    logits = model(imgs)
                    loss   = criterion(logits, labels)

                total_loss += loss.item() * imgs.size(0)
                probs = torch.softmax(logits, dim=1)[:, 1]

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(logits.argmax(1).cpu().numpy())

        n     = len(all_labels)
        loss  = total_loss / n
        acc   = sum(p == l for p, l in zip(all_preds, all_labels)) / n
        auroc = roc_auc_score(all_labels, all_probs)
        f1    = f1_score(all_labels, all_preds, average='weighted')

        return loss, acc, auroc, f1, \
               np.array(all_labels), np.array(all_probs), np.array(all_preds)


    # ── Main loop ─────────────────────────────────────────────────────────────
    history = []
    print(f"\nStarting from epoch {start_epoch+1}")
    print("=" * 65)

    for epoch in range(start_epoch, CFG['num_epochs']):

        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc, val_auroc, val_f1, _, _, _ = evaluate(val_loader, "Val")

        wandb.log({
            "epoch"      : epoch+1,
            "train/loss" : train_loss, "train/acc": train_acc,
            "val/loss"   : val_loss,   "val/acc"  : val_acc,
            "val/auroc"  : val_auroc,  "val/f1"   : val_f1,
            "lr"         : scheduler.get_last_lr()[0]
        })

        history.append({
            'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
            'val_auroc': val_auroc, 'val_f1': val_f1
        })

        print(f"Epoch {epoch+1:>2}/{CFG['num_epochs']} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} "
              f"AUROC {val_auroc:.4f} F1 {val_f1:.4f}")

        # Save full checkpoint — safe to kill any time and resume
        torch.save({
            'epoch'          : epoch+1,
            'model_state'    : model.state_dict(),
            'optim_state'    : optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state'   : scaler.state_dict(),
            'val_auroc'      : val_auroc,
            'val_f1'         : val_f1,
            'cfg'            : CFG
        }, MODELS_DIR / f'epoch_{epoch+1:02d}_auroc{val_auroc:.4f}.pt')

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                'epoch'      : epoch+1,
                'model_state': model.state_dict(),
                'val_auroc'  : val_auroc,
                'cfg'        : CFG
            }, MODELS_DIR / 'best_model.pt')
            print(f"  ✅ New best AUROC {val_auroc:.4f} → best_model.pt")

    # ── Save history + plots ──────────────────────────────────────────────────
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUTPUT_DIR / 'training_history.csv', index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(hist_df['epoch'], hist_df['train_loss'], label='Train', color='#4e79a7')
    axes[0].plot(hist_df['epoch'], hist_df['val_loss'],   label='Val',   color='#f28e2b')
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(hist_df['epoch'], hist_df['train_acc'], label='Train', color='#4e79a7')
    axes[1].plot(hist_df['epoch'], hist_df['val_acc'],   label='Val',   color='#f28e2b')
    axes[1].set_title("Accuracy"); axes[1].legend()

    axes[2].plot(hist_df['epoch'], hist_df['val_auroc'], color='#59a14f', linewidth=2)
    axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_title("Val AUROC"); axes[2].set_ylim([0.4, 1.05])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\nLoading best model for test evaluation...")
    best_ckpt = torch.load(MODELS_DIR / 'best_model.pt', map_location=DEVICE)
    model.load_state_dict(best_ckpt['model_state'])

    test_loss, test_acc, test_auroc, test_f1, \
    test_labels, test_probs, test_preds = evaluate(test_loader, "Test")

    print(f"\n{'='*55}")
    print(f"  FINAL TEST RESULTS")
    print(f"  Accuracy : {test_acc*100:.2f}%")
    print(f"  AUROC    : {test_auroc:.4f}")
    print(f"  F1       : {test_f1:.4f}")
    print(f"{'='*55}")
    print(classification_report(test_labels, test_preds, target_names=['Human','AI']))

    # Per-source breakdown
    print("Per-source accuracy:")
    test_df_eval            = test_df.copy()
    test_df_eval['pred']    = test_preds
    test_df_eval['correct'] = (test_df_eval['pred'] == test_df_eval['label']).astype(int)
    for source in sorted(test_df_eval['source'].unique()):
        acc  = test_df_eval[test_df_eval['source']==source]['correct'].mean()
        flag = "⚠️" if acc < 0.80 else "✅"
        print(f"  {flag} {source:<25} {acc:.4f}")

    # Save results
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump({
            'best_epoch'    : int(best_ckpt['epoch']),
            'best_val_auroc': float(best_auroc),
            'test_accuracy' : float(test_acc),
            'test_auroc'    : float(test_auroc),
            'test_f1'       : float(test_f1),
        }, f, indent=2)

    wandb.log({"test/acc": test_acc, "test/auroc": test_auroc, "test/f1": test_f1})
    wandb.finish()

    print(f"\n✅ Phase 2 complete")
    print(f"   Best val AUROC : {best_auroc:.4f}")
    print(f"   Test AUROC     : {test_auroc:.4f}")
    print(f"   Outputs        : D:/ArtLens/outputs/")