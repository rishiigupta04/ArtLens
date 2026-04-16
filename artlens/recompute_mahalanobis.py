# recompute_mahalanobis.py
# Run with: uv run python recompute_mahalanobis.py
# Uses full training set embeddings — fixes the rank-deficiency problem

import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path('D:/ArtLens/outputs')
MODELS_DIR = OUTPUT_DIR / 'models'
EMBED_DIR  = OUTPUT_DIR / 'embeddings'

# Load cached embeddings (all 31594 images in order: train, val, test)
emb_cache = np.load(EMBED_DIR / 'phase3_embeddings.npz')['embeddings']

# Load splits to get indices
train_df = pd.read_csv(OUTPUT_DIR / 'split_train.csv')
val_df   = pd.read_csv(OUTPUT_DIR / 'split_val.csv')
n_train  = len(train_df)
n_val    = len(val_df)

# Use train + val for statistics — more samples = better covariance estimate
# Never use test set for this — that's the bug we're fixing
train_val_df = pd.concat([train_df, val_df]).reset_index(drop=True)
train_val_emb = emb_cache[:n_train + n_val]   # first N rows = train+val

print(f"Computing stats on {len(train_val_df):,} train+val embeddings")

def remap_path(row):
    from pathlib import Path
    fname = Path(row['path']).name
    if row['label'] == 0:
        return 'human'
    return 'midjourney' if row['generator'] == 'midjourney' else 'stable_diffusion'

train_val_df['class'] = train_val_df.apply(remap_path, axis=1)

class_stats = {}

for cls in ['human', 'midjourney', 'stable_diffusion']:
    mask   = train_val_df['class'] == cls
    embeds = train_val_emb[mask.values]

    print(f"\n{cls}: {len(embeds):,} embeddings")

    mean = embeds.mean(axis=0)

    # Use shrinkage estimator (Ledoit-Wolf) instead of raw covariance
    # Critical when n_samples is close to n_dimensions
    # Ledoit-Wolf shrinks the covariance toward a diagonal matrix
    # reducing estimation error from finite samples
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf(assume_centered=False)
    lw.fit(embeds)

    # Get the precision matrix (= inverse covariance) directly
    # More numerically stable than computing cov then inverting
    precision = lw.precision_   # this IS the inverse covariance

    # Sanity check: compute distance to own mean (should be ~sqrt(768) ≈ 27.7)
    diff      = mean - mean   # zero vector
    d_self    = float(np.sqrt(diff @ precision @ diff))
    print(f"  Distance to own mean: {d_self:.2f} (should be 0.00)")

    # Compute a few sample distances to verify scale
    sample_dists = []
    for i in range(min(100, len(embeds))):
        d = embeds[i] - mean
        dist = float(np.sqrt(d @ precision @ d))
        sample_dists.append(dist)
    print(f"  Sample distances — mean: {np.mean(sample_dists):.2f}  "
          f"std: {np.std(sample_dists):.2f}  "
          f"95th pct: {np.percentile(sample_dists, 95):.2f}")
    print(f"  Suggested threshold: {np.percentile(sample_dists, 95) * 1.5:.2f}")

    class_stats[cls] = {
        'mean'     : mean,
        'precision': precision,   # save precision not cov_inv for consistency
        'n_samples': len(embeds),
        'dist_mean': float(np.mean(sample_dists)),
        'dist_std' : float(np.std(sample_dists)),
        'dist_p95' : float(np.percentile(sample_dists, 95)),
    }

# Save — new format with precision matrix and distance statistics
np.savez(
    MODELS_DIR / 'mahalanobis_stats_v2.npz',
    **{f'{cls}_{k}': v for cls, stats in class_stats.items()
       for k, v in stats.items() if isinstance(v, np.ndarray)}
)

# Save thresholds separately as JSON for easy reading
import json
thresholds = {
    cls: {
        'mean_dist'        : stats['dist_mean'],
        'std_dist'         : stats['dist_std'],
        'p95_dist'         : stats['dist_p95'],
        'recommended_threshold': stats['dist_p95'] * 1.5,
        'n_samples'        : stats['n_samples'],
    }
    for cls, stats in class_stats.items()
}
with open(MODELS_DIR / 'mahalanobis_thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)

print("\n✅ Recomputed stats saved:")
print(f"   {MODELS_DIR}/mahalanobis_stats_v2.npz")
print(f"   {MODELS_DIR}/mahalanobis_thresholds.json")
print("\nUse these thresholds in app.py for open-set detection")