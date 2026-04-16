from huggingface_hub import HfApi, create_repo
from pathlib import Path

REPO_ID   = "rishigupta04/ArtLens"   # ← replace
MODELS_DIR = Path('D:/ArtLens/outputs/models')

# Create repo if it doesn't exist
create_repo(REPO_ID, repo_type="model", exist_ok=True)

api = HfApi()

# Upload all model files
files_to_push = [
    'phase3_best_model.pt',
    'lgb_model.txt',
    'feature_scaler.pkl',
    'phase3_mahalanobis_stats.npz',
    'temperature.json',
]

for fname in files_to_push:
    path = MODELS_DIR / fname
    if path.exists():
        print(f"Uploading {fname}...")
        api.upload_file(
            path_or_fileobj = str(path),
            path_in_repo    = fname,
            repo_id         = REPO_ID,
            repo_type       = "model"
        )
        print(f"  ✅ {fname} uploaded")
    else:
        print(f"  ⚠️  {fname} not found — skipping")

print(f"\n✅ All files pushed to https://huggingface.co/{REPO_ID}")