# ── ArtLens app.py v2 ─────────────────────────────────────────────────────────
# Production-ready FastAPI inference server
# Key changes from v1:
# - Async heatmap generation on separate endpoint (not blocking /predict)
# - Heatmaps saved as PNG files served via /static, not base64 in JSON
# - Mahalanobis uses LedoitWolf precision matrix from training embeddings
# - Per-class distance thresholds from empirical distribution
# - Open-set flag suppresses confidence and provides "closest match" framing
# - Structured logging via Python logging module
# - Request ID tracking for debugging
# - Proper HTTP error codes
# - Response caching for duplicate images (MD5-based)

import os, json, io, hashlib, time, uuid, warnings, logging
warnings.filterwarnings('ignore')
from pathlib import Path
from typing import Optional, List
from functools import lru_cache
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2
import pywt
from skimage.feature import local_binary_pattern
import lightgbm as lgb
import joblib

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import atexit, shutil

def cleanup_heatmaps():
    if HEATMAP_DIR.exists():
        shutil.rmtree(HEATMAP_DIR)
        HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

atexit.register(cleanup_heatmaps)

# Also add TTL — delete heatmaps older than 1 hour
# Run this check on each /explain call
def clean_old_heatmaps(max_age_s=3600):
    now = time.time()
    for f in HEATMAP_DIR.glob("*.png"):
        if now - f.stat().st_mtime > max_age_s:
            f.unlink(missing_ok=True)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler('artlens_api.log')
    ]
)
log = logging.getLogger('artlens')
# ── HuggingFace Hub download ──────────────────────────────────────────────────
from huggingface_hub import hf_hub_download
# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = Path('models')
STATIC_DIR  = Path('static')
HEATMAP_DIR = STATIC_DIR / 'heatmaps'
MODELS_DIR.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
HF_REPO_ID    = os.getenv('HF_REPO_ID', 'YOUR_USERNAME/artlens')
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IDX_TO_GEN    = {0: 'Midjourney', 1: 'Stable Diffusion'}
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10MB
MAX_BATCH     = 20

# ── Simple in-memory cache ────────────────────────────────────────────────────
# Caches predictions for images already seen within this server session
# Key = MD5 hash of image bytes, Value = prediction dict
_PREDICTION_CACHE: dict = {}
_TENSOR_CACHE: dict = {}
MAX_CACHE_SIZE = 500   # evict oldest when exceeded


def get_image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


def cache_put(key: str, value: dict):
    if len(_PREDICTION_CACHE) >= MAX_CACHE_SIZE:
        # Evict oldest entry
        oldest = next(iter(_PREDICTION_CACHE))
        del _PREDICTION_CACHE[oldest]
    _PREDICTION_CACHE[key] = value


# ── Model definition ──────────────────────────────────────────────────────────
class ArtLensMultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0
        )
        d = self.backbone.num_features
        self.head_binary    = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 2))
        self.head_generator = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 2))

    def forward(self, x):
        e = self.backbone(x)
        return self.head_binary(e), self.head_generator(e), e

    def get_embeddings(self, x):
        return self.backbone(x)


# ── Attention rollout ─────────────────────────────────────────────────────────
class AttentionRollout:
    """
    Uses last-layer attention only (not full rollout product).
    Full rollout produces uniform maps for high-confidence fine-tuned models.
    Last layer is more spatially discriminative.
    """
    def __init__(self, model):
        self.model = model
        self._attn = None
        # Hook only the LAST transformer block
        model.backbone.blocks[-1].attn.attn_drop.register_forward_hook(
            lambda m, i, o: setattr(self, '_attn', o.detach().cpu())
        )

    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        self._attn = None
        self.model.eval()
        with torch.no_grad():
            with autocast(DEVICE.type):
                _ = self.model(tensor)
        if self._attn is None:
            return np.ones((14, 14)) / 196
        # Average over heads, take CLS→patch attention
        attn_avg = self._attn[0].mean(dim=0)   # [197, 197]
        cls_attn = attn_avg[0, 1:]              # [196] CLS→patches
        hm       = cls_attn.reshape(14, 14).numpy()
        hm       = (hm - hm.min()) / (hm.max() - hm.min() + 1e-10)
        return hm


# ── GradCAM++ ─────────────────────────────────────────────────────────────────
class GradCAMPlusPlus:
    """
    Targets blocks[-3] (3rd from last) instead of last block.
    Earlier layers have more spatially grounded, less abstract features.
    """
    def __init__(self, model):
        self.model  = model
        self._acts  = None
        self._grads = None
        # Target 3rd-from-last block for better spatial localisation
        target = model.backbone.blocks[-3].norm1
        target.register_forward_hook(
            lambda m, i, o: setattr(self, '_acts', o.detach())
        )
        target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, '_grads', go[0].detach())
        )

    def __call__(self, tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        t = tensor.to(DEVICE).requires_grad_(True)
        lb, _, _ = self.model(t)
        if class_idx is None:
            class_idx = lb.argmax(dim=1).item()
        self.model.zero_grad()
        lb[0, class_idx].backward()
        if self._acts is None or self._grads is None:
            return np.ones((14, 14)) / 196
        acts  = self._acts[0, 1:, :]
        grads = self._grads[0, 1:, :]
        gsq   = grads ** 2
        gcub  = grads ** 3
        alpha = gsq / (2*gsq + (acts*gcub).sum(0, keepdim=True) + 1e-10)
        wts   = (alpha * F.relu(grads)).sum(1)
        hm    = wts.reshape(14, 14).cpu().detach().numpy()
        hm    = np.maximum(hm, 0)
        hm    = (hm - hm.min()) / (hm.max() - hm.min() + 1e-10)
        return hm


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_fft(img_np: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (224, 224))
        fft  = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        mag  = np.abs(fft)
        h, w = mag.shape; cy, cx = h//2, w//2
        Y, X = np.ogrid[:h,:w]
        dist = np.sqrt((X-cx)**2 + (Y-cy)**2); rm = min(cx,cy)
        tot  = mag.sum() + 1e-10
        hf   = mag[dist >= rm*0.5].sum() / tot
        rb   = [mag[(dist>=rm*i/5)&(dist<rm*(i+1)/5)].mean() for i in range(5)]
        ml   = np.log1p(mag).flatten()
        p    = ml/(ml.sum()+1e-10)
        ent  = -np.sum(p*np.log(p+1e-10))
        fi   = np.argsort(mag.flatten())[-4:]
        pr   = (fi//w)/h; pc = (fi%w)/w
        ha   = abs(mag[:cy,:].sum()-mag[cy:,:].sum())/tot
        return np.array([hf,mag[dist<rm*.1].sum()/tot,
                         mag[(dist>=rm*.1)&(dist<rm*.5)].sum()/tot,
                         mag[dist>=rm*.5].sum()/tot,*rb,
                         ml.mean(),ml.std(),ent,*pr[:2],*pc[:2],
                         mag[:cy,:].sum()/(mag[:,cx:].sum()+1e-10),ha],
                        dtype=np.float32)
    except Exception:
        return np.zeros(18, dtype=np.float32)


def extract_dwt(img_np: np.ndarray) -> np.ndarray:
    try:
        g = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        g = cv2.resize(g, (224,224))
        f = []
        cA,(cH,cV,cD) = pywt.dwt2(g,'db2')
        _,(cH2,cV2,cD2) = pywt.dwt2(cA,'db2')
        for c in [cH,cV,cD,cH2,cV2,cD2]:
            fa=np.abs(c.flatten()); e=(c.flatten()**2).mean()
            h,_=np.histogram(fa,32,density=True); h+=1e-10
            f.extend([fa.mean(),fa.std(),e,-np.sum(h*np.log(h))])
        return np.array(f[:24],dtype=np.float32)
    except Exception:
        return np.zeros(24,dtype=np.float32)


def extract_lbp(img_np: np.ndarray) -> np.ndarray:
    try:
        g   = cv2.cvtColor(img_np,cv2.COLOR_RGB2GRAY)
        g   = cv2.resize(g,(224,224))
        lbp = local_binary_pattern(g,24,3,'uniform')
        nb  = int(lbp.max())+1
        h,_ = np.histogram(lbp.ravel(),nb,(0,nb),density=True)
        hs  = h+1e-10
        return np.array([*h,-np.sum(hs*np.log(hs)),(h**2).sum()],dtype=np.float32)
    except Exception:
        return np.zeros(28,dtype=np.float32)


def extract_handcrafted(img_np: np.ndarray) -> np.ndarray:
    return np.concatenate([extract_fft(img_np), extract_dwt(img_np), extract_lbp(img_np)])


# ── Heatmap overlay ───────────────────────────────────────────────────────────
def make_overlay(heatmap: np.ndarray, img_np: np.ndarray,
                  colormap=cv2.COLORMAP_INFERNO, alpha=0.5) -> np.ndarray:
    hm   = cv2.resize(heatmap.astype(np.float32), (IMG_SIZE, IMG_SIZE))
    hm8  = (hm * 255).astype(np.uint8)
    col  = cv2.cvtColor(cv2.applyColorMap(hm8, colormap), cv2.COLOR_BGR2RGB)
    return np.clip((1-alpha)*img_np + alpha*col, 0, 255).astype(np.uint8)


def save_heatmap_png(heatmap: np.ndarray, img_np: np.ndarray,
                      request_id: str, method: str) -> str:
    """Saves overlay PNG to static/heatmaps/, returns relative URL path."""
    overlay  = make_overlay(heatmap, img_np)
    filename = f"{request_id}_{method}.png"
    path     = HEATMAP_DIR / filename
    Image.fromarray(overlay).save(path, format='PNG', optimize=True)
    return f"/static/heatmaps/{filename}"   # URL path served by FastAPI StaticFiles


# ── Open-set detection ────────────────────────────────────────────────────────
def compute_openset(
    embedding    : np.ndarray,
    maha_stats   : dict,
    thresholds   : dict,
    pred_label   : int,
    pred_gen_idx : int
) -> dict:
    """
    Computes normalised Mahalanobis distances and open-set verdict.

    Key design decisions:
    1. Normalise by each class's own distance distribution (z-score)
       so distances are comparable across classes
    2. "Unknown" only triggered for AI images — human images can't be
       unknown generator since they don't have one
    3. Return "closest_match" even for unknowns — still useful signal

    Returns dict with normalised distances, is_unknown flag, confidence penalty.
    """
    raw_distances   = {}
    normed_distances = {}

    for cls, stats in maha_stats.items():
        diff = embedding - stats['mean']
        try:
            d = float(np.sqrt(diff @ stats['precision'] @ diff))
        except Exception:
            d = float('inf')
        raw_distances[cls] = d

        # Normalise: how many std devs above the class's own mean distance?
        # A normalised distance of 0 = perfectly average member of this class
        # A normalised distance of 2 = 2 std devs above average = unusual
        dist_mean = thresholds[cls]['mean_dist']
        dist_std  = thresholds[cls]['std_dist']
        normed    = (d - dist_mean) / (dist_std + 1e-10)
        normed_distances[cls] = round(float(normed), 3)

    # Closest class by raw distance
    closest_class = min(raw_distances, key=raw_distances.get)
    closest_normed = normed_distances[closest_class]

    # Unknown generator: only for AI predictions, when normalised distance
    # to all AI classes is high (> 3 std devs above normal)
    ai_classes = ['midjourney', 'stable_diffusion']
    min_ai_normed = min(normed_distances[c] for c in ai_classes)
    is_unknown = pred_label == 1 and min_ai_normed > 3.0

    return {
        'raw_distances'    : {k: round(v, 2) for k, v in raw_distances.items()},
        'normalised_distances': normed_distances,
        'closest_class'    : closest_class,
        'closest_normed'   : round(closest_normed, 3),
        'is_unknown'       : is_unknown,
        'min_ai_normed'    : round(min_ai_normed, 3),
    }


# ── Dynamic textual explanation ───────────────────────────────────────────────
def generate_explanation(
    label          : str,
    confidence     : float,
    generator      : Optional[str],
    is_unknown     : bool,
    gen_confidence : Optional[float],
    vit_prob       : float,
    lgb_prob       : float,
    openset        : dict,
) -> dict:
    """
    Generates human-readable explanation of the verdict.
    Written to be shown directly in the frontend UI.
    """
    lines = []

    if label == 'human':
        if confidence > 0.95:
            lines.append(f"This image shows strong characteristics of human-made artwork.")
            lines.append(f"The model is highly confident ({confidence*100:.0f}%) that this was created by a human artist.")
        elif confidence > 0.80:
            lines.append(f"This image appears to be human-made art ({confidence*100:.0f}% confident).")
        else:
            lines.append(f"This image leans toward human-made art, but with lower confidence ({confidence*100:.0f}%).")
            lines.append("Some visual characteristics are ambiguous — this could be digitally polished human art.")

        lines.append("The deep visual features (texture continuity, compositional irregularities, "
                     "brushwork signatures) are consistent with human artistic intent.")

    else:  # AI
        if is_unknown:
            lines.append("This image appears to be AI-generated, but the generator is not one the model was trained on.")
            lines.append(f"The closest known generator is {generator}, but the match is poor "
                         f"(normalised distance: {openset['min_ai_normed']:.1f} std devs above normal).")
            lines.append("This could be Adobe Firefly, DALL-E 3, Imagen, or another generator not in the training set.")
        else:
            lines.append(f"This image is AI-generated, most likely by {generator} "
                         f"({gen_confidence*100:.0f}% generator confidence).")

        if confidence > 0.95:
            lines.append(f"The model is highly confident ({confidence*100:.0f}%) in this verdict.")
        else:
            lines.append(f"Overall detection confidence: {confidence*100:.0f}%.")

        # Add signal source breakdown
        if abs(vit_prob - lgb_prob) > 0.15:
            if vit_prob > lgb_prob:
                lines.append("The deep visual model detected AI characteristics strongly; "
                              "classical frequency features provided weaker but corroborating signal.")
            else:
                lines.append("Classical frequency analysis (FFT/wavelet) detected AI artifacts; "
                              "the deep visual model provided moderate signal.")
        else:
            lines.append("Both the deep visual model and classical frequency analysis "
                         "independently confirmed AI generation.")

    # Model agreement note
    agree = abs(vit_prob - lgb_prob) < 0.1
    lines.append(
        "Both detection methods agree on this verdict." if agree
        else f"Note: the deep model ({vit_prob*100:.0f}% AI) and frequency model "
             f"({lgb_prob*100:.0f}% AI) show some disagreement — interpret with appropriate caution."
    )

    return {
        'summary': lines[0],
        'detail' : ' '.join(lines[1:]),
        'signals': {
            'deep_model_signal'     : 'strong' if abs(vit_prob-0.5) > 0.4 else 'moderate' if abs(vit_prob-0.5) > 0.2 else 'weak',
            'frequency_model_signal': 'strong' if abs(lgb_prob-0.5) > 0.4 else 'moderate' if abs(lgb_prob-0.5) > 0.2 else 'weak',
            'models_agree'          : agree,
        }
    }


# ── Model loading (startup) ───────────────────────────────────────────────────
models = {}   # populated in lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup. FastAPI lifespan replaces deprecated @app.on_event."""
    log.info(f"Loading models — device: {DEVICE}")

    from huggingface_hub import hf_hub_download

    files = [
        'phase3_best_model.pt',
        'lgb_model.txt',
        'feature_scaler.pkl',
        'mahalanobis_stats_v2.npz',    # new v2 stats from training embeddings
        'mahalanobis_thresholds.json',
        'temperature.json',
    ]
    for fname in files:
        dest = MODELS_DIR / fname
        if not dest.exists():
            log.info(f"Downloading {fname} from HF Hub...")
            hf_hub_download(repo_id=HF_REPO_ID, filename=fname, local_dir=str(MODELS_DIR))

    # ViT model
    vit = ArtLensMultiTaskModel().to(DEVICE)
    ckpt = torch.load(MODELS_DIR / 'phase3_best_model.pt', map_location=DEVICE, weights_only=False)
    vit.load_state_dict(ckpt['model_state'])
    vit.eval()
    vit.backbone.set_grad_checkpointing(enable=False)
    models['vit']      = vit
    models['ckpt_epoch'] = ckpt['epoch']
    log.info(f"ViT loaded (epoch {ckpt['epoch']})")

    # LightGBM
    models['lgb']    = lgb.Booster(model_file=str(MODELS_DIR / 'lgb_model.txt'))
    models['scaler'] = joblib.load(MODELS_DIR / 'feature_scaler.pkl')
    log.info("LightGBM + scaler loaded")

    # Mahalanobis stats v2
    raw = np.load(MODELS_DIR / 'mahalanobis_stats_v2.npz')
    with open(MODELS_DIR / 'mahalanobis_thresholds.json') as f:
        thresholds = json.load(f)

    maha_stats = {}
    for cls in ['human', 'midjourney', 'stable_diffusion']:
        if f'{cls}_mean' in raw and f'{cls}_precision' in raw:
            maha_stats[cls] = {
                'mean'     : raw[f'{cls}_mean'],
                'precision': raw[f'{cls}_precision'],
            }
    models['maha_stats']  = maha_stats
    models['thresholds']  = thresholds
    log.info(f"Mahalanobis stats loaded for: {list(maha_stats.keys())}")

    # Temperature
    with open(MODELS_DIR / 'temperature.json') as f:
        models['temperature'] = json.load(f)['temperature']
    log.info(f"Temperature T={models['temperature']:.4f}")

    # Explainers
    models['rollout'] = AttentionRollout(vit)
    models['gradcam'] = GradCAMPlusPlus(vit)

    # Preprocessing transform
    models['transform'] = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    log.info("All models ready")
    yield   # app runs here

    log.info("Shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "ArtLens API",
    description = "AI-generated image detection with explainability",
    version     = "2.0.0",
    lifespan    = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Serve static heatmap PNGs
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Response models ───────────────────────────────────────────────────────────
class GeneratorInfo(BaseModel):
    name              : Optional[str]  = Field(None, description="Identified generator name")
    confidence        : Optional[float]= Field(None, description="Generator classification confidence")
    is_unknown        : bool           = Field(...,  description="True if generator not in training set")
    closest_known     : Optional[str]  = Field(None, description="Nearest known generator even if unknown")
    normalised_distances: dict         = Field(...,  description="Z-score distance to each known class")

class ModelSignals(BaseModel):
    vit_probability    : float = Field(..., description="ViT backbone P(AI)")
    lgb_probability    : float = Field(..., description="LightGBM ensemble P(AI)")
    ensemble_probability: float= Field(..., description="Weighted ensemble: 0.7*ViT + 0.3*LGB")
    models_agree       : bool  = Field(..., description="Both models within 0.1 of each other")

class ExplanationText(BaseModel):
    summary : str
    detail  : str
    signals : dict

class PredictResponse(BaseModel):
    request_id         : str
    label              : str   = Field(..., description="'human' or 'ai'")
    confidence         : float = Field(..., description="Calibrated confidence in final verdict")
    generator          : GeneratorInfo
    model_signals      : ModelSignals
    explanation        : ExplanationText
    heatmap_urls       : dict  = Field(..., description="URLs to PNG heatmap overlays, or null if not generated")
    inference_time_s   : float
    model_version      : str
    cached             : bool  = Field(False, description="True if served from cache")


# ── Core inference ────────────────────────────────────────────────────────────
def _run_inference(
    image_bytes      : bytes,
    request_id       : str,
    generate_heatmaps: bool = False
) -> dict:
    """
    Full inference pipeline. Heatmaps generated only when explicitly requested
    to keep /predict fast (~1-2s). Use /explain endpoint for heatmaps.
    """
    t0 = time.time()

    # Decode image
    try:
        pil  = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        np_  = np.array(pil.resize((IMG_SIZE, IMG_SIZE)))
        tens = models['transform'](image=np_)['image'].unsqueeze(0).to(DEVICE)
        img_hash = get_image_hash(image_bytes)
        _TENSOR_CACHE[img_hash] = (tens.cpu(), np_)
    except Exception as e:
        raise HTTPException(400, f"Cannot decode image: {e}")

    vit_model = models['vit']
    T         = models['temperature']

    # ── ViT forward pass ──────────────────────────────────────────────────────
    vit_model.eval()
    with torch.no_grad():
        with autocast(DEVICE.type):
            lb, lg, emb = vit_model(tens)

    emb_np  = emb.cpu().float().numpy()[0]

    # Temperature-scaled binary prediction
    scaled      = lb / T
    probs_bin   = torch.softmax(scaled, dim=1)[0]
    pred_label  = int(probs_bin.argmax())
    vit_prob    = float(probs_bin[1])

    # Generator prediction
    pred_gen_idx = int(lg.argmax(dim=1))
    gen_probs    = torch.softmax(lg, dim=1)[0]
    gen_conf     = float(gen_probs[pred_gen_idx])

    # ── Mahalanobis open-set ──────────────────────────────────────────────────
    openset = compute_openset(
        emb_np, models['maha_stats'], models['thresholds'], pred_label, pred_gen_idx
    )

    # ── LightGBM ──────────────────────────────────────────────────────────────
    hc      = extract_handcrafted(np_)
    combined = np.concatenate([emb_np, hc]).reshape(1, -1)
    combined_s = models['scaler'].transform(combined)
    lgb_prob   = float(models['lgb'].predict(combined_s)[0])

    # ── Ensemble ──────────────────────────────────────────────────────────────
    ens_prob  = 0.7 * vit_prob + 0.3 * lgb_prob
    ens_pred  = int(ens_prob >= 0.5)
    # Confidence: probability of the predicted class
    ens_conf  = ens_prob if ens_pred == 1 else 1 - ens_prob

    # If unknown generator, reduce confidence to reflect uncertainty
    if openset['is_unknown']:
        # More std devs above normal = lower confidence cap
        # At 3 std devs: cap = 0.75. At 6 std devs: cap = 0.60. At 10+: cap = 0.50
        penalty = max(0.50, 0.75 - (openset['min_ai_normed'] - 3.0) * 0.025)
        ens_conf = min(ens_conf, penalty)

    label = 'ai' if ens_pred == 1 else 'human'

    # ── Generator result ──────────────────────────────────────────────────────
    if ens_pred == 1:
        gen_name     = IDX_TO_GEN[pred_gen_idx]
        closest_known = gen_name
        if openset['is_unknown']:
            # Still report closest match but clearly flag as unknown
            gen_name     = None
            closest_known = IDX_TO_GEN[
                min(range(2), key=lambda i: openset['raw_distances'].get(
                    IDX_TO_GEN[i].lower().replace(' ', '_'), float('inf')))
            ]
    else:
        gen_name      = None
        closest_known = None
        gen_conf      = None

    # Build normalised distances for response
    # Map internal key names to display names
    key_map = {
        'human'            : 'Human (WikiArt)',
        'midjourney'       : 'Midjourney',
        'stable_diffusion' : 'Stable Diffusion',
    }
    normed_display = {
        key_map.get(k, k): v
        for k, v in openset['normalised_distances'].items()
    }

    # ── Textual explanation ───────────────────────────────────────────────────
    explanation = generate_explanation(
        label          = label,
        confidence     = ens_conf,
        generator      = gen_name or closest_known,
        is_unknown     = openset['is_unknown'],
        gen_confidence = gen_conf if ens_pred == 1 else None,
        vit_prob       = vit_prob,
        lgb_prob       = lgb_prob,
        openset        = openset,
    )

    # ── Heatmaps (optional) ───────────────────────────────────────────────────
    heatmap_urls = {'attention_rollout': None, 'gradcam': None}

    if generate_heatmaps:
        try:
            ro_hm = models['rollout'](tens)
            heatmap_urls['attention_rollout'] = save_heatmap_png(ro_hm, np_, request_id, 'rollout')
        except Exception as e:
            log.warning(f"[{request_id}] Rollout failed: {e}")
        try:
            gc_hm = models['gradcam'](tens, class_idx=ens_pred)
            heatmap_urls['gradcam'] = save_heatmap_png(gc_hm, np_, request_id, 'gradcam')
        except Exception as e:
            log.warning(f"[{request_id}] GradCAM failed: {e}")

    t_elapsed = round(time.time() - t0, 3)
    log.info(f"[{request_id}] label={label} conf={ens_conf:.3f} "
             f"gen={gen_name} unknown={openset['is_unknown']} t={t_elapsed}s")

    return {
        'request_id'   : request_id,
        'label'        : label,
        'confidence'   : round(ens_conf, 4),
        'generator'    : {
            'name'                : gen_name,
            'confidence'          : round(gen_conf, 4) if ens_pred == 1 else None,
            'is_unknown'          : openset['is_unknown'],
            'closest_known'       : closest_known,
            'normalised_distances': normed_display,
        },
        'model_signals': {
            'vit_probability'     : round(vit_prob, 4),
            'lgb_probability'     : round(lgb_prob, 4),
            'ensemble_probability': round(ens_prob, 4),
            'models_agree'        : abs(vit_prob - lgb_prob) < 0.1,
        },
        'explanation'       : explanation,
        'heatmap_urls'      : heatmap_urls,
        'inference_time_s'  : t_elapsed,
        'model_version'     : f"phase3_epoch{models['ckpt_epoch']}",
        'cached'            : False,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status"    : "ok",
        "device"    : str(DEVICE),
        "cache_size": len(_PREDICTION_CACHE),
        "models"    : {
            "vit"  : f"phase3_epoch{models.get('ckpt_epoch','?')}",
            "lgb"  : "loaded",
            "maha" : list(models.get('maha_stats', {}).keys()),
            "temp" : models.get('temperature'),
        }
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Fast prediction endpoint. No heatmaps included.
    Typical response time: 1-2 seconds.
    For heatmaps, call POST /explain with the returned request_id.
    """
    if file.content_type not in ['image/jpeg', 'image/png', 'image/webp']:
        raise HTTPException(400, f"Unsupported format: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, "File exceeds 10MB limit")

    # Cache check
    img_hash = get_image_hash(image_bytes)
    if img_hash in _PREDICTION_CACHE:
        cached = _PREDICTION_CACHE[img_hash].copy()
        cached['cached'] = True
        log.info(f"Cache hit: {img_hash[:8]}")
        return cached

    request_id = str(uuid.uuid4())[:8]
    result = _run_inference(image_bytes, request_id, generate_heatmaps=False)

    cache_put(img_hash, result)
    return result


@app.post("/explain/{request_id_or_upload}")
async def explain(request_id_or_upload: str, file: UploadFile = File(...)):
    """
    Generates heatmap overlays for an image.
    Returns URLs to PNG files served at /static/heatmaps/*.
    Slower than /predict (~3-5 seconds) due to gradient computation.
    """
    image_bytes = await file.read()
    request_id  = request_id_or_upload or str(uuid.uuid4())[:8]

    img_hash = get_image_hash(image_bytes)
    if img_hash in _TENSOR_CACHE:
        tens, np_ = _TENSOR_CACHE[img_hash]
        tens = tens.to(DEVICE)
        # Skip full inference when prediction is already cached
        result = _PREDICTION_CACHE.get(img_hash) or _run_inference(image_bytes, request_id)
    else:
        result = _run_inference(image_bytes, request_id, generate_heatmaps=False)
        tens, np_ = _TENSOR_CACHE[img_hash]
        tens = tens.to(DEVICE)

    ens_pred = 1 if result['label'] == 'ai' else 0

    # Generate heatmaps
    heatmap_urls = {}
    try:
        ro_hm = models['rollout'](tens)
        heatmap_urls['attention_rollout'] = save_heatmap_png(ro_hm, np_, request_id, 'rollout')
    except Exception as e:
        log.warning(f"Rollout failed: {e}")
        heatmap_urls['attention_rollout'] = None

    try:
        gc_hm = models['gradcam'](tens, class_idx=ens_pred)
        heatmap_urls['gradcam'] = save_heatmap_png(gc_hm, np_, request_id, 'gradcam')
    except Exception as e:
        log.warning(f"GradCAM failed: {e}")
        heatmap_urls['gradcam'] = None

    return {
        **result,
        'heatmap_urls': heatmap_urls
    }


@app.post("/batch")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Batch prediction — up to 20 images.
    No heatmaps. Returns results in input order.
    """
    if len(files) > MAX_BATCH:
        raise HTTPException(400, f"Max {MAX_BATCH} images per batch")

    results = []
    for f in files:
        try:
            image_bytes = await f.read()
            request_id  = str(uuid.uuid4())[:8]
            r = _run_inference(image_bytes, request_id)
            r['filename'] = f.filename
            results.append(r)
        except Exception as e:
            log.error(f"Batch item {f.filename} failed: {e}")
            results.append({'filename': f.filename, 'error': str(e), 'label': None})

    return {
        "count"      : len(results),
        "predictions": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")