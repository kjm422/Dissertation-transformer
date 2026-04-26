# One-label mineral ID classifier (classes 0..94 after shift; original data 1..95 with 0 dropped)
# Overfit-ready + optional cosine LR with warmup via --use_cosine
#
# UPDATES INCLUDED:
# 1. PLOTTING: Saves 'training_curves.png' at the end.
# 2. RAW DATA: Saves 'training_history.csv' (Loss/Acc per epoch).
# 3. VARIANCE REPORT: Prints % variance captured by PCA.
# 4. COMPATIBILITY: Freezes attn_bias if physics_init is False.
# 5. TRANSPARENCY: Saves per-class metrics and confusion matrix to CSV.
# 6. PRIOR MONITOR: Saves 'prior_evolution.npy' to track physics drift.  (ALL HEADS LOGGED)
# 7. TIMER: Prints total execution time.
# 8. PARTIAL FREEZE: Correctly freezes only PCA heads, allowing random heads to learn.
#
# FIXES IMPLEMENTED
# A) Optimizer: put attn_bias in its own AdamW param group with weight_decay=0.0
# B) Slice-freezing: restore frozen slice after opt.step() (prevents any drift)
# C) Prior logging: save an epoch-0 snapshot (prior_evolution.npy has shape [Epoch0..EpochN, Heads, Bands])
#
# LUSI FIXES IMPLEMENTED
# D) LUSI augmentation is applied in RAW reflectance domain (denorm -> augment -> clamp -> renorm)
# E) LUSI KL is computed with dropout OFF for BOTH teacher+student passes (model.eval() inside LUSI)
# F) LUSI is NOT computed inside CUDA autocast; runs in FP32 for stability
#
# r14 CHANGES
# G) Vectorized attention: replaced Python head loop with batched einsum (single GPU kernel)
# H) Water-vapor mask: PCA priors zero out H₂O absorption bands (~1.4/1.9 µm) via --wv_mask
# I) Per-head attention export: attention CSVs now include per-head breakdowns
#
# r15 CHANGES
# J) Consistent non-PCA head initialization: when physics_init + hybrid (H > k), the
#    remaining (H - k) heads are now initialized to ZERO instead of 0.01 * N(0, I) noise.
#    Rationale: symmetry breaking across heads already comes from the random query vectors
#    q_h ~ N(0, 1), so explicit noise on the bias is unnecessary. This also makes the
#    non-PCA heads behave identically to the physics_init=False case (all zeros), removing
#    a design inconsistency that was hard to justify in the paper.

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # macOS MPS memory tweak

import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from contextlib import nullcontext

# --------------------------- LUSI IMPLEMENTATION ---------------------------

def lusi_consistency_loss(model, x_clean_norm, med_t, iqr_t, T=2.0, wv_mask_t=None):
    """
    Computes Vapnik's LUSI (Illumination/Continuum Invariance) Consistency Loss.

    Corrected to:
      (1) apply physics in the RAW REFLECTANCE domain (denorm -> augment -> clamp -> renorm)
      (2) disable dropout for BOTH teacher + student passes (model.eval()) so KL measures physics, not dropout
      (3) run in float32 with AMP disabled for stability
      (4) if wv_mask_t is provided, only augment good bands; preserve original values at H2O bands
    """
    B, P = x_clean_norm.shape
    device = x_clean_norm.device

    # --- STEP 0: SAVE/SET MODEL MODE FOR STABILITY ---
    # We disable dropout for BOTH passes so KL divergence reflects augmentation disagreement only.
    was_training = model.training
    model.eval()

    # Disable AMP explicitly if CUDA (even though we call this outside autocast in the loop)
    amp_off = torch.cuda.amp.autocast(enabled=False) if device.type == "cuda" else nullcontext()

    with amp_off:
        x_clean_norm = x_clean_norm.float()

        # --- STEP 1: DENORMALIZE TO REFLECTANCE SPACE ---
        # Reverse the robust z-score: x_raw = (x_norm * iqr) + med
        x_raw = x_clean_norm * (iqr_t + 1e-6) + med_t

        # --- STEP 2: APPLY PHYSICS AUGMENTATION ---
        # alpha: random global scaling in [0.8, 1.2] representing illumination/shadow changes
        alpha = torch.rand(B, 1, device=device) * 0.4 + 0.8

        # beta_raw: random continuum tilt (slope) applied in RAW units
        slope_base = torch.linspace(-1.0, 1.0, P, device=device)
        beta_scale = torch.randn(B, 1, device=device) * 0.01  # smaller than 0.05 (less aggressive; adjust if desired)
        beta_raw = slope_base.unsqueeze(0) * beta_scale

        x_raw_aug = (x_raw * alpha) + beta_raw

        # --- STEP 2b: PRESERVE ORIGINAL VALUES AT WATER VAPOR BANDS ---
        # Only augment bands where we have reliable signal; leave H2O bands unchanged
        # so LUSI doesn't enforce invariance on atmospheric noise.
        if wv_mask_t is not None:
            bad_bands = ~wv_mask_t                              # True at H2O/edge bands
            x_raw_aug[:, bad_bands] = x_raw[:, bad_bands]       # restore original

        # --- STEP 3: CLAMP (PHYSICAL BOUNDARY) ---
        # Reflectance cannot be negative; clamp in RAW domain (not normalized domain).
        x_raw_aug = torch.clamp(x_raw_aug, min=1e-6, max=10.0)

        # --- STEP 4: RE-NORMALIZE FOR THE AI ---
        # Convert augmented reflectance back to normalized space for the model.
        x_aug_norm = (x_raw_aug - med_t) / (iqr_t + 1e-6)

        # --- STEP 5: GET STABLE TARGET ("TEACHER" PASS) ---
        # Teacher target is stop-grad (no_grad), evaluated with dropout OFF.
        with torch.no_grad():
            logits_clean = model(x_clean_norm).clamp(-50, 50)

        # --- STEP 6: GET AUGMENTED PREDICTION ("STUDENT" PASS) ---
        # Student pass keeps dropout OFF too (still allows gradients).
        logits_aug = model(x_aug_norm.float()).clamp(-50, 50)

        # --- STEP 7: COMPUTE KL DIVERGENCE ---
        # Temperature T softens sharp distributions; multiply by T^2 (standard distillation scaling).
        probs_clean = F.softmax(logits_clean / T, dim=-1)
        log_probs_aug = F.log_softmax(logits_aug / T, dim=-1)
        loss = F.kl_div(log_probs_aug, probs_clean, reduction="batchmean") * (T * T)

    # Restore training mode if needed
    if was_training:
        model.train()

    return loss

# --------------------------- args ---------------------------
def get_args():
    p = argparse.ArgumentParser()

    # paths / core
    p.add_argument("--data", type=str, required=False,
                   default="/Users/kmccoy/Documents/USC/Research/Dissertation/Data/TOApixel_balanced_W_gb1ID_gb2ID_train7500.npy",
                   help="Path to npy with 285 features + 2 label cols")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=4096)

    # model
    p.add_argument("--d_model", type=int, default=192, help="token/hidden dim")
    p.add_argument("--heads", type=int, default=4, help="attention heads")
    p.add_argument("--attn_tau", type=float, default=1.0, help="attention temperature")

    # spectral encoding upgrades
    p.add_argument("--use_derivatives", action="store_true",
                   help="If set, embed x, dx, ddx per band (shape-aware tokenization).")

    # ---- physics-informed attention prior ----
    p.add_argument("--physics_init", action="store_true",
                   help="Enable physics-informed initialization of per-head band attention priors.")
    p.add_argument("--physics_mode", type=str, default="manual",
                   choices=["manual", "pca"],
                   help="manual: gaussian bumps at diagnostic wavelengths; pca: derive band-importance from PCA on ref spectra.")
    p.add_argument("--physics_alpha", type=float, default=1.5,
                   help="Strength of bias added to attention logits (typ 0.5..3).")
    p.add_argument("--physics_freeze_prior_epochs", type=int, default=3,
                   help="Freeze the attn_bias parameters for this many initial epochs, then unfreeze.")
    p.add_argument("--wavelengths", type=str, default=None,
                   help="Optional path to a 285-vector of wavelengths (nm). If omitted, a linear 381..2493 nm grid is assumed.")
    p.add_argument("--ref_spectra", type=str, default=None,
                   help="Path to reference spectra for physics_mode=pca. Expected .npy of shape (M,285) OR .npz with key 'X' (M,285).")
    p.add_argument("--ref_pca_k", type=int, default=4,
                   help="How many PCA components to use to build priors (limited by heads and by M-1).")

    # continuum removal options (ref set only)
    p.add_argument("--ref_continuum", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply convex-hull continuum removal to ref spectra before PCA (default: True).")
    p.add_argument("--ref_use_absdepth", action=argparse.BooleanOptionalAction, default=True,
                   help="After continuum removal, convert to absorption depth (1 - cr). Default: True.")
    p.add_argument("--ref_eps", type=float, default=1e-6,
                   help="Numerical epsilon for continuum division.")

    # --- ref spectra sanitization (prevents PCA failure / bad priors) ---
    p.add_argument("--ref_sanitize", action=argparse.BooleanOptionalAction, default=True,
                   help="Sanitize ref spectra: convert fill/sentinels to NaN, interpolate over wl, clip floor. Default: True.")
    p.add_argument("--ref_absmax", type=float, default=1e20,
                   help="Treat |value| >= this as fill/sentinel in ref spectra. Default 1e20 (catches -1e34).")
    p.add_argument("--ref_min_finite", type=int, default=30,
                   help="Min finite points required per ref spectrum to interpolate. Default 30.")
    p.add_argument("--ref_clip_floor", type=float, default=1e-6,
                   help="Clip ref reflectance to at least this after sanitize. Default 1e-6.")
    p.add_argument("--ref_drop_bad_rows", action=argparse.BooleanOptionalAction, default=False,
                   help="If True, drop ref spectra rows that are too sparse; else raise error.")
    p.add_argument("--wv_mask", type=str, default=None,
                   help="Path to water-vapor mask .npy (bool, shape 285). True=good band. "
                        "PCA priors will be zeroed at masked bands to avoid atmospheric features.")

    # ---- LUSI (Physics Invariance) ----
    p.add_argument("--use_lusi", action="store_true",
                   help="Enable Vapnik LUSI regularization (Scaling + Continuum Invariance).")
    p.add_argument("--lusi_weight", type=float, default=1.0,
                   help="Strength of LUSI regularization term (Lambda).")

    # LR / regularization
    p.add_argument("--lr_max", type=float, default=1e-3, help="Base LR (fixed) or peak LR (if --use_cosine)")
    p.add_argument("--wd", type=float, default=0.0, help="(legacy) weight decay")
    p.add_argument("--weight-decay", dest="weight_decay",
                   type=float, default=None,
                   help="Alias for weight decay; overrides --wd if set.")
    p.add_argument("--dropout", type=float, default=0.1, help="Head dropout")
    p.add_argument("--label-smoothing", dest="label_smoothing",
                   type=float, default=0.0, help="CE label smoothing in [0,1)")
    p.add_argument("--use_cosine", action="store_true",
                   help="Use cosine LR with 5%% warmup; otherwise fixed LR")

    # data limiting / sanity
    p.add_argument("--smoke", action="store_true", help="Quick end-to-end test")
    p.add_argument("--limit", type=int, default=20000, help="Row limit if --smoke")
    p.add_argument("--limit-train", dest="limit_train", type=int, default=None,
                   help="Use only this many training samples.")
    p.add_argument("--limit-val", dest="limit_val", type=int, default=None,
                   help="Use only this many validation samples.")
    p.add_argument("--overfit-single-split", dest="overfit_single_split",
                   action="store_true",
                   help="Use SAME train split as validation (overfit test).")
    p.add_argument("--smoke-override", action="store_true",
                   help="If set, --smoke won't clamp when explicit limits are provided.")

    # attention export
    p.add_argument("--dump_attn", action="store_true",
                   help="Export global and per-class band attention CSVs after training")
    p.add_argument("--attn_out", type=str, default="attn_outputs",
                   help="Folder to write attention CSVs")

    args = p.parse_args()
    args.weight_decay = args.weight_decay if args.weight_decay is not None else args.wd
    return args

# --------------------------- dataset ---------------------------
class PixelSet(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()   # [N, 285]
        self.y = torch.from_numpy(y).long()    # [N] (0..94 after filtering+shift)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# --------------------------- model ---------------------------
NCLS = 95  # after shifting labels to 0..94
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def default_wavelengths_nm(n_bands=285, lo=381.0, hi=2493.0):
    return np.linspace(lo, hi, n_bands).astype(np.float32)

def load_wavelengths(path_or_none, n_bands=285):
    if path_or_none is None:
        return default_wavelengths_nm(n_bands)
    arr = np.load(path_or_none)
    arr = np.asarray(arr).astype(np.float32).reshape(-1)
    if arr.shape[0] != n_bands:
        raise ValueError(f"--wavelengths must have length {n_bands}, got {arr.shape[0]}")
    return arr

def load_ref_spectra(path):
    if path is None:
        return None
    if path.endswith(".npz"):
        z = np.load(path)
        if "X" not in z:
            raise ValueError("Ref .npz must contain key 'X' with shape (M,285)")
        X = z["X"]
    else:
        X = np.load(path)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 285:
        raise ValueError(f"--ref_spectra must be (M,285). Got {X.shape}")
    return X

# ---------- ref sanitization ----------
def _find_fill_mask(X, absmax=1e20):
    X = np.asarray(X)
    return (~np.isfinite(X)) | (np.abs(X) >= absmax) | (X == -9999.0)

def sanitize_ref_spectra_for_pca(
    X_ref: np.ndarray,
    wl_nm: np.ndarray,
    *,
    absmax: float = 1e20,
    min_finite: int = 30,
    clip_floor: float = 1e-6,
    drop_bad_rows: bool = False,
    verbose: bool = True,
):
    wl = np.asarray(wl_nm, dtype=np.float64).reshape(-1)
    X = np.asarray(X_ref, dtype=np.float64).copy()

    if X.ndim != 2 or X.shape[1] != wl.shape[0]:
        raise ValueError(f"sanitize_ref_spectra_for_pca: X {X.shape} vs wl {wl.shape} mismatch")

    fill = _find_fill_mask(X, absmax=absmax)
    bad_band_mask = fill.any(axis=0)

    if verbose:
        print(f"[REF_SANITIZE] raw bad/fill count = {int(fill.sum())} "
              f"(bands with any bad = {int(bad_band_mask.sum())}/{bad_band_mask.size})")

    X[fill] = np.nan

    keep_rows = []
    for i in range(X.shape[0]):
        y = X[i]
        ok = np.isfinite(y)
        n_ok = int(ok.sum())

        if n_ok == y.size:
            keep_rows.append(i)
            continue

        if n_ok < min_finite:
            msg = f"[REF_SANITIZE] row {i} has only {n_ok} finite points (<{min_finite})."
            if drop_bad_rows:
                if verbose:
                    print(msg + " Dropping row.")
                continue
            raise ValueError(msg + " Fix ref set or run with --ref_drop_bad_rows True.")

        X[i] = np.interp(wl, wl[ok], y[ok])
        keep_rows.append(i)

    X = X[keep_rows]
    X = np.maximum(X, float(clip_floor))

    if not np.isfinite(X).all():
        raise ValueError("[REF_SANITIZE] Still non-finite after sanitize/interp. Check thresholds.")

    if verbose and bad_band_mask.any():
        bad_wl = wl[bad_band_mask]
        print("[REF_SANITIZE] Example missing wavelengths (nm):",
              ", ".join([f"{v:.1f}" for v in bad_wl[:15]]),
              ("..." if bad_wl.size > 15 else ""))

    return X.astype(np.float32), bad_band_mask.astype(bool)

# ---------- Continuum removal (convex hull) ----------
def _upper_hull_indices(x, y, eps=1e-12):
    stack = []
    for i in range(len(x)):
        stack.append(i)
        while len(stack) >= 3:
            a, b, c = stack[-3], stack[-2], stack[-1]
            xa, xb, xc = x[a], x[b], x[c]
            ya, yb, yc = y[a], y[b], y[c]
            if abs(xc - xa) < eps:
                y_line = max(ya, yc)
            else:
                t = (xb - xa) / (xc - xa)
                y_line = ya + t * (yc - ya)
            if yb <= y_line + eps:
                stack.pop(-2)
            else:
                break
    return np.array(stack, dtype=np.int64)

def continuum_remove_convex_hull(X, wl_nm, eps=1e-6, clip_min=1e-6, clip_max=None):
    wl = np.asarray(wl_nm, dtype=np.float32)
    M, P = X.shape
    Xcr = np.empty_like(X, dtype=np.float32)

    for m in range(M):
        y = X[m].astype(np.float32)
        y = np.maximum(y, clip_min)

        hull_idx = _upper_hull_indices(wl, y)
        hull_wl  = wl[hull_idx]
        hull_y   = y[hull_idx]

        cont = np.interp(wl, hull_wl, hull_y).astype(np.float32)
        cont = np.maximum(cont, eps)

        cr = y / cont

        if clip_max is not None:
            cr = np.minimum(cr, clip_max)
        cr = np.maximum(cr, clip_min)

        Xcr[m] = cr

    return Xcr

# ---------- PCA for priors (stable + variance) ----------
def compute_pca_components(X, k, ridge=1e-8):
    """
    Computes top k Principal Components.
    Returns (pcs, explained_variance_ratio)
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    denom = max(1, Xc.shape[0] - 1)
    C = (Xc.T @ Xc) / denom
    C.flat[::C.shape[0] + 1] += ridge

    evals, evecs = np.linalg.eigh(C)

    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    evals = np.maximum(evals, 0)
    total_variance = evals.sum()
    explained_variance_ratio = evals / (total_variance + 1e-12)

    r = min(k, evecs.shape[1])
    pcs = evecs[:, :r].T.astype(np.float32)

    pcs /= (np.linalg.norm(pcs, axis=1, keepdims=True) + 1e-8)

    return pcs, explained_variance_ratio[:r]

def make_pca_priors_from_ref(
    X_ref, wl_nm, n_heads, ref_pca_k,
    do_continuum=True, use_absdepth=True, eps=1e-6,
    *,
    ref_sanitize=True,
    ref_absmax=1e20,
    ref_min_finite=30,
    ref_clip_floor=1e-6,
    ref_drop_bad_rows=False,
    wv_mask=None,
):
    wl = np.asarray(wl_nm, dtype=np.float32)
    bad_band_mask = np.zeros(wl.shape[0], dtype=bool)
    Xp = np.asarray(X_ref, dtype=np.float32)

    if ref_sanitize:
        Xp, bad_band_mask = sanitize_ref_spectra_for_pca(
            Xp, wl,
            absmax=ref_absmax,
            min_finite=ref_min_finite,
            clip_floor=ref_clip_floor,
            drop_bad_rows=ref_drop_bad_rows,
            verbose=True,
        )

    if do_continuum:
        Xcr = continuum_remove_convex_hull(Xp, wl, eps=eps)
        Xp = (1.0 - Xcr) if use_absdepth else Xcr

    M, P = Xp.shape
    k = min(ref_pca_k, M - 1, n_heads)
    if k <= 0:
        raise ValueError(f"Ref PCA rank too small: M={M} so max_rank={M-1}")

    pcs, var_ratios = compute_pca_components(Xp, k=k)

    print(f"\n[PCA REPORT] Extracting Top {k} Components from {M} Reference Spectra")
    print(f"-------------------------------------------------------------")
    total_captured = 0.0
    for i, ratio in enumerate(var_ratios):
        pct = ratio * 100.0
        total_captured += pct
        print(f"  PC{i+1} (assigned to Head {i}): {pct:.2f}% variance")
    print(f"  TOTAL Variance Captured:    {total_captured:.2f}%")
    print(f"-------------------------------------------------------------\n")

    priors = np.zeros((n_heads, P), dtype=np.float32)
    for i in range(k):
        v = np.abs(pcs[i]).astype(np.float32)
        if bad_band_mask.any():
            v[bad_band_mask] = 0.0
        v = (v - v.mean()) / (v.std() + 1e-6)
        if bad_band_mask.any():
            v[bad_band_mask] = 0.0
        priors[i] = v

    # r15: non-PCA heads kept at zero (no explicit noise). Symmetry breaking
    # comes from the random query vectors q_h; consistent with physics_init=False.
    # priors[k:] is already zero from the np.zeros initialization above.

    # Zero out water-vapor / low-SNR bands so attention priors don't point at atmospheric features
    if wv_mask is not None:
        wv = np.asarray(wv_mask, dtype=bool)
        n_masked = int((~wv).sum())
        priors[:, ~wv] = 0.0
        print(f"[PCA PRIOR] Zeroed {n_masked} water-vapor/edge bands in attention priors")

    return priors

def gaussian_bump(wl_nm, center_nm, sigma_nm):
    return np.exp(-0.5 * ((wl_nm - center_nm) / max(1e-6, sigma_nm)) ** 2).astype(np.float32)

def make_manual_priors(wl_nm, n_heads, n_bands):
    priors = np.zeros((n_heads, n_bands), dtype=np.float32)
    priors[0] += gaussian_bump(wl_nm, 700.0, 250.0)
    if n_heads >= 2: priors[1] += gaussian_bump(wl_nm, 860.0, 25.0)
    if n_heads >= 3: priors[2] += gaussian_bump(wl_nm, 920.0, 25.0)
    if n_heads >= 4: priors[3] += gaussian_bump(wl_nm, 1000.0, 80.0)
    for h in range(4, n_heads):
        c = 600.0 + 80.0 * (h - 4)
        priors[h] += gaussian_bump(wl_nm, c, 60.0)
    return priors

class QoIAttnBackbone(nn.Module):
    """
    scores = dot(K, q_h)/sqrt(d) + attn_bias[h, :]
    a = softmax(scores / tau)
    """
    def __init__(self, n_bands=285, d=192, n_heads=4, attn_tau=1.0, use_derivatives=False):
        super().__init__()
        self.n_bands = n_bands
        self.d = d
        self.n_heads = n_heads
        self.tau = attn_tau
        self.use_derivatives = use_derivatives

        dh = d // 2
        in_ch = 3 if use_derivatives else 1
        self.val_fc   = nn.Linear(in_ch, dh)
        self.band_emb = nn.Embedding(n_bands, dh)
        self.proj_in  = nn.Linear(dh * 2, d)

        self.q  = nn.Parameter(torch.randn(n_heads, d))
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)

        self.ln = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.ones(n_bands))
        self.bias  = nn.Parameter(torch.zeros(n_bands))

        self.attn_bias = nn.Parameter(torch.zeros(n_heads, n_bands))

    def init_band_priors(self, priors_hp, alpha=1.5):
        pri = torch.from_numpy(priors_hp).float() if isinstance(priors_hp, np.ndarray) else priors_hp.float()
        if pri.shape != self.attn_bias.shape:
            raise ValueError(f"priors must be shape {tuple(self.attn_bias.shape)}, got {tuple(pri.shape)}")
        with torch.no_grad():
            self.attn_bias.copy_(alpha * pri.to(self.attn_bias.device))

    def forward(self, x, return_attn=False):
        B, P = x.shape
        x = x * self.scale + self.bias

        if self.use_derivatives:
            dx = torch.zeros_like(x)
            dx[:, 1:] = x[:, 1:] - x[:, :-1]
            ddx = torch.zeros_like(x)
            ddx[:, 2:] = dx[:, 2:] - dx[:, 1:-1]
            feats = torch.stack([x, dx, ddx], dim=-1)
        else:
            feats = x.unsqueeze(-1)

        band_ids = torch.arange(P, device=x.device).unsqueeze(0).expand(B, P)
        tok = torch.cat([ self.val_fc(feats), self.band_emb(band_ids) ], dim=-1)
        tok = self.proj_in(tok)
        tok = self.ln(tok)

        K = self.Wk(tok)                                           # (B, P, d)
        V = self.Wv(tok)                                           # (B, P, d)

        # Vectorized multi-head cross-attention (no Python loop)
        scale = K.size(-1) ** 0.5
        scores = torch.einsum('hd,bpd->bhp', self.q, K) / scale   # (B, H, P)
        scores = scores + self.attn_bias.unsqueeze(0)              # (B, H, P)
        a = F.softmax(scores / self.tau, dim=-1)                   # (B, H, P)
        z = torch.einsum('bhp,bpd->bhd', a, V)                    # (B, H, d)
        zcat = z.reshape(B, -1)                                    # (B, H*d)

        if return_attn:
            return zcat, a                                         # a: (B, H, P)
        return zcat

class MineralModel(nn.Module):
    def __init__(self, d=192, n_heads=4, attn_tau=1.0, dropout=0.1, use_derivatives=False, n_bands=285):
        super().__init__()
        self.bb   = QoIAttnBackbone(n_bands=n_bands, d=d, n_heads=n_heads, attn_tau=attn_tau, use_derivatives=use_derivatives)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d * n_heads, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, NCLS)
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            z, A = self.bb(x, return_attn=True)
            logits = self.head(z)
            return logits, A
        z = self.bb(x, return_attn=False)
        return self.head(z)

# --------------------------- utils ---------------------------
def _rng():
    return np.random.default_rng(0)

def take_limit(X, y, k):
    if (k is None) or (k >= len(X)):
        return X, y
    idx = _rng().choice(len(X), size=k, replace=False)
    return X[idx], y[idx]

def measure_head_collapse(model):
    q = model.bb.q.detach()
    n_heads = q.shape[0]
    if n_heads <= 1:
        return 1.0
    q_norm = F.normalize(q, p=2, dim=1)
    sim_matrix = torch.mm(q_norm, q_norm.t())
    mask = ~torch.eye(n_heads, dtype=torch.bool, device=q.device)
    return sim_matrix[mask].mean().item()

# --------------------------- main ---------------------------
def main():
    #Remove after tests
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    start_time = time.time()

    args = get_args()

    print("==> Loading:", args.data)
    Jt = np.load(args.data, mmap_mode=None)
    print("File shape:", Jt.shape, "(expect N x 287: 285 features + 2 labels; only using 1 right now)")
    n_bands = Jt.shape[1] - 2  # Dynamically detect number of features (total cols - 2 labels)
    X = Jt[:, :n_bands].astype(np.float32)
    y = Jt[:, -2].astype(np.int64)

    badX = (~np.isfinite(X)).any(axis=1) | (np.abs(X) >= 1e20).any(axis=1)
    if badX.any():
        print(f"[DATA] Dropping {int(badX.sum())} rows with non-finite/fill in X")
        X = X[~badX]
        y = y[~badX]

    if args.smoke and not args.smoke_override and (args.limit_train is None and args.limit_val is None):
        n = min(args.limit, len(X))
        idx = np.random.default_rng(0).permutation(len(X))[:n]
        X, y = X[idx], y[idx]
        print(f"[SMOKE] Using {len(X)} rows.")

    # Split
    N = len(X)
    perm = np.random.default_rng(42).permutation(N)
    cut  = int(0.9 * N)
    tr_idx, va_idx = perm[:cut], perm[cut:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    # Drop GroupID==0
    keep_tr = ytr != 0
    keep_va = yva != 0
    Xtr, ytr = Xtr[keep_tr], ytr[keep_tr]
    Xva, yva = Xva[keep_va], yva[keep_va]

    # Label shift 1..95 -> 0..94
    ytr = ytr - 1
    yva = yva - 1

    # Limits/overfit
    if args.overfit_single_split:
        if args.limit_train is not None:
            Xtr, ytr = take_limit(Xtr, ytr, args.limit_train)
        Xva, yva = Xtr, ytr
        print(f"[OVERFIT] train==val with {len(Xtr)} samples")
    else:
        Xtr, ytr = take_limit(Xtr, ytr, args.limit_train)
        Xva, yva = take_limit(Xva, yva, args.limit_val)
        if (args.limit_train is not None) or (args.limit_val is not None):
            print(f"[LIMIT] train={len(Xtr)} | val={len(Xva)}")

    print("Label min/max:", int(ytr.min()), int(ytr.max()))
    print("Unique classes in train set:", len(np.unique(ytr)))
    print("Head logits size (NCLS):", NCLS)
    print(f"Train size (no zeros): {len(Xtr)} | Val size (no zeros): {len(Xva)}")
    print("Device:", DEVICE, "| MPS:", torch.backends.mps.is_available(), "| CUDA:", torch.cuda.is_available())

    # Robust normalization (train stats only)
    med = np.median(Xtr, axis=0).astype(np.float32)
    iqr = (np.percentile(Xtr, 75, axis=0) - np.percentile(Xtr, 25, axis=0)).astype(np.float32)
    iqr[iqr <= 0] = 1.0

    def norm(A): return (A - med) / (iqr + 1e-6)
    Xtr, Xva = norm(Xtr), norm(Xva)

    train_dl = DataLoader(PixelSet(Xtr, ytr), batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=False, drop_last=False)
    val_dl   = DataLoader(PixelSet(Xva, yva), batch_size=args.batch, shuffle=False,
                          num_workers=0, pin_memory=False, drop_last=False)
    print("Batches per epoch:", len(train_dl))

    model  = MineralModel(d=args.d_model, n_heads=args.heads, attn_tau=args.attn_tau,
                          dropout=args.dropout, use_derivatives=args.use_derivatives, n_bands=n_bands).to(DEVICE)
    crit   = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Create med_t / iqr_t ONCE (used by LUSI and stays on the same device as the model)
    med_t = torch.tensor(med, device=DEVICE, dtype=torch.float32)
    iqr_t = torch.tensor(iqr, device=DEVICE, dtype=torch.float32)

    # Load water-vapor mask (used by PCA priors AND LUSI augmentation)
    wv_mask = None
    wv_mask_t = None
    if args.wv_mask is not None:
        wv_mask = np.load(args.wv_mask).astype(bool)
        if wv_mask.shape[0] != n_bands:
            raise ValueError(f"--wv_mask length {wv_mask.shape[0]} != n_bands {n_bands}")
        wv_mask_t = torch.tensor(wv_mask, device=DEVICE, dtype=torch.bool)
        print(f"[WV_MASK] Loaded {args.wv_mask}: {int(wv_mask.sum())} good / {int((~wv_mask).sum())} masked bands")

    if not args.physics_init:
        model.bb.attn_bias.requires_grad_(False)
        print("[COMPATIBILITY] Physics disabled: attn_bias frozen (acts like old script).")

    # PHYSICS INIT
    if args.physics_init:
        wl_nm = load_wavelengths(args.wavelengths, n_bands=n_bands)

        if args.physics_mode == "pca":
            Xref = load_ref_spectra(args.ref_spectra)
            if Xref is None:
                raise ValueError("physics_mode=pca requires --ref_spectra")

            pri_hp = make_pca_priors_from_ref(
                X_ref=Xref,
                wl_nm=wl_nm,
                n_heads=args.heads,
                ref_pca_k=args.ref_pca_k,
                do_continuum=args.ref_continuum,
                use_absdepth=args.ref_use_absdepth,
                eps=args.ref_eps,
                ref_sanitize=args.ref_sanitize,
                ref_absmax=args.ref_absmax,
                ref_min_finite=args.ref_min_finite,
                ref_clip_floor=args.ref_clip_floor,
                ref_drop_bad_rows=args.ref_drop_bad_rows,
                wv_mask=wv_mask,
            )
            model.bb.init_band_priors(pri_hp, alpha=args.physics_alpha)

            max_rank = Xref.shape[0] - 1
            print(f"[PHYSICS_INIT] mode=pca | alpha={args.physics_alpha} | ref_pca_k={args.ref_pca_k} "
                  f"| Xref={Xref.shape} | max_rank={max_rank} | continuum={args.ref_continuum} "
                  f"| absdepth={args.ref_use_absdepth}")

        if args.physics_freeze_prior_epochs > 0:
            k = min(args.ref_pca_k, args.heads)
            if args.physics_freeze_prior_epochs >= args.epochs:
                print(f"[PHYSICS_INIT] First {k} heads FROZEN for entire run (epochs={args.epochs}); remaining {args.heads - k} heads trainable")
            else:
                print(f"[PHYSICS_INIT] First {k} heads frozen for first {args.physics_freeze_prior_epochs} epochs; all {args.heads} heads trainable after")

    # Optimizer param groups
    main_params = []
    attn_bias_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("attn_bias"):
            attn_bias_params.append(p)
        else:
            main_params.append(p)

    param_groups = [{"params": main_params, "weight_decay": args.weight_decay}]
    if attn_bias_params:
        param_groups.append({"params": attn_bias_params, "weight_decay": 0.0})

    opt = torch.optim.AdamW(param_groups, lr=args.lr_max)

    epochs = 2 if args.smoke else args.epochs

    sched = None
    if args.use_cosine:
        total_steps = epochs * max(1, len(train_dl))
        warmup_steps = max(1, int(0.05 * total_steps))
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * t))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        print(f"[COSINE] total_steps={total_steps}, warmup_steps={warmup_steps}, base_lr={args.lr_max}")

    use_cuda_amp = torch.cuda.is_available()
    best = -1.0

    history = {
        'epoch': [],
        'train_total_loss': [],
        'train_ce_loss': [],
        'train_lusi_loss': [],
        'val_acc1': [],
        'val_acc3': [],
    }

    prior_history = []
    prior_history.append(model.bb.attn_bias.detach().cpu().numpy().copy())  # epoch 0

    print(f"\n[TRAIN] Beginning training for {epochs} epochs. LUSI enabled: {args.use_lusi}")

    for epoch in range(1, epochs + 1):
        if args.physics_init and args.physics_freeze_prior_epochs > 0 and epoch == (args.physics_freeze_prior_epochs + 1):
            print(f"[PHYSICS_INIT] Physics constraints lifted at epoch {epoch}. All heads now learning.")

        model.train()
        running_total = 0.0
        running_ce = 0.0
        running_lusi = 0.0

        for ib, (xb, yb) in enumerate(train_dl, 1):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            # ---------------- CE LOSS (optionally under CUDA autocast) ----------------
            if use_cuda_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb)
                    loss_ce = crit(logits, yb)
            else:
                logits = model(xb)
                loss_ce = crit(logits, yb)

            # ---------------- LUSI LOSS (NOT inside CUDA autocast; FP32 stable) ----------------
            if args.use_lusi:
                loss_lusi = lusi_consistency_loss(model, xb, med_t, iqr_t, T=2.0, wv_mask_t=wv_mask_t)
                loss = loss_ce + (args.lusi_weight * loss_lusi)
            else:
                loss_lusi = xb.new_tensor(0.0)
                loss = loss_ce

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # Slice-freezing logic
            freeze_active = (args.physics_init and args.physics_freeze_prior_epochs > 0 and epoch <= args.physics_freeze_prior_epochs)
            frozen_bias_slice = None

            if freeze_active:
                k = min(args.ref_pca_k, args.heads)
                if k > 0:
                    frozen_bias_slice = model.bb.attn_bias[:k].detach().clone()
                    if model.bb.attn_bias.grad is not None:
                        model.bb.attn_bias.grad[:k].zero_()

                    if ib == 1:
                        print(f"  [FREEZE] Epoch {epoch}: Physics heads (0-{k-1}) frozen; heads {k}-{args.heads-1} trainable")

            opt.step()

            if frozen_bias_slice is not None:
                with torch.no_grad():
                    model.bb.attn_bias[:k].copy_(frozen_bias_slice)

            if sched is not None:
                sched.step()

            running_total += loss.item()
            running_ce += loss_ce.item()
            running_lusi += loss_lusi.item()

            if args.smoke and ib >= 10:
                break

        if DEVICE == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        current_prior = model.bb.attn_bias.detach().cpu().numpy().copy()
        prior_history.append(current_prior)

        # Validation
        model.eval()
        top1 = top3 = total = 0
        with torch.no_grad():
            for vb, (xb, yb) in enumerate(val_dl, 1):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                if use_cuda_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(xb)
                else:
                    logits = model(xb)

                pred1 = logits.argmax(1)
                top1 += (pred1 == yb).sum().item()
                t3 = logits.topk(3, dim=1).indices
                top3 += t3.eq(yb.unsqueeze(1)).any(dim=1).sum().item()
                total += yb.numel()

                if args.smoke and vb >= 5:
                    break

        acc1 = top1 / max(1, total)
        acc3 = top3 / max(1, total)

        n_batches = max(1, len(train_dl))
        avg_total = running_total / n_batches
        avg_ce = running_ce / n_batches
        avg_lusi = running_lusi / n_batches

        head_sim = measure_head_collapse(model)
        prior_rms = float(model.bb.attn_bias.detach().pow(2).mean().sqrt().cpu())

        history['epoch'].append(epoch)
        history['train_total_loss'].append(avg_total)
        history['train_ce_loss'].append(avg_ce)
        history['train_lusi_loss'].append(avg_lusi)
        history['val_acc1'].append(acc1)
        history['val_acc3'].append(acc3)

        lusi_str = f" | lusi={avg_lusi:.4f}" if args.use_lusi else ""
        print(f"epoch {epoch:02d} | loss={avg_total:.4f} (ce={avg_ce:.4f}{lusi_str}) "
              f"| val_top1={acc1:.3f} | val_top3={acc3:.3f} "
              f"| head_sim={head_sim:.3f} | prior_rms={prior_rms:.3f}")

        if acc1 > best:
            best = acc1
            torch.save({
                "epoch": epoch,
                "backbone": model.bb.state_dict(),
                "head": model.head.state_dict(),
                "median": med.astype(np.float32),
                "iqr": iqr.astype(np.float32),
                "d_model": args.d_model,
                "heads": args.heads,
                "attn_tau": args.attn_tau,
                "use_derivatives": args.use_derivatives,
                "physics_init": args.physics_init,
                "physics_mode": args.physics_mode,
                "physics_alpha": args.physics_alpha,
                "ref_continuum": args.ref_continuum,
                "ref_use_absdepth": args.ref_use_absdepth,
            }, "ckpt_best_qoiattn_nozeros.pt")

    # SAVE RAW HISTORY
    df_history = pd.DataFrame(history)
    df_history.to_csv("training_history.csv", index=False)
    print("\n  ✔ Saved training_history.csv (includes CE and LUSI breakdown)")

    torch.save({
        "backbone": model.bb.state_dict(),
        "median": med.astype(np.float32),
        "iqr": iqr.astype(np.float32),
        "d_model": args.d_model,
        "heads": args.heads,
        "attn_tau": args.attn_tau,
        "use_derivatives": args.use_derivatives,
        "physics_init": args.physics_init,
        "physics_mode": args.physics_mode,
        "physics_alpha": args.physics_alpha,
        "ref_continuum": args.ref_continuum,
        "ref_use_absdepth": args.ref_use_absdepth,
    }, "spectral_stage1_qoiattn_nozeros.pt")

    prior_arr = np.array(prior_history)
    np.save("prior_evolution.npy", prior_arr)

    # PLOTTING
    print("==> Generating training plots ...")
    plt.switch_backend('Agg')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(history['epoch'], history['train_total_loss'], 'b-o', label='Total Loss')
    if args.use_lusi:
        ax1.plot(history['epoch'], history['train_ce_loss'], 'g--', label='CE Loss')
        ax1.plot(history['epoch'], history['train_lusi_loss'], 'r--', label='LUSI Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.plot(history['epoch'], history['val_acc1'], 'g-o', label='Val Top-1')
    ax2.plot(history['epoch'], history['val_acc3'], 'm--s', label='Val Top-3')
    ax2.set_ylabel('Accuracy'); ax2.set_xlabel('Epoch')
    ax2.grid(True, alpha=0.3); ax2.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("  ✔ Saved training_curves.png")

    # TRANSPARENCY / PER-CLASS REPORT
    print("\n==> Generating Transparency Report (Per-Class Metrics) from BEST checkpoint...")
    best_ckpt = torch.load("ckpt_best_qoiattn_nozeros.pt", map_location=DEVICE, weights_only=False)
    model.bb.load_state_dict(best_ckpt["backbone"])
    model.head.load_state_dict(best_ckpt["head"])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(DEVICE)
            if use_cuda_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb)
            else:
                logits = model(xb)

            preds = logits.argmax(dim=1).cpu().numpy()
            targets = yb.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

    cls_report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(cls_report).transpose()
    df_report.to_csv("transparency_report_metrics.csv")

    cm = confusion_matrix(all_targets, all_preds)
    df_cm = pd.DataFrame(cm)
    df_cm.to_csv("transparency_report_confusion_matrix.csv")

    df_filtered = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    df_sorted = df_filtered.sort_values(by="f1-score", ascending=True)
    print("\n[ALERT] Bottom 5 Performing Classes (Lowest F1-Score):")
    print(df_sorted[['precision', 'recall', 'f1-score', 'support']].head(5))

    if getattr(args, "dump_attn", False):
        os.makedirs(args.attn_out, exist_ok=True)
        n_h = args.heads
        print(f"==> Collecting per-head attention on validation set ({n_h} heads) ...")

        # Per-head accumulators: (H, P) for global, (H, NCLS, P) for per-class
        global_sum_ph = np.zeros((n_h, n_bands), dtype=np.float64)
        total_cnt  = 0
        class_sum_ph = np.zeros((n_h, NCLS, n_bands), dtype=np.float64)
        class_cnt  = np.zeros(NCLS, dtype=np.int64)

        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE)
                if use_cuda_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        _, A = model(xb, return_attn=True)
                else:
                    _, A = model(xb, return_attn=True)

                # A shape: (B, H, P)
                A_np = A.cpu().numpy()                          # (B, H, P)
                y_np = yb.cpu().numpy()                         # (B,)

                global_sum_ph += A_np.sum(axis=0)               # (H, P)
                total_cnt += A_np.shape[0]

                for c in np.unique(y_np):
                    if c < 0 or c >= NCLS:
                        continue
                    mask = (y_np == c)
                    if np.any(mask):
                        class_sum_ph[:, c, :] += A_np[mask].sum(axis=0)   # (H, P)
                        class_cnt[c] += int(mask.sum())

        # Head-averaged (backward-compatible)
        global_mean = global_sum_ph.sum(axis=0) / (n_h * max(1, total_cnt))  # (P,)
        class_mean = np.zeros((NCLS, n_bands), dtype=np.float64)
        nz = class_cnt > 0
        class_mean[nz] = class_sum_ph.sum(axis=0)[nz] / (n_h * class_cnt[nz, None])

        gpath = os.path.join(args.attn_out, "band_attention_global.csv")
        cpath = os.path.join(args.attn_out, "band_attention_by_class.csv")
        np.savetxt(gpath, global_mean.reshape(1, -1), delimiter=",", fmt="%.6f")
        np.savetxt(cpath, class_mean, delimiter=",", fmt="%.6f")

        # Per-head exports
        for h in range(n_h):
            gh_mean = global_sum_ph[h] / max(1, total_cnt)     # (P,)
            ch_mean = np.zeros((NCLS, n_bands), dtype=np.float64)
            ch_mean[nz] = class_sum_ph[h, nz, :] / class_cnt[nz, None]

            gh_path = os.path.join(args.attn_out, f"band_attention_global_head{h}.csv")
            ch_path = os.path.join(args.attn_out, f"band_attention_by_class_head{h}.csv")
            np.savetxt(gh_path, gh_mean.reshape(1, -1), delimiter=",", fmt="%.6f")
            np.savetxt(ch_path, ch_mean, delimiter=",", fmt="%.6f")

        print(f"  Saved head-averaged + {n_h} per-head attention CSVs to {args.attn_out}/")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n[TIMER] Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)  # macOS/Python 3.13 safe
    except RuntimeError:
        pass
    main()
