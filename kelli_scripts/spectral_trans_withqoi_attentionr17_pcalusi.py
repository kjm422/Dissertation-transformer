# r17 — adds physics_mode="precomputed" with forgiving bias loader
#
# UPDATES INCLUDED (carried from r15/r16):
# 1. PLOTTING: Saves 'training_curves.png' at the end.
# 2. RAW DATA: Saves 'training_history.csv' (Loss/Acc per epoch).
# 3. VARIANCE REPORT: Prints % variance captured by PCA.
# 4. COMPATIBILITY: Freezes attn_bias if physics_init is False.
# 5. TRANSPARENCY: Saves per-class metrics and confusion matrix to CSV.
# 6. PRIOR MONITOR: Saves 'prior_evolution.npy' to track physics drift.
# 7. TIMER: Prints total execution time.
# 8. PARTIAL FREEZE: Correctly freezes only prior heads, allowing remaining heads to learn.
#
# FIXES IMPLEMENTED
# A) Optimizer: put attn_bias in its own AdamW param group with weight_decay=0.0
# B) Slice-freezing: restore frozen slice after opt.step() (prevents any drift)
# C) Prior logging: save an epoch-0 snapshot
#
# LUSI FIXES IMPLEMENTED
# D) LUSI augmentation in raw-reflectance domain (denorm -> augment -> clamp -> renorm)
# E) LUSI KL with dropout OFF for both teacher+student passes (model.eval() inside LUSI)
# F) LUSI runs in fp32 (autocast disabled) for stability
#
# r14 CHANGES
# G) Vectorized attention via batched einsum
# H) Water-vapor mask zero-out for priors (--wv_mask)
# I) Per-head attention export
#
# r15 CHANGES
# J) Non-PCA hybrid heads init to ZERO (symmetry breaking comes from random q_h)
#
# r16 CHANGES
# K) Manual weak spectral prior at user-specified diagnostic bands.
#    Calibrated defaults: --manual_prior_normalize max, --physics_alpha 0.5,
#    --physics_freeze_prior_epochs 1, --manual_prior_bands "480,535,670,860,920".
#
# r17 CHANGES
# L) New --physics_mode precomputed: load a precomputed (k, 285) attention bias
#    from --prior_bias_path. Loader auto-pads (k -> n_heads with zeros) or
#    truncates (k -> n_heads, dropping extras), so one bias file works at any
#    --heads count. Only heads with non-zero rows are frozen during the
#    physics_freeze_prior_epochs warm-up; padded zero rows stay trainable.
#
#    Intended use: produce the bias offline with build_group1_ferric_pca_prior.py
#    (curated library + row-zscore + PCA-on-valid-bands), then point at the
#    saved .npy. Avoids the broken bad-band-mask behavior of physics_mode=pca.

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

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
    """Vapnik LUSI consistency loss (illumination + continuum invariance)."""
    B, P = x_clean_norm.shape
    device = x_clean_norm.device

    was_training = model.training
    model.eval()

    amp_off = torch.cuda.amp.autocast(enabled=False) if device.type == "cuda" else nullcontext()

    with amp_off:
        x_clean_norm = x_clean_norm.float()

        x_raw = x_clean_norm * (iqr_t + 1e-6) + med_t

        alpha = torch.rand(B, 1, device=device) * 0.4 + 0.8

        slope_base = torch.linspace(-1.0, 1.0, P, device=device)
        beta_scale = torch.randn(B, 1, device=device) * 0.01
        beta_raw = slope_base.unsqueeze(0) * beta_scale

        x_raw_aug = (x_raw * alpha) + beta_raw

        if wv_mask_t is not None:
            bad_bands = ~wv_mask_t
            x_raw_aug[:, bad_bands] = x_raw[:, bad_bands]

        x_raw_aug = torch.clamp(x_raw_aug, min=1e-6, max=10.0)
        x_aug_norm = (x_raw_aug - med_t) / (iqr_t + 1e-6)

        with torch.no_grad():
            logits_clean = model(x_clean_norm).clamp(-50, 50)

        logits_aug = model(x_aug_norm.float()).clamp(-50, 50)

        probs_clean = F.softmax(logits_clean / T, dim=-1)
        log_probs_aug = F.log_softmax(logits_aug / T, dim=-1)
        loss = F.kl_div(log_probs_aug, probs_clean, reduction="batchmean") * (T * T)

    if was_training:
        model.train()

    return loss


# --------------------------- args ---------------------------

def get_args():
    p = argparse.ArgumentParser()

    # paths / core
    p.add_argument(
        "--data", type=str, required=False,
        default="/Users/kmccoy/Documents/USC/Research/Dissertation/Data/TOApixel_balanced_W_gb1ID_gb2ID_train7500.npy",
        help="Path to npy with 285 features + 2 label cols",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=4096)

    # model
    p.add_argument("--d_model", type=int, default=192, help="token/hidden dim")
    p.add_argument("--heads", type=int, default=4, help="attention heads")
    p.add_argument("--attn_tau", type=float, default=1.0, help="attention temperature")

    # spectral encoding
    p.add_argument("--use_derivatives", action="store_true",
                   help="If set, embed x, dx, ddx per band (shape-aware tokenization).")

    # physics-informed prior
    p.add_argument("--physics_init", action="store_true",
                   help="Enable physics-informed initialization of per-head band attention priors.")

    p.add_argument(
        "--physics_mode", type=str, default="manual",
        choices=["manual", "pca", "precomputed"],
        help=(
            "manual: Gaussian bumps at diagnostic wavelengths (default — weak prior). "
            "pca: build PCA priors inline from --ref_spectra (legacy; uses old bad-band-mask). "
            "precomputed: load a (k, 285) attention bias from --prior_bias_path "
            "(produced by build_group1_ferric_pca_prior.py); auto-pads/truncates to --heads."
        ),
    )

    # ---- precomputed-bias options (r17) ----
    p.add_argument(
        "--prior_bias_path", type=str, default=None,
        help="Path to a precomputed attention-bias .npy of shape (k, 285). "
             "Used when --physics_mode precomputed. "
             "k is auto-padded with zero rows to n_heads (or truncated if k > n_heads).",
    )

    # ---- manual weak prior options ----
    p.add_argument(
        "--manual_prior_bands", type=str,
        default="480,535,670,860,920",
        help=(
            "Comma-separated diagnostic wavelengths in nm for physics_mode=manual. "
            "Defaults are canonical Jiang 2014 / Clark 1999 Fe3+ bands: "
            "480 (goethite CT), 535 (hematite CT), 670 (Fe3+ crystal-field), "
            "860 (hematite CF NIR), 920 (goethite CF NIR)."
        ),
    )
    p.add_argument("--manual_prior_sigma", type=float, default=25.0,
                   help="Gaussian bump width in nm for manual spectral prior.")
    p.add_argument(
        "--manual_prior_assign", type=str, default="round_robin",
        choices=["shared", "round_robin", "one_per_head"],
        help=(
            "How manual bands are assigned to heads. "
            "shared: all heads get all bands (sanity baseline; head divergence "
            "relies on random q_h); round_robin: bands distributed cyclically across heads; "
            "one_per_head: first H bands assigned one per head."
        ),
    )
    p.add_argument(
        "--manual_prior_normalize", type=str, default="max",
        choices=["zscore", "max", "none"],
        help=(
            "How to normalize manual prior before multiplying by physics_alpha. "
            "max (default): peak amplitude=1, off-bands=0 — clean weak prior. "
            "zscore: mean=0/std=1 over active bands; produces negative bias on tails "
            "(use only with very small physics_alpha ~0.2). "
            "none: raw Gaussian amplitudes."
        ),
    )

    p.add_argument(
        "--physics_alpha", type=float, default=0.5,
        help=(
            "Strength of bias added to attention logits. "
            "Default 0.5 calibrated for manual+max (peak logit bias ~0.5). "
            "For manual+zscore use ~0.2; for precomputed+zabs use ~0.2; "
            "for precomputed+max use ~0.5; for legacy PCA use 1.0."
        ),
    )

    p.add_argument(
        "--physics_freeze_prior_epochs", type=int, default=1,
        help=(
            "Freeze the prior-bearing attn_bias heads for this many initial epochs. "
            "Default 1 is appropriate for a weak prior; bump to 3 for stronger priors."
        ),
    )

    # ---- legacy inline-PCA options ----
    p.add_argument("--wavelengths", type=str, default=None)
    p.add_argument("--ref_spectra", type=str, default=None,
                   help="Path to ref spectra for physics_mode=pca. (M,285) npy or .npz with key 'X'.")
    p.add_argument("--ref_pca_k", type=int, default=4)
    p.add_argument("--ref_continuum", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ref_use_absdepth", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ref_eps", type=float, default=1e-6)
    p.add_argument("--ref_sanitize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ref_absmax", type=float, default=1e20)
    p.add_argument("--ref_min_finite", type=int, default=30)
    p.add_argument("--ref_clip_floor", type=float, default=1e-6)
    p.add_argument("--ref_drop_bad_rows", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--wv_mask", type=str, default=None,
                   help="Path to water-vapor mask .npy, bool shape 285. True=good band.")

    # LUSI
    p.add_argument("--use_lusi", action="store_true")
    p.add_argument("--lusi_weight", type=float, default=1.0)

    # LR / regularization
    p.add_argument("--lr_max", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", dest="label_smoothing", type=float, default=0.0)
    p.add_argument("--use_cosine", action="store_true")

    # data limiting / sanity
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--limit-train", dest="limit_train", type=int, default=None)
    p.add_argument("--limit-val", dest="limit_val", type=int, default=None)
    p.add_argument("--overfit-single-split", dest="overfit_single_split", action="store_true")
    p.add_argument("--smoke-override", action="store_true")

    # attention export
    p.add_argument("--dump_attn", action="store_true")
    p.add_argument("--attn_out", type=str, default="attn_outputs")

    args = p.parse_args()
    args.weight_decay = args.weight_decay if args.weight_decay is not None else args.wd
    return args


# --------------------------- dataset ---------------------------

class PixelSet(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


# --------------------------- model constants ---------------------------

NCLS = 95
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------- helper functions ---------------------------

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


# ---------- ref sanitization (legacy pca mode) ----------

def _find_fill_mask(X, absmax=1e20):
    X = np.asarray(X)
    return (~np.isfinite(X)) | (np.abs(X) >= absmax) | (X == -9999.0)


def sanitize_ref_spectra_for_pca(
    X_ref, wl_nm, *, absmax=1e20, min_finite=30, clip_floor=1e-6,
    drop_bad_rows=False, verbose=True,
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
        y = X[i]; ok = np.isfinite(y); n_ok = int(ok.sum())
        if n_ok == y.size:
            keep_rows.append(i); continue
        if n_ok < min_finite:
            msg = f"[REF_SANITIZE] row {i} has only {n_ok} finite points (<{min_finite})."
            if drop_bad_rows:
                if verbose: print(msg + " Dropping row.")
                continue
            raise ValueError(msg + " Fix ref set or run with --ref_drop_bad_rows True.")
        X[i] = np.interp(wl, wl[ok], y[ok])
        keep_rows.append(i)
    X = X[keep_rows]
    X = np.maximum(X, float(clip_floor))
    if not np.isfinite(X).all():
        raise ValueError("[REF_SANITIZE] Still non-finite after sanitize/interp.")

    if verbose and bad_band_mask.any():
        bad_wl = wl[bad_band_mask]
        print("[REF_SANITIZE] Example missing wavelengths (nm):",
              ", ".join([f"{v:.1f}" for v in bad_wl[:15]]),
              ("..." if bad_wl.size > 15 else ""))
    return X.astype(np.float32), bad_band_mask.astype(bool)


# ---------- continuum removal ----------

def _upper_hull_indices(x, y, eps=1e-12):
    stack = []
    for i in range(len(x)):
        stack.append(i)
        while len(stack) >= 3:
            a, b, c = stack[-3], stack[-2], stack[-1]
            xa, xb, xc = x[a], x[b], x[c]; ya, yb, yc = y[a], y[b], y[c]
            if abs(xc - xa) < eps:
                y_line = max(ya, yc)
            else:
                t = (xb - xa) / (xc - xa); y_line = ya + t * (yc - ya)
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
        y = np.maximum(X[m].astype(np.float32), clip_min)
        hull_idx = _upper_hull_indices(wl, y)
        cont = np.interp(wl, wl[hull_idx], y[hull_idx]).astype(np.float32)
        cont = np.maximum(cont, eps)
        cr = y / cont
        if clip_max is not None: cr = np.minimum(cr, clip_max)
        cr = np.maximum(cr, clip_min)
        Xcr[m] = cr
    return Xcr


# ---------- inline PCA priors (legacy pca mode) ----------

def compute_pca_components(X, k, ridge=1e-8):
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    denom = max(1, Xc.shape[0] - 1)
    C = (Xc.T @ Xc) / denom
    C.flat[::C.shape[0] + 1] += ridge
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals = np.maximum(evals[idx], 0.0); evecs = evecs[:, idx]
    total = evals.sum(); ratios = evals / (total + 1e-12)
    r = min(k, evecs.shape[1])
    pcs = evecs[:, :r].T.astype(np.float32)
    pcs /= (np.linalg.norm(pcs, axis=1, keepdims=True) + 1e-8)
    return pcs, ratios[:r]


def make_pca_priors_from_ref(
    X_ref, wl_nm, n_heads, ref_pca_k,
    do_continuum=True, use_absdepth=True, eps=1e-6,
    *, ref_sanitize=True, ref_absmax=1e20, ref_min_finite=30,
    ref_clip_floor=1e-6, ref_drop_bad_rows=False, wv_mask=None,
):
    wl = np.asarray(wl_nm, dtype=np.float32)
    bad_band_mask = np.zeros(wl.shape[0], dtype=bool)
    Xp = np.asarray(X_ref, dtype=np.float32)

    if ref_sanitize:
        Xp, bad_band_mask = sanitize_ref_spectra_for_pca(
            Xp, wl, absmax=ref_absmax, min_finite=ref_min_finite,
            clip_floor=ref_clip_floor, drop_bad_rows=ref_drop_bad_rows, verbose=True,
        )

    if do_continuum:
        Xcr = continuum_remove_convex_hull(Xp, wl, eps=eps)
        Xp = (1.0 - Xcr) if use_absdepth else Xcr

    M, P = Xp.shape
    k = min(ref_pca_k, M - 1, n_heads)
    if k <= 0:
        raise ValueError(f"Ref PCA rank too small: M={M} so max_rank={M - 1}")

    pcs, var_ratios = compute_pca_components(Xp, k=k)

    print(f"\n[PCA REPORT] Extracting Top {k} Components from {M} Reference Spectra")
    print("-------------------------------------------------------------")
    total_captured = 0.0
    for i, ratio in enumerate(var_ratios):
        pct = ratio * 100.0; total_captured += pct
        print(f"  PC{i + 1} assigned to Head {i}: {pct:.2f}% variance")
    print(f"  TOTAL Variance Captured: {total_captured:.2f}%")
    print("-------------------------------------------------------------\n")

    priors = np.zeros((n_heads, P), dtype=np.float32)
    for i in range(k):
        v = np.abs(pcs[i]).astype(np.float32)
        if bad_band_mask.any(): v[bad_band_mask] = 0.0
        v = (v - v.mean()) / (v.std() + 1e-6)
        if bad_band_mask.any(): v[bad_band_mask] = 0.0
        priors[i] = v

    if wv_mask is not None:
        wv = np.asarray(wv_mask, dtype=bool)
        if wv.shape[0] != P:
            raise ValueError(f"wv_mask length {wv.shape[0]} != number of bands {P}")
        n_masked = int((~wv).sum())
        priors[:, ~wv] = 0.0
        print(f"[PCA PRIOR] Zeroed {n_masked} water-vapor/edge bands in attention priors")

    return priors


# ---------- manual weak spectral priors ----------

def parse_float_list_csv(s):
    if s is None: return []
    s = str(s).replace(";", ",")
    return [float(part.strip()) for part in s.split(",") if part.strip()]


def gaussian_bump(wl_nm, center_nm, sigma_nm):
    return np.exp(-0.5 * ((wl_nm - center_nm) / max(1e-6, sigma_nm)) ** 2).astype(np.float32)


def normalize_prior_rows(priors, valid_mask=None, mode="max"):
    """Per-head prior normalization."""
    priors = np.asarray(priors, dtype=np.float32).copy()
    if valid_mask is None:
        valid_mask = np.ones(priors.shape[1], dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    for h in range(priors.shape[0]):
        v = priors[h]
        active = valid_mask & np.isfinite(v)
        if not np.any(active): continue
        if np.max(np.abs(v[active])) <= 0: continue

        if mode == "zscore":
            mu = v[active].mean(); sd = v[active].std()
            priors[h, active] = (v[active] - mu) / (sd + 1e-6)
            priors[h, ~active] = 0.0
        elif mode == "max":
            mx = np.max(np.abs(v[active]))
            priors[h, active] = v[active] / (mx + 1e-8)
            priors[h, ~active] = 0.0
        elif mode == "none":
            priors[h, ~active] = 0.0
        else:
            raise ValueError(f"Unknown prior normalization mode: {mode}")

    return priors.astype(np.float32)


def make_manual_priors(
    wl_nm, n_heads, n_bands,
    centers_nm=(480, 535, 670, 860, 920),
    sigma_nm=25.0,
    assign="round_robin",
    normalize="max",
    wv_mask=None,
):
    """Manual weak spectral prior using Gaussian bumps at diagnostic wavelengths."""
    wl = np.asarray(wl_nm, dtype=np.float32).reshape(-1)
    if wl.shape[0] != n_bands:
        raise ValueError(f"wl_nm length {wl.shape[0]} != n_bands {n_bands}")

    centers_nm = list(centers_nm)
    priors = np.zeros((n_heads, n_bands), dtype=np.float32)

    if len(centers_nm) == 0:
        print("[MANUAL PRIOR] No manual prior bands provided; returning zeros.")
        return priors

    # wv-mask sanity: warn if any center falls in a masked band
    if wv_mask is not None:
        wv = np.asarray(wv_mask, dtype=bool)
        for c in centers_nm:
            idx = int(np.argmin(np.abs(wl - c)))
            if not wv[idx]:
                print(f"[MANUAL PRIOR][warn] center {c:.0f} nm maps to band {idx} "
                      f"({wl[idx]:.1f} nm) which is wv-masked; this bump will be zeroed")

    if assign == "shared":
        v = np.zeros(n_bands, dtype=np.float32)
        for c in centers_nm:
            v += gaussian_bump(wl, c, sigma_nm)
        priors[:] = v[None, :]
    elif assign == "round_robin":
        for i, c in enumerate(centers_nm):
            h = i % n_heads
            priors[h] += gaussian_bump(wl, c, sigma_nm)
    elif assign == "one_per_head":
        for h, c in enumerate(centers_nm[:n_heads]):
            priors[h] += gaussian_bump(wl, c, sigma_nm)
    else:
        raise ValueError(f"Unknown manual prior assignment mode: {assign}")

    valid_mask = np.ones(n_bands, dtype=bool)
    if wv_mask is not None:
        wv = np.asarray(wv_mask, dtype=bool)
        if wv.shape[0] != n_bands:
            raise ValueError(f"wv_mask length {wv.shape[0]} != n_bands {n_bands}")
        valid_mask &= wv
        priors[:, ~wv] = 0.0
        print(f"[MANUAL PRIOR] Zeroed {int((~wv).sum())} water-vapor/edge bands")

    priors = normalize_prior_rows(priors, valid_mask=valid_mask, mode=normalize)

    print("\n[MANUAL PRIOR REPORT]")
    print(f"  bands_nm  = {centers_nm}")
    print(f"  sigma_nm  = {sigma_nm}")
    print(f"  assign    = {assign}")
    print(f"  normalize = {normalize}")
    print(f"  heads     = {n_heads}")
    for h in range(n_heads):
        if assign == "shared":
            assigned = centers_nm
        elif assign == "round_robin":
            assigned = [c for i, c in enumerate(centers_nm) if (i % n_heads) == h]
        elif assign == "one_per_head":
            assigned = [centers_nm[h]] if h < len(centers_nm) else []
        else:
            assigned = []
        nonzero = int(np.count_nonzero(np.abs(priors[h]) > 1e-8))
        peak = float(np.max(np.abs(priors[h]))) if nonzero > 0 else 0.0
        print(f"  head {h}: assigned={assigned} | nonzero bands={nonzero} | peak |v|={peak:.3f}")
    print("-------------------------------------------------------------\n")

    return priors.astype(np.float32)


# ---------- precomputed-bias loader (r17) ----------

def load_precomputed_bias(path, n_heads, n_bands, verbose=True):
    """
    Load a (k, P) attention-bias .npy. Auto-pads with zeros to (n_heads, P)
    or truncates if k > n_heads. Returns the (n_heads, n_bands) array.
    """
    pri_loaded = np.load(path).astype(np.float32)
    if pri_loaded.ndim != 2 or pri_loaded.shape[1] != n_bands:
        raise ValueError(
            f"--prior_bias_path must be shape (k, {n_bands}); got {pri_loaded.shape}"
        )

    k_loaded, P = pri_loaded.shape

    if k_loaded == n_heads:
        if verbose:
            print(f"[PRECOMPUTED] Loaded ({k_loaded}, {P}); exact match for heads={n_heads}")
        return pri_loaded

    if k_loaded < n_heads:
        pad = np.zeros((n_heads - k_loaded, n_bands), dtype=np.float32)
        out = np.vstack([pri_loaded, pad]).astype(np.float32)
        if verbose:
            print(f"[PRECOMPUTED] Loaded ({k_loaded}, {P}); padded with "
                  f"{n_heads - k_loaded} zero rows to ({n_heads}, {P}). "
                  f"Heads 0-{k_loaded - 1} get prior; heads {k_loaded}-{n_heads - 1} stay random.")
        return out

    # k_loaded > n_heads -> truncate
    out = pri_loaded[:n_heads].astype(np.float32)
    if verbose:
        print(f"[PRECOMPUTED] Loaded ({k_loaded}, {P}); truncated to ({n_heads}, {P}). "
              f"Dropped {k_loaded - n_heads} prior rows beyond head {n_heads - 1}.")
    return out


# --------------------------- model ---------------------------

class QoIAttnBackbone(nn.Module):
    def __init__(self, n_bands=285, d=192, n_heads=4, attn_tau=1.0, use_derivatives=False):
        super().__init__()
        self.n_bands = n_bands; self.d = d; self.n_heads = n_heads
        self.tau = attn_tau; self.use_derivatives = use_derivatives

        dh = d // 2
        in_ch = 3 if use_derivatives else 1

        self.val_fc   = nn.Linear(in_ch, dh)
        self.band_emb = nn.Embedding(n_bands, dh)
        self.proj_in  = nn.Linear(dh * 2, d)

        self.q  = nn.Parameter(torch.randn(n_heads, d))
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)

        self.ln    = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.ones(n_bands))
        self.bias  = nn.Parameter(torch.zeros(n_bands))

        self.attn_bias = nn.Parameter(torch.zeros(n_heads, n_bands))

    def init_band_priors(self, priors_hp, alpha=0.5):
        pri = torch.from_numpy(priors_hp).float() if isinstance(priors_hp, np.ndarray) else priors_hp.float()
        if pri.shape != self.attn_bias.shape:
            raise ValueError(f"priors must be shape {tuple(self.attn_bias.shape)}, got {tuple(pri.shape)}")
        with torch.no_grad():
            self.attn_bias.copy_(alpha * pri.to(self.attn_bias.device))

    def forward(self, x, return_attn=False):
        B, P = x.shape
        x = x * self.scale + self.bias

        if self.use_derivatives:
            dx = torch.zeros_like(x);  dx[:, 1:]  = x[:, 1:]  - x[:, :-1]
            ddx = torch.zeros_like(x); ddx[:, 2:] = dx[:, 2:] - dx[:, 1:-1]
            feats = torch.stack([x, dx, ddx], dim=-1)
        else:
            feats = x.unsqueeze(-1)

        band_ids = torch.arange(P, device=x.device).unsqueeze(0).expand(B, P)
        tok = torch.cat([self.val_fc(feats), self.band_emb(band_ids)], dim=-1)
        tok = self.proj_in(tok); tok = self.ln(tok)

        K = self.Wk(tok); V = self.Wv(tok)
        scale = K.size(-1) ** 0.5
        scores = torch.einsum("hd,bpd->bhp", self.q, K) / scale
        scores = scores + self.attn_bias.unsqueeze(0)
        a = F.softmax(scores / self.tau, dim=-1)
        z = torch.einsum("bhp,bpd->bhd", a, V)
        zcat = z.reshape(B, -1)

        if return_attn:
            return zcat, a
        return zcat


class MineralModel(nn.Module):
    def __init__(self, d=192, n_heads=4, attn_tau=1.0, dropout=0.1, use_derivatives=False, n_bands=285):
        super().__init__()
        self.bb = QoIAttnBackbone(n_bands=n_bands, d=d, n_heads=n_heads,
                                  attn_tau=attn_tau, use_derivatives=use_derivatives)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d * n_heads, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, NCLS),
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
    if (k is None) or (k >= len(X)): return X, y
    idx = _rng().choice(len(X), size=k, replace=False)
    return X[idx], y[idx]


def measure_head_collapse(model):
    q = model.bb.q.detach()
    n_heads = q.shape[0]
    if n_heads <= 1: return 1.0
    q_norm = F.normalize(q, p=2, dim=1)
    sim_matrix = torch.mm(q_norm, q_norm.t())
    mask = ~torch.eye(n_heads, dtype=torch.bool, device=q.device)
    return sim_matrix[mask].mean().item()


# --------------------------- main ---------------------------

def main():
    SEED = 42
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start_time = time.time()
    args = get_args()

    print("==> Loading:", args.data)
    Jt = np.load(args.data, mmap_mode=None)
    print("File shape:", Jt.shape, "(expect N x 287: 285 features + 2 labels)")

    n_bands = Jt.shape[1] - 2
    X = Jt[:, :n_bands].astype(np.float32)
    y = Jt[:, -2].astype(np.int64)

    badX = (~np.isfinite(X)).any(axis=1) | (np.abs(X) >= 1e20).any(axis=1)
    if badX.any():
        print(f"[DATA] Dropping {int(badX.sum())} rows with non-finite/fill in X")
        X = X[~badX]; y = y[~badX]

    if args.smoke and not args.smoke_override and (args.limit_train is None and args.limit_val is None):
        n = min(args.limit, len(X))
        idx = np.random.default_rng(0).permutation(len(X))[:n]
        X, y = X[idx], y[idx]
        print(f"[SMOKE] Using {len(X)} rows.")

    N = len(X); perm = np.random.default_rng(42).permutation(N); cut = int(0.9 * N)
    tr_idx, va_idx = perm[:cut], perm[cut:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    keep_tr = ytr != 0; keep_va = yva != 0
    Xtr, ytr = Xtr[keep_tr], ytr[keep_tr]
    Xva, yva = Xva[keep_va], yva[keep_va]
    ytr = ytr - 1; yva = yva - 1

    if args.overfit_single_split:
        if args.limit_train is not None: Xtr, ytr = take_limit(Xtr, ytr, args.limit_train)
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

    model = MineralModel(d=args.d_model, n_heads=args.heads, attn_tau=args.attn_tau,
                         dropout=args.dropout, use_derivatives=args.use_derivatives, n_bands=n_bands).to(DEVICE)
    crit  = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    med_t = torch.tensor(med, device=DEVICE, dtype=torch.float32)
    iqr_t = torch.tensor(iqr, device=DEVICE, dtype=torch.float32)

    wv_mask = None; wv_mask_t = None
    if args.wv_mask is not None:
        wv_mask = np.load(args.wv_mask).astype(bool)
        if wv_mask.shape[0] != n_bands:
            raise ValueError(f"--wv_mask length {wv_mask.shape[0]} != n_bands {n_bands}")
        wv_mask_t = torch.tensor(wv_mask, device=DEVICE, dtype=torch.bool)
        print(f"[WV_MASK] Loaded {args.wv_mask}: "
              f"{int(wv_mask.sum())} good / {int((~wv_mask).sum())} masked bands")

    if not args.physics_init:
        model.bb.attn_bias.requires_grad_(False)
        print("[COMPATIBILITY] Physics disabled: attn_bias frozen.")

    physics_frozen_heads = 0

    # --------------------------- PHYSICS INIT ---------------------------
    if args.physics_init:
        wl_nm = load_wavelengths(args.wavelengths, n_bands=n_bands)

        if args.physics_mode == "manual":
            manual_bands = parse_float_list_csv(args.manual_prior_bands)
            pri_hp = make_manual_priors(
                wl_nm=wl_nm, n_heads=args.heads, n_bands=n_bands,
                centers_nm=manual_bands, sigma_nm=args.manual_prior_sigma,
                assign=args.manual_prior_assign, normalize=args.manual_prior_normalize,
                wv_mask=wv_mask,
            )
            model.bb.init_band_priors(pri_hp, alpha=args.physics_alpha)
            physics_frozen_heads = int(np.any(np.abs(pri_hp) > 1e-8, axis=1).sum())
            print(f"[PHYSICS_INIT] mode=manual | alpha={args.physics_alpha} | "
                  f"bands={manual_bands} | sigma={args.manual_prior_sigma} | "
                  f"assign={args.manual_prior_assign} | normalize={args.manual_prior_normalize} | "
                  f"prior_heads={physics_frozen_heads}/{args.heads}")

        elif args.physics_mode == "pca":
            Xref = load_ref_spectra(args.ref_spectra)
            if Xref is None: raise ValueError("physics_mode=pca requires --ref_spectra")
            pri_hp = make_pca_priors_from_ref(
                X_ref=Xref, wl_nm=wl_nm, n_heads=args.heads, ref_pca_k=args.ref_pca_k,
                do_continuum=args.ref_continuum, use_absdepth=args.ref_use_absdepth, eps=args.ref_eps,
                ref_sanitize=args.ref_sanitize, ref_absmax=args.ref_absmax,
                ref_min_finite=args.ref_min_finite, ref_clip_floor=args.ref_clip_floor,
                ref_drop_bad_rows=args.ref_drop_bad_rows, wv_mask=wv_mask,
            )
            model.bb.init_band_priors(pri_hp, alpha=args.physics_alpha)
            physics_frozen_heads = int(np.any(np.abs(pri_hp) > 1e-8, axis=1).sum())
            max_rank = Xref.shape[0] - 1
            print(f"[PHYSICS_INIT] mode=pca | alpha={args.physics_alpha} | "
                  f"ref_pca_k={args.ref_pca_k} | Xref={Xref.shape} | max_rank={max_rank} | "
                  f"continuum={args.ref_continuum} | absdepth={args.ref_use_absdepth} | "
                  f"prior_heads={physics_frozen_heads}/{args.heads}")

        elif args.physics_mode == "precomputed":
            if args.prior_bias_path is None:
                raise ValueError("physics_mode=precomputed requires --prior_bias_path")
            pri_hp = load_precomputed_bias(
                args.prior_bias_path, n_heads=args.heads, n_bands=n_bands, verbose=True
            )
            model.bb.init_band_priors(pri_hp, alpha=args.physics_alpha)
            physics_frozen_heads = int(np.any(np.abs(pri_hp) > 1e-8, axis=1).sum())
            print(f"[PHYSICS_INIT] mode=precomputed | alpha={args.physics_alpha} | "
                  f"path={args.prior_bias_path} | shape={pri_hp.shape} | "
                  f"prior_heads={physics_frozen_heads}/{args.heads}")

        else:
            raise ValueError(f"Unknown physics_mode: {args.physics_mode}")

        if args.physics_freeze_prior_epochs > 0 and physics_frozen_heads > 0:
            if args.physics_freeze_prior_epochs >= args.epochs:
                print(f"[PHYSICS_INIT] First {physics_frozen_heads} prior heads FROZEN for entire run "
                      f"(epochs={args.epochs}); remaining {args.heads - physics_frozen_heads} heads trainable")
            else:
                print(f"[PHYSICS_INIT] First {physics_frozen_heads} prior heads frozen for first "
                      f"{args.physics_freeze_prior_epochs} epochs; all {args.heads} heads trainable after")

    # --------------------------- Optimizer ---------------------------
    main_params = []; attn_bias_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
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
            if step < warmup_steps: return (step + 1) / warmup_steps
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * t))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        print(f"[COSINE] total_steps={total_steps}, warmup_steps={warmup_steps}, base_lr={args.lr_max}")

    use_cuda_amp = torch.cuda.is_available()
    best = -1.0

    history = {"epoch": [], "train_total_loss": [], "train_ce_loss": [],
               "train_lusi_loss": [], "val_acc1": [], "val_acc3": []}

    prior_history = []
    prior_history.append(model.bb.attn_bias.detach().cpu().numpy().copy())  # epoch 0

    print(f"\n[TRAIN] Beginning training for {epochs} epochs. LUSI enabled: {args.use_lusi}")

    # --------------------------- Training loop ---------------------------
    for epoch in range(1, epochs + 1):
        if (args.physics_init and args.physics_freeze_prior_epochs > 0
            and epoch == (args.physics_freeze_prior_epochs + 1)):
            print(f"[PHYSICS_INIT] Physics constraints lifted at epoch {epoch}. All heads now learning.")

        model.train()
        running_total = 0.0; running_ce = 0.0; running_lusi = 0.0

        for ib, (xb, yb) in enumerate(train_dl, 1):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            if use_cuda_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb); loss_ce = crit(logits, yb)
            else:
                logits = model(xb); loss_ce = crit(logits, yb)

            if args.use_lusi:
                loss_lusi = lusi_consistency_loss(model, xb, med_t, iqr_t, T=2.0, wv_mask_t=wv_mask_t)
                loss = loss_ce + (args.lusi_weight * loss_lusi)
            else:
                loss_lusi = xb.new_tensor(0.0)
                loss = loss_ce

            opt.zero_grad(set_to_none=True)
            loss.backward()

            freeze_active = (args.physics_init and args.physics_freeze_prior_epochs > 0
                             and epoch <= args.physics_freeze_prior_epochs)
            frozen_bias_slice = None

            if freeze_active:
                k = physics_frozen_heads
                if k > 0:
                    frozen_bias_slice = model.bb.attn_bias[:k].detach().clone()
                    if model.bb.attn_bias.grad is not None:
                        model.bb.attn_bias.grad[:k].zero_()
                    if ib == 1:
                        if k >= args.heads:
                            print(f"  [FREEZE] Epoch {epoch}: All {args.heads} heads frozen "
                                  f"(no trainable heads during warmup)")
                        else:
                            print(f"  [FREEZE] Epoch {epoch}: Physics-prior heads "
                                  f"(0-{k - 1}) frozen; heads {k}-{args.heads - 1} trainable")

            opt.step()

            if frozen_bias_slice is not None:
                with torch.no_grad():
                    model.bb.attn_bias[:k].copy_(frozen_bias_slice)

            if sched is not None: sched.step()

            running_total += loss.item(); running_ce += loss_ce.item(); running_lusi += loss_lusi.item()

            if args.smoke and ib >= 10: break

        if DEVICE == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

        prior_history.append(model.bb.attn_bias.detach().cpu().numpy().copy())

        # Validation
        model.eval()
        top1 = 0; top3 = 0; total = 0
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
                if args.smoke and vb >= 5: break

        acc1 = top1 / max(1, total); acc3 = top3 / max(1, total)
        n_batches = max(1, len(train_dl))
        avg_total = running_total / n_batches
        avg_ce = running_ce / n_batches
        avg_lusi = running_lusi / n_batches

        head_sim = measure_head_collapse(model)
        prior_rms = float(model.bb.attn_bias.detach().pow(2).mean().sqrt().cpu())

        history["epoch"].append(epoch)
        history["train_total_loss"].append(avg_total)
        history["train_ce_loss"].append(avg_ce)
        history["train_lusi_loss"].append(avg_lusi)
        history["val_acc1"].append(acc1); history["val_acc3"].append(acc3)

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
                "physics_frozen_heads": physics_frozen_heads,
                "manual_prior_bands": args.manual_prior_bands,
                "manual_prior_sigma": args.manual_prior_sigma,
                "manual_prior_assign": args.manual_prior_assign,
                "manual_prior_normalize": args.manual_prior_normalize,
                "prior_bias_path": args.prior_bias_path,
                "ref_continuum": args.ref_continuum,
                "ref_use_absdepth": args.ref_use_absdepth,
            }, "ckpt_best_qoiattn_nozeros.pt")

    # --------------------------- save history + final ckpt ---------------------------
    df_history = pd.DataFrame(history)
    df_history.to_csv("training_history.csv", index=False)
    print("\n  ✔ Saved training_history.csv")

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
        "physics_frozen_heads": physics_frozen_heads,
        "manual_prior_bands": args.manual_prior_bands,
        "manual_prior_sigma": args.manual_prior_sigma,
        "manual_prior_assign": args.manual_prior_assign,
        "manual_prior_normalize": args.manual_prior_normalize,
        "prior_bias_path": args.prior_bias_path,
        "ref_continuum": args.ref_continuum,
        "ref_use_absdepth": args.ref_use_absdepth,
    }, "spectral_stage1_qoiattn_nozeros.pt")

    prior_arr = np.array(prior_history)
    np.save("prior_evolution.npy", prior_arr)
    print("  ✔ Saved prior_evolution.npy")

    # --------------------------- training plots ---------------------------
    print("==> Generating training plots ...")
    plt.switch_backend("Agg")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(history["epoch"], history["train_total_loss"], "b-o", label="Total Loss")
    if args.use_lusi:
        ax1.plot(history["epoch"], history["train_ce_loss"], "g--", label="CE Loss")
        ax1.plot(history["epoch"], history["train_lusi_loss"], "r--", label="LUSI Loss")
    ax1.set_ylabel("Loss"); ax1.set_title("Training Progress")
    ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.plot(history["epoch"], history["val_acc1"], "g-o", label="Val Top-1")
    ax2.plot(history["epoch"], history["val_acc3"], "m--s", label="Val Top-3")
    ax2.set_ylabel("Accuracy"); ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.3); ax2.legend()
    plt.tight_layout(); plt.savefig("training_curves.png", dpi=150)
    print("  ✔ Saved training_curves.png")

    # --------------------------- transparency report ---------------------------
    print("\n==> Generating Transparency Report from BEST checkpoint...")
    best_ckpt = torch.load("ckpt_best_qoiattn_nozeros.pt", map_location=DEVICE, weights_only=False)
    model.bb.load_state_dict(best_ckpt["backbone"])
    model.head.load_state_dict(best_ckpt["head"])
    model.eval()

    all_preds = []; all_targets = []
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
            all_preds.extend(preds); all_targets.extend(targets)

    cls_report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(cls_report).transpose()
    df_report.to_csv("transparency_report_metrics.csv")
    cm = confusion_matrix(all_targets, all_preds)
    pd.DataFrame(cm).to_csv("transparency_report_confusion_matrix.csv")
    print("  ✔ Saved transparency_report_metrics.csv")
    print("  ✔ Saved transparency_report_confusion_matrix.csv")

    df_filtered = df_report.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    df_sorted = df_filtered.sort_values(by="f1-score", ascending=True)
    print("\n[ALERT] Bottom 5 Performing Classes:")
    print(df_sorted[["precision", "recall", "f1-score", "support"]].head(5))

    # --------------------------- attention export ---------------------------
    if getattr(args, "dump_attn", False):
        os.makedirs(args.attn_out, exist_ok=True)
        n_h = args.heads
        print(f"==> Collecting per-head attention on validation set ({n_h} heads) ...")

        global_sum_ph = np.zeros((n_h, n_bands), dtype=np.float64)
        total_cnt = 0
        class_sum_ph = np.zeros((n_h, NCLS, n_bands), dtype=np.float64)
        class_cnt = np.zeros(NCLS, dtype=np.int64)

        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE)
                if use_cuda_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        _, A = model(xb, return_attn=True)
                else:
                    _, A = model(xb, return_attn=True)

                A_np = A.cpu().numpy()
                y_np = yb.cpu().numpy()

                global_sum_ph += A_np.sum(axis=0)
                total_cnt += A_np.shape[0]

                for c in np.unique(y_np):
                    if c < 0 or c >= NCLS: continue
                    mask = (y_np == c)
                    if np.any(mask):
                        class_sum_ph[:, c, :] += A_np[mask].sum(axis=0)
                        class_cnt[c] += int(mask.sum())

        global_mean = global_sum_ph.sum(axis=0) / (n_h * max(1, total_cnt))
        class_mean = np.zeros((NCLS, n_bands), dtype=np.float64)
        nz = class_cnt > 0
        class_mean[nz] = class_sum_ph.sum(axis=0)[nz] / (n_h * class_cnt[nz, None])

        np.savetxt(os.path.join(args.attn_out, "band_attention_global.csv"),
                   global_mean.reshape(1, -1), delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(args.attn_out, "band_attention_by_class.csv"),
                   class_mean, delimiter=",", fmt="%.6f")

        for h in range(n_h):
            gh_mean = global_sum_ph[h] / max(1, total_cnt)
            ch_mean = np.zeros((NCLS, n_bands), dtype=np.float64)
            ch_mean[nz] = class_sum_ph[h, nz, :] / class_cnt[nz, None]
            np.savetxt(os.path.join(args.attn_out, f"band_attention_global_head{h}.csv"),
                       gh_mean.reshape(1, -1), delimiter=",", fmt="%.6f")
            np.savetxt(os.path.join(args.attn_out, f"band_attention_by_class_head{h}.csv"),
                       ch_mean, delimiter=",", fmt="%.6f")

        print(f"  Saved head-averaged + {n_h} per-head attention CSVs to {args.attn_out}/")

    elapsed = time.time() - start_time
    print(f"\n[TIMER] Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
