#!/usr/bin/env python3
"""
Build a curated Group-1 ferric-mineral PCA prior from EMIT-resampled USGS .npy spectra.

This version reads filenames directly from --npy_dir (no mineral matrix required).
Each .npy file is classified by keyword matching on its filename, then filtered by --mode.

Outputs (under --out_dir):
    selected_group1_ferric_metadata.csv             which files were used + their category
    missing_or_dropped_selected_spectra.csv         files that failed sanity checks
    pca_band_missingness_report.csv                 per-band stats incl. used_for_pca flag
    selected_group1_ferric_spectra_raw.npy          (M, 285) sanitized but not preprocessed
    selected_group1_ferric_spectra_preprocessed.npy (M, 285) after row_zscore (or chosen)
    pca_components_signed.npy                       (k, 285) signed eigenvectors
    pca_scores.npy                                  (M, k)
    pca_valid_mask.npy                              (285,) bool, bands used by PCA
    pca_explained_variance.csv                      per-PC variance ratios
    pca_top_wavelengths.csv                         top-10 bands per PC
    pca_attention_bias_zabs_H{N}.npy                (N, 285) bias = z(|v_h|), padded to N heads
    pca_attention_bias_max_H{N}.npy                 (N, 285) bias = |v_h|/max|v_h|, padded
    pca_group1_ferric_prior.png                     diagnostic plot

Example:
python build_group1_ferric_pca_prior.py \
  --npy_dir /Users/kmccoy/Documents/USC/Research/Dissertation/Spectra/group1_all \
  --wavelengths /Users/kmccoy/Documents/USC/Research/Dissertation/Data/emit_wavelength_centers_nm.npy \
  --wv_mask /Users/kmccoy/Documents/USC/Research/Dissertation/Spectra/group1_all/water_vapor_mask_285.npy \
  --out_dir /Users/kmccoy/Documents/USC/Research/Dissertation/Spectra/group1_ferric_pca_prior \
  --mode ferric_with_hard_negatives \
  --preprocess row_zscore \
  --k 4 --n_heads 4
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Selection keywords
# ============================================================

TARGET_KEYWORDS = [
    "hematite", "nanohematite", "goethite", "goet",
]

FERRIC_CONFUSER_KEYWORDS = [
    "maghemite", "ferrihydrite", "lepidocrocite", "lepidochros", "lepidocrosite",
    "jarosite", "schwertmannite",
    "fe_hydroxide", "fe-hydroxide", "fe hydroxide",
    "fe3+mix",
    "acid_mine", "acid mine", "amd",
    "coquimbite", "copiapite",
    "limonite", "pitchlimon", "pitch_limonite",
    "desert", "varnish",
    "mn-coating", "black_mn", "tailngs", "tailings",
]

HARD_NEGATIVE_KEYWORDS = [
    "magnetite", "pyrite",
    "chlorite", "clinochlore", "thuringite",
    "actinolite", "tremolite", "cummingtonite", "hornfels",
    "pyroxene", "hypersthene", "diopside", "pigeonite", "augite", "bronzite",
    "enstatite", "hedenbergite",
    "olivine",
    "epidote", "nontronite",
    "siderite", "riebeckite", "chromite",
]

EXCLUDE_KEYWORDS = [
    "plastic", "tarp",
    "sulfur", "neodymium", "samarium", "ree",
    "azurite", "malachite", "chrysocolla", "cuprite", "copper",
    "blue_efflor", "green_slime",
    "rhodonite", "rhodochrosite",
    "feldspar", "albite", "orthoclase", "microcline", "bytownite",
    "lazurite", "cinnabar", "pectolite",
    "almandine", "staurolite", "axinite", "jadeite",
]


# ============================================================
# Helpers
# ============================================================

def normalize_text(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).lower()


def contains_any(text, keywords):
    text = normalize_text(text)
    return any(k.lower() in text for k in keywords)


def classify_filename(filename):
    """
    Classify a .npy filename by keyword match. Returns one of:
      target, ferric_confuser, hard_negative, excluded, other
    """
    text = normalize_text(filename)

    if contains_any(text, EXCLUDE_KEYWORDS):
        return "excluded"
    if contains_any(text, TARGET_KEYWORDS):
        return "target"
    if contains_any(text, FERRIC_CONFUSER_KEYWORDS):
        return "ferric_confuser"
    if contains_any(text, HARD_NEGATIVE_KEYWORDS):
        return "hard_negative"
    return "other"


def scan_and_select(npy_dir, mode):
    """
    List every *.npy in npy_dir, classify by filename, return the full DataFrame
    and the subset selected for the given mode.
    """
    npy_dir = Path(npy_dir)
    if not npy_dir.exists():
        raise FileNotFoundError(f"--npy_dir does not exist: {npy_dir}")

    all_files = sorted(npy_dir.glob("*.npy"))
    if not all_files:
        raise FileNotFoundError(f"No .npy files found in {npy_dir}")

    rows = []
    for path in all_files:
        rows.append({
            "Filename": path.name,
            "Name": path.stem,
            "npy_path": str(path),
            "prior_category": classify_filename(path.name),
        })
    df = pd.DataFrame(rows)

    if mode == "ferric_targets":
        keep = df["prior_category"].isin(["target"])
    elif mode == "ferric_with_confusers":
        keep = df["prior_category"].isin(["target", "ferric_confuser"])
    elif mode == "ferric_with_hard_negatives":
        keep = df["prior_category"].isin(["target", "ferric_confuser", "hard_negative"])
    elif mode == "all_group1_valid":
        keep = ~df["prior_category"].isin(["excluded"])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df, df[keep].copy()


def sanitize_spectrum_with_report(
    x, wl, absmax=1e20, min_finite=30, max_missing_frac=0.25, clip_floor=1e-6,
):
    """
    Handles NaN, Inf, -9999, and huge fill values.
    Small gaps are interpolated.
    Spectra with too much missingness are dropped (raise ValueError).
    """
    x = np.asarray(x, dtype=np.float64).squeeze()

    if x.ndim != 1:
        raise ValueError(f"Expected 1D spectrum after squeeze, got shape {x.shape}")
    if x.shape[0] != wl.shape[0]:
        raise ValueError(f"Spectrum length {x.shape[0]} != wavelength length {wl.shape[0]}")

    missing = (~np.isfinite(x)) | (np.abs(x) >= absmax) | (x == -9999.0)
    missing_frac = float(missing.mean())
    ok = ~missing
    n_ok = int(ok.sum())

    if n_ok < min_finite:
        raise ValueError(f"only {n_ok} finite points; min_finite={min_finite}")
    if missing_frac > max_missing_frac:
        raise ValueError(
            f"missing fraction {missing_frac:.3f} exceeds max_missing_frac={max_missing_frac}"
        )

    if missing.any():
        x[missing] = np.interp(wl[missing], wl[ok], x[ok])

    x = np.maximum(x, clip_floor)
    if not np.isfinite(x).all():
        raise ValueError("non-finite values remain after interpolation")

    return x.astype(np.float32), missing.astype(bool), missing_frac


def preprocess_spectra(X, valid_mask, method):
    """
    Recommended method: row_zscore -- removes albedo dominance, emphasizes spectral shape.
    """
    X = np.asarray(X, dtype=np.float32).copy()

    if method == "raw":
        return X
    if method == "row_center":
        mu = X[:, valid_mask].mean(axis=1, keepdims=True)
        return (X - mu).astype(np.float32)
    if method == "row_zscore":
        mu = X[:, valid_mask].mean(axis=1, keepdims=True)
        sd = X[:, valid_mask].std(axis=1, keepdims=True)
        return ((X - mu) / (sd + 1e-6)).astype(np.float32)
    if method == "l2":
        norm = np.linalg.norm(X[:, valid_mask], axis=1, keepdims=True)
        return (X / (norm + 1e-8)).astype(np.float32)

    raise ValueError(f"Unknown preprocess method: {method}")


def compute_pca_svd(X, k):
    """
    PCA by SVD.

    X: M x P
    Returns:
      pcs: k x P signed loadings (rows are eigenvectors)
      ratios: k explained variance ratios
      scores: M x k
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    eigvals = (S ** 2) / max(1, X.shape[0] - 1)
    ratios = eigvals / (eigvals.sum() + 1e-12)

    k = min(k, Vt.shape[0])
    pcs = Vt[:k].astype(np.float32)
    scores = (U[:, :k] * S[:k]).astype(np.float32)

    pcs /= np.linalg.norm(pcs, axis=1, keepdims=True) + 1e-8
    return pcs, ratios[:k], scores


def make_attention_bias_zabs(pcs_full, valid_mask):
    """beta_h(lambda) = zscore(|v_h(lambda)|)."""
    pcs_full = np.asarray(pcs_full, dtype=np.float32)
    bias = np.zeros_like(pcs_full, dtype=np.float32)
    for h in range(pcs_full.shape[0]):
        v = np.abs(pcs_full[h])
        vv = v[valid_mask]
        z = np.zeros_like(v)
        z[valid_mask] = (vv - vv.mean()) / (vv.std() + 1e-6)
        bias[h] = z
    return bias


def make_attention_bias_max(pcs_full, valid_mask):
    """beta_h(lambda) = |v_h(lambda)| / max|v_h|. Peaks at 1, off-bands at 0."""
    pcs_full = np.asarray(pcs_full, dtype=np.float32)
    bias = np.zeros_like(pcs_full, dtype=np.float32)
    for h in range(pcs_full.shape[0]):
        v = np.abs(pcs_full[h])
        mx = float(v[valid_mask].max()) if v[valid_mask].size > 0 else 0.0
        if mx <= 0:
            continue
        bias[h] = v / (mx + 1e-8)
        bias[h, ~valid_mask] = 0.0
    return bias


def pad_or_truncate_bias(bias, n_heads):
    """Force the bias to (n_heads, P) by zero-padding or truncating along axis 0."""
    k_actual, P = bias.shape
    if n_heads == k_actual:
        return bias
    if n_heads < k_actual:
        print(f"[BIAS] Truncating from k={k_actual} to n_heads={n_heads}")
        return bias[:n_heads].astype(np.float32)
    pad = np.zeros((n_heads - k_actual, P), dtype=np.float32)
    print(f"[BIAS] Padding from k={k_actual} to n_heads={n_heads} (extra rows = zero)")
    return np.vstack([bias, pad]).astype(np.float32)


def plot_pca_prior(wl, X, meta, pcs_full, bias_zabs, bias_max, ratios, valid_mask, out_png):
    k = pcs_full.shape[0]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    category_style = {
        "target": 0.90, "ferric_confuser": 0.50,
        "hard_negative": 0.28, "other": 0.15,
    }
    for cat, alpha in category_style.items():
        idx = np.where(meta["prior_category"].values == cat)[0]
        if len(idx) == 0:
            continue
        first = True
        for i in idx:
            ax1.plot(wl, X[i], linewidth=0.8, alpha=alpha, label=cat if first else None)
            first = False

    ax1.set_ylabel("Preprocessed spectrum")
    ax1.set_title("Selected Group-1 spectra used for ferric PCA prior")
    ax1.grid(alpha=0.2); ax1.legend(loc="upper right")

    for h in range(k):
        v = pcs_full[h].copy()
        denom = np.max(np.abs(v[valid_mask])) + 1e-8
        ax2.plot(wl, v / denom, color=colors[h % len(colors)], linewidth=1.3,
                 label=f"PC{h+1} signed ({ratios[h] * 100:.1f}% var)")
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("Signed PC loading\n(normalized)")
    ax2.set_title("Signed PCA coefficients")
    ax2.grid(alpha=0.2); ax2.legend(loc="upper right")

    for h in range(k):
        ax3.plot(wl, bias_zabs[h], color=colors[h % len(colors)], linewidth=1.3,
                 label=rf"$\beta_{{{h+1}}}(\lambda)=z(|v_{h+1}|)$")
    ax3.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax3.set_ylabel(r"z-scored $|v_h|$")
    ax3.set_title("Attention-bias init (zabs normalization)")
    ax3.grid(alpha=0.2); ax3.legend(loc="upper right")

    for h in range(k):
        ax4.plot(wl, bias_max[h], color=colors[h % len(colors)], linewidth=1.3,
                 label=rf"$\beta_{{{h+1}}}(\lambda)=|v_{h+1}|/\max$")
    ax4.set_ylabel(r"$|v_h|/\max|v_h|$")
    ax4.set_xlabel("Wavelength (nm)")
    ax4.set_title("Attention-bias init (max normalization -- peaks=1, off-bands=0)")
    ax4.grid(alpha=0.2); ax4.legend(loc="upper right")

    # Canonical Jiang 2014 / Clark 1999 markers (matches paper's tab:fe_diagnostics)
    diagnostic_markers = [480, 535, 670, 860, 920]
    for ax in axes:
        for x in diagnostic_markers:
            ax.axvline(x, color="black", linestyle="--", linewidth=0.8, alpha=0.25)
            ax.text(x, 0.98, f"{x}", rotation=90, va="top", ha="right",
                    fontsize=8, alpha=0.7, transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npy_dir", required=True,
                    help="Directory containing EMIT-resampled *.npy spectra")
    ap.add_argument("--wavelengths", required=True,
                    help="Path to emit_wavelength_centers_nm.npy")
    ap.add_argument("--wv_mask", required=True,
                    help="Path to water_vapor_mask_285.npy; True=good band")
    ap.add_argument("--out_dir", required=True, help="Output directory")

    ap.add_argument("--mode", default="ferric_with_hard_negatives",
                    choices=["ferric_targets", "ferric_with_confusers",
                             "ferric_with_hard_negatives", "all_group1_valid"])
    ap.add_argument("--preprocess", default="row_zscore",
                    choices=["raw", "row_center", "row_zscore", "l2"])

    ap.add_argument("--k", type=int, default=4,
                    help="Number of PCA components to extract.")
    ap.add_argument("--n_heads", type=int, default=4,
                    help="Pad the saved bias to (n_heads, 285). PCA still uses k components; "
                         "rows beyond k are zero so those heads stay randomly initialized.")

    ap.add_argument("--absmax", type=float, default=1e20)
    ap.add_argument("--min_finite", type=int, default=30)
    ap.add_argument("--max_missing_frac", type=float, default=0.25)
    ap.add_argument("--max_band_missing_frac", type=float, default=0.25)
    ap.add_argument("--clip_floor", type=float, default=1e-6)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wl = np.load(args.wavelengths).astype(np.float32).reshape(-1)
    wv_mask = np.load(args.wv_mask).astype(bool).reshape(-1)
    if wv_mask.shape[0] != wl.shape[0]:
        raise ValueError(f"wv_mask length {wv_mask.shape[0]} != wavelength length {wl.shape[0]}")

    # ---- scan + classify ----
    all_df, selected = scan_and_select(args.npy_dir, args.mode)

    print(f"[SCAN] Found {len(all_df)} .npy files in {args.npy_dir}")
    print("[SCAN] Category counts (all files):")
    print(all_df["prior_category"].value_counts())

    print(f"\n[SELECT] mode={args.mode} kept {len(selected)} files")
    print("[SELECT] Category counts (kept):")
    print(selected["prior_category"].value_counts())

    if len(selected) < 3:
        raise RuntimeError(
            f"Only {len(selected)} files matched mode={args.mode}. "
            f"Check filenames or try --mode ferric_with_hard_negatives."
        )

    # ---- load + sanitize each spectrum ----
    spectra = []
    missing_masks = []
    kept_rows = []
    missing_rows = []

    for _, row in selected.iterrows():
        path = row["npy_path"]
        try:
            x_raw = np.load(path).squeeze()
            x, missing_mask, missing_frac = sanitize_spectrum_with_report(
                x_raw, wl,
                absmax=args.absmax, min_finite=args.min_finite,
                max_missing_frac=args.max_missing_frac, clip_floor=args.clip_floor,
            )
        except Exception as e:
            missing_rows.append({
                "Filename": row.get("Filename", ""),
                "Name": row.get("Name", ""),
                "prior_category": row.get("prior_category", ""),
                "npy_path": str(path),
                "reason": str(e),
            })
            continue

        spectra.append(x)
        missing_masks.append(missing_mask)
        kept = row.copy()
        kept["missing_frac_before_interp"] = missing_frac
        kept_rows.append(kept)

    if len(spectra) < 3:
        raise RuntimeError(
            f"Only {len(spectra)} usable spectra after sanitization. "
            f"Check missingness thresholds."
        )

    meta = pd.DataFrame(kept_rows).reset_index(drop=True)
    X_raw = np.vstack(spectra).astype(np.float32)
    missing_masks = np.vstack(missing_masks).astype(bool)

    missing_frac_by_band = missing_masks.mean(axis=0)
    band_quality_mask = missing_frac_by_band <= args.max_band_missing_frac
    valid_mask = wv_mask & band_quality_mask

    print(f"\n[SPECTRA] Loaded usable spectra: {X_raw.shape}")
    print("[SPECTRA] Category counts after sanitization:")
    print(meta["prior_category"].value_counts())
    print(f"\n[MASK] Water-vapor good bands: {int(wv_mask.sum())}/{len(wv_mask)}")
    print(f"[MASK] Band-quality good bands: {int(band_quality_mask.sum())}/{len(band_quality_mask)}")
    print(f"[MASK] PCA uses bands: {int(valid_mask.sum())}/{len(valid_mask)}")

    if valid_mask.sum() < 5:
        raise RuntimeError("Too few valid PCA bands. Check masks and missingness thresholds.")

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(
            out_dir / "missing_or_dropped_selected_spectra.csv", index=False
        )
        print(f"\n[WARN] Missing/dropped spectra: {len(missing_rows)}")
        print(f"[WARN] Saved: {out_dir / 'missing_or_dropped_selected_spectra.csv'}")

    band_report = pd.DataFrame({
        "band_index": np.arange(len(wl)),
        "wavelength_nm": wl,
        "wv_mask_good": wv_mask,
        "missing_frac_by_band_before_interp": missing_frac_by_band,
        "band_quality_good": band_quality_mask,
        "used_for_pca": valid_mask,
    })
    band_report.to_csv(out_dir / "pca_band_missingness_report.csv", index=False)

    # ---- preprocess ----
    X_for_pca = preprocess_spectra(X_raw, valid_mask=valid_mask, method=args.preprocess)

    if args.preprocess in ("row_center", "row_zscore"):
        per_band_std = X_for_pca[:, valid_mask].std(axis=0)
        wl_valid = wl[valid_mask]
        top5 = np.argsort(per_band_std)[-5:][::-1]
        print(f"\n[PREPROC] After {args.preprocess}, top-5 most-variable bands across library:")
        for j, idx in enumerate(top5, 1):
            print(f"  {j}. {wl_valid[idx]:7.1f} nm  std={per_band_std[idx]:.4f}")

    # ---- PCA ----
    pcs_valid, ratios, scores = compute_pca_svd(X_for_pca[:, valid_mask], k=args.k)

    pcs_full = np.zeros((pcs_valid.shape[0], wl.shape[0]), dtype=np.float32)
    pcs_full[:, valid_mask] = pcs_valid

    bias_zabs = make_attention_bias_zabs(pcs_full, valid_mask=valid_mask)
    bias_max  = make_attention_bias_max( pcs_full, valid_mask=valid_mask)

    bias_zabs_h = pad_or_truncate_bias(bias_zabs, args.n_heads)
    bias_max_h  = pad_or_truncate_bias(bias_max,  args.n_heads)

    # ---- save outputs ----
    meta.to_csv(out_dir / "selected_group1_ferric_metadata.csv", index=False)
    np.save(out_dir / "selected_group1_ferric_spectra_raw.npy", X_raw)
    np.save(out_dir / "selected_group1_ferric_spectra_preprocessed.npy", X_for_pca)
    np.save(out_dir / "pca_components_signed.npy", pcs_full)
    np.save(out_dir / "pca_scores.npy", scores)
    np.save(out_dir / "pca_valid_mask.npy", valid_mask)

    zabs_path = out_dir / f"pca_attention_bias_zabs_H{args.n_heads}.npy"
    max_path  = out_dir / f"pca_attention_bias_max_H{args.n_heads}.npy"
    np.save(zabs_path, bias_zabs_h)
    np.save(max_path,  bias_max_h)

    ev = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(ratios))],
        "explained_variance_ratio": ratios,
        "explained_variance_percent": ratios * 100.0,
    })
    ev.to_csv(out_dir / "pca_explained_variance.csv", index=False)
    print("\n[PCA EXPLAINED VARIANCE]")
    print(ev.to_string(index=False))

    top_rows = []
    for h in range(bias_zabs.shape[0]):
        idx = np.argsort(bias_zabs[h])[-20:][::-1]
        idx = [i for i in idx if valid_mask[i]]
        print(f"\n[TOP WAVELENGTHS] PC{h+1}")
        for i in idx[:10]:
            print(f"  {wl[i]:8.2f} nm | signed={pcs_full[h, i]: .4f} "
                  f"| bias_zabs={bias_zabs[h, i]: .2f} | bias_max={bias_max[h, i]: .2f}")
            top_rows.append({
                "PC": f"PC{h+1}",
                "band_index": int(i),
                "wavelength_nm": float(wl[i]),
                "signed_loading": float(pcs_full[h, i]),
                "attention_bias_zabs": float(bias_zabs[h, i]),
                "attention_bias_max":  float(bias_max[h, i]),
            })
    pd.DataFrame(top_rows).to_csv(out_dir / "pca_top_wavelengths.csv", index=False)

    plot_pca_prior(
        wl=wl, X=X_for_pca, meta=meta,
        pcs_full=pcs_full,
        bias_zabs=bias_zabs, bias_max=bias_max,
        ratios=ratios, valid_mask=valid_mask,
        out_png=out_dir / "pca_group1_ferric_prior.png",
    )

    print("\n[DONE]")
    print(f"Saved outputs to: {out_dir}")
    print(f"Use one of these to seed the transformer attention bias (matches your --heads):")
    print(f"  zabs (z(|v|), needs small physics_alpha ~0.2): {zabs_path}")
    print(f"  max  (peaks=1, needs physics_alpha ~0.5):       {max_path}")


if __name__ == "__main__":
    main()
