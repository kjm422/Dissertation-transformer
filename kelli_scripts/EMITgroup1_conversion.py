"""
Convolve USGS splib06/splib07 ASCII spectra (.asc) to EMIT band centers + FWHM
(using Gaussian bandpasses derived from FWHM), then build a stacked matrix
of ALL Group 1 spectra ready for PCA / Tetracorder pipeline.

Water-vapor absorption bands (~1.4 µm, ~1.9 µm) and the noisy long-wave edge
(>2450 nm) are masked out before the final concatenation, since:
  - USGS reference spectra are measured at surface level (no atmosphere)
  - EMIT L1B TOA reflectance contains atmospheric H₂O absorption at these bands
  - PCA on unmasked data would capture atmospheric features, not mineral features

Outputs:
  - Per-spectrum *_EMIT.npy  (285 bands, full)
  - water_vapor_mask_285.npy (boolean mask: True = GOOD band)
  - ALL_group1_spectra.npy   (N_spectra x N_good_bands)
  - ALL_group1_spectra.json  (filenames + metadata)

Run:
  python EMITgroup1_conversion.py
"""

from __future__ import annotations

import json
import os
import glob
import numpy as np
from netCDF4 import Dataset


# ========================= CONFIG =========================

EMIT_NC = "/Users/kmccoy/Documents/USC/Research/Dissertation/Data/EMIT_L1B_RAD_001_20230227T111506_2305807_012.nc"
IN_ASC = "/Users/kmccoy/Documents/USC/Research/Dissertation/Spectra/group1_all"
OUT_DIR = "/Users/kmccoy/Documents/USC/Research/Dissertation/Spectra/group1_all"

# Water-vapor / low-SNR mask thresholds (nm)
# Bands whose centers fall in these ranges are masked OUT
H2O_MASK_RANGES_NM = [
    (1340.0, 1500.0),   # 1.4 µm H₂O absorption
    (1800.0, 1960.0),   # 1.9 µm H₂O absorption
    (2450.0, np.inf),   # long-wave edge, low SNR
]


# ========================= EMIT read =========================

def read_emit_centers_fwhm_nm(nc_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read EMIT band centers and FWHM (nm) from:
      group 'sensor_band_parameters' variables 'wavelengths' and 'fwhm'
    Handles _FillValue = -9999 by converting to NaN.
    """
    with Dataset(nc_path, "r") as ds:
        g = ds.groups["sensor_band_parameters"]
        centers_nm = np.array(g.variables["wavelengths"][:], dtype=float)
        fwhm_nm = np.array(g.variables["fwhm"][:], dtype=float)

        fill_centers = getattr(g.variables["wavelengths"], "_FillValue", -9999.0)
        fill_fwhm = getattr(g.variables["fwhm"], "_FillValue", -9999.0)

        bad = (
            ~np.isfinite(centers_nm) | ~np.isfinite(fwhm_nm) |
            (centers_nm == fill_centers) | (fwhm_nm == fill_fwhm) |
            (centers_nm <= 0) | (fwhm_nm <= 0)
        )
        centers_nm[bad] = np.nan
        fwhm_nm[bad] = np.nan

    return centers_nm, fwhm_nm


def build_water_vapor_mask(
    centers_nm: np.ndarray,
    mask_ranges_nm: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Build a boolean mask for EMIT bands: True = GOOD (keep), False = masked out.

    Bands are masked if their center falls in any of the specified ranges,
    or if the center wavelength is NaN.
    """
    if mask_ranges_nm is None:
        mask_ranges_nm = H2O_MASK_RANGES_NM

    good = np.isfinite(centers_nm)
    for lo, hi in mask_ranges_nm:
        good &= ~((centers_nm >= lo) & (centers_nm <= hi))
    return good


# ========================= USGS ASCII read =========================

def load_usgs_ascii_um(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a USGS ASCII spectrum file (.asc).

    Returns:
      wl_um  : wavelengths in microns (um)
      refl   : reflectance (unitless)
    """
    wl = []
    refl = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                wl.append(float(parts[0]))
                refl.append(float(parts[1]))
            except ValueError:
                continue

    wl_um = np.asarray(wl, dtype=float)
    r = np.asarray(refl, dtype=float)

    m = np.isfinite(wl_um) & np.isfinite(r)
    wl_um = wl_um[m]
    r = r[m]
    if wl_um.size == 0:
        return wl_um, r

    idx = np.argsort(wl_um)
    return wl_um[idx], r[idx]


# ========================= Convolution =========================

def convolve_to_emit_gaussian(
    spec_wl_nm: np.ndarray,
    spec_refl: np.ndarray,
    emit_centers_nm: np.ndarray,
    emit_fwhm_nm: np.ndarray,
    *,
    k_sigma_window: float = 4.0,
    require_full_coverage: bool = True,
) -> np.ndarray:
    """
    Convolve a high-res spectrum to EMIT bands using a Gaussian SRF per band.

    For band b:
      sigma_b = FWHM_b / (2*sqrt(2*ln2))
      S_b(λ)  = exp(-0.5 * ((λ - mu_b)/sigma_b)^2)
      R_b ≈ (Σ R(λ_j) S_b(λ_j) Δλ_j) / (Σ S_b(λ_j) Δλ_j)

    Returns:
      emit_refl: array of shape (n_bands,) (285 for EMIT)
    """
    spec_wl_nm = np.asarray(spec_wl_nm, float)
    spec_refl = np.asarray(spec_refl, float)
    emit_centers_nm = np.asarray(emit_centers_nm, float)
    emit_fwhm_nm = np.asarray(emit_fwhm_nm, float)

    m = np.isfinite(spec_wl_nm) & np.isfinite(spec_refl)
    spec_wl_nm = spec_wl_nm[m]
    spec_refl = spec_refl[m]
    if spec_wl_nm.size < 5:
        return np.full_like(emit_centers_nm, np.nan, dtype=float)

    order = np.argsort(spec_wl_nm)
    spec_wl_nm = spec_wl_nm[order]
    spec_refl = spec_refl[order]

    dlam = np.gradient(spec_wl_nm)
    sigma = emit_fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    out = np.full_like(emit_centers_nm, np.nan, dtype=float)
    wl_min, wl_max = spec_wl_nm[0], spec_wl_nm[-1]

    for i, (mu, s) in enumerate(zip(emit_centers_nm, sigma)):
        if not np.isfinite(mu) or not np.isfinite(s) or s <= 0:
            continue

        lo = mu - k_sigma_window * s
        hi = mu + k_sigma_window * s

        if require_full_coverage and (lo < wl_min or hi > wl_max):
            continue

        j0 = np.searchsorted(spec_wl_nm, lo, side="left")
        j1 = np.searchsorted(spec_wl_nm, hi, side="right")
        if j1 - j0 < 3:
            continue

        wl_seg = spec_wl_nm[j0:j1]
        r_seg = spec_refl[j0:j1]
        d_seg = dlam[j0:j1]

        w = np.exp(-0.5 * ((wl_seg - mu) / s) ** 2)
        den = np.sum(w * d_seg)
        if den > 0:
            out[i] = np.sum(r_seg * w * d_seg) / den

    return out


# ========================= Batch driver =========================

def convolve_folder_usgs_to_emit(
    emit_nc_path: str,
    input_asc_folder: str,
    output_folder: str,
    *,
    pattern: str = "*.asc",
    save_csv: bool = False,
    require_full_coverage: bool = True,
    k_sigma_window: float = 4.0,
    recursive: bool = False,
    save_emit_wavelengths: bool = True,
    emit_wavelengths_filename: str = "emit_wavelength_centers_nm.npy",
    emit_fwhm_filename: str = "emit_fwhm_nm.npy",
    emit_wavelengths_csv: bool = True,
    emit_wavelengths_csv_filename: str = "emit_wavelength_centers_fwhm_nm.csv",
) -> None:
    """
    Batch convert all .asc files in a folder to EMIT 285-band vectors.
    Prints a sanity check on the first file only.
    """
    emit_centers_nm, emit_fwhm_nm = read_emit_centers_fwhm_nm(emit_nc_path)

    os.makedirs(output_folder, exist_ok=True)

    # Save wavelength centers/FWHM once
    if save_emit_wavelengths:
        out_wl = os.path.join(output_folder, emit_wavelengths_filename)
        out_fw = os.path.join(output_folder, emit_fwhm_filename)
        np.save(out_wl, emit_centers_nm.astype(np.float32))
        np.save(out_fw, emit_fwhm_nm.astype(np.float32))
        print(f"WROTE: {out_wl}")
        print(f"WROTE: {out_fw}")

        if emit_wavelengths_csv:
            out_csv = os.path.join(output_folder, emit_wavelengths_csv_filename)
            arr = np.column_stack([emit_centers_nm, emit_fwhm_nm])
            np.savetxt(
                out_csv, arr, delimiter=",",
                header="emit_center_nm,emit_fwhm_nm", comments="", fmt="%.6f",
            )
            print(f"WROTE: {out_csv}")

    # Save water-vapor mask
    wv_mask = build_water_vapor_mask(emit_centers_nm)
    mask_path = os.path.join(output_folder, "water_vapor_mask_285.npy")
    np.save(mask_path, wv_mask)
    n_good = wv_mask.sum()
    n_masked = (~wv_mask).sum()
    print(f"WROTE: {mask_path}  ({n_good} good bands, {n_masked} masked)")

    if recursive:
        asc_files = sorted(glob.glob(os.path.join(input_asc_folder, "**", pattern), recursive=True))
    else:
        asc_files = sorted(glob.glob(os.path.join(input_asc_folder, pattern)))

    if not asc_files:
        raise FileNotFoundError(f"No files matching {pattern} found in {input_asc_folder}")

    printed_sanity = False

    for asc_path in asc_files:
        wl_um, refl = load_usgs_ascii_um(asc_path)
        if wl_um.size == 0:
            print(f"SKIP (empty/parse fail): {asc_path}")
            continue

        wl_nm = wl_um * 1000.0

        # Sanity check (first file only)
        if not printed_sanity:
            print("\n--- sanity check (first file only) ---")
            print(
                f"[{os.path.basename(asc_path)}] USGS wl min/max: "
                f"{np.nanmin(wl_um):.4f} .. {np.nanmax(wl_um):.4f} um (n={wl_um.size})"
            )
            print(
                f"[{os.path.basename(asc_path)}] Converted wl min/max: "
                f"{np.nanmin(wl_nm):.1f} .. {np.nanmax(wl_nm):.1f} nm"
            )

            emit_min = np.nanmin(emit_centers_nm)
            emit_max = np.nanmax(emit_centers_nm)
            print(
                f"EMIT centers min/max: {emit_min:.1f} .. {emit_max:.1f} nm "
                f"(bands={np.isfinite(emit_centers_nm).sum()}/{emit_centers_nm.size})"
            )

            ov_min = max(np.nanmin(wl_nm), emit_min)
            ov_max = min(np.nanmax(wl_nm), emit_max)
            print(
                f"Overlap (spectrum vs EMIT centers): {ov_min:.1f} .. {ov_max:.1f} nm  ->  "
                f"{'OK' if (ov_max > ov_min) else 'NO OVERLAP'}"
            )

            covered = (
                np.isfinite(emit_centers_nm)
                & (emit_centers_nm >= np.nanmin(wl_nm))
                & (emit_centers_nm <= np.nanmax(wl_nm))
            )
            print(f"EMIT band centers within spectrum range: {covered.sum()}/{emit_centers_nm.size}")
            print(f"After H₂O masking: {(covered & wv_mask).sum()} usable bands")
            print("--- end sanity check ---\n")

            printed_sanity = True

        y = convolve_to_emit_gaussian(
            wl_nm, refl,
            emit_centers_nm, emit_fwhm_nm,
            k_sigma_window=k_sigma_window,
            require_full_coverage=require_full_coverage,
        )

        base = os.path.splitext(os.path.basename(asc_path))[0]
        out_npy = os.path.join(output_folder, base + "_EMIT.npy")
        np.save(out_npy, y)

        if save_csv:
            out_csv = os.path.join(output_folder, base + "_EMIT.csv")
            arr = np.column_stack([emit_centers_nm, y])
            np.savetxt(
                out_csv, arr, delimiter=",",
                header="emit_center_nm,emit_reflectance", comments="",
            )

        print(f"WROTE: {out_npy}" + (" (+csv)" if save_csv else ""))


# ========================= Concatenation =========================

def load_as_row(path: str) -> np.ndarray:
    """Load a .npy and return as 2D row(s) [n_rows, n_bands]."""
    arr = np.load(path)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected array ndim={arr.ndim} in {path} (shape={arr.shape})")


def concat_all_group1(
    out_dir: str,
    wv_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Stack ALL per-spectrum *_EMIT.npy files into a single matrix.

    Args:
        out_dir: folder containing the *_EMIT.npy files
        wv_mask: boolean mask (285,). If provided, columns where mask=False
                 are dropped from the output matrix.

    Returns:
        X_full:  (N_spectra, 285)          — full 285-band matrix
        X_masked: (N_spectra, N_good_bands) — water-vapor-masked matrix
        names:   list of source filenames
    """
    all_npy = sorted(glob.glob(os.path.join(out_dir, "*_EMIT.npy")))

    # Exclude utility files (wavelength centers, fwhm, masks, concat outputs)
    skip_prefixes = (
        "emit_wavelength", "emit_fwhm", "water_vapor_mask",
        "ALL_", "goethite_hematite",
    )
    all_npy = [p for p in all_npy if not os.path.basename(p).startswith(skip_prefixes)]

    if not all_npy:
        raise FileNotFoundError(f"No *_EMIT.npy spectrum files found in: {out_dir}")

    rows = []
    names = []
    expected_bands = None

    for p in all_npy:
        a = load_as_row(p)
        if expected_bands is None:
            expected_bands = a.shape[1]
        elif a.shape[1] != expected_bands:
            raise ValueError(
                f"Band mismatch in {p}: got {a.shape[1]} bands, expected {expected_bands}"
            )
        rows.append(a)
        names.extend([os.path.basename(p)] * a.shape[0])

    X_full = np.vstack(rows)

    if wv_mask is not None:
        X_masked = X_full[:, wv_mask]
    else:
        X_masked = X_full

    return X_full, X_masked, names


# ========================= Main =========================

if __name__ == "__main__":

    # --- Step 1: Convolve all .asc spectra to EMIT 285 bands ---
    print("=" * 60)
    print("STEP 1: Convolve USGS spectra to EMIT bands")
    print("=" * 60)

    convolve_folder_usgs_to_emit(
        emit_nc_path=EMIT_NC,
        input_asc_folder=IN_ASC,
        output_folder=OUT_DIR,
        pattern="*.asc",
        save_csv=False,
        require_full_coverage=True,
        k_sigma_window=4.0,
        recursive=False,
        save_emit_wavelengths=True,
    )

    # --- Step 2: Load water-vapor mask and concatenate all spectra ---
    print("\n" + "=" * 60)
    print("STEP 2: Concatenate all Group 1 spectra (with H₂O mask)")
    print("=" * 60)

    wv_mask = np.load(os.path.join(OUT_DIR, "water_vapor_mask_285.npy"))
    emit_centers = np.load(os.path.join(OUT_DIR, "emit_wavelength_centers_nm.npy"))

    X_full, X_masked, filenames = concat_all_group1(OUT_DIR, wv_mask=wv_mask)

    n_spectra, n_bands_full = X_full.shape
    n_bands_masked = X_masked.shape[1]

    # Save full (285-band) matrix
    out_full_npy = os.path.join(OUT_DIR, "ALL_group1_spectra_full.npy")
    np.save(out_full_npy, X_full)

    # Save masked matrix (ready for PCA)
    out_masked_npy = os.path.join(OUT_DIR, "ALL_group1_spectra_masked.npy")
    np.save(out_masked_npy, X_masked)

    # Save masked wavelength centers (for plotting / band identification)
    out_masked_wl = os.path.join(OUT_DIR, "emit_wavelength_centers_masked_nm.npy")
    np.save(out_masked_wl, emit_centers[wv_mask])

    # Save metadata
    out_json = os.path.join(OUT_DIR, "ALL_group1_spectra.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "full_npy": os.path.basename(out_full_npy),
                "masked_npy": os.path.basename(out_masked_npy),
                "masked_wavelengths_npy": os.path.basename(out_masked_wl),
                "shape_full": list(X_full.shape),
                "shape_masked": list(X_masked.shape),
                "n_spectra": n_spectra,
                "n_bands_full": n_bands_full,
                "n_bands_masked": n_bands_masked,
                "mask_ranges_nm": [[lo, hi] for lo, hi in H2O_MASK_RANGES_NM],
                "files": filenames,
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {out_full_npy}")
    print(f"  shape: {X_full.shape}  (spectra x 285 full bands)")
    print(f"Saved: {out_masked_npy}")
    print(f"  shape: {X_masked.shape}  (spectra x {n_bands_masked} masked bands)")
    print(f"Saved: {out_masked_wl}")
    print(f"Saved: {out_json}")

    # Summary of what was masked
    masked_wl = emit_centers[~wv_mask]
    print(f"\nMasked {(~wv_mask).sum()} bands in ranges:")
    for lo, hi in H2O_MASK_RANGES_NM:
        in_range = masked_wl[(masked_wl >= lo) & (masked_wl <= hi)]
        if len(in_range) > 0:
            print(f"  {lo:.0f}-{hi:.0f} nm: {len(in_range)} bands "
                  f"({in_range[0]:.1f} .. {in_range[-1]:.1f} nm)")

    print(f"\n>>> Use ALL_group1_spectra_masked.npy ({X_masked.shape}) for PCA <<<")
