#!/usr/bin/env python3
"""
Aggregate SouthwestUS granule-level full-image inference metrics across the
four 4-head attn_outputs_* configurations.

For every (config, granule) pair where a prediction file
  Data/attn_outputs_<config>/SW/Xformer_predictSW_<scene_id>.npy
exists, this script:
  1. Maps <scene_id> -> GT row by the sorted index of the matching TOA cube
     in /Volumes/LaCie/Dissertation/data/SouthwestUS/TOA_reflectance_SW/.
  2. Loads ground truth from group1_mineralIDSW.npy[row].
  3. Computes top-1, top-3, macro precision/recall/F1 on the mineral-pixel
     mask (label > 0).

Writes Data/granule_inference_summary_SW.csv and prints config x granule
pivots for top-1, top-3, macro_f1, macro_recall (same layout as the original
Africa summary).

Usage:
    python kelli_scripts/granule_inference_summary_SW.py
"""
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

DATA   = "/Users/kmccoy/Documents/USC/Research/Dissertation/Data"
SW     = "/Volumes/LaCie/Dissertation/data/SouthwestUS"
TOADIR = os.path.join(SW, "TOA_reflectance_SW")
GT     = os.path.join(SW, "group1_mineralIDSW.npy")
OUT    = os.path.join(DATA, "granule_inference_summary_SW.csv")

CONFIGS = [
    "Trans4_diffnowv", "Manual4_diff", "PCA4zabs_diff", "Lusi4_diffnowv",
    "Trans8_diffnowv", "Manual8_diff", "PCAzabs8_diff", "Lusi8_diffnowv",
]
LABEL_MAP = {
    "Trans4_diffnowv": "Trans 4H",
    "Manual4_diff":    "Manual 4H",
    "PCA4zabs_diff":   "PCA-zabs 4H",
    "Lusi4_diffnowv":  "LUSI 4H",
    "Trans8_diffnowv": "Trans 8H",
    "Manual8_diff":    "Manual 8H",
    "PCAzabs8_diff":   "PCA-zabs 8H",
    "Lusi8_diffnowv":  "LUSI 8H",
}

SCENE_RE = re.compile(r"(\d{8}T\d{6}_\d{7}_\d{3})")
PRED_RE  = re.compile(r"^Xformer_predictSW_(\d{8}T\d{6}_\d{7}_\d{3})\.npy$")

# Granule eligibility: include only granules whose Group-1 mineral pixels make up
# at least this fraction of total pixels in the GT row. This excludes near-empty
# granules whose tiny mineral-pixel counts produce noisy per-granule accuracies.
MIN_MINERAL_FRAC = 0.20
MAX_GRANULES     = 20  # cap so the summary table is bounded
PIXELS_PER_ROW   = None  # computed on first GT load


def build_scene_to_row():
    """Sort TOA cubes the same way the inference loop did; index = GT row."""
    files = sorted(f for f in os.listdir(TOADIR) if f.startswith("TOAref_") and f.endswith(".npy"))
    mapping = {}
    for row, fname in enumerate(files):
        m = SCENE_RE.search(fname)
        if m:
            mapping[m.group(1)] = row
    return mapping


def compute_metrics(pred_path, y_true):
    preds = np.load(pred_path)
    if preds.shape[0] != y_true.size:
        return {"error": f"shape mismatch preds={preds.shape[0]} gt={y_true.size}"}

    top1 = preds[:, 0].astype(np.int32)
    top2 = preds[:, 2].astype(np.int32)
    top3 = preds[:, 4].astype(np.int32)

    mask = (y_true > 0)
    if not mask.any():
        return None

    yt = y_true[mask]
    p1 = top1[mask]
    p2 = top2[mask]
    p3 = top3[mask]

    acc1 = float(np.mean(p1 == yt))
    acc3 = float(np.mean((p1 == yt) | (p2 == yt) | (p3 == yt)))
    p, r, f1, _ = precision_recall_fscore_support(yt, p1, average="macro", zero_division=0)

    return {
        "n_mineral_px":    int(mask.sum()),
        "top1":            acc1,
        "top3":            acc3,
        "macro_precision": float(p),
        "macro_recall":    float(r),
        "macro_f1":        float(f1),
    }


def main():
    scene_to_row = build_scene_to_row()
    gt_full = np.load(GT)
    print(f"GT shape: {gt_full.shape}  ({len(scene_to_row)} scenes mapped)")

    # Pre-compute per-row mineral-pixel fraction and select the eligible row set.
    n_total = gt_full.shape[1]
    eligible_rows = []
    for r in range(gt_full.shape[0]):
        row_arr = gt_full[r].astype(np.float32)
        row_arr[np.isnan(row_arr)] = 0
        n_mineral = int(np.sum(row_arr.astype(np.int32) > 0))
        if n_mineral / n_total >= MIN_MINERAL_FRAC:
            eligible_rows.append(r)
    eligible_set = set(eligible_rows[:MAX_GRANULES])
    print(f"Eligible rows (>= {MIN_MINERAL_FRAC:.0%} mineral, capped at {MAX_GRANULES}): "
          f"{sorted(eligible_set)}  ({len(eligible_set)} granules)")

    rows = []
    for cfg_key in CONFIGS:
        cfg_label = LABEL_MAP[cfg_key]
        sw_dir = os.path.join(DATA, f"attn_outputs_{cfg_key}", "SW")
        if not os.path.isdir(sw_dir):
            print(f"[SKIP] {cfg_label}: no SW folder at {sw_dir}")
            continue

        for fname in sorted(os.listdir(sw_dir)):
            m = PRED_RE.match(fname)
            if not m:
                continue
            scene_id = m.group(1)
            row = scene_to_row.get(scene_id)
            if row is None:
                print(f"[SKIP] {cfg_label} {scene_id}: scene not in TOA folder")
                continue
            if row not in eligible_set:
                continue  # below MIN_MINERAL_FRAC threshold; quietly skip

            y_true = gt_full[row].astype(np.float32)
            y_true[np.isnan(y_true)] = 0
            y_true = y_true.astype(np.int32).flatten()

            metrics = compute_metrics(os.path.join(sw_dir, fname), y_true)
            if metrics is None:
                print(f"[SKIP] {cfg_label} row={row} {scene_id}: no mineral pixels")
                continue
            if "error" in metrics:
                print(f"[ERR ] {cfg_label} row={row} {scene_id}: {metrics['error']}")
                continue

            rows.append({
                "config":        cfg_label,
                "config_folder": cfg_key,
                "row":           row,
                "scene_id":      scene_id,
                **metrics,
            })
            print(f"[OK]   {cfg_label:14s} row={row:<3d} {scene_id}  "
                  f"top1={metrics['top1']:.4f}  top3={metrics['top3']:.4f}  "
                  f"macroF1={metrics['macro_f1']:.4f}  n={metrics['n_mineral_px']:,}")

    if not rows:
        print("No prediction files found.")
        return

    df = pd.DataFrame(rows).sort_values(["row", "config"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(df)} rows)")

    for metric in ("top1", "top3", "macro_f1", "macro_recall"):
        print(f"\n=== {metric} ===")
        pivot = df.pivot(index="config", columns="row", values=metric).round(4)
        # Mean is computed only over rows where EVERY config has a value, so the
        # mean comparison is apples-to-apples even if some config has extra runs.
        common_rows = list(pivot.dropna(axis=1).columns)
        pivot["mean"] = pivot[common_rows].mean(axis=1).round(4)

        # Pixel-weighted mean over the same common rows (uses Trans 4H's n_mineral_px
        # as the weight per row, since per-granule pixel counts are config-invariant).
        weight_lookup = (df[df["config"] == LABEL_MAP[CONFIGS[0]]]
                         .set_index("row")["n_mineral_px"]
                         .reindex(common_rows))
        weights = weight_lookup.values
        total_w = weights.sum() if weights.size else 0
        if total_w > 0:
            wmean = (pivot[common_rows].values * weights).sum(axis=1) / total_w
            pivot["pixel_weighted_mean"] = np.round(wmean, 4)

        order = [LABEL_MAP[c] for c in CONFIGS if LABEL_MAP[c] in pivot.index]
        pivot = pivot.reindex(order)
        print(pivot)
        print(f"  (mean over {len(common_rows)} common granules; "
              f"pixel-weighted mean uses {int(total_w):,} mineral pixels)")


if __name__ == "__main__":
    main()
