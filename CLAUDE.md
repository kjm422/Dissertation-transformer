# CLAUDE.md

## Project
Physics-informed transformer for hyperspectral mineral classification from EMIT L1B TOA reflectance, bypassing RTM atmospheric correction. USC Viterbi dissertation (Kelli McCoy).

## Architecture
- Cross-attention transformer with learned query vectors (not self-attention) — O(L) not O(L^2)
- Three modular enhancements: PCA-based attention bias (USGS ref spectra), LUSI consistency regularization, spectral derivatives input
- 285 EMIT bands, 95 mineral classes (Group 1), per-pixel classification
- Training script lineage: spectral_trans_withqoi_attentionr{N}_pcalusi.py (current: r14)

## Key Data Paths
- Training data: `/Users/kmccoy/Documents/USC/Research/Dissertation/Data/TOApixel_balanced_W_gb1ID_gb2ID_train7500.npy`
- EMIT NetCDF: `/Volumes/big24Tb/USC/Research/Dissertation/Data/Africa_MiddleEast/` (L1B/, L2B/, TOArefl_pca_output/band/)
- USGS ref spectra (93, Group 1): `Spectra/group1_all/*.asc` and `ALL_group1_spectra_full.npy`
- Original 23 iron-bearing ref spectra: `Spectra/from_minmatrix_all/ALL_spectra_concat_23x285.npy`
- Water vapor mask: `Spectra/group1_all/water_vapor_mask_285.npy`
- EMIT wavelengths: `Data/emit_wavelength_centers_nm.npy`
- Mineral grouping matrix: `Data/mineral_grouping_matrix_20230503.csv` (293 entries, Group 1 = 95, Group 2 = 199)

## Important Conventions
- USGS spectra are surface reflectance (no atmosphere); EMIT L1B is TOA reflectance (atmosphere present)
- Water vapor bands (~1340-1500 nm, ~1800-1960 nm) and edge (>2450 nm) must be masked before PCA on ref spectra
- Labels in training data: column -2 is Group 1 ID (1..95), shifted to 0..94; class 0 is dropped
- Robust z-score normalization: (x - median) / (IQR + 1e-6), computed on train split only
- With full ~1M training data, 23-spectrum and 93-spectrum ref sets perform comparably (top1=0.804). The 93-spectrum set underperforms only on small subsets (~1000 samples).

## Attention Head Analysis (r14, 4-head, PCA+derivatives+continuum)
Per-head attention reveals distinct specialization:
- **Head 0**: Atmospheric water vapor proxy (1506 nm) — not spectroscopically valid, identical for hematite/goethite
- **Head 1**: SWIR mineral context (2219, 2330 nm) — clay/carbonate features, detects mineral assemblage associations
- **Head 2**: Fe3+ charge transfer (507, 567, 582 nm) — spectroscopically correct, matches diagnostic bands at ~530 and ~670 nm
- **Head 3**: Fe crystal field NIR (873, 1074 nm) — matches hematite ~860 nm diagnostic feature

Key findings:
- 2/4 heads use spectroscopically valid mineral features (Heads 2, 3)
- 2/4 heads use empirical shortcuts — atmospheric proxies or co-occurring mineral associations (Heads 0, 1)
- Only Heads 2 and 3 differentiate hematite from goethite; Heads 0 and 1 show identical patterns for both
- Prior evolution plots show model learns to suppress H2O bands (negative drift at ~1.4/1.9 um), wasting capacity that the wv_mask provides for free
- r14 vectorized attention: 201 min -> 141 min (30% faster), same accuracy

## LUSI-only Attention Head Analysis (r14, 4-head, no PCA priors)
LUSI-only run: top1=0.787, top3=0.961 (1.7% below PCA-only)

Per-head attention (no physics initialization — purely data-driven):
- **Head 0**: Single-band NIR (1119 nm) — not H₂O like PCA run, but still non-discriminative (identical for hematite/goethite)
- **Head 1**: Fe3+ VNIR (492, 522, 544 nm) — spectroscopically valid, found without PCA guidance
- **Head 2**: NIR (768, 1193, 1208 nm) — goethite shows 768 nm that hematite doesn't (spectroscopically meaningful shift)
- **Head 3**: Diffuse NIR (753, 1081, 1246 nm) — broadly distributed, less focused than PCA Head 3

Key comparison (PCA vs LUSI):
- LUSI-only model CAN discover Fe3+ features (Head 1) without physics priors
- PCA priors produce more focused, spectroscopically complete feature sets (860 nm crystal field in PCA Head 3 vs diffuse NIR in LUSI)
- Both configurations waste ~1 head on non-discriminative features
- **Conclusion: physics priors don't change what the model CAN learn, but they change what it DOES learn**
- 4 heads is insufficient — 1 head wasted in every configuration. 8-head experiment (4 PCA + 4 free) is next priority.

## Ablation Results Summary

| Config | Top-1 | Top-3 | Time (min) |
|--------|-------|-------|------------|
| PCA + derivatives (r14) | 0.804 | 0.966 | 141 |
| LUSI + derivatives (r14) | 0.787 | 0.961 | ~141 |
| PCA + LUSI + derivatives | TBD | TBD | TBD |
| 8-head PCA + derivatives | TBD | TBD | TBD |

## Output Folders Convention
Training outputs are moved to `Data/attn_outputs_{config}/` e.g.:
- `Data/attn_outputs_PCA4_diffwt1_cont/` — r14 PCA 4-head, derivatives, continuum removal
- `Data/attn_outputs_Lusi4_diffwt1/` — r14 LUSI-only 4-head, derivatives

## Git
- Repo: https://github.com/kjm422/Dissertation-transformer
- Only add files by name — never `git add .` (large notebooks and data files will blow up the push)
- .npy, .pt, .nc, .asc files are gitignored except for specific exceptions in Spectra/group1_all/

## Running Training
Typical invocation (r14):
```
python kelli_scripts/spectral_trans_withqoi_attentionr14_pcalusi.py \
  --data "Data/TOApixel_balanced_W_gb1ID_gb2ID_train7500.npy" \
  --physics_init --physics_mode pca \
  --ref_spectra "Spectra/from_minmatrix_all/ALL_spectra_concat_23x285.npy" \
  --wavelengths "Data/emit_wavelength_centers_nm.npy" \
  --wv_mask "Spectra/group1_all/water_vapor_mask_285.npy" \
  --ref_pca_k 4 --ref_continuum --ref_use_absdepth \
  --physics_alpha 1 --physics_freeze_prior_epochs 3 \
  --epochs 120 --batch 1024 --lr_max 5e-4 \
  --weight-decay 3e-3 --dropout 0.1 --label-smoothing 0.05 \
  --d_model 192 --heads 4 --attn_tau 0.9 \
  --use_cosine --dump_attn --use_derivatives
```

## Device
macOS with MPS (Apple Silicon). CUDA paths exist for HPC. num_workers=0 for MPS compatibility.
