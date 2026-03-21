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
- 4 heads is insufficient — 1 head wasted in every configuration
- 8-head + wv mask is the recommended configuration: most spectroscopically interpretable with top accuracy

## 8-Head PCA + Water Vapor Mask Analysis
Best spectroscopic validation of any configuration:
- **Head 2** (PCA): top-5 peaks at 544, 537, 530, 470, 873 nm — matches hematite diagnostic at ~530/860 nm and goethite at ~480 nm
- **Head 3** (PCA): 873 nm Fe crystal field + 1514/1208 nm NIR
- **Heads 0, 1** (PCA): SWIR features (2004, 1588 nm) — mineral overtones, not iron-specific
- **Heads 4–7** (random): various SWIR/NIR data-driven patterns
- Head-averaged attention peaks at 925 and 470 nm — genuine iron oxide features, not atmospheric artifacts
- Prior evolution: without mask 6/8 heads suppress H₂O; with mask 3/8 heads drift at Fe³⁺ wavelengths
- WV mask reduces PCA head drift (priors start closer to optimal): PC1 drift 0.81→0.57, PC2 1.01→0.65
- 8-head einsum: 127 min (faster than 4-head 141 min due to better GPU utilization)

## Ablation Results Summary

| Config | Heads | Ref spectra | WV mask | Top-1 | Top-3 | Time (min) |
|--------|-------|-------------|---------|-------|-------|------------|
| Transformer only + derivatives | 4 | — | No | 0.791 | 0.962 | — |
| LUSI + derivatives | 4 | — | No | 0.787 | 0.961 | ~141 |
| PCA + derivatives | 4 | 93 | Yes | 0.804 | 0.966 | 141 |
| PCA + derivatives | 8 | 23 | No | 0.812 | 0.968 | — |
| PCA + derivatives | 8 | 93 | Yes | 0.806 | 0.966 | 127 |
| PCA + LUSI + derivatives | 8 | 93 | Yes | 0.805 | 0.965 | ~127 |

## Transformer-Only Baseline Analysis
Pure transformer (no PCA, no LUSI, no wv mask): top1=0.791, top3=0.962
- Head 0: 1119 nm single-band NIR (wasted, identical for hematite/goethite)
- Head 1: 552/567/589 nm Fe3+ charge transfer (spectroscopically valid, discovered without physics)
- Head 2: 738/820/992/1216 nm broad NIR (wider coverage than any PCA head)
- Head 3: 850/1074/1260 nm Fe crystal field — **only head across all configs with significant Spearman r=0.323 (p<0.001)**
- Spearman works for diffuse attention (transformer-only); peak coincidence works for sharp attention (PCA)
- Model naturally discovers Fe3+ features regardless of initialization — PCA sharpens but doesn't create them

## Output Folders Convention
Training outputs are moved to `Data/attn_outputs_{config}/` e.g.:
- `Data/attn_outputs_PCA4_diffwt1_cont/` — r14 PCA 4-head, derivatives, continuum removal
- `Data/attn_outputs_Lusi4_diffwt1/` — r14 LUSI-only 4-head, derivatives
- `Data/attn_outputs_PCA8_diffwt1_cont/` — r14 PCA 8-head, derivatives, continuum, wv mask
- `Data/attn_outputs_Trans4_diff/` — r14 transformer-only 4-head baseline, derivatives
- `Data/attn_outputs_PCALUSI8_diffcont_wts1/` — r14 PCA+LUSI 8-head, derivatives, continuum, wv mask, lusi_weight=1

## Documentation
Spectra_interrogationr2.ipynb contains three mathematical write-ups in markdown cells:
1. **General Transformer Architecture** — Sections 1-8: preprocessing, tokenization, cross-attention backbone, classification head, optimization, monitoring, outputs
2. **Physics-Informed Priors** — PCA on USGS ref spectra, continuum removal, hybrid head allocation, freeze schedule, water vapor masking, prior evolution tracking
3. **LUSI Consistency Regularization** — Vapnik connection, scale/slope predicates, KL divergence loss, empirical findings

Notation conventions: $\mathbf{W}_p$ for pixel reflectance, $\hat{w}$ for normalized, $Q_1=95$ for mineral classes, $\tau$ for softmax temperature (not $T$, which is the physical transformation)

## LUSI Findings
LUSI does not improve accuracy in any configuration tested:
- 4-head LUSI-only (0.787) underperforms transformer baseline (0.791)
- 8-head PCA+LUSI (0.805) matches PCA-only (0.806) within noise
- LUSI loss is small (~0.03) at convergence — model already achieves brightness/slope invariance through training
- Conclusion: explicit invariance regularization is redundant at ~1M pixel data scale

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
