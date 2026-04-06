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
- Balanced index pool: `Data/balanced_g1_g2_indices_MINperID_7500.npy` (1,260,203 indices; 7,500 cap/ID, built by `kelli_scripts/Africa_pixel_GroupID.ipynb`)

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

| Config | Heads (PCA/Total) | Top-1 | Top-3 | Time (min) |
|--------|-------------------|-------|-------|------------|
| Transformer only | 4 | 0.791 | 0.962 | 117 |
| **Transformer only** | **8** | **0.813** | **0.968** | **136** |
| PCA | 4/4 | 0.804 | 0.966 | 141 |
| PCA | 4/8 | 0.806 | 0.966 | 127 |
| PCA | 8/8 | 0.805 | 0.966 | 150 |
| LUSI | 4 | 0.787 | 0.961 | 269 |
| LUSI | 8 | 0.796 | 0.963 | 293 |
| PCA+LUSI | 4/8 | 0.805 | 0.965 | 311 |

Best result: 8-head transformer-only (0.813) — no PCA, no LUSI, no wv mask needed.
PCA ceiling: ~0.805-0.806 regardless of head count or seeding strategy.
LUSI: hurts accuracy in every configuration and roughly doubles training time.

## Transformer-Only 8-Head Analysis (best result)
8-head transformer (no PCA, no LUSI): top1=0.813, top3=0.968, 136 min
- Outperforms all PCA configurations (0.805-0.806) by discovering features in native TOA domain
- Head 2: 477/515/544 nm Fe3+ charge transfer (same region as PCA, discovered independently)
- Head 1: 835 nm near crystal field diagnostic (~860 nm)
- Head 4: 955 nm near goethite diagnostic (~920 nm)
- Head 7: 679 nm near Fe3+ diagnostic (~670 nm) — NOT found by any PCA configuration
- head_sim=0.010 (lowest of any run — maximally diverse heads)
- PCA priors constrain heads away from features the model would otherwise discover (679, 835 nm)

## Transformer-Only 4-Head Analysis
Pure transformer (no PCA, no LUSI, no wv mask): top1=0.791, top3=0.962, 117 min
- Head 1: 552/567/589 nm Fe3+ charge transfer (spectroscopically valid, discovered without physics)
- Head 3: 850/1074/1260 nm Fe crystal field — **only head across all configs with significant Spearman r=0.323 (p<0.001)**
- Spearman works for diffuse attention (transformer-only); peak coincidence works for sharp attention (PCA)
- Model naturally discovers Fe3+ features regardless of initialization — PCA sharpens but doesn't create them

## No-Continuum PCA Finding
PCA on raw reflectance (no continuum removal) breaks through the continuum-removed PCA ceiling:
- 4/4H no-continuum: 0.810 (vs 0.804 with continuum)
- **8/8H no-continuum: 0.821** — best overall result, beats 8H transformer (0.813)
- Eliminates representation mismatch (eigenvectors in reflectance space, not absorption depth space)
- Rescues Head 0: no-continuum PC1 maps to VNIR Fe3+ instead of atmospheric proxy
- Continuum removal was counterproductive: it discards albedo information useful for TOA classification

## Cross-Scene Generalization (SW US desert, trained on Africa/Middle East)
| Model | Train region | SW US (new) | Change |
|-------|-------------|-------------|--------|
| LUSI 8H | 0.901 | 0.927 | +2.6% |
| Transformer 8H | 0.917 | 0.900 | -1.7% |
| PCA 8H nocont | 0.917 | 0.859 | -5.8% |

LUSI is the only model that improves on the new scene — consistent with Vapnik's theory that statistical invariants help when test conditions differ from training. PCA degrades most due to region-specific priors.

## Output Folders Convention
Training outputs are in `Data/attn_outputs_{config}/`:
- `Data/attn_outputs_Trans4_diff/` — transformer-only 4H (0.791)
- `Data/attn_outputs_Trans8_diff/` — transformer-only 8H (0.813)
- `Data/attn_outputs_PCA4_diffwt1_cont/` — PCA 4/4H continuum (0.804)
- `Data/attn_outputs_PCA4_diffwt1_nocont/` — PCA 4/4H no-continuum (0.810)
- `Data/attn_outputs_PCA4_8_diffwt1_cont/` — PCA 4/8H continuum (0.806)
- `Data/attn_outputs_PCA4_8_diffwt1_nocont/` — PCA 4/8H no-continuum (0.805)
- `Data/attn_outputs_Lusi4_diffwt1/` — LUSI 4H (0.787)
- `Data/attn_outputs_Lusi8_diffwt1/` — LUSI 8H (0.796)
- `Data/attn_outputs_PCALUSI4_8_diffcont_wts1/` — PCA+LUSI 4/8H (0.805)
- `Data/attn_outputs_PCALUSI8_diffcont_wts1/` — r14 PCA+LUSI 8-head, derivatives, continuum, wv mask, lusi_weight=1

## Dissertation Structure
Two-part structure with shared intro/background (Chapters 1-3) and closing (Chapters 10-11):
- **Part I (Chapters 4-6):** Statistical learning methods — PCA, NMF, autoencoder, PLoM, knockoffs, conditional expectation, random forest, KNN. Predicts both mineral group banddepth and mineral ID. Key contribution: full-image prediction in milliseconds. Includes PACE extensibility demonstration (Ch 6).
- **Part II (Chapters 7-9):** Physics-informed transformer — cross-attention backbone, PCA priors, LUSI, spectral derivatives. Predicts mineral ID only (95 Group 1 classes). Key contribution: spectroscopically interpretable attention validated against literature.
- Unifying paradigm: RTM-bypass (collect small SDS-processed set → learn W→Q mapping → apply without RTM)
- Both Parts are **post-hoc** methods: trained on L1B/L2B pairs already processed through EMIT SDS (ISOFIT + Tetracorder). No RTM at inference. Training quality bounded by SDS labels.
- LaTeX source: `Disseratation_txt/dissertation.tex` with `references.bib`
- USC formatting: 1" margins, double-spaced, Roman numeral prelim pages, Arabic body pages

## Dissertation Chapters Status (as of 2026-04-05)

Now 12 chapters (previously noted as 11). Chapter numbering shifted: ablation is Ch 10, generalization is Ch 11, conclusions is Ch 12.

- **Ch 1:** EMIT pipeline bottlenecks (ISOFIT OE deficiencies, AOD masking, H2O biases, shadowing, soil fraction threshold)
- **Ch 2:** Expanded background — data lifecycle, RTM emulation taxonomy (forward vs retrieval, in-line vs post-hoc), OE cost function, spatial paradigms (pixel/superpixel/full-image), Malsky transformer RT emulator, positioning table
- **Ch 3:** Instrument specs table, Dyson design, vicarious calibration, geometric/radiometric performance table, spectral confusion/unmixing table
- **Ch 4:** PCA section complete; NMF, autoencoder, comparison sections still TODO
- **Ch 5:** PLoM section complete (diffusion maps, intrinsic dimension=85, 100k synthetic from 99 originals). Knockoff feature selection math complete (exchangeability, FDR control). KNN section TODO.
- **Ch 6:** Training data section (sec:training_data) now fully detailed — 505,430 G1 pixels (93 classes) + 754,773 G2 pixels (177 classes) = 1,260,203 total, 7,500 cap per ID, 90/10 split. PACE extensibility present. Random forest, pixel-level results, full-image results still TODO.
- **Ch 7:** Substantially expanded — comprehensive spectral transformer literature review with DETR/Perceiver/SpecTf architectural lineage; Hughes phenomenon; gap analysis identifying this as first spectral transformer on TOA reflectance without atmospheric correction. Full architecture description: preprocessing, tokenization (token/embedding definitions, content + positional embeddings), attention pooling (Query/Key/Value definitions, cross-attention, einsum implementation), classification head (hidden layer bottleneck 1536→192→94, dropout, prediction equation $\hat{q}_1^{(i)}$ linking to QoI framework), parameter summary table (307K for H=4, 457K for H=8), optimization (cross-entropy with label smoothing, backpropagation, AdamW with decoupled weight decay, cosine LR schedule).
- **Ch 8:** Complete — PCA-derived attention bias, reference spectra sanitization, continuum removal vs raw reflectance finding (raw wins: 0.821 vs 0.805), water vapor masking, hybrid head allocation, freeze schedule.
- **Ch 9:** Complete — LUSI theoretical foundation (Vapnik-Izmailov predicates, invariants, RKHS), physical predicates (scale ρ∈U(0.8,1.2), slope ξ~N(0,0.01)), domain-correct augmentation (denormalize→transform→renormalize), KL divergence consistency loss.
- **Ch 10:** Complete — full ablation table (all configs), per-head attention validation, Spearman + peak coincidence metrics, 4 key findings about physics priors.
- **Ch 11:** Partial — cross-region generalization results (LUSI +2.6% on new scene). Part I vs Part II comparison, full-image inference timing, PACE transformer extensibility still TODO.
- **Ch 12:** Complete — summary of contributions, key findings, limitations (4 items), future work (5 items).

### Remaining TODOs

- Ch 4: NMF, autoencoder, comparison methods
- Ch 5: KNN for missing data
- Ch 6: Random forest details, pixel-level results, full-image results
- Ch 11: Part I vs Part II generalization comparison, full-image inference, PACE transformer extensibility

## Documentation
Spectra_interrogationr2.ipynb contains three mathematical write-ups in markdown cells:
1. **General Transformer Architecture** — Sections 1-8: preprocessing, tokenization, cross-attention backbone, classification head, optimization, monitoring, outputs
2. **Physics-Informed Priors** — PCA on USGS ref spectra, continuum removal, hybrid head allocation, freeze schedule, water vapor masking, prior evolution tracking
3. **LUSI Consistency Regularization** — Vapnik connection, scale/slope predicates, KL divergence loss, empirical findings

Notation conventions: $\mathbf{W}_p \in \R^{N \times L}$ for pixel-level reflectance matrix, $\mathbf{w}^{(i)} \in \R^L$ for single pixel $i$, $w^{(i)}_j$ for pixel $i$ band $j$, $\hat{w}$ for normalized, $Q_1=95$ for mineral classes, $\hat{q}_1^{(i)}$ for predicted Group 1 mineral ID of pixel $i$, $\tau$ for softmax temperature (not $T$, which is the physical transformation), $\mathbf{f}_1$/$\mathbf{f}_2$ for feedforward layer outputs (not $\mathbf{h}$, which collides with head index $h$). Key symbol choices to avoid collisions: $\delta_j$ for learnable per-band shift (not $\beta_j$, which collides with attention bias $\boldsymbol{\beta}_{h,j}$), $\rho$ for LUSI scale factor and $\xi$ for LUSI tilt factor (not $\alpha$/$\beta$, which collide with PCA strength $\alpha$ and attention bias), $\omega_j$ for knockoff test statistic (not $W_j$, which collides with feature matrix), $g$ for Part I PCA band grouping index (not $j$, reserved for band index in Part II)

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
