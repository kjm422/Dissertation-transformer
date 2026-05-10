# CLAUDE.md

## Project
Physics-informed transformer for hyperspectral mineral classification from EMIT L1B TOA reflectance, bypassing RTM atmospheric correction. USC Viterbi dissertation (Kelli McCoy).

## Architecture
- Cross-attention transformer with learned query vectors (not self-attention) — O(L) not O(L^2)
- Three modular enhancements: PCA-based attention bias (USGS ref spectra), LUSI consistency regularization, spectral derivatives input
- 285 EMIT bands, 95 mineral classes (Group 1), per-pixel classification
- Training script lineage: spectral_trans_withqoi_attentionr{N}_pcalusi.py (current: r17)
- r15 vs r14: non-PCA hybrid heads now init to zero instead of 0.01·N(0,I) noise — symmetry breaking comes from random query vectors q_h ~ N(0,1), so explicit bias noise is unnecessary. This makes non-PCA-head behavior consistent between physics_init=True (hybrid) and physics_init=False (all zeros).
- r16 vs r15: adds `--physics_mode manual` for a hand-curated weak Gaussian-bump prior at user-specified diagnostic wavelengths (defaults: 480/535/670/860/920 nm). Calibrated defaults: `--manual_prior_normalize max`, `--physics_alpha 0.5`, `--physics_freeze_prior_epochs 1`. Wavelengths cycle across heads via `round_robin` (or `shared` / `one_per_head`). Includes wv-mask sanity warning if a center falls in a masked band.
- r17 vs r16: adds `--physics_mode precomputed` with a forgiving loader (`load_precomputed_bias`). Loads a `(k, 285)` attention-bias `.npy` from `--prior_bias_path` and auto-pads (k < n_heads) or truncates (k > n_heads) to match the model's head count. Padded zero rows stay trainable from epoch 1 (only the rows with non-zero loadings count toward `physics_frozen_heads`). Intended companion to `build_group1_ferric_pca_prior.py`.

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
- Water vapor bands (~1340-1500 nm, ~1800-1960 nm) and edge (>2450 nm) must be masked before PCA on ref spectra. The mask isn't "no mineral information at all" — hydrous minerals (clays, sulfates, carbonates, gypsum) genuinely have OH/H₂O combination/overtone features there in surface reflectance (kaolinite 1.4 μm doublet, smectite OH stretch, gypsum 1.45/1.95 μm). But at TOA the atmospheric H₂O absorption reaches ~95–99% extinction in those windows, so mineral signal is overwhelmed by spatially/temporally variable atmospheric water vapor; ISOFIT/ATCOR skip these bands for the same reason. For Group 1 specifically, the diagnostic Fe-oxide features are at 480–920 nm (well below the H₂O windows) and the hydrous-mineral features in the masked region belong to Group 2 classes not predicted here. Empirically, prior-evolution analysis shows that without the mask 6 of 8 heads spend learning capacity actively suppressing H₂O bands — the model treats them as noise to filter out anyway, so the mask gives that suppression for free. For Group 2 / hydrated-sulfate studies, revisit this trade-off.
- Labels in training data: column -2 is Group 1 ID (1..95), shifted to 0..94; class 0 is dropped
- Robust z-score normalization: (x - median) / (IQR + 1e-6), computed on train split only
- With full ~1M training data, 23-spectrum and 93-spectrum ref sets perform comparably (top1=0.804). The 93-spectrum set underperforms only on small subsets (~1000 samples).

## Group 1 Class Distribution (verified from raw 396 granules)
- 92 of 95 Group 1 IDs are populated. 3 are absent: ID 1 (green plastic tarp; artificial-material reference), ID 72 (staurolite; Fe2+ metamorphic silicate), ID 74 (augite; Fe2+ pyroxene). All three are unexpected in arid dust-source-region surfaces, so absence is consistent with regional mineralogy.
- Top 9 most prevalent Group 1 classes are all iron-bearing minerals (consistent with EMIT mission focus):
  1. ID 8: Goethite thin film (162.6M pixels)
  2. ID 0: unclassified (87.2M; not a mineral, dropped before training)
  3. ID 46: Nano-hematite + fine-goethite mix (84.8M)
  4. ID 45: Nano-hematite (69.6M)
  5. ID 47: Nano-hematite variant (49.7M)
  6. ID 4: Goethite + Quartz (37.2M)
  7. ID 40: Weathered basalt (25.3M)
  8. ID 20: Cummingtonite — Fe/Mg amphibole (13.1M)
  9. ID 28: Fe-Hydroxide amorphous (12.6M)
- After 7,500 per-class cap: 20 of 92 populated classes fall short of the target — IDs (count): 83(1), 94(15), 63(71), 58(91), 34(105), 81(108), 35(116), 26(135), 12(195), 71(218), 93(289), 91(299), 88(365), 84(580), 85(992), 54(1000), 39(1024), 87(1205), 78(1219), 90(1328). The other 72 reach the cap.
- Three rarest Group 1 classes are geologically/spectroscopically explainable: rhodochrosite (ID 63, MnCO3) and rhodonite (ID 94, MnSiO3) are localized Mn ore minerals genuinely uncommon on arid surfaces; desert varnish (ID 83, GDS78A on rhyolite) is physically common but rarely matches at 60m pixel scale because (i) sub-mm coatings get spectrally averaged with substrate, (ii) the reference is hyper-specific (varnish-on-rhyolite-from-single-sample). Tetracorder match counts ≠ surface mineralogical abundance.
- Group 2 raw: 581.5M pixels (8.16% ID 0). Sampled: 1.26M pixels (10.71% ID 0). Most common G2 class is ID 168 with 77M pixels.

## Figures inventory (Dissertation_images/)
- `architecture_diagram.pdf` — full-pipeline TikZ diagram (paper + dissertation), built from `PaperofPartII/test_diagram.tex`
- `G1_orig_samphist.png` — Group 1 raw vs. stratified-sample distribution (paper + dissertation)
- `G2_orig_samphist.png` — Group 2 raw vs. stratified-sample distribution, ID 0 excluded (dissertation only)
- `G1_hist.png`, `G2_hist.png` — single-panel raw histograms (no longer used in current docs; kept for reference)
- `Africa-collection.png` — map of 396 granules over Africa/Middle East, used in EMIT Instrument section (wrapfig)
- `Data_level_depiction.png` — RTM-bypass vs. SDS pipeline schematic, used in RTM-bypass paradigm subsection

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

## Dissertation Chapters Status (as of 2026-05-01)

Now 11 chapters. The former Ch 8 (Physics-Informed Spectral Priors) and Ch 9 (LUSI Consistency Regularization) were merged into a single new Ch 8 (Optional Physics-Informed Extensions) with §8.1 = Physics priors, §8.2 = LUSI. Ablation/generalization/conclusions chapters renumbered down by one. Existing labels preserved: `chap:phys_extensions` (new parent), `chap:physics` (now §8.1, label retained), `chap:lusi` (now §8.2, label retained). All `Chapter~\ref{chap:physics}` and `Chapter~\ref{chap:lusi}` inline references converted to `Section~\ref{...}`. Paper mirrors this structure under `\section{Optional physics-informed extensions}` (label `sec:phys_extensions`).

- **Ch 1:** EMIT pipeline bottlenecks (ISOFIT OE deficiencies, AOD masking, H2O biases, shadowing, soil fraction threshold). Limitations paragraph generalized to "operational SDS pipelines" (with EMIT as concrete example); now mentions the L2A→L2B mineral-identification step (per-pixel spectral-library matching against tens to hundreds of reference spectra) as a second per-pixel cost contributor on top of atmospheric correction. RTM-bypass section retitled "A Next-Generation Remote Sensing Pipeline." New "post-hoc direct retrieval" terminology adopted alongside "RTM bypass" — both terms now used: "post-hoc direct retrieval" is the recognized architectural category (citing Keely 2025, Li 2022 as direct-retrieval analogues), "RTM bypass" is the paper-internal shorthand emphasizing the no-RTM-at-training distinction. High-AOD recovery argument added to contributions and conclusions: post-hoc direct retrieval potentially recovers pixels EMIT's L2 pipeline discards at AOD > 0.5, since the model never invokes the failing ISOFIT inversion. Validation paths flagged in conclusions: AOD-binned accuracy curves, cross-sensor validation (PRISMA, EnMAP, CHIME), in-situ at high-AOD targets (Bodélé Depression, Lake Eyre).
- **Ch 2:** Expanded background — data lifecycle, RTM emulation taxonomy (forward vs retrieval, in-line vs post-hoc), OE cost function, spatial paradigms (pixel/superpixel/full-image), Malsky transformer RT emulator, positioning table. New §2.5 "Approaches to Physics-Informed Retrieval" (paper §2.3 mirror) added: brief PINN/theory-guided ML context (Raissi 2019, Karpatne 2017), Wang et al. 2021 distinguished as PCA+attention analogue, LUSI/Vapnik invariance positioning. Wang/Vapnik intro paragraphs in §8.1/§8.2 (paper §6.1/§6.2) shortened with forward-references to avoid redundancy.
- **Ch 3:** Instrument specs table, Dyson design, vicarious calibration, geometric/radiometric performance table, spectral confusion/unmixing table
- **Ch 4:** PCA section complete; NMF, autoencoder, comparison sections still TODO
- **Ch 5:** PLoM section complete (diffusion maps, intrinsic dimension=85, 100k synthetic from 99 originals). Knockoff feature selection math complete (exchangeability, FDR control). KNN section TODO.
- **Ch 6:** Training data section (sec:training_data) now fully detailed — 505,430 G1 pixels across 92 of 95 populated classes + 754,773 G2 pixels (177 classes) = 1,260,203 total, 7,500 cap per ID, 90/10 split. New additions: raw G1 distribution paragraph naming the three missing IDs (1=plastic tarp, 72=staurolite, 74=augite); identification of top-9 G1 classes as all iron-bearing (goethite/hematite-dominated, consistent with EMIT mission focus); G2 distribution paragraph with ID 0 stats (8.16% raw, 10.71% sampled); shortfall table listing 20 G1 classes that fell short of 7,500 cap (1 to 1,328 pixels). Figures: G1_orig_samphist (raw vs sample), G2_orig_samphist (dissertation only). PACE extensibility present. Random forest, pixel-level results, full-image results still TODO.
- **Ch 7:** Substantially expanded — comprehensive spectral transformer literature review with Tsai \& Philpot / Perceiver / SpecTf architectural lineage (DETR removed in favor of Tsai \& Philpot 1998 as the source for spectral-derivative tokenization, since DETR's contribution was object queries on images, not anything used here); Hughes phenomenon; gap analysis identifying this as first spectral transformer on TOA reflectance without atmospheric correction. Architectural-foundations paragraphs tightened to 3 sentences each: Tsai \& Philpot (finite divided differences for shape-aware tokenization, Savitzky--Golay alternative noted), Perceiver (asymmetric cross-attention, $O(L^2) \to O(ML)$, $H$ learned query vectors), SpecTf (per-pixel cloud detection on EMIT, outperformed operational baseline with fewer parameters, attention emphasized physical absorption regions). Full architecture description: preprocessing, tokenization (token/embedding definitions, content + positional embeddings), attention pooling (Query/Key/Value definitions, cross-attention, einsum implementation), classification head (hidden layer bottleneck 1536→192→95, dropout, prediction equation $\hat{q}_{1,i}$ linking to QoI framework), parameter summary table (307,373 for H=4, 456,737 for H=8), optimization (cross-entropy with label smoothing rationale, label-smoothing predicate explanation citing Szegedy 2016, AdamW with both parameter groups enumerated, decoupled weight decay rationale, cosine LR schedule). Softmax-interpretation paragraph added: explicit framing that softmax values are *model* probabilities not real-world probabilities; top-1 / top-3 retrievals reported alongside; calibration disclaimer. Brief plain-language explanations of crystal-field (d-d) and charge-transfer (LMCT) transitions added so non-spectroscopist readers understand why Fe$^{3+}$ shows diagnostic features at the cited wavelengths. Wavelengths standardized across paper and dissertation: hematite ~535 nm, goethite ~480 nm derivative-spectrum minima, Fe$^{3+}$ charge-transfer ~530/670 nm, Fe crystal-field NIR ~860--920 nm (citing Jiang 2014 and Clark 1999 for the canonical iron-oxide diagnostics).
- **Ch 8:** Optional Physics-Informed Extensions — merged former Ch 8 + Ch 9. §8.1 Physics-Informed Spectral Priors (PCA-derived attention bias, reference spectra sanitization, continuum vs raw reflectance finding 0.821 vs 0.805, water vapor masking, hybrid head allocation, freeze schedule, prior evolution). §8.2 LUSI Consistency Regularization (formal Vapnik-Izmailov framework: predicates, statistical invariants, RKHS solution; followed by neural-approximation disclaimer; physical predicates: scale ρ∈U(0.8,1.2), tilt ξ~N(0,0.01); domain-correct augmentation pipeline; KL divergence consistency loss; training-loop integration). New chapter intro frames both as user-toggleable extensions to the base Ch 7 architecture.
  - **§8.1 PCA-priors prose now mirrors paper:** TOA decomposition with full 6S form (Eq. `eq:toa_full`) reducing to first-order approximation (Eq. `eq:toa_decomp`) under $S\rho_{\text{surface}} \ll 1$; cited Vermote 1997 as canonical 6S source; Wang et al. 2021 distinguished as a related-but-distinct PCA+attention construction; "what the prior is NOT" disclaimer paragraph; orthogonality caveat noting absolute-value profiles aren't orthogonal but are derived from orthogonal modes; cautious framing on visible Fe³⁺ feature preservation (530/670 nm sit in Rayleigh-affected region; 860 nm is favorable).
  - **§8.2 LUSI prose substantially revised** (and mirrored to paper): theoretical foundation now explicitly framed as a *transformation-based neural approximation* of formal LUSI rather than a literal RKHS implementation — closes the gap between Vapnik's hard-constraint setup and the soft consistency loss actually used. Teacher/student terminology introduced in the theory subsection (with `\cite{hinton2015distilling}`) before being used in the consistency loss. KL ordering corrected: was `D_KL(p_trans||p_orig)` (reverse KL, mode-seeking) — now `D_KL(p_orig^sg||p_trans)` (forward KL, mode-covering, matches r15 implementation). Stop-gradient annotation `^sg` added to teacher term. Tilt-predicate framing softened — drops the overclaim that aerosol+Rayleigh is "approximately linear over VSWIR" and the λ⁻⁴ derivation; now framed as a "conservative label-preserving perturbation, not a full atmospheric model." TOA-vs-surface disclaimer added in theory subsection: predicates are label-preserving directions in TOA-input space, not literal atmospheric corrections; explicitly does not assume `ρ_toa ≈ ρ_surface`. All "nuisance" jargon (5 distinct phrasings across both files) replaced with "label-preserving" or simplified prose.
- **Ch 9:** Ablation Study and Spectral Attention Analysis — full ablation table (all configs), per-head attention validation, Spearman + peak coincidence metrics, 4 key findings about physics priors. (formerly Ch 10)
- **Ch 10:** Generalization and Cross-Platform Analysis — partial. Cross-region generalization results (LUSI +2.6% on new scene). Part I vs Part II comparison, full-image inference timing, PACE transformer extensibility still TODO. (formerly Ch 11)
- **Ch 11:** Conclusions and Future Work — summary of contributions, key findings, limitations (4 items), future work (5 items). (formerly Ch 12)

### Remaining TODOs

- Ch 4: NMF, autoencoder, comparison methods
- Ch 5: KNN for missing data
- Ch 6: Random forest details, pixel-level results, full-image results
- Ch 10: Part I vs Part II generalization comparison, full-image inference, PACE transformer extensibility
- Paper §6.2.x (LUSI) and §5.x (Transformer-only results placeholder, `sec:trans_only_results`) — empirical content TBD; `sec:ablation` label in paper still unresolved

## Documentation
Spectra_interrogationr2.ipynb contains three mathematical write-ups in markdown cells:
1. **General Transformer Architecture** — Sections 1-8: preprocessing, tokenization, cross-attention backbone, classification head, optimization, monitoring, outputs
2. **Physics-Informed Priors** — PCA on USGS ref spectra, continuum removal, hybrid head allocation, freeze schedule, water vapor masking, prior evolution tracking
3. **LUSI Consistency Regularization** — Vapnik connection, scale/slope predicates, KL divergence loss, empirical findings

Notation conventions (updated 2026-04-14): $\mathbf{W}^p \in \R^{N \times L}$ for pixel-level reflectance matrix, $\mathbf{w}_i \in \R^L$ for single pixel $i$ (subscript, not superscript), $w_{ij}$ for pixel $i$ band $j$, $\hat{w}$ for normalized, $Q_1=95$ for mineral classes, $\hat{q}_{1,i}$ for predicted Group 1 mineral ID of pixel $i$. Radiance: $R_j^M$ (measured at-sensor, superscript M for "measured"). Attention: vectorized equations — $\mathbf{s}_h, \mathbf{A}_h, \bm{\beta}_h \in \R^L$ (vectors over bands, not element-wise $A_{h,j}$). Temperatures: $\tau_a$ for attention softmax (0.9), $\tau_l$ for LUSI KL divergence (2.0). Use $d$ instead of $d_{\text{model}}$ (defined once as $d = d_{\text{model}} = 192$ in notation table). Use $\bm{\beta}_h$ (bm package) not $\boldsymbol{\beta}_{h,j}$. Use $\oplus$ for vector concatenation ($\mathbf{z}_{\text{cat}}$), $[\cdot]^T$ for stacking into matrices ($\mathbf{E}$). No $\cdot$ dot for scalar-vector multiplication (juxtaposition only). Cross-entropy labels: $q_c$ / $\tilde{q}_c$ (not $y_c$). Key symbol choices to avoid collisions: $\delta_j$ for learnable per-band shift (not $\beta_j$), $\rho$ for LUSI scale factor and $\xi$ for LUSI tilt factor, $\omega_j$ for knockoff test statistic, $g$ for Part I PCA band grouping index

## Journal Paper
- `Disseratation_txt/PaperofPartII/partII_paper.tex` — journal paper, **target venue PNAS** (Research Report) as of 2026-05-04. Was previously formatted for Remote Sensing of Environment (elsarticle class); rewrite to PNAS LaTeX template and ~4,000-word length is in progress.
- Shares `references.bib` with dissertation (copied to both directories)
- Covers Part II content: architecture, manual + PCA spectral priors, LUSI, ablation, cross-scene generalization
- Continuum-removed-vs-raw subsection was in the paper but has been removed (kept in dissertation only) per 2026-05-03 prune
- Pruned 2026-05-04: §6 rhetorical flourish, §5 4-bullet "Key architectural features" list (replaced with one-paragraph integration), `tab:manual_ablation` (4-row table — superset is in dissertation `tab:ablation`), and TODO blocks for cross-scene-generalization and discussion sections (full content lives in dissertation Ch 10–11)

### PNAS submission target (2026-05-04)
- **Article type**: Research Report (~6 PNAS pages, ~4,000 words main text, ≤250-word abstract, **50–120-word Significance Statement required**, 3–5 keywords, ~4 figures/tables, ≤50 references)
- **Section structure** (PNAS preferred order): Introduction → Results → Discussion → Materials and Methods → Acknowledgments → References
- **Strategic framing**: position the work as the transformer instantiation of the **physics-informed deep learning** paradigm (Karpatne 2017 theory-guided ML + Raissi 2019 PINN). Two physics-injection mechanisms in parallel:
  - **Architectural prior**: spectroscopic priors injected through attention bias $\bm{\beta}_h$ (manual mode at literature wavelengths or PCA-curated mode on USGS library)
  - **Loss-based prior**: LUSI consistency regularization, statistical invariants of atmospheric transfer adapted from Vapnik-Izmailov
- **Polished thesis paragraph** (end-of-Introduction; cite `karpatne2017tgds`, `raissi2019pinn`, `vapnik2020`):

  > "We present a cross-attention transformer that is physics-informed in the sense of Karpatne et al. and Raissi et al.: spectroscopic priors are injected through the attention bias $\bm{\beta}_h$, and statistical invariants drawn from atmospheric transfer enter through a consistency loss adapted from Vapnik and Izmailov. The two mechanisms realize the two foundational ingredients of physics-informed deep learning — architectural prior initialization and physics-driven loss regularization — in an attention-based architecture, made natural and inspectable by the attention bias's role as an explicit physics-injection interface. We demonstrate this on EMIT top-of-atmosphere hyperspectral observations: a 4-head transformer with a manual spectral prior matches the accuracy of an 8-head transformer-only baseline at 14% lower training cost, and consistency regularization is the only configuration that maintains accuracy on a geographically disjoint test scene where unregularized configurations degrade."

- **Significance Statement draft** (118 words, broad-audience, the most distinctive PNAS element):

  > "Mapping Earth's surface mineralogy from orbiting hyperspectral sensors is critical for understanding mineral dust's role in climate, but the standard pipeline rejects the very high-aerosol pixels it was designed to study because the underlying radiative transfer inversion fails. We show that a cross-attention transformer trained directly on uncorrected top-of-atmosphere reflectance replaces this multi-step pipeline with a single forward pass, recovering canonical iron-oxide absorption features through interpretable attention. The approach realizes physics-informed deep learning for an attention-based architecture: spectroscopic priors enter through attention bias initialization, and statistical invariants drawn from atmospheric transfer enter through consistency regularization. The result is a transferable framework for injecting domain knowledge into transformer models in any physical-science setting where labeled data is bounded by an upstream physical model."

- **Results structure** (3 subsections, with priors and LUSI as parallel physics-injection mechanisms — *not* as method + side-experiment):
  1. Cross-attention transformer recovers Fe³⁺ diagnostics from TOA reflectance (uses Fig: per-head attention)
  2. Two complementary physics-injection mechanisms: (2a) attention-bias initialization substitutes for architectural capacity (Table: ablation); (2b) consistency regularization improves cross-region generalization where unregularized configurations degrade (Fig: cross-scene bar chart)
  3. Optional integrative result tying both mechanisms to the high-AOD-recovery argument

- **LUSI in/out-of-distribution narrative**: re-frame the existing observation (LUSI hurts in-distribution by ~0.4 pp but is the only configuration that *improves* on the SW US scene by +2.6 pp) as a *feature* aligned with PINN literature: physics-driven regularization can trade in-distribution training error for better extrapolation to physics-respecting test conditions. This is PNAS-friendly because it demonstrates the framework, not just the numbers.

- **Main-text figures/tables (max 4)**:
  1. Architecture pipeline (`architecture_diagram.pdf`)
  2. Per-head attention plot (`4Hxformer_attnplot.png`)
  3. Headline ablation table (Trans 4H/8H × Manual 4H/8H + curated PCA 4H)
  4. Cross-scene generalization bar chart (LUSI +2.6%, Trans −1.7%, PCA −5.8%) — needs to be created

- **Move to SI Appendix (single PDF)**:
  - Tables: `tab:emulation_spectrum`, `tab:fe_diagnostics`, `tab:notation`, `tab:arch_inspirations`, `tab:phys_interventions`
  - Figure: `fig:g1_hist`
  - Equations: TOA decomposition, full LUSI loss derivation, full attention/pooling equations
  - Detailed methods: training data sampling, sanitization pipeline, full ablation table with PCA-legacy + LUSI rows, prior-evolution diagnostics
  - Per-head walkthrough text (just one-line summary in main)

- **Required PNAS-specific elements to add**: ORCID iD for corresponding author (and ideally all authors), narrative author contributions statement, competing-interest declaration, funding acknowledgment, **Data and software availability statement** (mention GitHub repo `kjm422/Dissertation-transformer` and the EMIT data source LP DAAC).

- **Two overclaim warnings to soften** when finalizing the thesis paragraph: "available only in transformers, not in CNNs or MLPs" → "made natural and inspectable by the attention bias's role as an explicit physics-injection interface" (CNN attention modules exist; PIT for fluid dynamics exists). "For the first time in an attention-based architecture" → "applied to top-of-atmosphere hyperspectral mineral classification" (other physics-informed transformers exist in fluid dynamics and atmospheric chemistry).

- **Workflow recommendation**: fork as `partII_paper_PNAS.tex` rather than editing in place — keeps the RSE-length version available for the dissertation companion or RSE fallback.

### Bib entries added during recent revisions
- `vermote1997sixs` — Vermote et al. 1997, IEEE TGRS 35(3): 675–686. Canonical 6S radiative transfer reference replacing the earlier `schott2007`/`schaepman2006` citations for the TOA decomposition. Both schott2007 and schaepman2006 are no longer cited in either document.
- `szegedy2016rethinking` — Szegedy et al. 2016, CVPR. Cited for label smoothing in the cross-entropy / optimization subsection.
- `wang2021attention_pca_guided` — Wang et al. 2021, Remote Sensing 13(23): 4927. Cited in PCA-priors subsection to distinguish this work's reference-library-derived attention bias from prior PCA+attention hyperspectral approaches.
- `lamminpaa2025` — Lamminpää et al. 2025, *Atmos. Meas. Tech.* 18: 673–694, doi:10.5194/amt-18-673-2025. **Replaces** the earlier preprint `lamminpaa2024`. Forward-emulator example (GP/OCO-2) in Table 1.
- `gao2023` — Gao et al. 2023, *Atmos. Meas. Tech.* 16(23): 5863–5881. FastMAPOL forward-emulator example for PACE/HARP2 retrieval (cascading NN forward model). (Note: original key had `number={12}` and `pages={5863}`; corrected to `{23}` and `5863--5881`.)
- `Keely2025ConditionalDiffusionCO2` — Keely et al. 2025, ICLR Climate Change AI Workshop, arXiv:2504.17074. Direct-retrieval (probabilistic diffusion) example in Table 1; the canonical "retrieval emulator" citation alongside Li.
- `Li2022NO2NN` — Li et al. 2022, *Journal of Remote Sensing* 2022: 9817134, doi:10.34133/2022/9817134. Direct-retrieval (deterministic NN) example for NO₂ from UV-Vis radiances; complements Keely as the deterministic instantiation of the row.
- `raissi2019pinn` — Raissi, Perdikaris, Karniadakis 2019, *J. Comput. Phys.* 378: 686–707. Canonical PINN reference cited in the new "Approaches to physics-informed retrieval" section.
- `karpatne2017tgds` — Karpatne et al. 2017, *IEEE TKDE* 29(10): 2318–2331. Theory-guided data science reference cited alongside Raissi 2019 in the new physics-informed-retrieval section.
- `braverman2021` — Braverman et al. 2021, *SIAM/ASA J. Uncertainty Quantification* 9(3): 1064–1093. Originally cited in Table 1's Post-hoc UQ row; that row was later removed from the table; still cited in dissertation prose at lines 439, 483, but no longer in the paper.
- `tsai1998derivative` — Tsai \& Philpot 1998, *Remote Sens. Environ.* 66(1): 41–51. Canonical hyperspectral derivative-analysis reference; describes Savitzky--Golay convolutional derivatives and finite divided differences along the wavelength axis. Replaces DETR as the cited source for shape-aware spectral-derivative tokenization in both files.
- `Jiang2014AlGoethite` — Jiang, Liu, Dekkers, Barron, Torrent 2014. Cited for hematite (~535 nm) and goethite (~480 nm) derivative-spectrum minima; used to standardize wavelength references throughout paper and dissertation.
- All `detr` / `carion2020detr` references removed from both bib files; no remaining citations to DETR in either document.

## Architecture Diagram
- Source: `Disseratation_txt/PaperofPartII/test_diagram.tex` (TikZ standalone) → renders to `test_diagram.pdf` → copied to `Dissertation_images/architecture_diagram.pdf` for use in both paper and dissertation
- All main forward-pass nodes are filled `gray!20`; color is reserved for the four group regions and for optional/loss boxes
- Four dashed-bordered group regions wrap the forward pass and align 1:1 with paper subsections (mirrored in dissertation): (1) `sec:preprocessing` light yellow, (2) `sec:tokenization` pink (Vaswani embedding color), (3) `sec:attn_pooling` light orange (Vaswani attention color), (4) `sec:classification` light cyan (Vaswani feed-forward color). Group titles bold-set in top-left of each region
- Yellow boxes (`bias` for spectral priors $\bm{\beta}_h$, `lusi` for $T_{\text{joint}}$ and $D_{\text{KL}}$) sit *outside* the four group regions because they augment rather than replace the base pass
- Pink/purple boxes for losses ($\mathcal{L}_{\text{CE}}$, $\mathcal{L}_{\text{total}}$); dashed arrows = optional/conditional pathways; solid = required data flow
- Side annotations (MLP head detail, spectral-priors detail, $T_{\text{joint}}$ formula, KL student-pass note) document internal operations without breaking out separate sub-blocks
- Robust z-score equation $\hat{w}_j = (w_j - \text{median}(\mathbf{w}))/(\text{IQR}(\mathbf{w}) + \epsilon)$ is rendered inside the `norm` block (not just labelled)
- Figure caption in both files names each region and links it to the matching subsection label
- Tikz libraries used: `positioning, arrows.meta, shapes.geometric, calc, fit, backgrounds`; the `fit` library + `on background layer` is what draws the four group regions cleanly behind the nodes
- Recompile after edits: `cd Disseratation_txt/PaperofPartII && pdflatex test_diagram.tex && cp test_diagram.pdf ../Dissertation_images/architecture_diagram.pdf`

## LUSI Findings
LUSI does not improve accuracy in any configuration tested:

- 4-head LUSI-only (0.773 under r17; legacy 0.787 under r14/r15) underperforms transformer baseline (0.791)
- 8-head PCA+LUSI (0.805) matches PCA-only (0.806) within noise
- LUSI loss is small (~0.03) at convergence — model already achieves brightness/slope invariance through training
- Conclusion: explicit invariance regularization is redundant at ~1M pixel data scale

## Broken-PCA-Prior Finding (2026-05-03)
The legacy `physics_mode=pca` pipeline (r14, retained in r17 for backward-compatibility runs) silently produces nonsense priors when run on the 93-spectrum Group-1 file. Root cause is in `make_pca_priors_from_ref` (the carried-forward implementation in [spectral_trans_withqoi_attentionr17_pcalusi.py](kelli_scripts/spectral_trans_withqoi_attentionr17_pcalusi.py)):
```python
fill = _find_fill_mask(X, absmax=absmax)
bad_band_mask = fill.any(axis=0)        # flags any band where ANY spectrum has a fill
# ...later...
v[bad_band_mask] = 0.0                  # zeros prior at all flagged bands
```
With the 93-spectrum library: 248 of 285 bands flagged → only ~37 bands survive, all clustered in 1500–1700 nm. The `prior_evolution.npy[0]` snapshot is essentially zero at every Fe³⁺ diagnostic wavelength (480/535/670/860/920 nm). This explains why every PCA ablation cell in the existing table (PCA 4/4H 0.804, PCA 8/8H 0.805, PCA+LUSI 0.805) underperforms 8H transformer-only (0.813): the bias was actively pushing attention toward non-diagnostic wavelengths.

Two replacement paths added (both avoid the bad-band-mask pathology):

**Option A — manual weak prior** (r16):
Skip the library entirely. Use Gaussian bumps at user-specified diagnostic wavelengths.
```bash
python kelli_scripts/spectral_trans_withqoi_attentionr17_pcalusi.py \
  --physics_init --physics_mode manual --heads 4 \
  --physics_alpha 0.5 --physics_freeze_prior_epochs 1 \
  --wavelengths Data/emit_wavelength_centers_nm.npy \
  --wv_mask Spectra/group1_all/water_vapor_mask_285.npy \
  ...
```

**Option B — curated PCA precomputed offline** (r17 + new build script):
Use `build_group1_ferric_pca_prior.py` to (a) curate library by keyword (target/ferric_confuser/hard_negative), (b) row-zscore preprocess, (c) PCA on valid bands only, (d) save `(n_heads, 285)` bias file. Then load via `--physics_mode precomputed --prior_bias_path ...`.
```bash
# Step 1: build the bias offline
python kelli_scripts/build_group1_ferric_pca_prior.py \
  --npy_dir Spectra/group1_all \
  --wavelengths Data/emit_wavelength_centers_nm.npy \
  --wv_mask Spectra/group1_all/water_vapor_mask_285.npy \
  --out_dir Spectra/group1_ferric_pca_prior \
  --mode ferric_with_hard_negatives --preprocess row_zscore \
  --k 4 --n_heads 4

# Step 2: load it during training
python kelli_scripts/spectral_trans_withqoi_attentionr17_pcalusi.py \
  --physics_init --physics_mode precomputed \
  --prior_bias_path Spectra/group1_ferric_pca_prior/pca_attention_bias_max_H4.npy \
  --physics_alpha 0.5 --physics_freeze_prior_epochs 1 ...
```

The build script reads `.npy` filenames directly from `--npy_dir` (no mineral matrix); each file is classified by keyword on its filename. Modes filter which categories survive into PCA. Saves both `zabs` and `max` bias variants per head count.

### Spectral-prior empirical results (2026-05-04, updated 2026-05-09)

The architectural prior was evaluated under two independent constructions of $\beta_h$, both at $\alpha = 1.0$ with a 3-epoch freeze of the prior-bearing heads. The locked 9-row ablation table (papers + dissertation, `tab:ablation`):

| Config | #Heads | Top-1 | Top-3 | head_sim | prior_rms | Time (min) |
|---|---:|---:|---:|---:|---:|---:|
| Transformer baseline (β trainable, zero-init, no wv-mask) | 4 | 0.789 | 0.961 | 0.043 | 0.968 | 119 |
| **Manual prior (Fe³⁺ bands, PAPER HEADLINE)** | **4** | **0.808** | **0.967** | 0.031 | 0.977 | **117** |
| PCA-curated (zabs) prior | 4 | 0.809 | 0.967 | 0.028 | 0.946 | 124 |
| LUSI-only (no wv-mask, trainable β zero-init) | 4 | 0.787 | 0.961 | 0.030 | 1.308 | 289 |
| PCA-curated (zabs) + LUSI | 4 | 0.790 | 0.962 | 0.039 | 1.648 | ~290 |
| Transformer baseline | 8 | 0.819 | 0.970 | 0.024 | 0.946 | 134 |
| Manual prior (Fe³⁺ bands) | 8 | 0.806 | 0.966 | 0.015 | 1.154 | 125 |
| PCA-curated (zabs) prior | 8 | 0.818 | 0.970 | 0.020 | 1.066 | 129 |
| LUSI-only | 8 | 0.796 | 0.962 | 0.004 | 0.000 | 326 |

**Headline finding (capacity substitution, refined 2026-05-09)**: The manual spectral prior at $H = 4$ reaches top-1 0.808, delivering 1.9 of the 3.0 percentage-point gain that doubling the head count produces (Trans 4H 0.789 → Trans 8H 0.819) — about 63% of the head-doubling benefit at zero additional training cost. The PCA-curated zabs prior at $H = 4$ reaches 0.809 from a largely non-overlapping band set, supporting the **capacity substitution rather than wavelength selection** interpretation. At $H = 8$ the priors diverge: Manual underperforms baseline by $-1.3$ pp ($0.819 \to 0.806$) because narrow Gaussian bumps over-constrain attention at saturated capacity; PCA-curated zabs ties baseline ($0.818$) because its diffuse loadings span $\sim$220 of 285 bands and align with what the unguided model would discover anyway. **Sharp-vs-diffuse refinement**: capacity-saturation harm depends on the prior's spectral support concentration.

**Capacity-vs-information control (2026-05-06)**: A parameter-matched Trans 4H run with $\bm{\beta}_h$ trainable but zero-initialized (`--physics_init --physics_mode manual --physics_alpha 0.0 --physics_freeze_prior_epochs 0`) reaches top-1 0.790 — splitting the prior's $+1.8$ pp gain at $H = 4$ into a $+0.6$ pp capacity component (extra trainable parameters) and a $+1.8$ pp information component (spectroscopic init). Information dominates capacity by $\sim 3\times$; rules out "prior = additional model capacity in disguise." `prior_rms = 0.872` (from zero init) confirms the bias drifted as expected when trainable.

**Early-epoch acceleration finding (2026-05-04)**: a 4-config first-5-epochs comparison (Manual / PCA-zabs / Trans / LUSI all at $H = 4$) shows:
- Manual prior leads at every epoch: epoch-1 top-1 = 0.331 vs Trans 0.305 (immediate +2.6 pp from initialization alone); the gap holds to ~2–3 pp through epoch 5.
- PCA-zabs shows a smaller but consistent push: epoch-1 top-1 = 0.313 (+0.8 pp over Trans), training-loss trajectory tracks Manual more closely than Trans.
- LUSI without architectural prior is *slower* than Trans early in training, consistent with its in-distribution underperformance — the architectural-vs-loss-based asymmetry reinforces the two-mechanism framing for the paper.

**Diagnostic signals**:
- `prior_rms` differences in the two paper configurations: manual held priors closest to initial scale (0.977, just −2.3%), zabs preserved the prior nearly as well (0.946, −5.4%). The zabs result is consistent with the negative-floor of z-score normalization providing additional inductive signal that the model doesn't want to wash out.
- At $H = 8$ the manual prior amplified to 1.154 (+15.4%), indicating the extra capacity went into prior amplification rather than novel feature discovery; `head_sim` stayed low (0.015) ruling out head collapse.

**Implication for the paper**: the architectural-vs-loss-based asymmetry during early training is the cleanest evidence yet of the two-mechanism framework. Architectural priors provide an *immediate* inductive head start (Manual at H=4 is +2.6 pp ahead by epoch 1); loss-based priors *delay* in-distribution convergence in exchange for cross-region robustness (LUSI is slower in-distribution but is the only configuration that improves on a disjoint test scene). Both fit the PINN-tradition pattern but at different points in the training trajectory. The 4-config head-averaged convergence figure (`train_top1_top3.png`, dissertation `fig:earlyepoch`) plus the 9-config full-trajectory figure (same filename, refreshed) tell the story across both head counts.

### Per-mineral Spearman analysis (2026-05-09)

Per-class attention vs per-mineral USGS absorption depth (`1 - ref_spec[ref_row]`), correcting the earlier mean-across-93 version. At $H = 4$, architectural priors achieve statistically significant per-mineral alignment:

- **PCA-manual H1 hematite: ρ = 0.389 (p < 10⁻⁸)** — top peaks at 530/544/559/873 nm matching canonical 535 (visible CT) and 860 (NIR crystal-field)
- **PCA-manual H0 goethite: ρ = 0.394 (p < 10⁻⁷)** — top peaks at 470/485/902 nm matching canonical 480 (CT) and 920 (CF)
- Trans/LUSI head-avg ρ ≈ 0.06–0.15, all non-significant per-mineral

**At $H = 8$ the ranking flips** (Spearman analog of the capacity-saturation finding):

| Config | Hematite avg ρ (4H → 8H) | Goethite avg ρ (4H → 8H) |
|---|---:|---:|
| Trans | 0.059 → **0.320** | 0.145 → 0.249 |
| PCA-manual | **0.238** → 0.038 | **0.202** → 0.074 |
| PCA-zabs | 0.216 → 0.246 | 0.170 → 0.138 |
| LUSI | 0.070 → 0.171 | 0.112 → **0.315** |

PCA-manual collapses at 8H to non-significance; per-head, the heads pinned at 480 and 535 nm end up *anti-correlated* with hematite absorption (ρ = $-0.203$, $-0.224$) — the trainable bias drifted to non-canonical wavelengths during the post-freeze phase ($\|\bm{\beta}\|_{\text{rms}} = 1.154$). PCA-zabs holds at 8H because broader spectral support anchors the bias structurally. Trans 8H rediscovers the same Fe³⁺ structure the manual prior scaffolded at 4H. LUSI 8H surprises on goethite (ρ = 0.315), strongest goethite alignment of any 8H config.

### Rarity analysis (2026-05-09)

Per-class precision/recall/F1 stratified by training-set support across three rarity bins (0–150, 150–1000, 1000–4000 pixels per class) shows the architectural prior's accuracy gain is concentrated in the rare-class regime:

- **150–1000 sample bin**: Manual 4H F1 = 0.62, statistically tied with Trans 8H F1 = 0.64, well above LUSI 4H F1 = 0.49 (a +13 pp F1 gap between architectural and loss-based priors in the rare regime)
- **1000–4000 sample bin**: all 8 configurations cluster at F1 = 0.82–0.86 with negligible separation

This localizes the headline "Manual 4H matches Trans 8H" claim to *rare-class equivalence*: the prior buys head-doubling-equivalent rare-class generalization at no extra training cost; on common classes both saturate to the same F1. Augmentation (LUSI) cannot substitute for examples in the rare regime — the consistency loss provides a regularization signal but no mineral-specific information. Figure: `comparison_rarity_grid.png` (dissertation `fig:rarity_grid`, original paper).

### Asymmetric wv_mask role (2026-05-08, locked)

The water-vapor mask was originally applied across all configurations. Empirical testing revealed an asymmetric role:

- **Prior-bearing runs (Manual, PCA)**: keep the mask. Acts as a bias-drift regularizer; without it, the trainable bias amplifies aggressively into H₂O bands and absorbs atmospheric noise as a free degree of freedom. Removing the mask from PCA-zabs 4H caused $\|\bm{\beta}\|_{\text{rms}}$ to amplify 54% and top-1 to drop 1.6 pp.
- **Transformer-only baseline**: drop the mask. With β frozen at 0, the mask is a no-op and just adds an unnecessary preprocessing flag. Empirically within noise (0.784 with mask vs 0.790 without).
- **LUSI-only**: drop the mask. With the mask, the consistency loss can't learn H₂O-band invariance (which is exactly where atmospheric variance is largest). Removing the mask from LUSI 4H gained +1.5 pp top-1 (0.773 → 0.788).

The mask is therefore a *regularizer specifically for the trainable attention bias* in the absence of other regularization (LUSI's consistency loss). When the bias is frozen (Trans) or another regularization is active (LUSI), the mask is at best redundant and at worst harmful.

### Curated PCA prior outputs (2026-05-04)

`build_group1_ferric_pca_prior.py` run on `Spectra/group1_all` (99 .npy files):
- Selected: 60 spectra (26 hard_negative, 22 target, 12 ferric_confuser); 56 survived sanitization
- PCA bands: 224 of 285 (vs broken-pipeline's 37 of 285)
- Variance explained: PC1 43.6%, PC2 26.1%, PC3 13.2%, PC4 6.1% (cumulative 89.0%)
- PC interpretations from top-loaded wavelengths:
  - PC1 (43.6%): 1967–2200 nm SWIR clay/carbonate axis → "Fe-oxide vs not-Fe" discrimination, driven by hard-negative chlorites/pyroxenes
  - PC2 (26.1%): 507–575 nm cluster, peak at 537 nm → matches the canonical hematite charge-transfer edge
  - PC3 (13.2%): 2315–2390 nm carbonate refinement
  - PC4 (6.1%): 381–447 nm UV/blue → goethite charge-transfer edge region
- Caveat: 670 nm crystal-field band is *not* in PC1–4 (likely PC5/PC6); k=4 may miss this canonical Fe³⁺ feature

Outputs saved to `Spectra/group1_ferric_pca_prior/`:
- `pca_attention_bias_max_H{N}.npy` — peaks=1, off-bands=0; pair with `--physics_alpha 0.5–1.0`
- `pca_attention_bias_zabs_H{N}.npy` — z-score(|v|), legacy normalization with negative floor; pair with `--physics_alpha 0.2`
- Plus `pca_components_signed.npy`, `pca_explained_variance.csv`, `pca_top_wavelengths.csv`, `pca_band_missingness_report.csv`, `selected_group1_ferric_metadata.csv`, and a 4-panel diagnostic PNG

The `zabs` normalization is the direct mathematical successor of the legacy in-line PCA pipeline (same formula); the `max` normalization is a new alternative that avoids the negative-floor on off-diagnostic bands.

### Paper / dissertation updates (2026-05-04)

- Section "Transformer-only results" added to both files with `4Hxformer_attnplot.png` figure and per-head spectroscopic interpretation. Calibration: framed as "supporting evidence rather than proof" (attention weights are not causal explanations).
- Section "Ablation: manual spectral prior vs transformer-only baseline" added to paper with `\label{sec:ablation}` (resolves all forward-references). Dissertation `tab:ablation` updated to add the two new manual-prior rows and explicitly tag the four legacy-PCA rows with a "(legacy)" qualifier; new `\label{sec:pca_legacy_caveat}` paragraph documents the bad-band-mask issue.
- `tab:fe_diagnostics` simplified in both files to charge-transfer/crystal-field region descriptions with one-line entries (was multi-clause technical entries with quantum term symbols and LMCT/d-d notation).
- Manual + PCA framed as two **modes of a single spectral-prior extension** (not three independent extensions). Section structure: `\subsection{Physics-informed spectral priors}` → `\subsubsection{Manual diagnostic-band priors}` + `\subsubsection{PCA-derived attention bias}`.
- Continuum-removed-vs-raw section removed from paper (still in dissertation).
- All paper tables standardized to Table 1 format: `\small`, `p{...\textwidth}` columns, `\rowcolor[gray]{0.85}` header, `\hline` between data rows.
- Hardware sentence in §5.4 names the actual chip: "single Apple M4 Max laptop (40-core GPU) using PyTorch's MPS backend."

## Git
- Repo: https://github.com/kjm422/Dissertation-transformer
- Only add files by name — never `git add .` (large notebooks and data files will blow up the push)
- .npy, .pt, .nc, .asc files are gitignored except for specific exceptions in Spectra/group1_all/

## Running Training
Use r17 by default. Three `--physics_mode` options:

**Manual weak prior (recommended starting point):**
```
python kelli_scripts/spectral_trans_withqoi_attentionr17_pcalusi.py \
  --data "Data/TOApixel_balanced_W_gb1ID_gb2ID_train7500.npy" \
  --physics_init --physics_mode manual \
  --wavelengths "Data/emit_wavelength_centers_nm.npy" \
  --wv_mask "Spectra/group1_all/water_vapor_mask_285.npy" \
  --epochs 120 --batch 1024 --lr_max 5e-4 \
  --weight-decay 3e-3 --dropout 0.1 --label-smoothing 0.05 \
  --d_model 192 --heads 4 --attn_tau 0.9 \
  --use_cosine --dump_attn --use_derivatives \
  --attn_out Data/attn_outputs_Manual4_diff
```
Defaults: `--manual_prior_bands "480,535,670,860,920" --manual_prior_normalize max --physics_alpha 0.5 --physics_freeze_prior_epochs 1`. Override `--physics_alpha 1.0` or `--physics_freeze_prior_epochs 3` for a stronger / longer-pinned prior.

**Curated PCA prior (loaded from disk):**
```
# build first
python kelli_scripts/build_group1_ferric_pca_prior.py \
  --npy_dir Spectra/group1_all \
  --wavelengths Data/emit_wavelength_centers_nm.npy \
  --wv_mask Spectra/group1_all/water_vapor_mask_285.npy \
  --out_dir Spectra/group1_ferric_pca_prior \
  --mode ferric_with_hard_negatives --preprocess row_zscore \
  --k 4 --n_heads 4
# then train
python kelli_scripts/spectral_trans_withqoi_attentionr17_pcalusi.py \
  --physics_init --physics_mode precomputed \
  --prior_bias_path Spectra/group1_ferric_pca_prior/pca_attention_bias_max_H4.npy \
  --physics_alpha 0.5 --physics_freeze_prior_epochs 1 \
  ... (other args same as above) ...
```

**Legacy inline PCA (kept for backward-comparable runs only — broken bad-band-mask):**
```
python kelli_scripts/spectral_trans_withqoi_attentionr17_pcalusi.py \
  --physics_init --physics_mode pca \
  --ref_spectra "Spectra/group1_all/ALL_group1_spectra_full.npy" \
  --ref_pca_k 4 --ref_continuum --ref_use_absdepth \
  --physics_alpha 1 --physics_freeze_prior_epochs 3 \
  ...
```

## Device
macOS with MPS (Apple Silicon). CUDA paths exist for HPC. num_workers=0 for MPS compatibility.
