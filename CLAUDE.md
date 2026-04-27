# CLAUDE.md

## Project
Physics-informed transformer for hyperspectral mineral classification from EMIT L1B TOA reflectance, bypassing RTM atmospheric correction. USC Viterbi dissertation (Kelli McCoy).

## Architecture
- Cross-attention transformer with learned query vectors (not self-attention) — O(L) not O(L^2)
- Three modular enhancements: PCA-based attention bias (USGS ref spectra), LUSI consistency regularization, spectral derivatives input
- 285 EMIT bands, 95 mineral classes (Group 1), per-pixel classification
- Training script lineage: spectral_trans_withqoi_attentionr{N}_pcalusi.py (current: r15)
- r15 vs r14: non-PCA hybrid heads now init to zero instead of 0.01·N(0,I) noise — symmetry breaking comes from random query vectors q_h ~ N(0,1), so explicit bias noise is unnecessary. This makes non-PCA-head behavior consistent between physics_init=True (hybrid) and physics_init=False (all zeros). Re-run ablation pending.

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

## Dissertation Chapters Status (as of 2026-04-28)

Now 11 chapters. The former Ch 8 (Physics-Informed Spectral Priors) and Ch 9 (LUSI Consistency Regularization) were merged into a single new Ch 8 (Optional Physics-Informed Extensions) with §8.1 = Physics priors, §8.2 = LUSI. Ablation/generalization/conclusions chapters renumbered down by one. Existing labels preserved: `chap:phys_extensions` (new parent), `chap:physics` (now §8.1, label retained), `chap:lusi` (now §8.2, label retained). All `Chapter~\ref{chap:physics}` and `Chapter~\ref{chap:lusi}` inline references converted to `Section~\ref{...}`. Paper mirrors this structure under `\section{Optional physics-informed extensions}` (label `sec:phys_extensions`).

- **Ch 1:** EMIT pipeline bottlenecks (ISOFIT OE deficiencies, AOD masking, H2O biases, shadowing, soil fraction threshold)
- **Ch 2:** Expanded background — data lifecycle, RTM emulation taxonomy (forward vs retrieval, in-line vs post-hoc), OE cost function, spatial paradigms (pixel/superpixel/full-image), Malsky transformer RT emulator, positioning table
- **Ch 3:** Instrument specs table, Dyson design, vicarious calibration, geometric/radiometric performance table, spectral confusion/unmixing table
- **Ch 4:** PCA section complete; NMF, autoencoder, comparison sections still TODO
- **Ch 5:** PLoM section complete (diffusion maps, intrinsic dimension=85, 100k synthetic from 99 originals). Knockoff feature selection math complete (exchangeability, FDR control). KNN section TODO.
- **Ch 6:** Training data section (sec:training_data) now fully detailed — 505,430 G1 pixels across 92 of 95 populated classes + 754,773 G2 pixels (177 classes) = 1,260,203 total, 7,500 cap per ID, 90/10 split. New additions: raw G1 distribution paragraph naming the three missing IDs (1=plastic tarp, 72=staurolite, 74=augite); identification of top-9 G1 classes as all iron-bearing (goethite/hematite-dominated, consistent with EMIT mission focus); G2 distribution paragraph with ID 0 stats (8.16% raw, 10.71% sampled); shortfall table listing 20 G1 classes that fell short of 7,500 cap (1 to 1,328 pixels). Figures: G1_orig_samphist (raw vs sample), G2_orig_samphist (dissertation only). PACE extensibility present. Random forest, pixel-level results, full-image results still TODO.
- **Ch 7:** Substantially expanded — comprehensive spectral transformer literature review with DETR/Perceiver/SpecTf architectural lineage; Hughes phenomenon; gap analysis identifying this as first spectral transformer on TOA reflectance without atmospheric correction. Full architecture description: preprocessing, tokenization (token/embedding definitions, content + positional embeddings), attention pooling (Query/Key/Value definitions, cross-attention, einsum implementation), classification head (hidden layer bottleneck 1536→192→95, dropout, prediction equation $\hat{q}_{1,i}$ linking to QoI framework), parameter summary table (307,373 for H=4, 456,737 for H=8), optimization (cross-entropy with label smoothing rationale, label-smoothing predicate explanation citing Szegedy 2016, AdamW with both parameter groups enumerated, decoupled weight decay rationale, cosine LR schedule).
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
- `Disseratation_txt/PaperofPartII/partII_paper.tex` — journal paper for Remote Sensing of Environment (elsarticle class)
- Shares `references.bib` with dissertation (copied to both directories)
- Covers Part II content: architecture, PCA priors, LUSI, ablation, cross-scene generalization
- Key additions vs dissertation: TOA decomposition equation (Eq. toa_decomp) connecting surface/TOA reflectance to PCA and LUSI motivation; forward model / retrieval framing from Thompson; sRTMnet/MODTRAN lineage (ISOFIT originally called MODTRAN, sRTMnet trained on MODTRAN outputs replaces those calls in the operational pipeline)
- Group 1 only — Group 2 distribution + per-class shortfall table are dissertation-exclusive; paper has compact prose summary
- Parameter summary table (`tab:params`) is dissertation-exclusive (space-saving in paper); paper retains prose stating 307,373 (H=4) / 456,737 (H=8) totals and lists dependencies ($d=192$, $L=285$, $Q_1=95$, $c_{in}=3$)
- Hybrid head init equation (`eq:hybrid_init`) is dissertation-exclusive (paper removed for space); paper retains prose description of PCA-vs-non-PCA head allocation
- Continuum-removed-vs-raw subsection sits in blue at the end of the paper (post-bibliography candidate, kept tentative pending space)

### Bib entries added during recent revisions
- `vermote1997sixs` — Vermote et al. 1997, IEEE TGRS 35(3): 675–686. Canonical 6S radiative transfer reference replacing the earlier `schott2007`/`schaepman2006` citations for the TOA decomposition. Both schott2007 and schaepman2006 are no longer cited in either document.
- `szegedy2016rethinking` — Szegedy et al. 2016, CVPR. Cited for label smoothing in the cross-entropy / optimization subsection.
- `wang2021attention_pca_guided` — Wang et al. 2021, Remote Sensing 13(23): 4927. Cited in PCA-priors subsection to distinguish this work's reference-library-derived attention bias from prior PCA+attention hyperspectral approaches.

## Architecture Diagram
- Source: `Disseratation_txt/PaperofPartII/test_diagram.tex` (TikZ standalone) → renders to `test_diagram.pdf` → copied to `Dissertation_images/architecture_diagram.pdf` for use in both paper and dissertation
- Vaswani-style color palette: pink=embedding, grey=linear projection, orange=attention, light green=softmax, cyan=feed-forward, yellow=optional physics-informed components (spectral priors + LUSI), purple=loss
- Dashed arrows = optional/conditional pathways; solid = required data flow
- Annotations to the side of dense blocks (MLP head, spectral priors, T_joint) document internal operations without breaking out separate sub-blocks
- Recompile after edits: `cd Disseratation_txt/PaperofPartII && pdflatex test_diagram.tex && cp test_diagram.pdf ../Dissertation_images/architecture_diagram.pdf`

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
Typical invocation (r15 — same flags as r14, only the script name changes):
```
python kelli_scripts/spectral_trans_withqoi_attentionr15_pcalusi.py \
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
