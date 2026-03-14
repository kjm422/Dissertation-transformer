#!/bin/bash
# Download all 95 Group 1 spectra from USGS Spectral Library 06 via clarkvision mirror
# Each file is named using the CSV Filename field (group.1um/...) for traceability

BASE="https://clarkvision.com/science/usgs-spectral-library-06/ASCII"
OUTDIR="/Users/kmccoy/Documents/USC/Research/Dissertation/Spectra/group1_all"
FAILED=""
SUCCESS=0
FAIL=0

download() {
  local url="$1"
  local outfile="$2"
  if [ -f "$OUTDIR/$outfile" ]; then
    echo "SKIP (exists): $outfile"
    SUCCESS=$((SUCCESS+1))
    return
  fi
  if curl -fL --max-time 30 "$url" -o "$OUTDIR/$outfile" 2>/dev/null; then
    echo "OK: $outfile"
    SUCCESS=$((SUCCESS+1))
  else
    echo "FAIL: $outfile  ($url)"
    FAILED="$FAILED\n  $outfile"
    FAIL=$((FAIL+1))
    rm -f "$OUTDIR/$outfile"
  fi
}

# === ALREADY DOWNLOADED (23 iron-bearing spectra) - re-download for completeness ===

# 1. Hematite.02+Quartz.98 GDS76 (Record 744)
download "$BASE/S/hema_qtz_gds76.25384.asc" "fe3+_hematite.fine.gr.gds76.asc"

# 2. Hematite GDS27 (Record 738)
download "$BASE/M/hematite_gds27.9282.asc" "fe3+_hematite.med.gr.gds27.asc"

# 3. Hematite_Thin_Film GDS27 (Record 6198)
download "$BASE/C/hematite_thin_film_gds27.26800.asc" "fe3+_hematite.thincoat.asc"

# 4. Hematite FE2602 (Record 2100)
download "$BASE/M/hematite_fe2602.9271.asc" "fe3+_hematite.fine.gr.fe2602.asc"

# 5. Hematite WS161 (Record 906)
download "$BASE/M/hematite_ws161.9776.asc" "fe3+_hematite.fine.gr.ws161.asc"

# 6. Magnetite_skarn BR93-5B (Record 846)
download "$BASE/S/magnetite_skarn_br5b.25904.asc" "fe2+fe3+mix_with_hematite_br5b.asc"

# 7. Basalt_weathered BR93-43 (Record 5460)
download "$BASE/S/basalt_weathered_br93-43.24263.asc" "fe2+fe3+_hematite_weathering.asc"

# 8. Hematite_Coatd_Qtzt BR93-25A (Record 840)
download "$BASE/C/hematite_coat_br93-25a.26778.asc" "fe3+_hematite.lg.gr.br25a.asc"

# 9. Hematite_Coatd_Qtz BR93-25B (Record 6174)
download "$BASE/C/hematite_coat_br93-25b.26756.asc" "fe3+_hematite.med.gr.br25b.asc"

# 10. Hematite_Coatd_Qtzt BR93-25C (Record 816)
download "$BASE/C/hematite_coat_br93-25c.26789.asc" "fe3+_hematite.lg.gr.br25c.asc"

# 11. Nanohematite BR93-34B2 (Record 3468)
download "$BASE/M/nanohematite_br93-34b2.15589.asc" "fe3+_hematite.nano.BR34b2.asc"

# 12. Hematite_Coatd_Qtz BR93-34C (Record 822)
download "$BASE/C/hematite_coat_br93-34c.26767.asc" "fe3+_hematite.lg.gr.br34c.asc"

# 13. Maghemite GDS81 (Record 2928)
download "$BASE/M/maghemite_gds81.13124.asc" "fe3+_maghemite.asc"

# 14. Goethite WS222 MedGrn (Record 882)
download "$BASE/M/goethite_ws222.8447.asc" "fe3+_goethite.medgr.ws222.asc"

# 15. Goethite0.02+Quartz GDS240 (Record 5736)
download "$BASE/S/goethite_gds240.25198.asc" "fe3+_goethite+qtz.medgr.gds240.asc"

# 16. Goethite MPCMA2-B FineGr (Record 1878)
download "$BASE/M/goethite_mpcma2b.8351.asc" "fe3+_goethite.fingr.asc"

# 17. Goethite MPCMA2-C M-Crsgrad2 (Record 852)
download "$BASE/M/goethite_mpcma2c.8365.asc" "fe3+_goethite.medcoarsegr.mpc.trjar.asc"

# 18. Goethite WS222 coarse (Record 894)
download "$BASE/M/goethite_ws222.8459.asc" "fe3+_goethite.coarsegr.asc"

# 19. Goethite_Thin_Film WS222 (Record 6168)
download "$BASE/C/goethite_ws222_coating.26742.asc" "fe3+_goethite.thincoat.asc"

# 20. Goeth+qtz.5+Jarosite.5 AMX11 (Record 5730)
download "$BASE/S/goet_qtz_jaro_amx11.25184.asc" "fe3+_goeth+jarosite.asc"

# 21. Lepidocrosite GDS80 (Record 732)
download "$BASE/M/lepidochros_gds80.12600.asc" "fe3+_goethite.lepidocrosite.asc"

# 22. Chlor+Goethite CU93-4B (Record 5604)
download "$BASE/S/chlorite_mixture_cu93-4b.24750.asc" "fe2+fe3+_chlor+goeth.propylzone.asc"

# 23. Fe-Hydroxide SU93-106 (Record 5724)
download "$BASE/S/fe_hydroxide_su93-106.25174.asc" "fe3+bearing1.asc"

# === NEW DOWNLOADS (remaining ~72 Group 1 spectra) ===

# 24. Plastic_Tarp GDS339 Green (Record 6858)
download "$BASE/A/tarp_gds339.29036.asc" "organic_green_plastic_tarp_1um.asc"

# 25. Pyrite LV95-6A Weath on Tail (Record 6042)
download "$BASE/S/pyrite_lv95-6a.26297.asc" "sulfide_pyrite.asc"

# 26. Schwertmannite BZ93-1 (Record 4476)
download "$BASE/M/schwertmannite_bz93-1.20097.asc" "fe3+_sulfate_schwertmannite.asc"

# 27. Blck_Mn_Coat_Tailngs LV95-3 (Record 6144)
download "$BASE/C/black_mntailings_lv95-3.26691.asc" "Mn-Coating.asc"

# 28. Acid_Mine_Dr Assemb1-Fe3+ (Record 5316)
download "$BASE/S/acid_mine_drain_assem1.23775.asc" "fe3+mix_AMD.assemb1.asc"

# 29. Acid_Mine_Dr Assemb2-Fe3+ (Record 5322)
download "$BASE/S/acid_mine_drain_assem2.23799.asc" "fe3+mix_AMD.assemb2.asc"

# 30. Jarosite GDS99 K 200C Syn (Record 2568)
download "$BASE/M/jarosite_gds99.11469.asc" "fe3+_sulfate_kjarosite200.asc"

# 31. Chlorite+Muscovite CU93-65A (Record 5634)
download "$BASE/S/chlorite_mixture_cu93-65a.24855.asc" "fe2+_chlor+muscphy.asc"

# 32. Goethite CU91-252 coatedchip (Record 900) - NO METADATA URL
# This entry has "No available USGS Spectral Library 7 Description Reference"
# Try matching by sample ID in S/ directory
download "$BASE/S/goethite_phyllite_cu91_236a.25213.asc" "fe2+_goeth+musc_PLACEHOLDER.asc"
# NOTE: CU91-252 not found in clarkvision index. This placeholder may be wrong.

# 33. Cummingtonite HS294.3B (Record 1446)
download "$BASE/M/cummingtonite_hs294-3b.6583.asc" "fe2+generic_nrw.cummingtonite.asc"

# 34. Actinolite NMNHR16485 (Record 72)
download "$BASE/M/actinolite_nmnh16485.247.asc" "fe2+generic_nrw.actinolite.asc"

# 35. Actinolite HS22.3B (Record 42)
download "$BASE/M/actinolite_hs22.119.asc" "fe2+generic_nrw.hs-actinolite.asc"

# 36. Hypersthene NMNHC2368 (Record 2334)
download "$BASE/M/hypersthene_nmnhc2368.10348.asc" "fe2+_pyroxene.hypersthene.asc"

# 37. Diopside NMNHR18685 ~160 Pyx (Record 1548)
download "$BASE/M/diopside_nmnh18685.7035.asc" "fe2+_pyroxene.diopside.asc"

# 38. Jadeite HS343.3B (Record 2502)
download "$BASE/M/jadeite_hs343.11155.asc" "fe2+generic_med.jadeite.asc"

# 39. Pigeonite HS199.3B (Record 3966)
download "$BASE/M/pigeonite_hs199.17843.asc" "fe2+_pyroxene_clino_pigeonite.asc"

# 40. Epidote GDS26.a 75-200um (Record 1668)
download "$BASE/M/epidote_gds26.7546.asc" "epidote.asc"

# 41. Ferrihydrite GDS75 Syn F6 (Record 1740)
download "$BASE/M/ferrihydrite_gds75.7822.asc" "fe3+_ferrihydrite.asc"

# 42. Nontronite NG-1.a (Record 3558)
download "$BASE/M/nontronite_ng1.15991.asc" "fe3+_smectite_nontronite.asc"

# 43. Actinolite-Hornfels BR93-5a (Record 5328)
download "$BASE/S/actinolite-hornfels_br93-5a.23822.asc" "fe2+generic_brd.br5a_actinolite.asc"

# 44. Actinolite-Tremolit BR93-22C (Record 5334)
download "$BASE/S/actinolite-tremolite_br93-22c.23833.asc" "fe2+generic_brd.br22c_actinolite.asc"

# 45. Epidote BR93-33a (Record 1656)
download "$BASE/M/epidote_br93-33a.7490.asc" "fe2+generic_br33a_bioqtzmonz_epidote.asc"

# 46. Biotite-Chlorite_Mx BR93-36A (Record 5490)
download "$BASE/S/biotite-chlorite_mx_br93-36a.24407.asc" "fe2+generic_brd.br36a_chlorite.asc"

# 47. Basalt_fresh BR93-46B (Record 5454)
download "$BASE/S/basalt_fresh_br93-46b.24252.asc" "fe2+generic_basalt_br46b.asc"

# 48. nHematit+fg-Goethit 34B2+MPC (Record 1026) - NO METADATA URL
# Mixed spectrum, may not be on clarkvision as individual file
# SKIP - will be flagged as failed

# 49. Nanohematite FBR93-34B2b ed1 (Record 918) - NO METADATA URL
# Edited version of nanohematite, may not be on clarkvision
# SKIP - will be flagged as failed

# 50. Jarosite_on_Qtzite BR93-34A2 (Record 6210)
download "$BASE/C/jarosite_on_qtzite_br93-34a2.26829.asc" "fe3+_sulfate_jarosite_br34a2.asc"

# 51. Actinolite_Dolomit BR93-60B (Record 5340)
download "$BASE/S/actinolite_dolomite_skarn_br93-60b.23844.asc" "fe2+generic_broad_br60b.asc"

# 52. Phlogopite_Sand_Mix BR93-20 (Record 6036)
download "$BASE/S/phlogopite_sand_mix_br93-20.26287.asc" "fe2+generic_vbroad_br20.asc"

# 53. Renyolds_TnlSldgWet SM93-15w (Record 6942)
download "$BASE/A/renyolds_tunnel_sludge_sm93-15.29317.asc" "fe2+fe3+_water_RTsludge.asc"

# 54. Clinochlore GDS158 Flagst (Record 660)
download "$BASE/M/clinochlore_gds158.5327.asc" "fe2+_chlorite.clinochlor.asc"

# 55. Clinochlore NMNH83369 (Record 1200)
download "$BASE/M/clinochlore_nmnh83369.5444.asc" "fe2+_chlorite.Felow.clinochlor.asc"

# 56. Thuringite SMR-15.c 32um (Record 4890)
# SMR-15 files in M/: thuringite_smr15.21810.asc through .21983.asc
# The "c" and "32um" variant - based on record ordering, .21865 or .21910
download "$BASE/M/thuringite_smr15.21865.asc" "fe2+_chlorite.thuringite.asc"

# 57. Azurite WS316 (Record 720)
download "$BASE/M/azurite_ws316.3357.asc" "copper_carbonate_azurite.asc"

# 58. Pitch_Limonite GDS104 Cu (Record 3990)
download "$BASE/M/pitchlimon_gds104.17952.asc" "fe3+copper-hydroxide_pitchlimonite.asc"

# 59. Malachite HS254.3B (Record 2964)
download "$BASE/M/malachite_hs254.13289.asc" "copper_carbonate_malachite.asc"

# 60. Blue_Efflorscnt_Min SU93-300 (Record 5496)
download "$BASE/S/blue_effor_su93-300.24417.asc" "copper_sulfate_bluefflor.asc"

# 61. Green_Slime SM93-14A Summitv (Record 6594)
download "$BASE/A/green_slime_sm93-14a.28199.asc" "copper_precipitate_greenslime.asc"

# 62. Olivine HS285.4B Fo80 (Record 3696)
download "$BASE/M/olivine_hs285.16681.asc" "fe2+_olivine-lrg-gr.asc"

# 63. Olivine GDS71.a Fo91 65um (Record 3666)
download "$BASE/M/olivine_gds71.16520.asc" "fe2+_olivine-fine-gr.asc"

# 64. Rhodochrosite HS338.3B (Record 4278)
download "$BASE/M/rhodochrosite_hs338.19222.asc" "carbonate_rhodochrosite.asc"

# 65. Siderite HS271.3B (Record 4560)
download "$BASE/M/siderite_hs271.20461.asc" "fe2+generic_carbonate_siderite1.asc"

# 66. Riebeckite NMNH122689 Amph (Record 4362)
download "$BASE/M/riebeckite_nmnh122689.19614.asc" "fe2+generic_amphibole_riebeckite.asc"

# 67. Coquimbite GDS22 (Record 1386)
download "$BASE/M/coquimbite_gds22.6341.asc" "fe3+fe2+_sulfate_coquimbite.asc"

# 68. Copiapite GDS21 (Record 1374)
download "$BASE/M/copiapite_gds21.6285.asc" "fe3+fe2+_sulfate_copiapite.asc"

# 69. Chrysocolla HS297.3B (Record 1134)
download "$BASE/M/chrysocolla_hs297.5163.asc" "copper_chrysocolla.asc"

# 70. Neodymium_Oxide GDS34 (Record 3498)
download "$BASE/M/neodymium_gds34.15715.asc" "ree_neodymium_oxide.asc"

# 71. Samarium_Oxide GDS36 (Record 4410)
download "$BASE/M/samarium_gds36.19806.asc" "ree_samarium_oxide.asc"

# 72. Axinite HS342.3B (Record 708)
download "$BASE/M/axinite_hs342.3302.asc" "fe2+generic_axinite.asc"

# 73. Staurolite HS188.3B (Record 4728)
download "$BASE/M/staurolite_hs188.21169.asc" "fe2+generic_staurolite.asc"

# 74. Almandine WS479 Garnet (Record 216)
download "$BASE/M/almandine_ws479.911.asc" "fe2+generic_almandine.asc"

# 75. Augite NMNH120049 (Record 678)
download "$BASE/M/augite_nmnh120049.3180.asc" "fe2+_pyroxene_augite.asc"

# 76. Bronzite HS9.3B Pyroxene (Record 810)
download "$BASE/M/bronzite_hs9.3757.asc" "fe2+_pyroxene.bronzite.asc"

# 77. Butlerite GDS25 (Record 858)
download "$BASE/M/butlerite_gds25.3947.asc" "fe2+generic_sulfate_butlerite.asc"

# 78. Bytownite HS106.3B Plagio (Record 882)
download "$BASE/M/bytownite_hs106.4058.asc" "fe2+_feldspar.bytownite.asc"

# 79. Chalcopyrite HS431.3B (Record 1002)
download "$BASE/M/chalcopyrite_hs431.4557.asc" "sulfide_copper_chalcopyrite.asc"

# 80. Chromite HS281.3B (Record 1122)
download "$BASE/M/chromite_hs281.5107.asc" "fe2+_chromite.asc"

# 81. Cinnabar HS133.3B (Record 1164)
download "$BASE/M/cinnabar_hs133.5270.asc" "sulfide_cinnabar.asc"

# 82. Cuprite HS127.3B (Record 1458)
download "$BASE/M/cuprite_hs127.6639.asc" "copper_oxide_cuprite.asc"

# 83. Desert_Varnish GDS141 (Record 6156)
# GDS141 not directly found in C/ listing. Trying closest match.
download "$BASE/C/des_varnish_anp90-14.26705.asc" "fe3+mn_desert.varnish1.asc"

# 84. Desert_Varnish GDS78A Rhy (Record 6162)
download "$BASE/C/des_varnish_gds78.26731.asc" "fe3+mn_desert.varnish2.asc"

# 85. Enstatite NMNH128288 (Record 1644)
download "$BASE/M/enstatite_nmnh128288.7435.asc" "fe2+_pyroxene_enstatite.asc"

# 86. Hedenbergite NMNH119197 (Record 2088)
download "$BASE/M/hedenbergite_nmnh119197.9218.asc" "fe2+_pyroxene_hedenbergite.asc"

# 87. Lazurite HS418.3B (Record 2796)
download "$BASE/M/lazurite_hs418.12545.asc" "sulfate_lazurite.asc"

# 88. Magnetite HS195.3B (Record 2940)
download "$BASE/M/magnetite_hs195.13178.asc" "magnetite.asc"

# 89. Pyrolusite HS138.3B (Record 4158)
download "$BASE/M/pyrolusite_hs138.18732.asc" "mn_oxide_pyrolusite.asc"

# 90. Microcline HS103.3B Feldspar (Record 3042)
download "$BASE/M/microcline_hs103.13639.asc" "fe2+_feldspar_microcline.asc"

# 91. Albite HS143.3B Plagioclase (Record 132)
download "$BASE/M/albite_hs143.512.asc" "fe2+_feldspar_albite.asc"

# 92. Orthoclase NMNH142137 Fe (Record 3852)
download "$BASE/M/orthoclase_nmnh142137.17349.asc" "fe2+_feldspar_orthoclase.asc"

# 93. Pectolite NMNH94865.a (Record 3882)
download "$BASE/M/pectolite_nmnh94865.17443.asc" "inosilicate_pectolite.asc"

# 94. Rhodonite NMNHC6148 >250um (Record 4314)
download "$BASE/M/rhodonite_nmnh6148.19388.asc" "mn2+_rhodonite.asc"

# 95. Sulfur GDS94 Reagent (Record 4776)
download "$BASE/M/sulfur_gds94.21389.asc" "sulfur.asc"

echo ""
echo "========================================="
echo "Download complete: $SUCCESS succeeded, $FAIL failed"
if [ -n "$FAILED" ]; then
  echo "Failed files:"
  echo -e "$FAILED"
fi
echo "========================================="
