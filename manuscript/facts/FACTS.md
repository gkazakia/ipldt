# FACTS -- the numbers of the ipldt / ORMIR-BQRL manuscript (n = 137)

Built by `manuscript/facts/build_facts.py` on 2026-09-25 (n = 137, public build) from the validation records only. `facts.json` holds every value at full precision under the key printed in the `key` column; this file rounds. Cite keys, not prose.

**Sources.** Patella configuration A: `validation/results/records.json` (105 = 21 subjects x 5 maps), `validation/results/sample_means.csv` and, for BV/TV, `validation/results/patella_bvtv_A/records.json` (`validation/patella_bvtv_A.py`: the ratio on IPL's SEG and IPL's rendered trabecular contour, with IPL's printed sheet value beside it). Patella configuration B, every stage (STEP 1, contours, SEG, dt maps, metrics, BV/TV): `validation/results/from_ipl_contour_ceil_dt/records/*.json`, the dt-inclusive run of 2026-09-18 under the shipped engine (LH pad offset 'ceil', float32); its SEG stage is asserted identical to the SEG-only run `validation/results/from_ipl_contour_ceil/records/*.json`. Radius/tibia, A and B: `validation/results/oslh_auto_v4/records/*.json` (63 files, 62 counted) and `validation/results/oslh_noedit_v3/records/*.json` (54 files, 54 counted) matching `^[A-Za-z]+_[A-Za-z]+_\d+\.json$`; `Diaphyseal_CKD_991161` is not used; n = 116. Study names: `validation/results/cohort_studies.csv` and meta.study_label. Parameters: `ipldt/step1.py`, `ipldt/ormir.py`, `ipldt/core.py`.

**Configurations.** A = IPL's own SEG and contour renderings in, ipldt's distance-transform engine and cortical pore cascade out (tests modules 6-7). B = IPL's periosteal contour in; ipldt's compartment separation, contour rendering, Laplace-Hamming segmentation, distance transforms and cortical pore cascade out (tests modules 3-7). The reference is IPL V5.42; the metric is voxels that differ by global position; 'exact' means 0 differing voxels. Cortical porosity is in `FACTS_BMD_CTPO.md` (`make_facts_bmd_ctpo.py`; the file name is historical).

**Tb.Th definitions.** `TbTh_trabseg` = dt_thickness on TRAB_SEG cropped to the trabecular contour (what Script 32 computes; exists for all 137 scans: patella IPL file `<base>_TRAB_TH_old`, radius/tibia file `<base>_TRAB_TH`). `TbTh_wholeseg` = dt_thickness on the whole SEG (internal; the patella IPL file `<base>_TRAB_TH`). `TbTh_as_delivered` = the definition of IPL's delivered TRAB_TH file per cohort (patella whole SEG, radius/tibia TRAB_SEG); The paper reports `TbTh_trabseg` (one definition, all 137).

**Site groups.** `site.UD_radius`, `site.UD_tibia`, `site.D_radius`, `site.D_tibia` (meta.bone_key of the record) and `site.diaphyseal` = D radius + D tibia pooled, the figures' 'diaphyseal' cohort. `oslh` = all radius/tibia scans; `pooled` = patellae + radius/tibia.

## 1. Cohort

Table 1 rows (`tables.cohort.table1`; every column read from the records):

| site | n | study (scans) | stack length, slices (scans) | STEP-1 parameter set (scans) | participants | key |
|---|---|---|---|---|---|---|
| Patella | 21 | PFJ | 168 (19), 320 (1), 504 (1) | tibia | 16 | cohort.patella.* |
| Ultradistal radius | 29 | CKD (9), REPRO (20) | 168 | radius | 29 | cohort.site.UD_radius.* |
| Ultradistal tibia | 29 | CKD (9), REPRO (20) | 168 | tibia | 29 | cohort.site.UD_tibia.* |
| Diaphyseal radius | 29 | CKD (9), REPRO (20) | 168 | radius | 29 | cohort.site.D_radius.* |
| Diaphyseal tibia | 29 | CKD (8), REPRO (20), BMAT (1) | 168 | radius (1), tibia (28) | 29 | cohort.site.D_tibia.* |
| Total | 137 | PFJ (21), CKD (35), REPRO (80), BMAT (1) | 168 (135), 320 (1), 504 (1) | tibia (78), radius (59) | 46 | cohort.n_total, cohort.participants.total |

- Study CKD (`cohort.oslh.study.CKD.*`): 35 scan(s), 9 participant(s) (1 participant(s) with 3 scan(s), 8 participant(s) with 4 scan(s)), same-site repeats 0; chronic kidney disease study ('CKDStudy' in the scan header): UD radius 9, UD tibia 9, D radius 9, D tibia 8.
- Study REPRO (`cohort.oslh.study.REPRO.*`): 80 scan(s), 20 participant(s) (20 participant(s) with 4 scan(s)), same-site repeats 0; reproducibility study ('REPROStudy' in the scan header): UD radius 20, UD tibia 20, D radius 20, D tibia 20.
- Study BMAT (`cohort.oslh.study.BMAT.*`): 1 scan(s), 1 participant(s) (1 participant(s) with 1 scan(s)), same-site repeats 0; a third study; its aim is not stated in the records ('BMATStudy' in the scan header): D tibia 1.
- Patellae: study PFJSTUDY (`cohort.patella.study`), 16 participants, 5 with both knees; STEP-1 preset tibia (`cohort.patella.preset`). Participants over all studies: 46 (`cohort.participants.total`; assumes no person took part in two of the studies, which the records cannot establish).
- Participant codes normalised for one-character typos (`cohort.oslh.participant_code_fixes`): Diaphyseal/CKD/433045 (subject prefix typo CDK -> CKD); Diaphyseal/REPRO/800721 (subject prefix typo REPR0 -> REPRO).
- Radius/tibia presets: tibia 57 / radius 59 (`cohort.oslh.preset.*`); preset source {'fallback (better rendering match)': 1, 'layout rule': 115}. Scans evaluated with the other site's parameter set: 1 -- Diaphyseal/BMAT/610892 (D tibia, radius preset, rendering voxels differing per preset tried {'tibia': 26799, 'radius': 0}) (`cohort.oslh.preset.other_site.*`).
- Checked individually (`cohort.oslh.preset.checked_scans`): Diaphyseal/BMAT/610892 D tibia (scanner site index 39): record preset radius by fallback (better rendering match), trials {'tibia': 26799, 'radius': 0}; Diaphyseal/REPRO/787166 D tibia (scanner site index 38): record preset tibia by layout rule, trials {'tibia': 0}; Diaphyseal/REPRO/950609 D radius (scanner site index 20): record preset radius by layout rule, trials {'radius': 0}.
- Scanner XtremeCT II (Scanco Medical AG); IPL V5.42 (`cohort.ipl_version`); nominal voxel size 0.0607 mm (header element sizes x 0.0606989-0.0607003 mm, z 0.0606964-0.0606984 mm).
- Slices per scan: patella {168: 19, 320: 1, 504: 1}; radius/tibia {168: 116} (`cohort.*.slices.counts`).
- Patella acquisition: 68 kVp, 1470 uA, 60.7 um isotropic, ~2 min per knee, ~5 uSv per knee; UCSF IRB, written informed consent (`cohort.patella.acquisition`).

### Voxels compared

| comparison | voxels | key |
|---|---|---|
| A, patella, each trabecular map (SEG grid) | 987,878,584 | A.patella.voxels_compared.seg_grid |
| A, patella, Ct.Th (CORT_MASK grid) | 992,041,120 | A.patella.voxels_compared.cort_grid |
| A, patella, four reported maps | 3,955,676,872 | A.patella.voxels_compared.total_four_maps |
| A, radius/tibia, Tb.Th (TRAB_SEG) map | 3,405,028,872 | A.oslh.map.TbTh_trabseg.voxels_compared |
| A, pooled 137, four reported maps | 18,348,991,192 | A.pooled.voxels_compared.four_maps |
| B, patella, SEG (union grids) | 987,878,584 | B.patella.SEG.voxels_compared |
| B, radius/tibia, SEG (union grids) | 3,656,692,200 | B.oslh.SEG.voxels_compared |
| B, pooled 137, SEG | 4,644,570,784 | B.pooled.SEG.voxels_compared |
| B, pooled 137, Tb.Th (TRAB_SEG) map | 4,395,715,712 | B.pooled.map.TbTh_trabseg.voxels_compared |

## 2. Agreement per module

### 2.1 Cortical / trabecular compartment separation (STEP 1) and contour rendering -- configuration B

| stage | exact scans | mismatching voxels | IPL set voxels | min Dice | key |
|---|---|---|---|---|---|
| patella, cortical mask (STEP 1) | 21/21 | 0 | 206,208,867 | 1 | B.patella.step1.cort.* |
| patella, trabecular mask (STEP 1) | 21/21 | 0 | 361,567,628 | 1 | B.patella.step1.trab.* |
| patella, cortical contour rendering | 21/21 | 0 | 206,199,654 | 1 | B.patella.contour.cort.* |
| patella, trabecular contour rendering | 21/21 | 0 | 361,536,565 | 1 | B.patella.contour.trab.* |
| radius/tibia, cortical contour rendering (STEP 1 -> contour) | 116/116 | 0 | 722,711,147 | 1 | B.oslh.rendering.cort.* |
| radius/tibia, trabecular contour rendering (STEP 1 -> contour) | 116/116 | 0 | 1,483,184,326 | 1 | B.oslh.rendering.trab.* |
|   UD radius, cortical / trabecular renderings | 29/29 / 29/29 | 0 / 0 | 73,809,493 / 450,396,770 |  | B.site.UD_radius.rendering.* |
|   UD tibia, cortical / trabecular renderings | 29/29 / 29/29 | 0 / 0 | 178,210,450 / 829,332,511 |  | B.site.UD_tibia.rendering.* |
|   D radius, cortical / trabecular renderings | 29/29 / 29/29 | 0 / 0 | 115,562,343 / 37,499,366 |  | B.site.D_radius.rendering.* |
|   D tibia, cortical / trabecular renderings | 29/29 / 29/29 | 0 / 0 | 355,128,861 / 165,955,679 |  | B.site.D_tibia.rendering.* |
|   diaphyseal, cortical / trabecular renderings | 58/58 / 58/58 | 0 / 0 | 470,691,204 / 203,455,045 |  | B.site.diaphyseal.rendering.* |
|   ultradistal, cortical / trabecular renderings | 58/58 / 58/58 | 0 / 0 | 252,019,943 / 1,279,729,281 |  | B.site.ultradistal.rendering.* |

Radius/tibia STEP-1 masks are validated through their renderings (IPL's raw mask rasters were not exported); pooled over 137 scans the cortical compartment is exact on 137/137 and the trabecular on 137/137, with 0 differing voxels.

### 2.2 Laplace-Hamming segmentation (SEG) -- configuration B

| cohort | exact | mismatching voxels | ours only | IPL only | per scan min / median / max | min Dice | median Dice | Dice >= 0.9999 | per million | key |
|---|---|---|---|---|---|---|---|---|---|---|
| patella | 2/21 | 50 | 23 | 27 | 0 / 2 / 5 | 0.9999998 | 0.9999999 | 21/21 | 0.051 | B.patella.SEG.* |
| radius/tibia (116) | 35/116 | 91,713 | 85,564 | 6,149 | 0 / 3 / 41,480 | 0.99843367 | 0.99999985 | 112/116 | 25.081 | B.oslh.SEG.* |
| pooled (137) | 37/137 | 91,763 | 85,587 | 6,176 | 0 / 2 / 41,480 | 0.99843367 | 0.99999988 | 133/137 | 19.757 | B.pooled.SEG.* |
|   UD radius | 0/29 | 7,187 | 3,163 | 4,024 | 1 / 132 / 2,232 | 0.99983811 | 0.99998731 | 28/29 | 6.862 | B.site.UD_radius.SEG.* |
|   UD tibia | 0/29 | 84,487 | 82,381 | 2,106 | 1 / 42 / 41,480 | 0.99843367 | 0.9999985 | 26/29 | 53.006 | B.site.UD_tibia.SEG.* |
|   D radius | 25/29 | 5 | 3 | 2 | 0 / 0 / 2 | 0.9999998 | 1 | 29/29 | 0.022 | B.site.D_radius.SEG.* |
|   D tibia | 10/29 | 34 | 17 | 17 | 0 / 1 / 4 | 0.99999978 | 0.99999996 | 29/29 | 0.043 | B.site.D_tibia.SEG.* |
|   diaphyseal | 35/58 | 39 | 20 | 19 | 0 / 0 / 4 | 0.99999978 | 1 | 58/58 | 0.038 | B.site.diaphyseal.SEG.* |
|   ultradistal | 0/58 | 91,674 | 85,544 | 6,130 | 1 / 72 / 41,480 | 0.99843367 | 0.99999594 | 54/58 | 34.709 | B.site.ultradistal.SEG.* |

- Radius/tibia: the three scans with the largest SEG residual carry 82,462 voxels = 89.9 % of the total (Distal_REPRO_2095 UD tibia 41,480, Distal_REPRO_2051 UD tibia 23,797, Distal_REPRO_612066 UD tibia 17,185); the other 113 scans total 9,251 voxels (median 2, max 2,232) (`B.oslh.SEG.top3_*`, `B.oslh.SEG.without_top3.*`).
- Scans with <= 10 / <= 100 / <= 1,000 differing SEG voxels, pooled: 89 / 111 / 132 of 137 (`B.pooled.SEG.per_scan.le_*`).
- Label mismatches (set in both, different label): patella 0, radius/tibia 0.
- Patella TRAB_SEG: 33 voxels (13 ours only / 20 IPL only), exact 6/21, min Dice 0.99999973; CORT_SEG: 17 voxels (10 / 7), exact 9/21, min Dice 0.9999998 (`B.patella.TRAB_SEG.*`, `B.patella.CORT_SEG.*`). Radius/tibia: no TRAB_SEG files delivered; CORT_SEG comparisons are in FACTS_BMD_CTPO (pore.B.cort_seg.*).

### 2.3 Distance-transform maps -- configuration A (IPL's SEG and contours in)

| map | cohort | exact scans | mismatching voxels | ours only | IPL only | voxels compared | IPL non-zero voxels | key |
|---|---|---|---|---|---|---|---|---|
| Tb.Th map (dt_thickness on TRAB_SEG, Script 32 definition) | patella | 21/21 | 0 | 0 | 0 | 987,878,584 | 122,648,995 | A.patella.map.TbTh_trabseg.* |
| Tb.Th map (dt_thickness on TRAB_SEG, Script 32 definition) | radius/tibia | 116/116 | 0 | 0 | 0 | 3,405,028,872 | 327,375,152 | A.oslh.map.TbTh_trabseg.* |
| Tb.Th map (dt_thickness on TRAB_SEG, Script 32 definition) | pooled 137 | 137/137 | 0 | 0 | 0 | 4,392,907,456 | 450,024,147 | A.pooled.map.TbTh_trabseg.* |
| Tb.Sp map (dt_spacing) | patella | 21/21 | 0 | 0 | 0 | 987,878,584 | 229,578,781 | A.patella.map.TbSp.* |
| Tb.Sp map (dt_spacing) | radius/tibia | 116/116 | 0 | 0 | 0 | 3,653,958,840 | 1,108,414,176 | A.oslh.map.TbSp.* |
| Tb.Sp map (dt_spacing) | pooled 137 | 137/137 | 0 | 0 | 0 | 4,641,837,424 | 1,337,992,957 | A.pooled.map.TbSp.* |
| 1/Tb.N map (dt_number) | patella | 21/21 | 0 | 0 | 0 | 987,878,584 | 332,011,162 | A.patella.map.TbN.* |
| 1/Tb.N map (dt_number) | radius/tibia | 116/116 | 0 | 0 | 0 | 3,653,958,840 | 1,373,513,425 | A.oslh.map.TbN.* |
| 1/Tb.N map (dt_number) | pooled 137 | 137/137 | 0 | 0 | 0 | 4,641,837,424 | 1,705,524,587 | A.pooled.map.TbN.* |
| Ct.Th map (dt_thickness on CORT_MASK) | patella | 21/21 | 0 | 0 | 0 | 992,041,120 | 202,220,940 | A.patella.map.CtTh.* |
| Ct.Th map (dt_thickness on CORT_MASK) | radius/tibia | 116/116 | 0 | 0 | 0 | 3,680,367,768 | 711,656,796 | A.oslh.map.CtTh.* |
| Ct.Th map (dt_thickness on CORT_MASK) | pooled 137 | 137/137 | 0 | 0 | 0 | 4,672,408,888 | 913,877,736 | A.pooled.map.CtTh.* |
| Tb.Th map (dt_thickness on the whole SEG; internal, not the Script 32 definition) | patella | 21/21 | 0 | 0 | 0 | 987,878,584 | 121,993,740 | A.patella.map.TbTh_wholeseg.* |
| Tb.Th map (dt_thickness on the whole SEG; internal, not the Script 32 definition) | radius/tibia | 0/116 | 13,368,670 | 127,688 | 3,368,034 | 3,653,958,840 | 327,375,152 | A.oslh.map.TbTh_wholeseg.* |

- The differing voxels of the pooled configuration A (four reported maps, 18,348,991,192 voxels compared, 0 differing; 548 of 548 scan-map comparisons exact): none (`A.oslh.map.TbTh_trabseg.nonexact_scans`).
- Per site, configuration A, Tb.Th (TRAB_SEG) / Tb.Sp / 1/Tb.N / Ct.Th exact scans: UD radius 29/29, 29, 29, 29; UD tibia 29/29, 29, 29, 29; D radius 29/29, 29, 29, 29; D tibia 29/29, 29, 29, 29; diaphyseal 58/58, 58, 58, 58; ultradistal 58/58, 58, 58, 58 (`A.site.*`).
- The whole-SEG Tb.Th object (internal) differs from IPL's delivered radius/tibia Tb.Th files by 13,368,670 voxels (0/116 exact): IPL's standard evaluation uses TRAB_SEG.

### 2.4 Distance-transform maps -- configuration B (whole chain from IPL's periosteal contour)

| map | cohort | exact scans | mismatching voxels | ours only | IPL only | voxels compared | key |
|---|---|---|---|---|---|---|---|
| Tb.Th map (dt_thickness on TRAB_SEG, Script 32 definition) | patella | 6/21 | 168 | 17 | 28 | 990,085,568 | B.patella.map.TbTh_trabseg.* |
| Tb.Th map (dt_thickness on TRAB_SEG, Script 32 definition) | radius/tibia | 72/116 | 20,743 | 7,647 | 224 | 3,405,630,144 | B.oslh.map.TbTh_trabseg.* |
| Tb.Th map (dt_thickness on TRAB_SEG, Script 32 definition) | pooled 137 | 78/137 | 20,911 | 7,664 | 252 | 4,395,715,712 | B.pooled.map.TbTh_trabseg.* |
| Tb.Sp map (dt_spacing) | patella | 6/21 | 199 | 32 | 27 | 987,878,584 | B.patella.map.TbSp.* |
| Tb.Sp map (dt_spacing) | radius/tibia | 67/116 | 14,148 | 3,141 | 682 | 3,656,692,200 | B.oslh.map.TbSp.* |
| Tb.Sp map (dt_spacing) | pooled 137 | 73/137 | 14,347 | 3,173 | 709 | 4,644,570,784 | B.pooled.map.TbSp.* |
| 1/Tb.N map (dt_number) | patella | 8/21 | 1,685 | 103 | 52 | 987,878,584 | B.patella.map.TbN.* |
| 1/Tb.N map (dt_number) | radius/tibia | 65/116 | 55,046 | 11,069 | 2,385 | 3,656,692,200 | B.oslh.map.TbN.* |
| 1/Tb.N map (dt_number) | pooled 137 | 73/137 | 56,731 | 11,172 | 2,437 | 4,644,570,784 | B.pooled.map.TbN.* |
| Ct.Th map (dt_thickness on CORT_MASK) | patella | 21/21 | 0 | 0 | 0 | 992,041,120 | B.patella.map.CtTh.* |
| Ct.Th map (dt_thickness on CORT_MASK) | radius/tibia | 116/116 | 0 | 0 | 0 | 3,680,367,768 | B.oslh.map.CtTh.* |
| Ct.Th map (dt_thickness on CORT_MASK) | pooled 137 | 137/137 | 0 | 0 | 0 | 4,672,408,888 | B.pooled.map.CtTh.* |

Per site (B): UD radius: Tb.Th 146 (10/29 exact), Tb.Sp 3,273 (5), 1/Tb.N 7,064 (5), Ct.Th 0 (29); UD tibia: Tb.Th 20,573 (10/29 exact), Tb.Sp 9,413 (11), 1/Tb.N 47,795 (11), Ct.Th 0 (29); D radius: Tb.Th 0 (29/29 exact), Tb.Sp 0 (29), 1/Tb.N 0 (29), Ct.Th 0 (29); D tibia: Tb.Th 24 (23/29 exact), Tb.Sp 1,462 (22), 1/Tb.N 187 (20), Ct.Th 0 (29); diaphyseal: Tb.Th 24 (52/58 exact), Tb.Sp 1,462 (51), 1/Tb.N 187 (49), Ct.Th 0 (58); ultradistal: Tb.Th 20,719 (20/58 exact), Tb.Sp 12,686 (16), 1/Tb.N 54,859 (16), Ct.Th 0 (58) (`B.site.*.map.*`).

## 3. Scalar metrics (ours vs IPL; IPL is x)

Statistics: mean +- SD of both, bias = mean(ours - IPL), SD of the differences, 95 % limits of agreement = bias +- 1.96 SD, max |diff|, max |rel diff| in %, exact = scans with identical values (to 12 significant digits; differences of a few 1e-16 between two identical maps averaged by different code paths are treated as 0, the bitwise count is `exact_bitwise`; when every pair is exact the fit is the identity), OLS slope / intercept of ours on IPL, R^2, ICC(2,1) (two-way random, absolute agreement, single measurement). Metric values at the nominal 0.0607 mm voxel size. Keys: `metrics.<A|B>.<patella|oslh|pooled|site.X>.<metric>.<stat>`.

### 3.1 Configuration B (the whole chain), pooled n = 137 and per cohort

| metric | cohort | n | IPL mean +- SD | ours mean +- SD | bias | SD diff | 95 % LoA | max |diff| | max |rel| % | exact | slope | intercept | R^2 | ICC(2,1) | key |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BV/TV | pooled | 137 | 0.197348 +- 0.105282 | 0.19735 +- 0.105283 | 1.895e-06 | 1.485e-05 | [-2.722e-05, 3.101e-05] | 1.322e-04 | 0.0506 | 85/137 | 1.000013 | -6.986e-07 | 1 | 1 | metrics.B.pooled.BVTV.* |
| BV/TV | patella | 21 | 0.355964 +- 0.042826 | 0.355964 +- 0.042826 | -1.342e-08 | 7.231e-08 | [-1.551e-07, 1.283e-07] | 1.934e-07 | 5.226e-05 | 11/21 | 1 | -4.008e-08 | 1 | 1 | metrics.B.patella.BVTV.* |
| BV/TV | radius/tibia | 116 | 0.168633 +- 0.08581 | 0.168635 +- 0.085813 | 2.241e-06 | 1.613e-05 | [-2.937e-05, 3.385e-05] | 1.322e-04 | 0.0506 | 74/116 | 1.000032 | -3.199e-06 | 1 | 1 | metrics.B.oslh.BVTV.* |
| BV/TV | UD radius | 29 | 0.236909 +- 0.03887 | 0.236909 +- 0.03887 | -4.031e-08 | 6.082e-08 | [-1.595e-07, 7.889e-08] | 1.883e-07 | 7.570e-05 | 12/29 | 1 | -4.693e-08 | 1 | 1 | metrics.B.site.UD_radius.BVTV.* |
| BV/TV | UD tibia | 29 | 0.244598 +- 0.037958 | 0.244607 +- 0.037965 | 9.004e-06 | 3.171e-05 | [-5.314e-05, 7.115e-05] | 1.322e-04 | 0.0506 | 10/29 | 1.000188 | -3.700e-05 | 0.9999993 | 0.9999996 | metrics.B.site.UD_tibia.BVTV.* |
| BV/TV | D radius | 29 | 0.072091 +- 0.053749 | 0.072091 +- 0.053749 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.B.site.D_radius.BVTV.* |
| BV/TV | D tibia | 29 | 0.120932 +- 0.039839 | 0.120932 +- 0.039839 | -1.780e-09 | 9.357e-08 | [-1.852e-07, 1.816e-07] | 2.799e-07 | 2.506e-04 | 23/29 | 0.999999 | 8.173e-08 | 1 | 1 | metrics.B.site.D_tibia.BVTV.* |
| BV/TV | diaphyseal | 58 | 0.096512 +- 0.052968 | 0.096512 +- 0.052968 | -8.899e-10 | 6.559e-08 | [-1.294e-07, 1.277e-07] | 2.799e-07 | 2.506e-04 | 52/58 | 1 | 1.839e-08 | 1 | 1 | metrics.B.site.diaphyseal.BVTV.* |
| BV/TV | ultradistal | 58 | 0.240754 +- 0.038276 | 0.240758 +- 0.038279 | 4.482e-06 | 2.269e-05 | [-3.998e-05, 4.895e-05] | 1.322e-04 | 0.0506 | 22/58 | 1.000103 | -2.030e-05 | 0.9999997 | 0.9999998 | metrics.B.site.ultradistal.BVTV.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | pooled | 137 | 0.208107 +- 0.022397 | 0.208109 +- 0.022398 | 1.306e-06 | 9.423e-06 | [-1.716e-05, 1.977e-05] | 8.904e-05 | 0.0405 | 81/137 | 1.000038 | -6.689e-06 | 0.9999998 | 0.9999999 | metrics.B.pooled.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | patella | 21 | 0.224255 +- 0.008337 | 0.224255 +- 0.008337 | 2.509e-08 | 1.121e-07 | [-1.946e-07, 2.448e-07] | 3.920e-07 | 1.700e-04 | 7/21 | 1.000002 | -5.082e-07 | 1 | 1 | metrics.B.patella.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | radius/tibia | 116 | 0.205184 +- 0.022911 | 0.205186 +- 0.022912 | 1.538e-06 | 1.023e-05 | [-1.851e-05, 2.159e-05] | 8.904e-05 | 0.0405 | 74/116 | 1.000052 | -9.103e-06 | 0.9999998 | 0.9999999 | metrics.B.oslh.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | UD radius | 29 | 0.205624 +- 0.008936 | 0.205624 +- 0.008936 | 2.513e-08 | 1.305e-07 | [-2.306e-07, 2.809e-07] | 5.970e-07 | 2.795e-04 | 11/29 | 1.000002 | -4.845e-07 | 1 | 1 | metrics.B.site.UD_radius.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | UD tibia | 29 | 0.218996 +- 0.012385 | 0.219003 +- 0.012386 | 6.080e-06 | 2.003e-05 | [-3.318e-05, 4.534e-05] | 8.904e-05 | 0.0405 | 11/29 | 1.000153 | -2.738e-05 | 0.9999974 | 0.9999986 | metrics.B.site.UD_tibia.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | D radius | 29 | 0.187944 +- 0.033272 | 0.187944 +- 0.033272 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.B.site.D_radius.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | D tibia | 29 | 0.208172 +- 0.017343 | 0.208172 +- 0.017343 | 4.467e-08 | 2.730e-07 | [-4.904e-07, 5.797e-07] | 1.405e-06 | 6.186e-04 | 23/29 | 1.000003 | -6.714e-07 | 1 | 1 | metrics.B.site.D_tibia.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | diaphyseal | 58 | 0.198058 +- 0.028207 | 0.198058 +- 0.028207 | 2.234e-08 | 1.926e-07 | [-3.553e-07, 3.999e-07] | 1.405e-06 | 6.186e-04 | 52/58 | 1.000001 | -1.614e-07 | 1 | 1 | metrics.B.site.diaphyseal.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | ultradistal | 58 | 0.21231 +- 0.012652 | 0.212313 +- 0.012654 | 3.053e-06 | 1.437e-05 | [-2.511e-05, 3.121e-05] | 8.904e-05 | 0.0405 | 22/58 | 1.000201 | -3.967e-05 | 0.9999988 | 0.9999993 | metrics.B.site.ultradistal.TbTh_trabseg.* |
| Tb.Sp | pooled | 137 | 1.000666 +- 0.824539 | 1.000666 +- 0.824539 | 1.471e-08 | 3.620e-06 | [-7.080e-06, 7.110e-06] | 2.782e-05 | 0.0038 | 73/137 | 1 | -2.165e-07 | 1 | 1 | metrics.B.pooled.TbSp.* |
| Tb.Sp | patella | 21 | 0.359076 +- 0.059595 | 0.359076 +- 0.059595 | -1.999e-08 | 4.582e-08 | [-1.098e-07, 6.983e-08] | 1.665e-07 | 5.003e-05 | 6/21 | 1 | 6.951e-08 | 1 | 1 | metrics.B.patella.TbSp.* |
| Tb.Sp | radius/tibia | 116 | 1.116816 +- 0.845351 | 1.116816 +- 0.845351 | 2.099e-08 | 3.936e-06 | [-7.694e-06, 7.736e-06] | 2.782e-05 | 0.0038 | 67/116 | 1 | -2.620e-07 | 1 | 1 | metrics.B.oslh.TbSp.* |
| Tb.Sp | UD radius | 29 | 0.575515 +- 0.131137 | 0.575516 +- 0.131138 | 3.057e-07 | 1.888e-06 | [-3.394e-06, 4.006e-06] | 9.355e-06 | 0.001 | 5/29 | 1.000007 | -3.560e-06 | 1 | 1 | metrics.B.site.UD_radius.TbSp.* |
| Tb.Sp | UD tibia | 29 | 0.670296 +- 0.11387 | 0.670295 +- 0.113871 | -1.142e-06 | 5.559e-06 | [-1.204e-05, 9.753e-06] | 2.552e-05 | 0.0038 | 11/29 | 1.000008 | -6.689e-06 | 1 | 1 | metrics.B.site.UD_tibia.TbSp.* |
| Tb.Sp | D radius | 29 | 2.062709 +- 0.96986 | 2.062709 +- 0.96986 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.B.site.D_radius.TbSp.* |
| Tb.Sp | D tibia | 29 | 1.158743 +- 0.725112 | 1.158744 +- 0.725113 | 9.203e-07 | 5.182e-06 | [-9.237e-06, 1.108e-05] | 2.782e-05 | 0.0021 | 22/29 | 1 | 6.463e-07 | 1 | 1 | metrics.B.site.D_tibia.TbSp.* |
| Tb.Sp | diaphyseal | 58 | 1.610726 +- 0.963441 | 1.610727 +- 0.96344 | 4.601e-07 | 3.662e-06 | [-6.717e-06, 7.637e-06] | 2.782e-05 | 0.0021 | 51/58 | 1 | 7.214e-07 | 1 | 1 | metrics.B.site.diaphyseal.TbSp.* |
| Tb.Sp | ultradistal | 58 | 0.622906 +- 0.130776 | 0.622905 +- 0.130776 | -4.181e-07 | 4.179e-06 | [-8.608e-06, 7.772e-06] | 2.552e-05 | 0.0038 | 16/58 | 1.000004 | -3.133e-06 | 1 | 1 | metrics.B.site.ultradistal.TbSp.* |
| Tb.N | pooled | 137 | 1.365424 +- 0.649167 | 1.365425 +- 0.649167 | 1.003e-06 | 1.078e-05 | [-2.013e-05, 2.214e-05] | 1.186e-04 | 0.0088 | 73/137 | 1 | 6.037e-07 | 1 | 1 | metrics.B.pooled.TbN.* |
| Tb.N | patella | 21 | 2.389915 +- 0.271834 | 2.389916 +- 0.271834 | 2.989e-07 | 2.785e-06 | [-5.159e-06, 5.757e-06] | 8.887e-06 | 3.318e-04 | 8/21 | 0.999998 | 3.896e-06 | 1 | 1 | metrics.B.patella.TbN.* |
| Tb.N | radius/tibia | 116 | 1.179955 +- 0.509078 | 1.179957 +- 0.509078 | 1.131e-06 | 1.166e-05 | [-2.173e-05, 2.399e-05] | 1.186e-04 | 0.0088 | 65/116 | 1.000001 | -3.303e-07 | 1 | 1 | metrics.B.oslh.TbN.* |
| Tb.N | UD radius | 29 | 1.640923 +- 0.291933 | 1.640923 +- 0.291934 | 8.947e-07 | 7.522e-06 | [-1.385e-05, 1.564e-05] | 3.705e-05 | 0.0019 | 5/29 | 1.000003 | -3.771e-06 | 1 | 1 | metrics.B.site.UD_radius.TbN.* |
| Tb.N | UD tibia | 29 | 1.397522 +- 0.185279 | 1.397526 +- 0.185278 | 3.667e-06 | 2.219e-05 | [-3.983e-05, 4.716e-05] | 1.186e-04 | 0.0088 | 11/29 | 0.999994 | 1.191e-05 | 1 | 1 | metrics.B.site.UD_tibia.TbN.* |
| Tb.N | D radius | 29 | 0.579814 +- 0.259544 | 0.579814 +- 0.259544 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.B.site.D_radius.TbN.* |
| Tb.N | D tibia | 29 | 1.101563 +- 0.478527 | 1.101563 +- 0.478527 | -3.731e-08 | 3.182e-07 | [-6.609e-07, 5.863e-07] | 1.510e-06 | 1.991e-04 | 20/29 | 1 | -1.425e-07 | 1 | 1 | metrics.B.site.D_tibia.TbN.* |
| Tb.N | diaphyseal | 58 | 0.840689 +- 0.463493 | 0.840688 +- 0.463493 | -1.865e-08 | 2.238e-07 | [-4.573e-07, 4.200e-07] | 1.510e-06 | 1.991e-04 | 49/58 | 1 | -4.130e-08 | 1 | 1 | metrics.B.site.diaphyseal.TbN.* |
| Tb.N | ultradistal | 58 | 1.519222 +- 0.271659 | 1.519225 +- 0.271658 | 2.281e-06 | 1.648e-05 | [-3.003e-05, 3.459e-05] | 1.186e-04 | 0.0088 | 16/58 | 0.999998 | 5.410e-06 | 1 | 1 | metrics.B.site.ultradistal.TbN.* |
| Ct.Th | pooled | 137 | 2.795437 +- 1.851162 | 2.795437 +- 1.851162 | 0 | 0 | [0, 0] | 0 | 0 | 137/137 | 1 | 0 | 1 | 1 | metrics.B.pooled.CtTh.* |
| Ct.Th | patella | 21 | 3.104652 +- 0.965622 | 3.104652 +- 0.965622 | 0 | 0 | [0, 0] | 0 | 0 | 21/21 | 1 | 0 | 1 | 1 | metrics.B.patella.CtTh.* |
| Ct.Th | radius/tibia | 116 | 2.739458 +- 1.967177 | 2.739458 +- 1.967177 | 0 | 0 | [0, 0] | 0 | 0 | 116/116 | 1 | 0 | 1 | 1 | metrics.B.oslh.CtTh.* |
| Ct.Th | UD radius | 29 | 0.77523 +- 0.250284 | 0.77523 +- 0.250284 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.B.site.UD_radius.CtTh.* |
| Ct.Th | UD tibia | 29 | 1.492257 +- 0.369268 | 1.492257 +- 0.369268 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.B.site.UD_tibia.CtTh.* |
| Ct.Th | D radius | 29 | 3.1993 +- 0.612932 | 3.1993 +- 0.612932 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.B.site.D_radius.CtTh.* |
| Ct.Th | D tibia | 29 | 5.491046 +- 1.285058 | 5.491046 +- 1.285058 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.B.site.D_tibia.CtTh.* |
| Ct.Th | diaphyseal | 58 | 4.345173 +- 1.527026 | 4.345173 +- 1.527026 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.B.site.diaphyseal.CtTh.* |
| Ct.Th | ultradistal | 58 | 1.133743 +- 0.478061 | 1.133743 +- 0.478061 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.B.site.ultradistal.CtTh.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | pooled | 137 | 0.208581 +- 0.02279 | 0.208582 +- 0.022791 | 1.306e-06 | 9.423e-06 | [-1.716e-05, 1.977e-05] | 8.904e-05 | 0.0405 | 80/137 | 1.000036 | -6.188e-06 | 0.9999998 | 0.9999999 | metrics.B.pooled.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | patella | 21 | 0.227345 +- 0.008764 | 0.227345 +- 0.008764 | 2.479e-08 | 1.129e-07 | [-1.965e-07, 2.460e-07] | 3.941e-07 | 1.690e-04 | 6/21 | 1.000002 | -4.768e-07 | 1 | 1 | metrics.B.patella.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | radius/tibia | 116 | 0.205184 +- 0.022911 | 0.205186 +- 0.022912 | 1.538e-06 | 1.023e-05 | [-1.851e-05, 2.159e-05] | 8.904e-05 | 0.0405 | 74/116 | 1.000052 | -9.103e-06 | 0.9999998 | 0.9999999 | metrics.B.oslh.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | UD radius | 29 | 0.205624 +- 0.008936 | 0.205624 +- 0.008936 | 2.513e-08 | 1.305e-07 | [-2.306e-07, 2.809e-07] | 5.970e-07 | 2.795e-04 | 11/29 | 1.000002 | -4.845e-07 | 1 | 1 | metrics.B.site.UD_radius.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | UD tibia | 29 | 0.218996 +- 0.012385 | 0.219003 +- 0.012386 | 6.080e-06 | 2.003e-05 | [-3.318e-05, 4.534e-05] | 8.904e-05 | 0.0405 | 11/29 | 1.000153 | -2.738e-05 | 0.9999974 | 0.9999986 | metrics.B.site.UD_tibia.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | D radius | 29 | 0.187944 +- 0.033272 | 0.187944 +- 0.033272 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.B.site.D_radius.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | D tibia | 29 | 0.208172 +- 0.017343 | 0.208172 +- 0.017343 | 4.467e-08 | 2.730e-07 | [-4.904e-07, 5.797e-07] | 1.405e-06 | 6.186e-04 | 23/29 | 1.000003 | -6.714e-07 | 1 | 1 | metrics.B.site.D_tibia.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | diaphyseal | 58 | 0.198058 +- 0.028207 | 0.198058 +- 0.028207 | 2.234e-08 | 1.926e-07 | [-3.553e-07, 3.999e-07] | 1.405e-06 | 6.186e-04 | 52/58 | 1.000001 | -1.614e-07 | 1 | 1 | metrics.B.site.diaphyseal.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | ultradistal | 58 | 0.21231 +- 0.012652 | 0.212313 +- 0.012654 | 3.053e-06 | 1.437e-05 | [-2.511e-05, 3.121e-05] | 8.904e-05 | 0.0405 | 22/58 | 1.000201 | -3.967e-05 | 0.9999988 | 0.9999993 | metrics.B.site.ultradistal.TbTh_as_delivered.* |
| Tb.Th (whole SEG; internal) | pooled | 137 | 0.208581 +- 0.02279 | 0.217068 +- 0.021355 | 0.008487 | 0.010345 | [-0.011789, 0.028764] | 0.06016 | 33.9105 | 6/137 | 0.836003 | 0.042694 | 0.7959561 | 0.8296738 | metrics.B.pooled.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | patella | 21 | 0.227345 +- 0.008764 | 0.227345 +- 0.008764 | 2.479e-08 | 1.129e-07 | [-1.965e-07, 2.460e-07] | 3.941e-07 | 1.690e-04 | 6/21 | 1.000002 | -4.768e-07 | 1 | 1 | metrics.B.patella.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | radius/tibia | 116 | 0.205184 +- 0.022911 | 0.215208 +- 0.022432 | 0.010024 | 0.010537 | [-0.010629, 0.030676] | 0.06016 | 33.9105 | 0/116 | 0.873537 | 0.035972 | 0.7960325 | 0.813282 | metrics.B.oslh.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | UD radius | 29 | 0.205624 +- 0.008936 | 0.208516 +- 0.009195 | 0.002892 | 7.044e-04 | [0.001512, 0.004273] | 0.004898 | 2.2962 | 0/29 | 1.026264 | -0.002508 | 0.9947819 | 0.9487994 | metrics.B.site.UD_radius.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | UD tibia | 29 | 0.218996 +- 0.012385 | 0.221793 +- 0.012482 | 0.002797 | 6.726e-04 | [0.001478, 0.004115] | 0.004352 | 1.9428 | 0/29 | 1.006392 | 0.001397 | 0.997136 | 0.9739486 | metrics.B.site.UD_tibia.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | D radius | 29 | 0.187944 +- 0.033272 | 0.211135 +- 0.037329 | 0.023191 | 0.012312 | [-9.403e-04, 0.047323] | 0.06016 | 33.9105 | 0/29 | 1.0609 | 0.011745 | 0.8941637 | 0.7744268 | metrics.B.site.D_radius.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | D tibia | 29 | 0.208172 +- 0.017343 | 0.219388 +- 0.017501 | 0.011215 | 0.003957 | [0.00346, 0.018971] | 0.022176 | 10.8816 | 0/29 | 0.983128 | 0.014727 | 0.9491585 | 0.8075913 | metrics.B.site.D_tibia.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | diaphyseal | 58 | 0.198058 +- 0.028207 | 0.215261 +- 0.029194 | 0.017203 | 0.010892 | [-0.004145, 0.038552] | 0.06016 | 33.9105 | 0/58 | 0.961037 | 0.02492 | 0.8622161 | 0.7875502 | metrics.B.site.diaphyseal.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | ultradistal | 58 | 0.21231 +- 0.012652 | 0.215154 +- 0.012763 | 0.002844 | 6.844e-04 | [0.001503, 0.004186] | 0.004898 | 2.2962 | 0/58 | 1.00741 | 0.001271 | 0.9971789 | 0.9741696 | metrics.B.site.ultradistal.TbTh_wholeseg.* |

### 3.2 Configuration A (IPL's SEG in), pooled and per cohort

| metric | cohort | n | IPL mean +- SD | ours mean +- SD | bias | SD diff | 95 % LoA | max |diff| | max |rel| % | exact | slope | intercept | R^2 | ICC(2,1) | key |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BV/TV | pooled | 137 | 0.197348 +- 0.105282 | 0.197348 +- 0.105282 | 0 | 0 | [0, 0] | 0 | 0 | 137/137 | 1 | 0 | 1 | 1 | metrics.A.pooled.BVTV.* |
| BV/TV | patella | 21 | 0.355964 +- 0.042826 | 0.355964 +- 0.042826 | 0 | 0 | [0, 0] | 0 | 0 | 21/21 | 1 | 0 | 1 | 1 | metrics.A.patella.BVTV.* |
| BV/TV | radius/tibia | 116 | 0.168633 +- 0.08581 | 0.168633 +- 0.08581 | 0 | 0 | [0, 0] | 0 | 0 | 116/116 | 1 | 0 | 1 | 1 | metrics.A.oslh.BVTV.* |
| BV/TV | UD radius | 29 | 0.236909 +- 0.03887 | 0.236909 +- 0.03887 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.UD_radius.BVTV.* |
| BV/TV | UD tibia | 29 | 0.244598 +- 0.037958 | 0.244598 +- 0.037958 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.UD_tibia.BVTV.* |
| BV/TV | D radius | 29 | 0.072091 +- 0.053749 | 0.072091 +- 0.053749 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.D_radius.BVTV.* |
| BV/TV | D tibia | 29 | 0.120932 +- 0.039839 | 0.120932 +- 0.039839 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.D_tibia.BVTV.* |
| BV/TV | diaphyseal | 58 | 0.096512 +- 0.052968 | 0.096512 +- 0.052968 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.A.site.diaphyseal.BVTV.* |
| BV/TV | ultradistal | 58 | 0.240754 +- 0.038276 | 0.240754 +- 0.038276 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.A.site.ultradistal.BVTV.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | pooled | 137 | 0.208107 +- 0.022397 | 0.208107 +- 0.022397 | 0 | 0 | [0, 0] | 0 | 0 | 137/137 | 1 | 0 | 1 | 1 | metrics.A.pooled.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | patella | 21 | 0.224255 +- 0.008337 | 0.224255 +- 0.008337 | 0 | 0 | [0, 0] | 0 | 0 | 21/21 | 1 | 0 | 1 | 1 | metrics.A.patella.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | radius/tibia | 116 | 0.205184 +- 0.022911 | 0.205184 +- 0.022911 | 0 | 0 | [0, 0] | 0 | 0 | 116/116 | 1 | 0 | 1 | 1 | metrics.A.oslh.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | UD radius | 29 | 0.205624 +- 0.008936 | 0.205624 +- 0.008936 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.UD_radius.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | UD tibia | 29 | 0.218996 +- 0.012385 | 0.218996 +- 0.012385 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.UD_tibia.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | D radius | 29 | 0.187944 +- 0.033272 | 0.187944 +- 0.033272 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.D_radius.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | D tibia | 29 | 0.208172 +- 0.017343 | 0.208172 +- 0.017343 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.D_tibia.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | diaphyseal | 58 | 0.198058 +- 0.028207 | 0.198058 +- 0.028207 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.A.site.diaphyseal.TbTh_trabseg.* |
| Tb.Th (TRAB_SEG, Script 32 definition) | ultradistal | 58 | 0.21231 +- 0.012652 | 0.21231 +- 0.012652 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.A.site.ultradistal.TbTh_trabseg.* |
| Tb.Sp | pooled | 137 | 1.000666 +- 0.824539 | 1.000666 +- 0.824539 | 0 | 0 | [0, 0] | 0 | 0 | 137/137 | 1 | 0 | 1 | 1 | metrics.A.pooled.TbSp.* |
| Tb.Sp | patella | 21 | 0.359076 +- 0.059595 | 0.359076 +- 0.059595 | 0 | 0 | [0, 0] | 0 | 0 | 21/21 | 1 | 0 | 1 | 1 | metrics.A.patella.TbSp.* |
| Tb.Sp | radius/tibia | 116 | 1.116816 +- 0.845351 | 1.116816 +- 0.845351 | 0 | 0 | [0, 0] | 0 | 0 | 116/116 | 1 | 0 | 1 | 1 | metrics.A.oslh.TbSp.* |
| Tb.Sp | UD radius | 29 | 0.575515 +- 0.131137 | 0.575515 +- 0.131137 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.UD_radius.TbSp.* |
| Tb.Sp | UD tibia | 29 | 0.670296 +- 0.11387 | 0.670296 +- 0.11387 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.UD_tibia.TbSp.* |
| Tb.Sp | D radius | 29 | 2.062709 +- 0.96986 | 2.062709 +- 0.96986 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.D_radius.TbSp.* |
| Tb.Sp | D tibia | 29 | 1.158743 +- 0.725112 | 1.158743 +- 0.725112 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.D_tibia.TbSp.* |
| Tb.Sp | diaphyseal | 58 | 1.610726 +- 0.963441 | 1.610726 +- 0.963441 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.A.site.diaphyseal.TbSp.* |
| Tb.Sp | ultradistal | 58 | 0.622906 +- 0.130776 | 0.622906 +- 0.130776 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.A.site.ultradistal.TbSp.* |
| Tb.N | pooled | 137 | 1.365424 +- 0.649167 | 1.365424 +- 0.649167 | 0 | 0 | [0, 0] | 0 | 0 | 137/137 | 1 | 0 | 1 | 1 | metrics.A.pooled.TbN.* |
| Tb.N | patella | 21 | 2.389915 +- 0.271834 | 2.389915 +- 0.271834 | 0 | 0 | [0, 0] | 0 | 0 | 21/21 | 1 | 0 | 1 | 1 | metrics.A.patella.TbN.* |
| Tb.N | radius/tibia | 116 | 1.179955 +- 0.509078 | 1.179955 +- 0.509078 | 0 | 0 | [0, 0] | 0 | 0 | 116/116 | 1 | 0 | 1 | 1 | metrics.A.oslh.TbN.* |
| Tb.N | UD radius | 29 | 1.640923 +- 0.291933 | 1.640923 +- 0.291933 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.UD_radius.TbN.* |
| Tb.N | UD tibia | 29 | 1.397522 +- 0.185279 | 1.397522 +- 0.185279 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.UD_tibia.TbN.* |
| Tb.N | D radius | 29 | 0.579814 +- 0.259544 | 0.579814 +- 0.259544 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.D_radius.TbN.* |
| Tb.N | D tibia | 29 | 1.101563 +- 0.478527 | 1.101563 +- 0.478527 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.D_tibia.TbN.* |
| Tb.N | diaphyseal | 58 | 0.840689 +- 0.463493 | 0.840689 +- 0.463493 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.A.site.diaphyseal.TbN.* |
| Tb.N | ultradistal | 58 | 1.519222 +- 0.271659 | 1.519222 +- 0.271659 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.A.site.ultradistal.TbN.* |
| Ct.Th | pooled | 137 | 2.795437 +- 1.851162 | 2.795437 +- 1.851162 | 0 | 0 | [0, 0] | 0 | 0 | 137/137 | 1 | 0 | 1 | 1 | metrics.A.pooled.CtTh.* |
| Ct.Th | patella | 21 | 3.104652 +- 0.965622 | 3.104652 +- 0.965622 | 0 | 0 | [0, 0] | 0 | 0 | 21/21 | 1 | 0 | 1 | 1 | metrics.A.patella.CtTh.* |
| Ct.Th | radius/tibia | 116 | 2.739458 +- 1.967177 | 2.739458 +- 1.967177 | 0 | 0 | [0, 0] | 0 | 0 | 116/116 | 1 | 0 | 1 | 1 | metrics.A.oslh.CtTh.* |
| Ct.Th | UD radius | 29 | 0.77523 +- 0.250284 | 0.77523 +- 0.250284 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.UD_radius.CtTh.* |
| Ct.Th | UD tibia | 29 | 1.492257 +- 0.369268 | 1.492257 +- 0.369268 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.UD_tibia.CtTh.* |
| Ct.Th | D radius | 29 | 3.1993 +- 0.612932 | 3.1993 +- 0.612932 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.D_radius.CtTh.* |
| Ct.Th | D tibia | 29 | 5.491046 +- 1.285058 | 5.491046 +- 1.285058 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.D_tibia.CtTh.* |
| Ct.Th | diaphyseal | 58 | 4.345173 +- 1.527026 | 4.345173 +- 1.527026 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.A.site.diaphyseal.CtTh.* |
| Ct.Th | ultradistal | 58 | 1.133743 +- 0.478061 | 1.133743 +- 0.478061 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.A.site.ultradistal.CtTh.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | pooled | 137 | 0.208581 +- 0.02279 | 0.208581 +- 0.02279 | 0 | 0 | [0, 0] | 0 | 0 | 137/137 | 1 | 0 | 1 | 1 | metrics.A.pooled.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | patella | 21 | 0.227345 +- 0.008764 | 0.227345 +- 0.008764 | 0 | 0 | [0, 0] | 0 | 0 | 21/21 | 1 | 0 | 1 | 1 | metrics.A.patella.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | radius/tibia | 116 | 0.205184 +- 0.022911 | 0.205184 +- 0.022911 | 0 | 0 | [0, 0] | 0 | 0 | 116/116 | 1 | 0 | 1 | 1 | metrics.A.oslh.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | UD radius | 29 | 0.205624 +- 0.008936 | 0.205624 +- 0.008936 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.UD_radius.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | UD tibia | 29 | 0.218996 +- 0.012385 | 0.218996 +- 0.012385 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.UD_tibia.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | D radius | 29 | 0.187944 +- 0.033272 | 0.187944 +- 0.033272 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.D_radius.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | D tibia | 29 | 0.208172 +- 0.017343 | 0.208172 +- 0.017343 | 0 | 0 | [0, 0] | 0 | 0 | 29/29 | 1 | 0 | 1 | 1 | metrics.A.site.D_tibia.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | diaphyseal | 58 | 0.198058 +- 0.028207 | 0.198058 +- 0.028207 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.A.site.diaphyseal.TbTh_as_delivered.* |
| Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG) | ultradistal | 58 | 0.21231 +- 0.012652 | 0.21231 +- 0.012652 | 0 | 0 | [0, 0] | 0 | 0 | 58/58 | 1 | 0 | 1 | 1 | metrics.A.site.ultradistal.TbTh_as_delivered.* |
| Tb.Th (whole SEG; internal) | pooled | 137 | 0.208581 +- 0.02279 | 0.217067 +- 0.021355 | 0.008486 | 0.010346 | [-0.011792, 0.028764] | 0.06016 | 33.9105 | 21/137 | 0.835963 | 0.042701 | 0.7959237 | 0.8296731 | metrics.A.pooled.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | patella | 21 | 0.227345 +- 0.008764 | 0.227345 +- 0.008764 | 0 | 0 | [0, 0] | 0 | 0 | 21/21 | 1 | 0 | 1 | 1 | metrics.A.patella.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | radius/tibia | 116 | 0.205184 +- 0.022911 | 0.215206 +- 0.022431 | 0.010022 | 0.010538 | [-0.010633, 0.030677] | 0.06016 | 33.9105 | 0/116 | 0.87348 | 0.035982 | 0.7959843 | 0.8132785 | metrics.A.oslh.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | UD radius | 29 | 0.205624 +- 0.008936 | 0.208516 +- 0.009195 | 0.002892 | 7.044e-04 | [0.001512, 0.004273] | 0.004898 | 2.2962 | 0/29 | 1.026261 | -0.002508 | 0.9947817 | 0.9488 | metrics.A.site.UD_radius.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | UD tibia | 29 | 0.218996 +- 0.012385 | 0.221786 +- 0.012479 | 0.00279 | 6.742e-04 | [0.001468, 0.004111] | 0.004352 | 1.9428 | 0/29 | 1.006205 | 0.001431 | 0.9971189 | 0.9740508 | metrics.A.site.UD_tibia.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | D radius | 29 | 0.187944 +- 0.033272 | 0.211135 +- 0.037329 | 0.023191 | 0.012312 | [-9.403e-04, 0.047323] | 0.06016 | 33.9105 | 0/29 | 1.0609 | 0.011745 | 0.8941637 | 0.7744268 | metrics.A.site.D_radius.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | D tibia | 29 | 0.208172 +- 0.017343 | 0.219388 +- 0.017501 | 0.011215 | 0.003957 | [0.00346, 0.018971] | 0.022176 | 10.8816 | 0/29 | 0.983125 | 0.014728 | 0.9491594 | 0.8075923 | metrics.A.site.D_tibia.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | diaphyseal | 58 | 0.198058 +- 0.028207 | 0.215261 +- 0.029194 | 0.017203 | 0.010892 | [-0.004145, 0.038552] | 0.06016 | 33.9105 | 0/58 | 0.961036 | 0.02492 | 0.8622158 | 0.7875503 | metrics.A.site.diaphyseal.TbTh_wholeseg.* |
| Tb.Th (whole SEG; internal) | ultradistal | 58 | 0.21231 +- 0.012652 | 0.215151 +- 0.01276 | 0.002841 | 6.854e-04 | [0.001498, 0.004184] | 0.004898 | 2.2962 | 0/58 | 1.007182 | 0.001316 | 0.9971658 | 0.9742152 | metrics.A.site.ultradistal.TbTh_wholeseg.* |

Per-scan pairs for the figures: `facts.json` -> `tables.metrics.A.per_scan`, `tables.metrics.B.per_scan` (137 rows each, columns `<metric>_ipl` / `<metric>_ours`, `cohort`, `study`).

- Configuration A BV/TV, patella (n = 21): ipldt's ratio on IPL's SEG and IPL's rendered trabecular contour equals the ratio taken on the same files by the B run's IPL side on 21/21 (BV 124,775,445 / TV 361,536,565 voxels pooled), so the configuration-A pair is identical by construction, as for the radius/tibia. IPL's printed sheet BV/TV (three decimals) exists for 21/21 and equals the rounded ratio on 18/21 (configuration B's ratio: 18/21); max |ratio - printed| 5.580e-04 (rounding bound 0.0005), the 3 differing ratios lie within 5.799e-05 of a rounding boundary. Table footnote: IPL's BV/TV is the same ratio taken on IPL's own files; Script 32 writes no BV/TV and the sheet prints three decimals (`A.patella.BVTV.*`; per scan `tables.A.patella.BVTV.per_scan`).

## 4. Timings (GPU backend in every record) and memory

| stage | median s | range s | key |
|---|---|---|---|
| radius/tibia: AIM read | 0.7 | 0.1 - 2.1 | B.oslh.time_s.read.* |
| radius/tibia: STEP 1 compartment separation | 12 | 1.6 - 32.6 | B.oslh.time_s.step1.* |
| radius/tibia: contour rendering (both compartments) | 4.2 | 0.6 - 11.3 | B.oslh.time_s.render_B.* |
| radius/tibia: Laplace-Hamming filter | 1.9 | 0.4 - 9.6 | B.oslh.time_s.lh.* |
| radius/tibia: SEG assembly | 2.9 | 0.4 - 7.1 | B.oslh.time_s.seg.* |
| radius/tibia: dt, five B maps | 28 | 4.2 - 69.8 | B.oslh.time_s.dt_B.* |
| radius/tibia: dt Tb.Th map (B) | 1.6 | 0.2 - 6.1 | B.oslh.map.TbTh_trabseg.time_s.* |
| radius/tibia: dt Tb.Sp map (B) | 4.8 | 0.4 - 14.2 | B.oslh.map.TbSp.time_s.* |
| radius/tibia: dt 1/Tb.N map (B) | 13.7 | 2.1 - 34.8 | B.oslh.map.TbN.time_s.* |
| radius/tibia: dt Ct.Th map (B) | 2.5 | 0.6 - 6.3 | B.oslh.map.CtTh.time_s.* |
| radius/tibia: dt, five A maps | 27.1 | 4.3 - 68.3 | A.oslh.time_s.dt_all_maps.* |
| radius/tibia: validation record total (A + B + comparisons) | 79.5 | 11.9 - 202.2 | B.oslh.time_s.total.* |
| radius/tibia: validation record total without the pore cascade | 74.6 | 11.3 - 188.3 | B.oslh.time_s.total_excl_porosity.* |
| radius/tibia: pore cascade, both configurations (116 records) | 4.9 | 0.6 - 13.9 | B.oslh.time_s.porosity.* |
| patella: AIM read | 0.3 | 0.2 - 0.8 | B.patella.time_s.read.* |
| patella: STEP 1 | 9.1 | 4.4 - 24.4 | B.patella.time_s.step1.* |
| patella: contour rendering | 4.1 | 1.6 - 10.5 | B.patella.time_s.render.* |
| patella: Laplace-Hamming + SEG | 5.6 | 3 - 13.7 | B.patella.time_s.seg.* |
| patella: read + STEP 1 + rendering + SEG | 18.9 | 9.3 - 49.5 | B.patella.time_s.total.* |
| patella: dt, five maps | 36.2 | 21.2 - 101.9 | B.patella.time_s.dt_run.dt.* |
| patella: dt Tb.Th map (A) | 4.4 | 1.7 - 11.5 | A.patella.map.TbTh_trabseg.time_s.* |
| patella: dt Tb.Sp map (A) | 8.4 | 4.3 - 30.2 | A.patella.map.TbSp.time_s.* |
| patella: dt 1/Tb.N map (A) | 27.7 | 17.3 - 90.7 | A.patella.map.TbN.time_s.* |
| patella: dt Ct.Th map (A) | 4.8 | 3.1 - 11.9 | A.patella.map.CtTh.time_s.* |

Radius/tibia medians per site group (s; the pooled medians above move with the site mix, because a diaphyseal greyscale is smaller than an ultradistal one):

| site group | grey voxels (median) | read | STEP 1 | rendering | LH filter | SEG | dt, five B maps | 1/Tb.N map | key |
|---|---|---|---|---|---|---|---|---|---|
| UD radius | 36,835,680 | 0.9 | 15.1 | 5.2 | 3.9 | 3.3 | 31.9 | 15.9 | B.site.UD_radius.time_s.* |
| UD tibia | 56,058,912 | 1.4 | 22.8 | 8.1 | 8 | 5.1 | 49.4 | 24.9 | B.site.UD_tibia.time_s.* |
| D radius | 8,415,792 | 0.2 | 2.7 | 1 | 0.5 | 0.7 | 7 | 3.6 | B.site.D_radius.time_s.* |
| D tibia | 28,566,720 | 0.5 | 9.5 | 3.6 | 1.9 | 2.7 | 24.8 | 12.4 | B.site.D_tibia.time_s.* |
| diaphyseal | 14,806,848 | 0.3 | 5.3 | 1.9 | 1.8 | 1.5 | 13.8 | 6.8 | B.site.diaphyseal.time_s.* |
| ultradistal | 45,211,320 | 1.1 | 18.1 | 6.5 | 4 | 3.9 | 39.1 | 19.5 | B.site.ultradistal.time_s.* |

- Peak resident memory of the radius/tibia validation worker: 10.35 GB; per validation run oslh_auto_v4 10.35 GB, oslh_noedit_v3 4.69 GB (`B.oslh.peak_rss_gb.*`; validation process, not the pipeline alone; each record carries its worker's running peak).
- GPU == CPU: tests/test_gpu_cpu.py (26 tests: maps, centres, reports identical for dt_thickness / dt_spacing / dt_number, versions 1-3, with and without gobj, assign_epsilon sweep; ridge_gpu == ridge, draw_spheres_gpu == CPU) and tests/test_ridge_precision.py (GPU bit identity of the ridge test) (`software.gpu_cpu_identity.evidence`). Test suite: 317 tests collected, measured 2026-09-25 (`software.tests.collected_total`).

## 5. IPL parameter values the package uses

| key | value | source |
|---|---|---|
| params.step1.sigma | 2.0 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.support | 3 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.lower_mgha | 500.0 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.upper_mgha | 3000.0 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.peel0 | 6 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.erode | 3 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.dilate | 3 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.close1 | 15 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.open_ | 15 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.corner_erode | 3 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.corner_min_tibia | 200000 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.corner_min_radius | 800 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.corner_dilate | 3 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.corner_max | 500000 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.close2_tibia | 50 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.close2_radius | 30 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.slicewise_lo_pct | 50.0 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.slicewise_up_pct | 100.0 | ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30) |
| params.step1.metric | IPL chamfer metric-11 (3-4-5 type) morphology with the 3N + 2 threshold and IPL's boundary rules; 6-connected components; slicewise 4-connected extraction | ipldt/ipl_ops.py docstrings; ipldt/ormir.py step3_trab_cort_seg docstring |
| params.step1.n_ipl_commands | 36 | ipldt/step1.py module docstring (36 IPL commands, 30 volume stages 00..29) |
| params.contour.command | /togobj_from_aim -curvature_smooth 1 followed by /gobj_to_aim (rasterisation); slice-wise chain smoothing, 4-connected fill | ipldt/contour/render.py, smooth.py |
| params.lh.lp_cut_off_freq | 0.3 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.laplace_eps | 0.45 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.hamming_amp | 1.0 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.window | Hann (amp 1: (1 - A/2) + (A/2) cos) | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.norm_max | 200000.0 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.short_max | 32767 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.threshold_permille_lower | 475 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.threshold_permille_upper | 1000 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.threshold_native_lower | 15564 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.threshold_native_upper | 32767 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.cc_min_voxels_cort | 35 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.cc_min_voxels_trab | 70 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.seg_value_cort | 127 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.seg_value_trab | 126 | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.border | 1-voxel duplicated border (bounding_box_cut -border 1, offset_add, fill_offset_duplicate) | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.pad_rule | next power of two per axis, edge-exclusive mirror ('reflect'); when the pad is odd the extra voxel goes BEFORE the data (LH_PAD_OFFSET = 'ceil') | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.fft_dtype | float32 (LH_DTYPE; float64 opt-in) | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.short_conversion | trunc(float32(f) * float32(32767 / 200000)) after clipping | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.lh.cut_off_physical | 0.3 / el_z cycles per mm (per-axis header element sizes) | ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE) |
| params.dt.ridge_epsilon | 0.9 | ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring |
| params.dt.assign_epsilon | 0.5 | ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring |
| params.dt.peel_iter | -1 | ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring |
| params.dt.version | 3 | ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring |
| params.dt.suppress_boundary | 2 | ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring |
| params.dt.ridge_tolerance | 1e-09 | ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring |
| params.dt.surface_distance | s = |p(v)|, p(u) = sign(u) max(|u| - 1/2, 0), exact integer 4 s^2 in int64, one float64 sqrt | ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring |
| params.dt.ridge_rule | x kept unless a 26-neighbour y inside the image has |x - y| + s_x - s_y <= ridge_epsilon | ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring |
| params.dt.diameter_v3 | R = floor(D + 1/2) with D = |P1 - P2| when the pinned midpoint |M| < 1, else floor(2 s_x + 1/2) | ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring |
| params.dt.drawing | every voxel with |c - x| <= D/2 + assign_epsilon takes the largest D (atomicMax on the GPU) | ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring |
| params.dt.field | Danielsson-type vector distance transform, slice-interleaved sweep (ipldt.field.sir_quad); outside the image is object | ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring |
| params.dt.tb_n | Tb.N = 1 / mean of the 1/Tb.N map (dt_number) inside the trabecular contour | ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring |
| params.dt.tbth_object.shipped_default | whole SEG (label 127 + 126) with the trabecular contour as gobj -- ipldt.ormir.ipl_morphometry key TRAB_TH, called by ormir_bqrl.stages.morphometry | ipldt/ormir.py ipl_morphometry; ormir_bqrl/stages.py morphometry |
| params.dt.tbth_object.script32 | TRAB_SEG (label 126, components >= 70 voxels, cropped to the trabecular contour) with the trabecular contour as gobj | ipldt/ormir.py ipl_morphometry docstring; validation summaries 'Tb.Th (TRAB_SEG, Scripts 32/33/34)' |
| params.dt.ctth_object | CORT_MASK raster (the cortical compartment) with the cortical contour as gobj | ipldt/ormir.py ipl_morphometry |
| params.autocontour | ORMIR-XCT 1.1.0 ormir_xct.core.segmentation.autocontour.autocontour (AutocontourKnee.get_periosteal_mask; Buie 2007 dual threshold); called with the AIM's own mu_water, rescale slope and intercept | ipldt/ormir.py step2_autocontour; installed ormir_xct 1.1.0 |
| params.bmd | ORMIR-XCT bmd_masked (HU -> mgHA/cm3 with the AIM calibration) inside the rendered trabecular and cortical contours | ipldt/ormir.py step5_bmd |
| params.calibration | density slope, intercept and mu_scaling read from the AIM processing log (ITK ScancoImageIO header as fallback); /seg_gauss thresholds 500 / 3000 mgHA -> native units per scan (e.g. 4524 / 17173 or 4507 / 17095 on the patellae) | ipldt/ormir.py step1_calibration; records step1.thresholds |

## 6. Software facts

| key | value | source |
|---|---|---|
| software.ipldt.version | 1.0.0 | ipldt/__init__.py (__version__); pyproject.toml |
| software.ormir_bqrl.version | 0.1.0 | ormir_bqrl/__init__.py (__version__); README.md |
| software.ormir_xct.version | 1.1.0 | installed package in the ormir env (ormir_xct.__version__) |
| software.tests.collected_total | 317 | pytest --collect-only -q tests (2026-09-25) |
| software.tests.gpu_cpu_identity | 26 | pytest --collect-only tests/test_gpu_cpu.py (26 of the 43 collected together with test_ridge_precision.py's 17; 2026-09-18) |
| software.tests.ridge_precision | 17 | pytest --collect-only tests/test_ridge_precision.py (2026-09-18) |
| software.gpu_cpu_identity.evidence | tests/test_gpu_cpu.py (26 tests: maps, centres, reports identical for dt_thickness / dt_spacing / dt_number, versions 1-3, with and without gobj, assign_epsilon sweep; ridge_gpu == ridge, draw_spheres_gpu == CPU) and tests/test_ridge_precision.py (GPU bit identity of the ridge test) | tests/ |
| software.backends | GPU: CuPy (ridge test and sphere drawing kernels); CPU: numpy + numba (the vector-distance sweep is numba on both) | ipldt/gpu.py, ipldt/field.py |
| software.records_backend | gpu | dt_backend of all 116 radius/tibia and 21 patella-B records; backend of all 105 patella-A records |
| software.lh_engine_state | pad offset 'ceil' + float32 (the shipped engine) in every configuration-B number of this sheet (B.patella.*, B.oslh.*, metrics.B.*) | records variants.lh_pad_offset / seg.lh_pad_offset |
| software.outputs | report JSON/CSV; masks and maps as NIfTI or AIM; Slicer-ready labelmaps (.seg.nrrd); manual-correction re-entry (ormir_bqrl.redo, Script-34 REDO semantics) | README.md; ipldt/ormir.py run_pipeline and ormir_bqrl/stages.py write_volumes (NIfTI or AIM: --map-format of ipldt-pipeline and ormir-bqrl, the pore map included; ORMIR-BQRL adds the Slicer files and the redo) |

### 6b. Execution environment and packaging (measured 2026-09-18 in the validation environment, not from the records)

Measured in the `ormir` conda env on the validation workstation (the only environment the pipeline runs in); the records store `dt_backend = gpu` but no library versions.

| key | value | source |
|---|---|---|
| software.environment.python | 3.11.15 | sys.version |
| software.environment.numpy | 2.3.5 | numpy.__version__ |
| software.environment.scipy | 1.15.3 | scipy.__version__ |
| software.environment.numba | 0.66.0 | numba.__version__ |
| software.environment.cupy | 14.2.0 (cupy-cuda12x) | cupy.__version__ |
| software.environment.cuda_runtime | 12.9 | cupy.cuda.runtime.runtimeGetVersion() = 12090 |
| software.environment.itk | 5.4.6 | itk.__version__ |
| software.environment.simpleitk | 2.5.5 | SimpleITK.__version__ |
| software.environment.gpu | NVIDIA GeForce RTX 4090 | cupy device properties |
| software.environment.gpu_memory_gb | 24 | totalGlobalMem 25.76e9 bytes |
| software.environment.cpu | AMD Ryzen Threadripper 9960X | Win32_Processor |
| software.environment.cpu_cores | 24 (48 logical processors) | Win32_Processor |
| software.environment.ram_gb | 96 | Win32_ComputerSystem TotalPhysicalMemory 95.3 GiB |
| software.environment.os | Windows 11 Pro | Win32_OperatingSystem |
| software.packaging.requires_python | >=3.10 | pyproject.toml requires-python |
| software.packaging.license | MIT | pyproject.toml classifiers; LICENSE |
| software.packaging.dependencies | numpy>=1.24, scipy>=1.10, numba>=0.58; extras gpu (cupy-cuda12x), io (itk, itk-ioscanco, SimpleITK), ormir (ormir-xct), bqrl (matplotlib>=3.7) | pyproject.toml |

## 9. Headline numbers for the abstract (with keys)

- n = 137 XtremeCT II scans: 21 patellae, 29 ultradistal radii, 29 ultradistal tibiae, 29 diaphyseal radii, 29 diaphyseal tibiae; IPL V5.42, 60.7 um  [cohort.*]
- Participants: 16 (patellae) + 9 (CKD) + 20 (REPRO) + 1 (BMAT) = 46  [cohort.patella.participants, cohort.oslh.study.*.participants]
- Configuration A (IPL's SEG in): 0 differing voxel(s) in 18,348,991,192 compared over the four maps of 137 scans; Tb.Sp, 1/Tb.N and Ct.Th maps identical on 137/137/137 of 137, Tb.Th (TRAB_SEG) on 137/137  [A.pooled.mismatches.four_maps, A.pooled.voxels_compared.four_maps, A.pooled.map.*.exact_scans]
- Configuration A, BV/TV, n = 137: identical values on 137/137 (the ratio on IPL's SEG and rendered contour; patella sheet values equal the rounded ratio on 18/21)  [metrics.A.pooled.BVTV.*, A.patella.BVTV.printed.*]
- Configuration A, patella: 0 differing voxels on all 5 maps x 21 subjects, 4,943,555,456 voxels compared  [A.patella.maps.all_exact, A.patella.voxels_compared.total_five_maps]
- Configuration B: compartment masks / contour renderings identical on 137/137 (cortical) and 137/137 (trabecular) scans (patella STEP-1 masks 21/21 and 21/21; radius/tibia renderings 116/116 and 116/116)  [B.patella.step1.*, B.oslh.rendering.*]
- Configuration B, Laplace-Hamming SEG pooled: 91,763 differing voxels in 4,644,570,784 (19.76 per million), exact on 37/137, min Dice 0.99843367, median Dice 0.99999988, Dice >= 0.9999 on 133/137; patella 50 voxels (23 ours / 27 IPL), radius/tibia 91,713 of which 82,462 in three scans  [B.pooled.SEG.*, B.patella.SEG.*, B.oslh.SEG.*]
- Configuration B, Ct.Th map identical on 137/137 (4,672,408,888 voxels); Tb.Th (TRAB_SEG) / Tb.Sp / 1/Tb.N maps differ by 20,911 / 14,347 / 56,731 voxels over 137 scans  [B.pooled.map.*]
- Configuration B, BV/TV, n = 137: slope 1.000013, R^2 1, ICC(2,1) 1, bias 1.895e-06 -, 95 % LoA [-2.722e-05, 3.101e-05], max |rel| 0.0506 %, exact 85/137  [metrics.B.pooled.BVTV.*]
- Configuration B, Tb.Th (TRAB_SEG), n = 137: slope 1.000038, R^2 0.9999998, ICC(2,1) 0.9999999, bias 1.306e-06 mm, 95 % LoA [-1.716e-05, 1.977e-05], max |rel| 0.0405 %, exact 81/137  [metrics.B.pooled.TbTh_trabseg.*]
- Configuration B, Tb.Sp, n = 137: slope 1, R^2 1, ICC(2,1) 1, bias 1.471e-08 mm, 95 % LoA [-7.080e-06, 7.110e-06], max |rel| 0.0038 %, exact 73/137  [metrics.B.pooled.TbSp.*]
- Configuration B, Tb.N, n = 137: slope 1, R^2 1, ICC(2,1) 1, bias 1.003e-06 1/mm, 95 % LoA [-2.013e-05, 2.214e-05], max |rel| 0.0088 %, exact 73/137  [metrics.B.pooled.TbN.*]
- Configuration B, Ct.Th, n = 137: identical values on 137/137 (bias 0, max |rel| 0 %)  [metrics.B.pooled.CtTh.*]
