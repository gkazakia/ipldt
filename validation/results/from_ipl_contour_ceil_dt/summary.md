# ipldt from IPL's periosteal contour vs Scanco IPL V5.42: cohort summary

21 subjects; site preset 'tibia' (Step1Params corner_min 200,000, close2 50, peel0 6); dt parameters {'ridge_epsilon': 0.9, 'assign_epsilon': 0.5, 'peel_iter': -1, 'version': 3, 'suppress_boundary': 2}; voxel size 0.0607 mm; backend gpu; ipldt 1.0.0; 2026-09-18 14:13.

Input per subject: IPL's periosteal contour (raw CORT_MASK | TRAB_MASK = IPL's /gobj_to_aim rendering) and the native greyscale; every later stage is ipldt's and is compared with the file IPL wrote, by global voxel position.

## Exactness per stage (subjects with 0 mismatching voxels) and cohort totals

| stage | subjects exact | of | mismatching voxels (cohort) | ipldt only | IPL only | min Dice | subjects with identical grids |
|---|---|---|---|---|---|---|---|
| CORT_MASK (step 1) | 21 | 21 | 0 | 0 | 0 | 1.000000 | 21 |
| TRAB_MASK (step 1) | 21 | 21 | 0 | 0 | 0 | 1.000000 | 21 |
| cort contour | 21 | 21 | 0 | 0 | 0 | 1.000000 | 21 |
| trab contour | 21 | 21 | 0 | 0 | 0 | 1.000000 | 21 |
| SEG | 2 | 21 | 50 | 23 | 27 | 1.000000 | 21 |
| TRAB_SEG | 6 | 21 | 33 | 13 | 20 | 1.000000 | 19 |
| CORT_SEG | 9 | 21 | 17 | 10 | 7 | 1.000000 | 19 |
| TRAB_TH map | 6 | 21 | 167 | 17 | 28 | - | 21 |
| TRAB_SP map | 6 | 21 | 199 | 32 | 27 | - | 21 |
| TRAB_1N map | 8 | 21 | 1,685 | 103 | 52 | - | 21 |
| CORT_TH map | 21 | 21 | 0 | 0 | 0 | - | 21 |
| TRAB_TH_old map | 6 | 21 | 168 | 17 | 28 | - | 11 |

Tb.Th is reported under BOTH definitions.  **Tb.Th (whole SEG, as ORMIR-BQRL ships)** is dt_thickness of the whole SEG cropped to the trabecular gobj, what the shipped pipeline computes, compared with IPL's delivered file `<base>_TRAB_TH`; it is the one reported first here.  **Tb.Th (TRAB_SEG, Scripts 32/33/34)** is what Scanco's evaluation scripts compute -- /dt_thickness on TRAB_SEG (IPL_FNAME5) cropped to the same gobj -- and it is compared with IPL's file `<base>_TRAB_TH_old`.  NOTE the trap: for this cohort IPL's filenames are the reverse of the intuitive reading -- TRAB_TH_old is the Script-32 (TRAB_SEG) map and TRAB_TH is the whole-SEG one.  The record / CSV keys stay `TbTh_old` (TRAB_SEG) and `TbTh` (whole SEG) for continuity.

## Metrics (ours - IPL over the cohort; IPL values derived from IPL's maps as in validation/results/sample_means.csv)

| metric | unit | IPL mean | ipldt mean | mean diff | mean |diff| | max |diff| | mean |rel| % | max |rel| % | subjects exact | of |
|---|---|---|---|---|---|---|---|---|---|---|
| BV/TV | - | 0.355964 | 0.355964 | -1.342e-08 | 4.328e-08 | 1.934e-07 | 0.00001 | 0.00005 | 11 | 21 |
| Tb.Th (whole SEG, as ORMIR-BQRL ships) | mm | 0.227345 | 0.227345 | +2.479e-08 | 5.674e-08 | 3.941e-07 | 0.00003 | 0.00017 | 6 | 21 |
| Tb.Sp | mm | 0.359076 | 0.359076 | -1.999e-08 | 2.812e-08 | 1.665e-07 | 0.00001 | 0.00005 | 6 | 21 |
| Tb.N | 1/mm | 2.389915 | 2.389916 | +2.989e-07 | 1.429e-06 | 8.887e-06 | 0.00006 | 0.00033 | 8 | 21 |
| Ct.Th | mm | 3.104652 | 3.104652 | +0.000e+00 | 0.000e+00 | 0.000e+00 | 0.00000 | 0.00000 | 21 | 21 |
| Tb.Th (TRAB_SEG, Scripts 32/33/34) | mm | 0.224255 | 0.224255 | +2.509e-08 | 5.576e-08 | 3.920e-07 | 0.00002 | 0.00017 | 7 | 21 |

IPL column check: max |IPL value here - sample_means.csv| = 0.000e+00 over 105 values (same formula, same maps; identical).

## Per-subject overview

| subject | base | CORT_MASK mism. | TRAB_MASK mism. | SEG mism. | + | - | TRAB_SEG mism. | CORT_SEG mism. | SEG Dice | TRAB_TH mism. | TRAB_SP mism. | TRAB_1N mism. | CORT_TH mism. | TRAB_TH_old mism. | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PFJ-ab6af9_R | X3931708 | 0 | 0 | 5 | 1 | 4 | 4 | 1 | 1.000000 | 10 | 21 | 114 | 0 | 10 | 151.4 |
| PFJ-0be66a_R | X2420448 | 0 | 0 | 2 | 1 | 1 | 2 | 0 | 1.000000 | 13 | 15 | 52 | 0 | 13 | 56.8 |
| PFJ-351dc7_R | X9463122 | 0 | 0 | 3 | 0 | 3 | 2 | 1 | 1.000000 | 3 | 3 | 19 | 0 | 3 | 124.8 |
| PFJ-8bcf88_L | X4121991 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 50.1 |
| PFJ-8bcf88_R | X4891377 | 0 | 0 | 2 | 1 | 1 | 1 | 1 | 1.000000 | 14 | 1 | 225 | 0 | 14 | 44.6 |
| PFJ-d81140_L | X5366335 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 54.3 |
| PFJ-d81140_R | X1130319 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 57.4 |
| PFJ-0fb201_R | X5578364 | 0 | 0 | 2 | 1 | 1 | 1 | 1 | 1.000000 | 1 | 2 | 0 | 0 | 1 | 57.3 |
| PFJ-a5ee4c_R | X3744142 | 0 | 0 | 3 | 2 | 1 | 2 | 1 | 1.000000 | 14 | 4 | 242 | 0 | 14 | 58.6 |
| PFJ-5f00b4_R | X5974634 | 0 | 0 | 5 | 2 | 3 | 2 | 3 | 1.000000 | 3 | 7 | 24 | 0 | 3 | 48.7 |
| PFJ-a91200_L | X3186488 | 0 | 0 | 3 | 1 | 2 | 3 | 0 | 1.000000 | 27 | 13 | 46 | 0 | 27 | 55.1 |
| PFJ-b5d5d9_L | X7775442 | 0 | 0 | 2 | 1 | 1 | 2 | 0 | 1.000000 | 6 | 11 | 44 | 0 | 2 | 43.4 |
| PFJ-42293d_L | X3623103 | 0 | 0 | 3 | 3 | 0 | 1 | 2 | 1.000000 | 11 | 1 | 126 | 0 | 11 | 64.8 |
| PFJ-a56aae_R | X5606590 | 0 | 0 | 3 | 0 | 3 | 3 | 0 | 1.000000 | 4 | 47 | 221 | 0 | 4 | 54.3 |
| PFJ-6f5538_R | X5492058 | 0 | 0 | 2 | 2 | 0 | 0 | 2 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 30.4 |
| PFJ-6b714e_L | X5484780 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 40.3 |
| PFJ-6b714e_R | X2663245 | 0 | 0 | 2 | 1 | 1 | 0 | 2 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 39.1 |
| PFJ-69bcb0_L | X1183449 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1.000000 | 1 | 3 | 0 | 0 | 1 | 74.5 |
| PFJ-69bcb0_R | X1346001 | 0 | 0 | 5 | 2 | 3 | 4 | 1 | 1.000000 | 55 | 66 | 538 | 0 | 55 | 76.4 |
| PFJ-411dfd_L | X3334670 | 0 | 0 | 2 | 1 | 1 | 2 | 0 | 1.000000 | 2 | 1 | 13 | 0 | 2 | 58.7 |
| PFJ-411dfd_R | X5143651 | 0 | 0 | 3 | 3 | 0 | 3 | 0 | 1.000000 | 3 | 4 | 21 | 0 | 8 | 54.1 |

## Per-subject metrics (ours | IPL | diff)

| subject | BV/TV ours | IPL | diff | Tb.Th (whole SEG, as ORMIR-BQRL ships) ours | IPL | diff | Tb.Sp ours | IPL | diff | Tb.N ours | IPL | diff | Ct.Th ours | IPL | diff | Tb.Th (TRAB_SEG, Scripts 32/33/34) ours | IPL | diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PFJ-ab6af9_R | 0.317621 | 0.317621 | -5.56e-08 | 0.220999 | 0.220999 | -2.56e-08 | 0.397112 | 0.397112 | -3.92e-08 | 2.164763 | 2.164762 | +6.05e-07 | 2.290532 | 2.290532 | +0.00e+00 | 0.217589 | 0.217589 | -2.61e-08 |
| PFJ-0be66a_R | 0.299471 | 0.299471 | +0.00e+00 | 0.227346 | 0.227346 | -1.22e-07 | 0.438002 | 0.438002 | -7.87e-09 | 2.036697 | 2.036697 | +2.78e-07 | 2.324955 | 2.324955 | +0.00e+00 | 0.224490 | 0.224490 | -1.22e-07 |
| PFJ-351dc7_R | 0.422863 | 0.422863 | -1.20e-07 | 0.237256 | 0.237256 | -1.61e-09 | 0.307749 | 0.307749 | -2.65e-08 | 2.561965 | 2.561965 | -1.39e-07 | 4.354225 | 4.354225 | +0.00e+00 | 0.233691 | 0.233691 | -2.64e-09 |
| PFJ-8bcf88_L | 0.357506 | 0.357506 | +0.00e+00 | 0.230923 | 0.230923 | +0.00e+00 | 0.339319 | 0.339319 | +0.00e+00 | 2.498224 | 2.498224 | +0.00e+00 | 2.917932 | 2.917932 | +0.00e+00 | 0.227566 | 0.227566 | +0.00e+00 |
| PFJ-8bcf88_R | 0.418801 | 0.418801 | +1.00e-07 | 0.244156 | 0.244156 | -1.39e-08 | 0.302201 | 0.302201 | +3.20e-08 | 2.678653 | 2.678644 | +8.89e-06 | 2.599435 | 2.599435 | +0.00e+00 | 0.239655 | 0.239655 | -1.71e-08 |
| PFJ-d81140_L | 0.334406 | 0.334406 | +0.00e+00 | 0.226428 | 0.226428 | +0.00e+00 | 0.420317 | 0.420317 | +0.00e+00 | 2.075448 | 2.075448 | +0.00e+00 | 2.923350 | 2.923350 | +0.00e+00 | 0.223671 | 0.223671 | +0.00e+00 |
| PFJ-d81140_R | 0.347787 | 0.347787 | +0.00e+00 | 0.229514 | 0.229514 | +0.00e+00 | 0.372337 | 0.372337 | +0.00e+00 | 2.303861 | 2.303861 | +0.00e+00 | 3.288247 | 3.288247 | +0.00e+00 | 0.226977 | 0.226977 | +0.00e+00 |
| PFJ-0fb201_R | 0.358343 | 0.358343 | -5.63e-08 | 0.222519 | 0.222519 | +6.49e-09 | 0.354539 | 0.354539 | -4.62e-09 | 2.370517 | 2.370517 | +0.00e+00 | 3.313352 | 3.313352 | +0.00e+00 | 0.220196 | 0.220196 | +6.09e-09 |
| PFJ-a5ee4c_R | 0.346567 | 0.346567 | +0.00e+00 | 0.213223 | 0.213223 | -1.46e-07 | 0.325515 | 0.325515 | +2.78e-08 | 2.559277 | 2.559280 | -3.46e-06 | 3.809212 | 3.809212 | +0.00e+00 | 0.211119 | 0.211119 | -1.45e-07 |
| PFJ-5f00b4_R | 0.412416 | 0.412416 | +0.00e+00 | 0.225512 | 0.225512 | +2.40e-08 | 0.277468 | 0.277468 | +1.97e-08 | 2.795134 | 2.795136 | -1.49e-06 | 3.595071 | 3.595071 | +0.00e+00 | 0.222381 | 0.222381 | +2.32e-08 |
| PFJ-a91200_L | 0.289015 | 0.289015 | -4.84e-08 | 0.220796 | 0.220796 | +2.49e-07 | 0.410249 | 0.410249 | -1.08e-07 | 2.194343 | 2.194342 | +8.88e-07 | 1.996168 | 1.996168 | +0.00e+00 | 0.218095 | 0.218095 | +2.48e-07 |
| PFJ-b5d5d9_L | 0.286567 | 0.286567 | +0.00e+00 | 0.218416 | 0.218416 | +3.36e-08 | 0.442654 | 0.442654 | -3.49e-08 | 2.063126 | 2.063125 | +1.17e-06 | 2.154060 | 2.154060 | +0.00e+00 | 0.214905 | 0.214905 | +0.00e+00 |
| PFJ-42293d_L | 0.350670 | 0.350670 | +4.67e-08 | 0.240616 | 0.240616 | +7.47e-08 | 0.384009 | 0.384009 | +5.96e-09 | 2.261839 | 2.261837 | +2.21e-06 | 3.137596 | 3.137596 | +0.00e+00 | 0.237481 | 0.237481 | +7.48e-08 |
| PFJ-a56aae_R | 0.370069 | 0.370069 | -1.93e-07 | 0.225314 | 0.225314 | +7.39e-08 | 0.332767 | 0.332767 | -1.66e-07 | 2.523181 | 2.523187 | -6.78e-06 | 2.895872 | 2.895872 | +0.00e+00 | 0.222796 | 0.222796 | +7.18e-08 |
| PFJ-6f5538_R | 0.439328 | 0.439328 | +0.00e+00 | 0.236644 | 0.236644 | +0.00e+00 | 0.277527 | 0.277527 | +0.00e+00 | 2.785962 | 2.785962 | +0.00e+00 | 6.378689 | 6.378689 | +0.00e+00 | 0.230468 | 0.230468 | +0.00e+00 |
| PFJ-6b714e_L | 0.377558 | 0.377558 | +0.00e+00 | 0.223286 | 0.223286 | +0.00e+00 | 0.303633 | 0.303633 | +0.00e+00 | 2.627299 | 2.627299 | +0.00e+00 | 2.981554 | 2.981554 | +0.00e+00 | 0.220136 | 0.220136 | +0.00e+00 |
| PFJ-6b714e_R | 0.387870 | 0.387870 | +0.00e+00 | 0.220562 | 0.220562 | +0.00e+00 | 0.284955 | 0.284955 | +0.00e+00 | 2.767578 | 2.767578 | +0.00e+00 | 2.301266 | 2.301266 | +0.00e+00 | 0.217452 | 0.217452 | +0.00e+00 |
| PFJ-69bcb0_L | 0.349265 | 0.349265 | -4.52e-08 | 0.241655 | 0.241655 | -1.52e-10 | 0.394534 | 0.394534 | -4.14e-08 | 2.216197 | 2.216197 | +0.00e+00 | 3.720585 | 3.720585 | +0.00e+00 | 0.238825 | 0.238825 | -5.24e-10 |
| PFJ-69bcb0_R | 0.316268 | 0.316268 | -7.61e-08 | 0.233263 | 0.233263 | +3.94e-07 | 0.490811 | 0.490811 | -5.67e-08 | 1.836392 | 1.836389 | +3.05e-06 | 3.093223 | 3.093223 | +0.00e+00 | 0.230643 | 0.230642 | +3.92e-07 |
| PFJ-411dfd_L | 0.343339 | 0.343339 | +0.00e+00 | 0.218525 | 0.218525 | -8.64e-09 | 0.348032 | 0.348032 | -1.70e-08 | 2.397223 | 2.397223 | +2.71e-08 | 2.836892 | 2.836892 | +0.00e+00 | 0.216428 | 0.216428 | -8.60e-09 |
| PFJ-411dfd_R | 0.349517 | 0.349517 | +1.67e-07 | 0.217286 | 0.217286 | -1.72e-08 | 0.336870 | 0.336870 | -2.40e-09 | 2.470549 | 2.470548 | +1.03e-06 | 2.285475 | 2.285475 | +0.00e+00 | 0.214792 | 0.214792 | +3.32e-08 |

## Subjects not exact at some stage

- PFJ-ab6af9_R (X3931708): SEG 5 (+1/-4), TRAB_SEG 4 (+1/-3), CORT_SEG 1 (+0/-1), TRAB_TH map 10, TRAB_SP map 21, TRAB_1N map 114, TRAB_TH_old map 10, BV/TV -5.56e-08, Tb.Th (whole SEG, as ORMIR-BQRL ships) -2.56e-08, Tb.Sp -3.92e-08, Tb.N +6.05e-07, Tb.Th (TRAB_SEG, Scripts 32/33/34) -2.61e-08
- PFJ-0be66a_R (X2420448): SEG 2 (+1/-1), TRAB_SEG 2 (+1/-1), TRAB_TH map 13, TRAB_SP map 15, TRAB_1N map 52, TRAB_TH_old map 13, Tb.Th (whole SEG, as ORMIR-BQRL ships) -1.22e-07, Tb.Sp -7.87e-09, Tb.N +2.78e-07, Tb.Th (TRAB_SEG, Scripts 32/33/34) -1.22e-07
- PFJ-351dc7_R (X9463122): SEG 3 (+0/-3), TRAB_SEG 2 (+0/-2), CORT_SEG 1 (+0/-1), TRAB_TH map 3, TRAB_SP map 3, TRAB_1N map 19, TRAB_TH_old map 3, BV/TV -1.20e-07, Tb.Th (whole SEG, as ORMIR-BQRL ships) -1.61e-09, Tb.Sp -2.65e-08, Tb.N -1.39e-07, Tb.Th (TRAB_SEG, Scripts 32/33/34) -2.64e-09
- PFJ-8bcf88_R (X4891377): SEG 2 (+1/-1), TRAB_SEG 1 (+1/-0), CORT_SEG 1 (+0/-1), TRAB_TH map 14, TRAB_SP map 1, TRAB_1N map 225, TRAB_TH_old map 14, BV/TV +1.00e-07, Tb.Th (whole SEG, as ORMIR-BQRL ships) -1.39e-08, Tb.Sp +3.20e-08, Tb.N +8.89e-06, Tb.Th (TRAB_SEG, Scripts 32/33/34) -1.71e-08
- PFJ-d81140_L (X5366335): SEG 1 (+1/-0), CORT_SEG 1 (+1/-0)
- PFJ-0fb201_R (X5578364): SEG 2 (+1/-1), TRAB_SEG 1 (+0/-1), CORT_SEG 1 (+1/-0), TRAB_TH map 1, TRAB_SP map 2, TRAB_TH_old map 1, BV/TV -5.63e-08, Tb.Th (whole SEG, as ORMIR-BQRL ships) +6.49e-09, Tb.Sp -4.62e-09, Tb.Th (TRAB_SEG, Scripts 32/33/34) +6.09e-09
- PFJ-a5ee4c_R (X3744142): SEG 3 (+2/-1), TRAB_SEG 2 (+1/-1), CORT_SEG 1 (+1/-0), TRAB_TH map 14, TRAB_SP map 4, TRAB_1N map 242, TRAB_TH_old map 14, Tb.Th (whole SEG, as ORMIR-BQRL ships) -1.46e-07, Tb.Sp +2.78e-08, Tb.N -3.46e-06, Tb.Th (TRAB_SEG, Scripts 32/33/34) -1.45e-07
- PFJ-5f00b4_R (X5974634): SEG 5 (+2/-3), TRAB_SEG 2 (+1/-1), CORT_SEG 3 (+1/-2), TRAB_TH map 3, TRAB_SP map 7, TRAB_1N map 24, TRAB_TH_old map 3, Tb.Th (whole SEG, as ORMIR-BQRL ships) +2.40e-08, Tb.Sp +1.97e-08, Tb.N -1.49e-06, Tb.Th (TRAB_SEG, Scripts 32/33/34) +2.32e-08
- PFJ-a91200_L (X3186488): SEG 3 (+1/-2), TRAB_SEG 3 (+1/-2), TRAB_TH map 27, TRAB_SP map 13, TRAB_1N map 46, TRAB_TH_old map 27, BV/TV -4.84e-08, Tb.Th (whole SEG, as ORMIR-BQRL ships) +2.49e-07, Tb.Sp -1.08e-07, Tb.N +8.88e-07, Tb.Th (TRAB_SEG, Scripts 32/33/34) +2.48e-07
- PFJ-b5d5d9_L (X7775442): SEG 2 (+1/-1), TRAB_SEG 2 (+1/-1), TRAB_TH map 6, TRAB_SP map 11, TRAB_1N map 44, TRAB_TH_old map 2, Tb.Th (whole SEG, as ORMIR-BQRL ships) +3.36e-08, Tb.Sp -3.49e-08, Tb.N +1.17e-06
- PFJ-42293d_L (X3623103): SEG 3 (+3/-0), TRAB_SEG 1 (+1/-0), CORT_SEG 2 (+2/-0), TRAB_TH map 11, TRAB_SP map 1, TRAB_1N map 126, TRAB_TH_old map 11, BV/TV +4.67e-08, Tb.Th (whole SEG, as ORMIR-BQRL ships) +7.47e-08, Tb.Sp +5.96e-09, Tb.N +2.21e-06, Tb.Th (TRAB_SEG, Scripts 32/33/34) +7.48e-08
- PFJ-a56aae_R (X5606590): SEG 3 (+0/-3), TRAB_SEG 3 (+0/-3), TRAB_TH map 4, TRAB_SP map 47, TRAB_1N map 221, TRAB_TH_old map 4, BV/TV -1.93e-07, Tb.Th (whole SEG, as ORMIR-BQRL ships) +7.39e-08, Tb.Sp -1.66e-07, Tb.N -6.78e-06, Tb.Th (TRAB_SEG, Scripts 32/33/34) +7.18e-08
- PFJ-6f5538_R (X5492058): SEG 2 (+2/-0), CORT_SEG 2 (+2/-0)
- PFJ-6b714e_L (X5484780): SEG 1 (+0/-1), CORT_SEG 1 (+0/-1)
- PFJ-6b714e_R (X2663245): SEG 2 (+1/-1), CORT_SEG 2 (+1/-1)
- PFJ-69bcb0_L (X1183449): SEG 1 (+0/-1), TRAB_SEG 1 (+0/-1), TRAB_TH map 1, TRAB_SP map 3, TRAB_TH_old map 1, BV/TV -4.52e-08, Tb.Th (whole SEG, as ORMIR-BQRL ships) -1.52e-10, Tb.Sp -4.14e-08, Tb.Th (TRAB_SEG, Scripts 32/33/34) -5.24e-10
- PFJ-69bcb0_R (X1346001): SEG 5 (+2/-3), TRAB_SEG 4 (+1/-3), CORT_SEG 1 (+1/-0), TRAB_TH map 55, TRAB_SP map 66, TRAB_1N map 538, TRAB_TH_old map 55, BV/TV -7.61e-08, Tb.Th (whole SEG, as ORMIR-BQRL ships) +3.94e-07, Tb.Sp -5.67e-08, Tb.N +3.05e-06, Tb.Th (TRAB_SEG, Scripts 32/33/34) +3.92e-07
- PFJ-411dfd_L (X3334670): SEG 2 (+1/-1), TRAB_SEG 2 (+1/-1), TRAB_TH map 2, TRAB_SP map 1, TRAB_1N map 13, TRAB_TH_old map 2, Tb.Th (whole SEG, as ORMIR-BQRL ships) -8.64e-09, Tb.Sp -1.70e-08, Tb.N +2.71e-08, Tb.Th (TRAB_SEG, Scripts 32/33/34) -8.60e-09
- PFJ-411dfd_R (X5143651): SEG 3 (+3/-0), TRAB_SEG 3 (+3/-0), TRAB_TH map 3, TRAB_SP map 4, TRAB_1N map 21, TRAB_TH_old map 8, BV/TV +1.67e-07, Tb.Th (whole SEG, as ORMIR-BQRL ships) -1.72e-08, Tb.Sp -2.40e-09, Tb.N +1.03e-06, Tb.Th (TRAB_SEG, Scripts 32/33/34) +3.32e-08
