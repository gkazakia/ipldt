**Table 1. The 137 XtremeCT II scans of the validation set.**

| Site | n | Study (scans) | Stack length (slices) | IPL parameter set | Participants |
|---|---|---|---|---|---|
| Patella | 21 | Patellofemoral osteoarthritis study (21) | 168 (19 scans), 320 (1), 504 (1) | Tibia | 16 |
| Ultradistal radius | 29 | Chronic kidney disease study (9); reproducibility study (20) | 168 | Radius | 29 |
| Ultradistal tibia | 29 | Chronic kidney disease study (9); reproducibility study (20) | 168 | Tibia | 29 |
| Diaphyseal radius | 29 | Chronic kidney disease study (9); reproducibility study (20) | 168 | Radius | 29 |
| Diaphyseal tibia | 29 | Chronic kidney disease study (8); reproducibility study (20); BMAT study (1) | 168 | Tibia (28 scans), radius (1) | 29 |
| Total | 137 | Patellofemoral osteoarthritis 21; chronic kidney disease 35; reproducibility 80; BMAT 1 | 168 (135 scans), 320 (1), 504 (1) | Tibia (78 scans), radius (59) | 46 |

Read from `manuscript/facts/facts.json` (`tables["cohort.table1"]`), which build_facts.py computes from the validation records. Participant counts are per study; the total adds them up, assuming that nobody was enrolled in two of the studies.
