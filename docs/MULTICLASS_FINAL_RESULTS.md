# Frozen multiclass final results

The frozen 34-class specialist changes test Macro-F1 by **+0.007445** against its same-runtime XGBoost control. Its development-selected mean-recall limit does not generalize to every target class on the official test. The 8-class comparison has not demonstrated an improvement. Both the F1 result and recall limitations are reported below.

Official published tests: 1,000,000 rows per task. Both task choices, seeds, software environment, recall policy and uncertainty analysis were frozen before either test was opened. No test-driven tuning or model/seed selection.

## Final test scores

These reference/selected pairs use the same current runtime (XGBoost 3.2.0). Each row includes seeds 42, 123 and 2024; std is population std.

| Classes | Frozen candidate | Macro-F1 mean ± std | Weighted-F1 | Accuracy | Benign FPR |
|---:|---|---:|---:|---:|---:|
| 8 | paper_fixed | 0.888413 ± 0.000000 | 0.995827 | 99.5865% | 4.131% |
| 34 | paper_fixed | 0.886536 ± 0.000000 | 0.994720 | 99.4797% | 3.965% |
| 34 | specialist_oof_protected_bias | 0.893982 ± 0.000000 | 0.994832 | 99.4944% | 3.438% |

The final seed fits share the same complete training/test cohorts; their std is not uncertainty on new captures. The saved prediction checks state whether predictions are identical across seeds. Development split variation is reported separately.

## Paper baseline reproduction

Predeclared references use Python 3.11.2, NumPy 2.1.2, pandas 2.3.0, sklearn 1.6.1 and XGBoost 3.0.2, with the released selected-feature parameter dictionaries and author default training-thread policy. GPU classification scores and two-thread CPU classification scores are both recorded; GPU latency was not measured.

| Classes | Pinned-runtime CPU Macro-F1 | GPU Macro-F1 | GPU/CPU prediction disagreements |
|---:|---:|---:|---:|
| 8 | 0.890307 | 0.890307 | 0 |
| 34 | 0.887564 | 0.887564 | 0 |

The controlled improvement is the selected model versus its same-runtime control above. The pinned-runtime references separately check reproduction of the paper; runtime differences are not attributed to the specialist.

## Target minority classes

All three 8-class target families and all twelve 34-class target labels are included. Recall changes are percentage points, averaged across the three fixed final seeds.

| Classes | Target | Test support | Reference F1 | Selected F1 | Selected recall | Recall change (pp) |
|---:|---|---:|---:|---:|---:|---:|
| 8 | BruteForce | 280 | 0.71706 | 0.71706 | 59.29% | +0.000 |
| 8 | Recon | 7739 | 0.86110 | 0.86110 | 86.28% | +0.000 |
| 8 | WebBased | 1199 | 0.69805 | 0.69805 | 65.55% | +0.000 |
| 34 | Backdoor_Malware | 200 | 0.65730 | 0.68712 | 56.00% | -2.500 |
| 34 | BrowserHijacking | 200 | 0.71930 | 0.75802 | 65.00% | +3.500 |
| 34 | CommandInjection | 200 | 0.67278 | 0.72327 | 57.50% | +2.500 |
| 34 | DictionaryBruteForce | 279 | 0.67360 | 0.69180 | 55.91% | -2.151 |
| 34 | Recon-HostDiscovery | 2876 | 0.85340 | 0.85315 | 86.96% | -0.174 |
| 34 | Recon-OSScan | 2103 | 0.70934 | 0.72414 | 67.90% | -0.333 |
| 34 | Recon-PingSweep | 200 | 0.65130 | 0.67622 | 59.00% | +2.500 |
| 34 | Recon-PortScan | 1761 | 0.71392 | 0.72570 | 71.44% | -0.625 |
| 34 | SqlInjection | 200 | 0.64375 | 0.66044 | 53.00% | +1.500 |
| 34 | Uploading_Attack | 200 | 0.61920 | 0.63750 | 51.00% | +1.000 |
| 34 | VulnerabilityScan | 800 | 0.99563 | 0.98640 | 99.75% | +0.000 |
| 34 | XSS | 200 | 0.65546 | 0.68675 | 57.00% | -1.500 |

![Target-class confusion comparison](../outputs/studies/final_results_v1/minority_confusions.png)

Rows are normalized over all predicted classes. Cell annotations are percentages; Other aggregates remaining labels. This descriptive analysis uses predeclared seed 42 and does not select a new model.

The development-selected operating point does not satisfy every corresponding test check. Target recall losses exceeding 0.5 percentage points: Backdoor_Malware: -2.500 pp; DictionaryBruteForce: -2.151 pp; Recon-PortScan: -0.625 pp; XSS: -1.500 pp. Benign FPR nonincreasing in every seed: True. This is reported as a generalization limitation; no replacement, threshold change or retuning was performed from the test.

## Paired uncertainty

Predeclared model seed 42, 2,000 paired true-label-stratified bootstrap draws, bootstrap seed 2024. Macro-F1 delta **+0.007445**, percentile 95% interval **[+0.004791, +0.010026]**.

These are conditional flow-level intervals, which can be too narrow with correlated flows/captures. Class intervals are descriptive without multiple-comparison correction. They do not quantify new-device/capture uncertainty and did not select the final model.

## CPU cost and fitting cost

Exact selected-head development benchmark: 100,000 common validation rows, full warm-ups, four balanced rotating passes, two CPU threads. Reference **130.71 ± 1.42 µs/flow**; selected specialist **134.16 ± 3.17 µs/flow**; observed median change **+2.64%**. Timing variation overlaps; this does not establish a precise latency increase. These are batched throughputs on published selected/scaled features, excluding loading and raw feature extraction.

Final full-training models also have full-million-row warm-up plus one timed test pass per model/seed in test_summary.csv. That scope differs from the balanced development benchmark, and the timings should not be mixed. The single-pass test timings varied substantially between runs and were measured in a fixed model order. They are recorded observations, not a stable estimate of architecture overhead; the balanced paired benchmark is the comparison for that purpose.

Windows recorded a lid/button sleep from 16:45 to 17:55 Cairo during the 34-class seed-2024 reference timed pass. The recorded pass took 4,495.4 seconds and contains 4,219.6 seconds of suspension. It is excluded from valid throughput means, while every classification score remains included. Original timing values are preserved; subtracting sleep yields an active-wall estimate, not CPU time or a replacement measurement. The earlier paired benchmark and all final training fits were completed before this interruption. See [timing audit](../outputs/studies/34class_v1/test_evaluation_v1/timing_audit.json).

| Classes | Candidate | Mean final full-training fit seconds | Std |
|---:|---|---:|---:|
| 8 | paper_fixed | 169.5 | 1.1 |
| 34 | paper_fixed | 794.3 | 55.7 |
| 34 | specialist_oof_protected_bias | 850.3 | 55.3 |

Additional one-time 34-class calibration fitting: cached base OOF fits **1843.7 s** plus specialist OOF fits **205.1 s**. Count the shared base OOF work once, even when reused across bias/stack/specialist methods. These are fit costs, excluding other search, prediction, data preparation and optimization time. The complete specialist calibration stage reused those base fits and took 476.5 s of wall time. Final fitting cost excludes that one-time calibration work.

## Development evidence and scope

The 8-class reference averaged 0.88371 ± 0.00381 Macro-F1 on three development splits. The expanded specialist averaged 0.88358 ± 0.00412; bias tuning reached 0.88431 ± 0.00460 with inconsistent paired gains and lower target recall. The frozen 8-class choice therefore remains XGBoost. This comparison has not demonstrated an 8-class improvement.

For 34 classes, reference development Macro-F1 was 0.88742 ± 0.00381, raw specialist 0.89279 ± 0.00233, and selected OOF-calibrated specialist 0.89624 ± 0.00210. All three matched splits improved. The user's chosen mean-recall policy permitted at most 0.5 percentage points loss per target class and no benign-FPR increase in any split. The largest mean recall loss was PortScan −0.19 points; the largest individual-split loss was Backdoor −3.125 points. Original strict guard failures are preserved. The tested XGBoost/CatBoost stack did not improve the reference.

The group expansion and increased specialist depth/rounds form a joint variant; their individual effects have not been separated. OOF optimization uses group/settings chosen on disjoint tuning data and is not a nested performance estimate. Published 23-feature inputs were selected/scaled upstream by the authors; feature selection is not fold-local in this track. The earlier raw 46-feature local pipeline has a different sample/protocol and is not directly comparable. No natural-distribution/new-capture test or information ceiling has been established. Saved research models require the published scaled 23-feature representation.

## Artifacts

- [Frozen joint protocol](../outputs/studies/final_protocol_v1/protocol.json)
- [Completed global execution](../outputs/studies/final_protocol_v1/report.json)
- [Final score table](../outputs/studies/final_results_v1/test_summary.csv)
- [Target class precision/recall/F1](../outputs/studies/final_results_v1/minority_test_metrics.csv)
- [34-class paired uncertainty](../outputs/studies/34class_v1/test_evaluation_v1/paired_uncertainty_seed42/REPORT.md)
- [Exact selected-head paired CPU measurement](../outputs/studies/34class_v1/selected_paired_timing_v1/REPORT.md)
- [Full development comparisons, confusions and trade-offs](../outputs/studies/comparison_v1/REPORT.md)
- [8-class compared-head CPU benchmarks](../outputs/studies/8class_v1/paired_timing_v1/REPORT.md)
- [34-class compared-head CPU benchmarks](../outputs/studies/34class_v1/paired_timing_v1/REPORT.md)
- [8-class full test models and metrics](../outputs/studies/8class_v1/test_evaluation_v1/report.json)
- [34-class full test models and metrics](../outputs/studies/34class_v1/test_evaluation_v1/report.json)

## Saved model use

Seed 42 is a predeclared example, not a seed selected from test performance. Both baseline and selected bundles were reloaded for the final evaluation.

- [8-class seed-42 XGBoost bundle](../outputs/studies/8class_v1/test_evaluation_v1/seed_42/paper_fixed/model.joblib)
- [34-class seed-42 selected specialist bundle](../outputs/studies/34class_v1/test_evaluation_v1/seed_42/specialist_oof_protected_bias/model.joblib)

From the project root, use the saved bundle without fitting:

```powershell
.\.venv\Scripts\python.exe -B src/predict_multiclass_study.py --model outputs/studies/34class_v1/test_evaluation_v1/seed_42/specialist_oof_protected_bias/model.joblib --input PATH_TO_PUBLISHED_FORMAT.csv --output outputs/predictions/final34.csv --input-representation published-scaled-23
```

The input must match the corresponding task's published selected/scaled feature columns. An untransformed raw 46-feature CSV is a different representation.
