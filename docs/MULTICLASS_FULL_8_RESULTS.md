# Full-source eight-class comparison: seed 42

The six-candidate comparison completed successfully on 5 November 2026. All
12 candidate/decision evaluations completed; no candidate failed. Total run time
was 7,305 seconds (about two hours), including source preparation, OOF fitting,
bias tuning, refits, evaluation, and repeated in-run timing.

## Protocol and scope

- Training-source pool: **5,491,971 valid flows**, all 46 original numeric features.
- Development: **4,393,576 flows**; independent validation: **1,098,395 flows**.
- Raw-subtype-stratified splits, seed 42, five OOF folds, inner early stopping.
- Feature selection/scaling fitted inside each fold's inner fitting partition.
- Classifier weight powers normalized to mean sample weight one. The selector's
  weighting stayed fixed for the controlled weight ablation.
- Biases fitted to development OOF predictions. Validation was used for
  candidate/decision selection, so this is not an independent final-test score.
- The benchmark guard permits no increase in validation Benign FPR over the
  reference's argmax result, checked independently for every declared seed.
- This is one training/split seed; three-seed confirmation is still needed.
  The original test CSV was not opened or evaluated by this experiment or its
  post-run checks. This is a controlled local benchmark, not a paper reproduction.

## Useful operating points

| Configuration | Macro-F1 | Benign FPR | WebBased recall | BruteForce recall |
|---|---:|---:|---:|---:|
| Reference: median features, inverse weights, argmax | 0.7904 | 7.65% | 85.46% | 77.27% |
| All 46, inverse weights, OOF bias | 0.8776 | 5.87% | 51.60% | 61.36% |
| All 46, power 0.75, argmax | 0.8276 | 5.56% | 73.05% | 70.45% |
| All 46, power 0.75, OOF bias: benchmark leader | **0.8779** | **2.18%** | **46.10%** | **60.71%** |

The recorded leader improves Macro-F1 by 0.0875 over the reference and reduces
Benign false alarms from 1,981 to 564 among 25,908 Benign validation rows. Its
weighted-F1 is 0.995817 and accuracy is 99.5917%.

Its Macro-F1 is only 0.00035 above the all-feature inverse-weight model with
bias tuning. That tiny difference is not evidence of a reproducible ranking.
The lower false-positive rate is the more substantial distinction in this seed.

Removing weights entirely performed poorly: 0.7735/0.7768 Macro-F1 for selected/all
features with argmax, and 0.7768/0.7858 with their OOF biases. Power 0.5 reached
0.8324 with argmax and 0.8353 with bias. None matched the best weighted/bias choices.

## Recall is a separate requirement

For the benchmark leader, BruteForce F1 rises from **0.3310 to 0.7480**, and
WebBased F1 from **0.3388 to 0.5936**. Their precision/recall changes are:

| Class | Reference precision | Leader precision | Reference recall | Leader recall |
|---|---:|---:|---:|---:|
| BruteForce | 21.06% | 97.40% | 77.27% | 60.71% |
| WebBased | 21.13% | 83.33% | 85.46% | 46.10% |

The gain comes from a different precision/recall balance; it does not mean more
actual WebBased attacks were detected. The selected-features inverse-weight
model with bias actually has higher WebBased F1 (0.6375) than the global benchmark
leader, illustrating why Macro-F1 alone cannot choose every target-class policy.

Per the user's choice, the current comparison finished before setting recall
limits. No recall constraint has been applied. Among the evaluated points,
power 0.75 without bias already provides at least 70% recall for both target
classes; it trades some Macro-F1 and false-positive reduction for that recall.
These are examples for review, not deployment recommendations.

## Inference and training cost

Original in-run timing medians varied with measurement conditions. A post-run
check reloaded the exact validation cohort, warmed each condition on all rows,
then measured five repeats with rotating order and two CPU threads. All saved
validation predictions were reproduced exactly.

| Condition | Median CPU us/flow | Observed min–max |
|---|---:|---:|
| Reference, argmax | 7.14 | 6.96–7.48 |
| Power 0.75, argmax | 8.88 | 8.47–9.06 |
| Power 0.75, bias | 8.51 | 8.40–8.81 |

The selected configuration costs about 19% more inference time than the reference
in this warmed check. The two selected-model conditions have overlapping timing
ranges; the lower median with bias is not evidence that bias makes inference
faster. These are CPU **batch throughput** measurements including transformation
and decision logic, not measured single-flow latency.

The reference's development early-stop/refit cost 112 seconds; the leader's cost
151 seconds. Their OOF fitting costs were 404 and 586 seconds, respectively,
with bias searches taking roughly 9 and 7 seconds. These are distinct from the
whole six-candidate research run. Only one classifier is used at prediction time.

## Overlap and verification

363,443 validation rows (33.09%) share numeric float32 feature vectors with
development. Most are DDoS. Overlap is 4.22% for BruteForce, 2.66% for WebBased,
and 3.99% for Recon. Feature matches do not prove that two records are the same
physical flow; the audit excludes labels from its feature hashes.

On the 734,952 validation rows whose feature hashes are absent from development,
the leader reaches **0.8767 Macro-F1**, versus **0.7900** for the reference, and
2.21% Benign FPR. This is an exploratory filtered-subset check of the same fitted
models, not grouped retraining, new-capture evaluation, or a final test.

All 12 validation results/confusion matrices were independently recomputed.
All six OOF arrays were checked for finite normalized probabilities, full fold
coverage, and row alignment. Frozen selection/config hashes and all captured
training-implementation fingerprints were verified. A reusable biased benchmark
bundle also reproduced saved predictions through the prediction-only interface.

The regression suite passed 78 tests with one optional CUDA test skipped; three
additional overlap-audit tests passed. Real GPU fitting was exercised by the
full comparison. The reporting figures were rendered and visually inspected.

## Evidence and next step

- [Complete report and figures](../outputs/experiments/8class_v1/report/REPORT.md)
- [All 12 validation results](../outputs/experiments/8class_v1/validation_summary.csv)
- [Target-class comparison](../outputs/experiments/8class_v1/report/target_class_comparison.csv)
- [Class precision/recall/F1 summaries](../outputs/experiments/8class_v1/report/per_class_summary.csv)
- [Paired timing measurements](../outputs/experiments/8class_v1/report/paired_benchmark_seed_42/summary.csv)
- [Exploratory unseen-vector results](../outputs/experiments/8class_v1/overlap_audit/seed_42/unseen_validation_comparison.csv)
- [Development benchmark bundle](../outputs/experiments/8class_v1/benchmark_models/seed_42/model.joblib)

The exported bundle is trained on development rows and applies the selected OOF
bias automatically. Its metadata marks recall-policy review as pending; it is
not a final full-training refit or a deployment selection.

Next, specify the minimum WebBased/BruteForce recall, then retune decision biases
using the cached OOF probabilities and validate the constrained choices. The
six base models do not need to be retrained for that step. Confirm the resulting
finalists across several seeds before final test evaluation. Stacking and
specialists remain conditional on the remaining errors.

Follow-up: the [baseline audit](MULTICLASS_BASELINE_AUDIT.md) identifies material
paper/local protocol differences. The [completed recall-constrained searches](MULTICLASS_RECALL_RESULTS.md)
show that protecting all three target families removes the apparent large F1 gain.
