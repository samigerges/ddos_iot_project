# First controlled pilots

Completed on 5 November 2026 using the new
[experiment workflow](MULTICLASS_EXPERIMENTS.md). Both runs completed all six
feature/weight candidates, five OOF folds, and both decision rules without a
candidate failure. The real test CSV was not used by either experiment.

## Scope

- Original training source: 5,491,971 valid rows and 46 numeric features.
- Pilot training pool: 56,611 rows, capped at 2,000 per **original subtype**.
- Seed: 42; development: 45,288 rows; independent validation: 11,323 rows.
- The two tasks used identical original source rows and OOF fold assignments.
- XGBoost GPU fits; at most 100 boosting rounds with inner early stopping.
  Learning rate: 0.1 for eight classes and 0.05 for 34 classes.
- Feature selection/scaling were fitted inside each fold's inner fitting split.
- Biases were optimized using development OOF predictions, never validation or test.

These are **single-seed, sampled-training validation results**. Capping changes
class prevalence substantially. They cannot be compared directly with the old
full-data runs or the paper's scores, and they do not establish statistical
significance. Validation was also used to select the winner.

## Observed trade-offs

Benign FPR means the fraction of actual Benign flows classified as any attack.

| Task | Configuration | Validation Macro-F1 | Benign FPR |
|---|---|---:|---:|
| 8 | Median features, power 1, argmax: pilot reference | 0.8788 | 12.00% |
| 8 | All 46, power 0.75, argmax | 0.8838 | 12.25% |
| 8 | All 46, power 0.75, OOF bias: highest Macro-F1 | 0.8867 | 19.00% |
| 34 | Median features, power 1, argmax: pilot reference | 0.8447 | 19.75% |
| 34 | All 46, power 0.75, argmax | 0.8496 | 17.25% |
| 34 | Median features, power 1, OOF bias: highest Macro-F1 | 0.8699 | 31.00% |

The predeclared selection rule maximizes validation Macro-F1. Its recorded
winners therefore accept more Benign false alarms. Those winners are experiment
selections, not deployment recommendations.

The subsequent strict FPR selection review keeps these historical choices
unchanged and instead compares alternatives against the reference rate. Under
zero allowed increase, the eight-class pilot falls back to the reference
(0.8788 Macro-F1, 12.00% FPR); the 34-class pilot chooses all 46 features, power
0.75, and argmax (0.8496 Macro-F1, 17.25% FPR). New full-data configurations use
that guarded rule and retain both winners in their reports.

All 46 features with power 0.75 and argmax is a useful configuration to confirm
on the full source: it improved Macro-F1 by approximately 0.005 in both pilots,
with near-reference eight-class FPR and lower 34-class FPR. This is a hypothesis
for the next run, not a guaranteed full-data gain. Removing weights entirely
did not produce the best Macro-F1, especially on the 34-class task.

## Per-class and methodology checks

The highest-Macro-F1 eight-class configuration increased BruteForce F1 from
0.7358 to 0.7853, while WebBased F1 decreased from 0.7658 to 0.7535. This shows
why Macro-F1 alone cannot establish improvement in every target class.

The chosen 34-class configuration recorded 187 within-family and 872
between-family errors. These are counts, not contributions to Macro-F1; rare
subtypes still need inspection in the per-class report. The exported family
confusion matrix supports that analysis.

The overlap audit found 100 validation rows (0.883%) whose numeric float32
feature vectors also occurred in development. Duplicates were retained to keep
this declared flow-level protocol. This is not evidence of independence across
captures, devices, or time.

All 24 validation reports were independently recomputed from exported
predictions. All OOF arrays were checked for finite normalized probabilities,
five-fold coverage, and matching source row IDs. Cross-task source rows and fold
assignments were verified equal. The regression suite passed 70 tests with one
optional CUDA test skipped; the pilots themselves exercised real GPU training.
The expanded experiment tests also passed with two seeds and frozen final
evaluation on synthetic fixtures. The full suite needed the inherited invalid
`SSLKEYLOGFILE` setting removed from the test process; no app setting was changed.

The pilot runtime was approximately four minutes for eight classes and eleven
minutes for 34 classes. Timing reports are preserved, but regression tests ran
concurrently with parts of the pilots; use an otherwise idle machine for
publishable latency comparisons.

## Saved evidence and next run

The full comparison tables are:

- [8-class validation summary](../outputs/experiments/8class_pilot_v1/validation_summary.csv)
- [34-class validation summary](../outputs/experiments/34class_pilot_v1/validation_summary.csv)

Each run also contains the six development-trained bundles, OOF probability
arrays and row identities, decision biases, per-class reports, confusion
matrices, paired seed deltas, and frozen selections.

The eight-class pilot was relocated into `outputs/experiments/` after completion
to separate the new report format from legacy dashboard artifacts. Its frozen
configuration retains the original output path as historical provenance; all
scores and selection hashes are unchanged.

Use the full-data configurations `configs/experiments/8class.json` and
`configs/experiments/34class.json` for confirmation, starting with eight classes.
Keep the six candidates and fixed model settings for the initial ablation, then
confirm promising configurations with several seeds. Decide the acceptable
Benign false-positive budget before choosing a model for final test evaluation.
The separate `evaluate-test` command is ready once that experiment choice is
settled. Stacking and specialists remain deferred until the full-data error
reports establish their value.
