# Staged training studies on the verified published subsets

The [frozen final evaluation is complete](MULTICLASS_FINAL_RESULTS.md). It shows
the 34-class Macro-F1 gain and held-out recall shortfalls, the reproduced paper
baselines, and the unchanged 8-class operating point. The commands below are
workflow documentation; existing final-test directories prevent automatic
retesting.

This workflow implements the next training experiments without changing the
earlier raw-46-feature runs or their saved selections. It supports 8 and 34
classes and keeps training, confirmation, conditional complexity, and final-test
evaluation as explicit stages.

## Sequential execution

```powershell
.\.venv\Scripts\python.exe -B src/run_multiclass_research.py --paper-python .cache/paper_runtime3112/python.exe --wait-existing
```

This one-time runner finishes both base studies, fits the software-matched
references, audits exact conflicting feature vectors, and runs bias, specialist,
stacking and confirmation stages in sequence. It reuses completed stages and
shared OOF caches. GPU training and CPU timing jobs do not overlap. Progress and
individual stage logs live in `outputs/studies/research_pipeline_v1/`.

The default command leaves both official test sets untouched. After reviewing
the development comparisons, resume with the same command plus `--final-test`.
That freezes each final protocol and all full-training models before opening
the corresponding official test. Failed or interrupted stage directories are
preserved for inspection and block an automatic overwrite.

## Data contract and limits

Input is the verified `data/raw/paper_v3/training_subsets/` inventory. Each task
has 3,999,998 rows and 23 author-selected/scaled numeric features. SHA-256, schema,
row counts, labels, and finite values are checked before fitting.

The authors selected/scaled these columns on their full original training split.
Resplitting that CSV cannot establish fold-local feature selection. Report these
results as development on a published representation, not a nested raw-data
experiment. A held-out published-test score can reproduce the selected-feature
benchmark; all-46-feature and new-capture tests require the corresponding raw
source and preprocessing lineage. The existing local raw test CSV is separate.

Research bundles consume **already selected/scaled values**. They are not raw
flow deployment bundles; an identity transform preserves the published values.
Use `predict_multiclass_study.py` with the explicit representation argument.

## Base training study

```powershell
.\.venv\Scripts\python.exe -B src/study_multiclass.py plan --config configs/studies/8class.json
.\.venv\Scripts\python.exe -B src/study_multiclass.py run --config configs/studies/8class.json
.\.venv\Scripts\python.exe -B src/study_multiclass.py run --config configs/studies/34class.json
```

Run one GPU study at a time. Outputs are protected from overwrite; use a new
configuration/output path for another run.

For each seed, split the published training source into **60% fitting, 20%
tuning, and 20% validation**, stratified by task labels. Tuning chooses model
parameters/rounds. Validation metrics never enter the search. Raw feature
selection is not relearned from these preprocessed inputs.

The first six trials keep inverse-frequency weights fixed:

| Trial | Depth | Rounds | Early stopping | Other settings |
|---|---:|---:|---|---|
| Paper-parameter reference | 10 | 500 | No | Default row/column sampling, 256 histogram bins |
| Shallow fixed reference | 6 | 500 | No | Same remaining settings |
| Shallow early stop | 6 | Up to 500 | Weighted tuning log loss | Same remaining settings |
| Intermediate early stop | 8 | Up to 500 | Weighted tuning log loss | Same remaining settings |
| Deep early stop | 10 | Up to 500 | Weighted tuning log loss | Same remaining settings |
| Regularized deep candidate | 10 | Up to 500 | Weighted tuning log loss | Joint min-child-weight=5, L2=5 combination |

The reference uses the pinned author's **selected-feature** depth-10/500-round
settings for both tasks. The author depth-14 setting applies to the 34-class
all-46-feature ablation, which is not silently substituted here. Obsolete
`predictor` and label-encoder arguments are omitted; library versions and the
effective parameters are recorded. Current XGBoost is 3.2.0, versus paper 3.0.2.

Next, the strongest capacity configuration is retrained with weight powers
0.75 and 0.5, keeping its other settings fixed. Finally, CatBoost is screened as
a diversity candidate. Training times, selected rounds, per-class reports, and
confusions are saved for every trial, including failed trials.

Report both the highest tuning Macro-F1 and the best choice preserving reference
recall/FPR. The guarded choice must preserve every protected task label's recall
and not increase Benign FPR on the same tuning cohort. For 8 classes this protects
WebBased, Recon and BruteForce; for 34 it protects all 12 constituent raw labels.
All 34 class metrics are still reported.

Freeze these choices and rounds, refit the reference/finalists on the combined
80% development rows, then evaluate the reserved validation rows. Validation
checks whether the tuning constraints transfer; it does not guarantee they do.
CPU timing uses the complete validation cohort as warm-up and repeated batched
prediction, with two threads by default. It excludes CSV loading/raw feature
extraction and is not a single-flow latency measurement. Paper GPU timing is
not directly comparable to these CPU measurements.

A separate fixed reference is also fitted to **all published training rows** and
saved before any future test evaluation. This model must never score the internal
validation partition, because it has trained on those rows.

## Three-seed confirmation

```powershell
.\.venv\Scripts\python.exe -B src/confirm_multiclass_study.py --run-dir outputs/studies/8class_v1
.\.venv\Scripts\python.exe -B src/confirm_multiclass_study.py --run-dir outputs/studies/34class_v1
```

Confirm fixed finalists on seeds 42, 123 and 2024. Reuse the original seed-42
result and refit the same parameters and selected round counts for the other
splits; do not repeat model search or choose a favorable seed. Save Macro-F1,
weighted-F1, accuracy, Benign FPR, per-class recall/F1, training cost and CPU
throughput with population mean/std. These are repeated splits of one dataset,
not independent device/capture samples.
The initial tuning labels can occur in another seed's validation partition;
these means/std are robustness diagnostics, not a new independent final estimate.
Additional seeds use a full-cohort warm-up and one timed CPU pass, while reusing
the original seed's three-repeat timing. The untouched published test remains
the final performance estimate.

The final simple-model choice maximizes mean validation Macro-F1 subject to the
reference recall/FPR checks in every seed. The fixed reference is an explicit
fallback. The selection and its fingerprint are persisted before test access.

## Single-model OOF decisions

```powershell
.\.venv\Scripts\python.exe -B src/bias_multiclass_study.py --run-dir outputs/studies/8class_v1
```

Generate five-fold OOF predictions on fitting partition A for the fixed reference
and strongest XGBoost. Fit unconstrained and reference-recall/FPR-constrained bias
vectors there, freeze them, then evaluate the development-refitted XGBoost on
validation. This supplies the XGBoost-plus-threshold baseline before crediting a
stack with improvement. Scores remain optimization diagnostics: base settings
were selected using B and upstream feature selection used the author's full
training partition.

Report the unconstrained, constrained coordinate and constrained joint searches
separately. The joint search changes the Benign bias and one protected-class
bias together: a useful move can otherwise be blocked when either coordinate
alone violates recall/FPR constraints. These finite searches are heuristics,
not guarantees of an optimal threshold vector. Select the protected vector
using OOF diagnostics before inspecting its validation score.

Complete OOF caches are shared with stacking through immutable views. Check
source, settings, versions, fitting-code fingerprint, row/fold alignment,
normalization and array SHA-256 before reuse. Reject incomplete/modified caches;
use a fresh study directory rather than overwrite them.

## Conditional specialist

```powershell
.\.venv\Scripts\python.exe -B src/specialize_multiclass_study.py --run-dir outputs/studies/8class_v1
```

Training-only errors determine whether a specialist is justified. At least half
of target errors must lie inside the Benign/target routing group, so the
specialist can reach them. Count both minority-to-Benign and within-target
confusions. If the gate fails, persist the reason and skip fitting.

Train a depth-6 specialist on the true Benign/target classes with weight powers
0.75, 1 and 1.25. Tune rounds/settings on the training-only tuning cohort. Route
samples whose base argmax is in the group. Reallocate that group's probability
mass using specialist probabilities, leaving outside-group mass unchanged; no
claim of natural-prior probability calibration is made.

Full-system evaluation includes target samples missed by the router and samples
from other classes incorrectly routed into it. Save those counts explicitly.
Freeze specialist choices before development refitting and validation. A single
seed's promising specialist result requires confirmation before a final model
can supersede the simple-model choice.

The separate `--expanded-fixed` variant first uses tuning mistakes to add any
outside attack family receiving at least 10% of target-class errors. Include all
represented labels from such a family. Its specialist uses depth 10 and 500 fixed
rounds, matching reference capacity. This is a joint group/capacity variant; a
gain cannot be attributed to either change individually. The original specialist
is preserved. Expansion never uses validation/test labels; depth, group, weight
power and rounds are frozen before validation and retained across confirmation.

## OOF decisions for a promising final specialist

When the repeated-split raw specialist improves mean Macro-F1, complete the
final decision-tuning stage on that frozen hierarchy:

```powershell
.\.venv\Scripts\python.exe -B src/bias_multiclass_specialist.py --run-dir outputs/studies/34class_v1 --specialist-run outputs/studies/34class_v1/specialist_expanded_v2 --raw-confirmation outputs/studies/34class_v1/specialist_expanded_comparison_v2
.\.venv\Scripts\python.exe -B src/confirm_multiclass_specialist_bias.py --base-confirmation outputs/studies/34class_v1/confirmation_v1 --raw-confirmation outputs/studies/34class_v1/specialist_expanded_comparison_v2 --calibration outputs/studies/34class_v1/specialist_bias_v1 --output-dir outputs/studies/34class_v1/specialist_bias_confirmation_v1
```

Reuse verified base OOF probabilities and their exact fold assignments. Fit only
the specialist on each scoring fold's complementary fitting rows, with frozen
group, power, depth and rounds. Route and score every held-out row, including
router mistakes. Fit unrestricted, recall/FPR-protected and joint bias vectors
on these routed OOF predictions, then freeze them before validation.

Confirm every frozen vector using the existing raw hierarchies and one shared
probability pass per additional seed. This repeats no hierarchy training and
retunes no decision vector. Keep component latency estimates separate from the
paired timing benchmark. Only a candidate preserving reference recall/FPR in
every seed and improving mean Macro-F1 is promoted automatically. Adaptive
development comparisons and upstream published feature selection remain
limitations; the official test is still the independent final evaluation.

## Conditional OOF stack

```powershell
.\.venv\Scripts\python.exe -B src/stack_multiclass_study.py --run-dir outputs/studies/8class_v1
```

Require complementary target corrections from CatBoost and an estimated sum of
base inference costs within five times XGBoost's cost. These are compute gates,
not evidence that stacking will improve F1. Record error correlation, corrections,
new errors and gate outcomes.

Generate **five-fold base OOF probabilities on the 60% fitting partition A** for
the strongest XGBoost and CatBoost. Each scoring fold is absent from its base
gradient fitting. Use fixed rounds selected on the separate tuning partition B;
no scoring fold supplies early stopping. Use log probabilities as meta features.

For meta calibration, cross-fit Logistic Regression over B. Each meta fold is
trained on base OOF features from A plus base-held-out features from the other B
folds. Its scoring B rows are excluded from all base gradient fits and its own
meta gradient fit. B labels previously selected base settings/rounds, so these
are **optimization diagnostics**, not fully nested performance estimates.
Published upstream feature selection is an additional CV limitation.

This first stack uses two learners. ExtraTrees, an MLP and LightGBM are deferred
until the measured residual errors justify another base model; adding them is
not an implemented or demonstrated gain in this experiment.

Use balanced multinomial loss and a uniform per-class training cap of 10,000
rows to bound CPU/memory cost. Report cap, class counts, iterations and whether
the optimizer reached its iteration limit. The current maximum is 2,000; earlier
300-iteration runs are retained and identified by their saved training reports.
When L-BFGS still reaches its limit, the solver-only repair uses `newton-cg`, with
a maximum of 100 Newton iterations, on the same log features, sample, class
weights, C=1 and multinomial L2 objective. This is a numerical repair, with no
new base learner or feature transformation. See the [solver compatibility
documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html).
Record convergence for every fold and retain interrupted optimizer artifacts.
Confirmation retains the source run's optimizer and iteration budget. Fit raw and recall-constrained biases
only to the cross-fitted meta probabilities. Freeze all decision vectors before
validation. Refit the meta learner using A-OOF plus B-held-out features and use
the development-refitted bases to predict validation.

Report the stack's base+meta probability cost and decision cost separately;
those separate measurements are not a paired end-to-end latency measurement.
Do not promote an unconfirmed stack solely because one seed improves.

## Confirming advanced candidates

```powershell
.\.venv\Scripts\python.exe -B src/confirm_multiclass_advanced.py --confirmation outputs/studies/8class_v1/confirmation_v1 --advanced-run outputs/studies/8class_v1/bias_v1 --output-dir outputs/studies/8class_v1/bias_confirmation_v1
```

The same command accepts specialist or stack runs. Skip further training when no
frozen protected choice improved validation Macro-F1 while preserving the
reference recall/FPR. For promising methods, reuse seed 42 and confirm on the
remaining seeds without retuning decision vectors or specialist settings. Stack
confirmation refits OOF bases/meta learners with fixed settings but retains the
original bias; single-model bias confirmation reuses the fixed base finalists.

Accept an advanced model only if it preserves reference recall/FPR in every seed
and improves mean validation Macro-F1 over the confirmed simple winner. Record
all failures and mean/std; three repeated splits do not establish cross-capture
generalization. Finalization can use that advanced confirmation or fall back to
the already frozen simple model.

To finish comparisons before setting final recall limits, use an explicit frozen
candidate with `--candidate NAME --compare-only`. This evaluates that candidate
on all confirmation seeds even if it fails the provisional recall/FPR guard.
It never promotes a model into the final-test selection. Report its mean/std,
per-class metrics and guard failures alongside the protected candidates.

## Software-matched paper reference

`requirements-paper-reference.txt` pins the paper's reported core versions. Use
Python 3.11.2 in an isolated environment. Supporting SciPy/joblib/threadpoolctl
versions are also pinned here, but were not all specified by the paper. Local
hardware still differs. The separate reference commands assert the reported
Python/NumPy/pandas/sklearn/XGBoost versions and keep the current study environment
unchanged:

```powershell
.\.cache\paper_runtime3112\python.exe -B src/train_published_reference.py --classes 8 --output-dir outputs/studies/paper_fidelity_v3/8class
```

This fits the fixed reference on every published training row and reads no test.
Declare it with `--paper-reference outputs/studies/paper_fidelity_v3/8class` when
freezing the final evaluation protocol. Only then score it in the matching
runtime with `score_published_reference.py`, using the exact test CSV and frozen
protocol. The reference uses the released selected-feature parameter dictionary
and default training thread policy. It reports GPU classification metrics and a
separate CPU comparison with two inference threads, including any GPU/CPU
prediction disagreements. Its CPU timing is not a reproduction of the paper's
GPU latency. The GPU result makes no latency claim.

## Once-only published-test stage

Only after the final model scope is settled and the three-seed selection frozen:

```powershell
.\.venv\Scripts\python.exe -B src/evaluate_multiclass_study.py --confirmation outputs/studies/8class_v1/confirmation_v1 --test-directory data/raw/paper_v3/final_tests/8class
```

This command verifies frozen source/selection fingerprints, saves every full-
training reference/winner for every declared seed, freezes the evaluation
protocol, and **then** extracts/parses the published test. It evaluates all
declared models/seeds and never chooses a seed or retunes from test results.
Existing evaluation/test directories block automatic repeated evaluation.
The base confirmation command freezes simple models; specialist/stack results
must complete their own confirmation before inclusion in a final protocol.
The final command also accepts an advanced confirmation. Confirmed specialists
are refitted with full-training bases; confirmed stacks retain their development-
fitted meta learners/frozen biases while refitting bases on all training rows.
Report that meta scope explicitly. Final timing defaults to a complete warm-up
plus one measured pass per model/seed; `--latency-repeats` increases repetitions.

Before test access, predeclare an additional uncertainty analysis for model seed
42: 2,000 paired bootstrap draws, random seed 2024, fixed class counts and a
percentile 95% interval for the Macro-F1 difference. Run
`report_multiclass_uncertainty.py` only on the frozen reference/selected test
prediction files. Joint prediction counts retain the pairing while resampling
within true-label strata; see [paired bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
and [stratified resampling](https://scikit-learn.org/1.8/modules/generated/sklearn.utils.resample.html).
These are conditional flow-level intervals, which can be too narrow when flows
are correlated. Class intervals are descriptive, with no multiple-comparison
correction. They are not uncertainty on new devices/captures and never select
new models, seeds or thresholds after the final test.

Use `finalize_multiclass_research.py freeze` to freeze both task confirmations
together after development and the recall policy are complete. Supply both
`--8class-confirmation` and `--34class-confirmation`, the completed worker
`--development-report`, a concrete `--recall-policy`, and a new `--output-dir`.
This binds both choices, three seeds, configuration/selection files, pinned
paper references, archive and final evaluation implementation before either
test is opened. A comparison-only run must first receive an explicit policy
selection; it cannot silently enter final evaluation. Run the resulting
directory with `finalize_multiclass_research.py evaluate --protocol-dir ...`.
The evaluator checks both tasks' hashes before opening either test, evaluates
the pinned-software reference in its isolated runtime, and runs the predeclared
uncertainty analysis on seed 42. It preserves partial final runs for inspection.

If the user settles a different recall policy after reviewing comparisons, use
`finalize_multiclass_research.py select` with `--evidence`, `--candidate`,
`--maximum-recall-loss`, `--scope`, `--policy-note`, and a new `--output-dir`.
The loss is a fraction (`0.01` means one percentage point); `each_seed` and `mean`
are different policies. The default remains zero loss in every seed. Selection
requires all three paired seeds, higher mean Macro-F1, and nonincreasing benign
FPR in every seed. It only selects already frozen candidates and preserves
their original strict guard failures. Record the settled user policy before
the joint freeze; these controls do not authorize choosing a new tolerance from
official-test outcomes.

## Reports and saved prediction

```powershell
.\.venv\Scripts\python.exe -B src/report_multiclass_study.py --run-dir outputs/studies/8class_v1
.\.venv\Scripts\python.exe -B src/predict_multiclass_study.py --model outputs/studies/8class_v1/seed_42/finalists/paper_fixed/model.joblib --input data/raw/paper_v3/training_subsets/8class_train_23features.csv --output outputs/predictions/published_8class.csv --input-representation published-scaled-23
```

Reports read metrics only. Optional provisional reports are explicitly separate.
Base, specialist and stack bundles preserve class mapping and their decision
vectors. Existing raw-feature prediction/dashboard workflows remain separate.

For a combined comparison, including per-class mean/std, target minority classes,
CPU cost, guard failures and false-alarm/F1 plots:

```powershell
.\.venv\Scripts\python.exe -B src/report_multiclass_research.py
```

This reads completed stage reports only and writes
`outputs/studies/comparison_v1/REPORT.md`. Running/failed stages remain visible
in the status inventory but their partial metrics do not enter the comparison.
It never opens a dataset, loads a model or changes a frozen selection. The
sequential runner also refreshes this report after completing its stages.

Separate-run timings can vary with system load and clock/thermal conditions.
Before making cost claims, benchmark the reference, bias, expanded specialist
and repaired stack together after research fitting finishes:

```powershell
.\.venv\Scripts\python.exe -B src/benchmark_multiclass_study.py --run-dir outputs/studies/8class_v1
.\.venv\Scripts\python.exe -B src/benchmark_multiclass_study.py --run-dir outputs/studies/34class_v1
```

Use 100,000 common stratified validation rows, full-cohort warm-ups and one full
rotation of measured passes per model (four models normally; five with final
specialist decision tuning). Every model occupies every position
equally. Record complete bundle prediction cost with two CPU threads. The timing
subset supplies no new accuracy estimate; the F1/latency plot uses each saved
model's existing complete-validation score. Preserve both this paired benchmark
and the earlier full-cohort timing scopes. No model fitting may overlap these
benchmarks. Outputs live in each study's `paired_timing_v1/`.
