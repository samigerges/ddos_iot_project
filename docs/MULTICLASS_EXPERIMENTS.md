# Controlled multiclass experiments

This workflow measures the cheap improvements before considering a stack or a
specialist. It is a new, CLI-driven experiment harness. The existing binary,
dashboard, paper-style eight-class, prediction, and SHAP workflows retain their
entry points. Existing saved model class/module paths remain compatible.

## Module boundaries

```text
src/experiment_multiclass.py              Thin plan/run/evaluate-test CLI
src/ddos_iot/weighting.py                 Normalized inverse-frequency powers
src/ddos_iot/experiments/
    config.py                           Validated, immutable run/candidate settings
    data.py                             Raw-row identities, common splits, overlap audit
    training.py                         Fold-local selection, early stopping, refit
    oof.py                              Cross-fitted probabilities and fold provenance
    decisions.py                        Multiclass bias search and decision rule
    evaluation.py                       Per-class/family errors and CPU timing
    selection.py                        Per-seed Benign FPR budgets and winner comparison
    reporting.py                        Read-only Markdown/CSV/PNG/SVG comparisons
    overlap_audit.py                    Bounded training-source hash and class overlap audit
    benchmarking.py                     Full-warm-up, paired post-training CPU timing
    storage.py                          Atomic reports and array exports
    runner.py                           Development selection and frozen final evaluation
configs/experiments/{8class,34class}.json Full-data starting configurations
tests/test_experiments.py                Isolation and end-to-end regressions
```

The harness reuses the existing feature transform, model builders, fitting code,
and `MulticlassBundle`. It does not duplicate the eight- and 34-class trainers.
`MulticlassBundle` and `FeatureTransform` stay in `ddos_iot.multiclass`, because
previous joblib files refer to that module. Bundles can now carry a decision bias;
old bundles without that field still predict normally.

## Development protocol

1. Load only the existing raw **training** CSV, checking all 46 numeric features.
   Retain original zero-based CSV data-row IDs, valid duplicates, and all 34 raw
   labels. Missing/nonfinite rows are removed with counts recorded.
2. Split development/validation on the **raw 34 subtypes**, even in eight-class
   mode. With matching inputs, cap, data seed, and model seeds, both tasks use the
   same raw rows for development, validation, and OOF folds. The eight-class
   family mapping changes only the model targets.
3. For each development OOF fold, create a separate inner early-stopping split.
   Fit the selector/scaler only on inner fitting rows. Select boosting rounds
   with inner early stopping, refit on the outer training fold, and predict the
   untouched OOF fold. No outer scoring or independent validation row enters
   fitted preprocessing, weighting, early stopping, or the model.
4. Tune class biases on development OOF probabilities. Select classes with OOF
   F1 below `bias_f1_cutoff`; anchor the largest training class at zero. Search
   positive and negative log-score offsets using a bounded coordinate grid.
   The decision is `argmax(log(max(p, 1e-12)) + bias)`. Exactly one class wins.
5. Refit each candidate on development rows, then evaluate argmax and OOF-tuned
   bias decisions on independent validation rows. Choose the candidate/decision
   with the highest mean validation Macro-F1 over all declared seeds, subject to
   the configured Benign FPR guardrail. Exact ties favor argmax, then candidate
   declaration order.
6. Record the winning configuration, per-seed biases, source SHA-256, and class
   mapping in a frozen `selection.json`. The test file has not been opened.

OOF scores after bias optimization are **training diagnostics**, not independent
performance estimates. Validation selects the experiment, so its winner's score
also has selection optimism. Final test metrics are the held-out result. If you
need an unbiased CV estimate of the entire selection procedure, add an outer CV
loop around this complete development workflow.

### False-positive-aware selection

New runs default to `selection_policy: "reference_fpr"`. The reference is the
argmax result of `reference_candidate`, or the first declared candidate when the
name is omitted. A candidate/decision is eligible only when it completes every
declared seed and its validation Benign FPR does not exceed that seed's reference
FPR plus `max_benign_fpr_increase`. The allowance is an **absolute fraction**:
`0.005` would mean half a percentage point; the full-data presets use `0.0`.

The constraint is checked **in every seed**, not only on the mean. Missing
reference results cannot silently disable it. The reference itself is a fallback;
a higher score that requires more false positives is reported but not chosen.
Exact rate boundaries are allowed, with a tiny floating-point tolerance.

Both the unconstrained winner and the guarded winner are retained in
`selection_report.json`, together with per-seed reference rates and the policy.
The comparison CSV contains each row's reference, limit, delta, and pass/fail;
the summary identifies complete and eligible candidate/decision groups. The
selected guarded result is frozen in `selection.json` for final evaluation.

OOF bias optimization itself remains unconstrained: the guardrail selects among
the resulting argmax/bias choices using independent validation, without retuning
on validation labels. It is a validation constraint, not a guarantee of the
false-positive rate on unseen traffic. Final test scores are reported without
changing the selected decision rule.

Set `selection_policy: "macro_f1"` to retain the earlier score-only selection;
the guarded alternative remains visible. The historical pilot presets explicitly
use that policy, and the original frozen pilot selections are unchanged. Older
frozen configurations remain loadable when they lack the new optional fields.

The selector always uses inverse-frequency weights with power 1, held constant
across the weight ablation. Candidate weight powers affect classifier fitting
and weighted early-stopping loss. This keeps feature selection from becoming an
unintended second change in a weight-only comparison. Median selection retains
gain strictly above the median of all 46 features; ties mean the count is not
guaranteed to equal 23. Every fold reports the actual selected columns.

## Starting experiments

The default six candidates are:

| Name | Features | Classifier weight power |
|---|---|---:|
| `median_balanced` | Above-median gain | 1.0 |
| `all_balanced` | All 46 | 1.0 |
| `median_unweighted` | Above-median gain | 0.0 |
| `all_unweighted` | All 46 | 0.0 |
| `all_power_050` | All 46 | 0.5 |
| `all_power_075` | All 46 | 0.75 |

Each power starts with `(N / (K * count)) ** power`, then is normalized so mean
**sample** weight is one. This holds the overall loss scale constant.

From the repository root, preview a run without reading either dataset or
starting training:

```powershell
.\.venv\Scripts\python.exe -B src/experiment_multiclass.py plan --config configs/experiments/8class.json
```

Run eight-class development, then the 34-class configuration separately:

```powershell
.\.venv\Scripts\python.exe -B src/experiment_multiclass.py run --config configs/experiments/8class.json
.\.venv\Scripts\python.exe -B src/experiment_multiclass.py run --config configs/experiments/34class.json
```

Defaults use XGBoost GPU training, two CPU threads, five OOF folds, seed 42, and
three CPU timing repeats. Six candidates require 72 model fits per seed because
every fold and development refit includes early stopping followed by refitting.
Set `device` to `cpu` on machines without CUDA. LightGBM, if configured as a
candidate, stays on CPU; CatBoost fits on the configured device and scores on CPU.

Input/output paths in JSON resolve against the repository root. Choose a fresh
`output_path` for each run. For a smaller pilot, copy a config and explicitly set
`max_train_per_raw_class` and a lower `n_estimators`; the sampling reservoir scans
the whole training CSV so late rare classes are retained. Such metrics describe
the sampled training experiment, not the full-data baseline. The `data_seed`
controls that reservoir independently of model/split seeds.

`8class_pilot.json` and `34class_pilot.json` are ready-made bounded runs with
at most 2,000 training rows per original subtype and 100 boosting rounds. They
still scan the complete source, use five OOF folds, and leave the test unread.
They deliberately change training class prevalence and cannot establish a gain
over previously saved full-data results.

Use a `candidates` array to run a subset, for example:

```json
{
  "num_classes": 8,
  "output_path": "outputs/experiments/8class_four_ablations",
  "device": "gpu",
  "candidates": [
    {"name": "median_balanced", "feature_selection": "median", "weight_power": 1},
    {"name": "all_balanced", "feature_selection": "none", "weight_power": 1},
    {"name": "median_unweighted", "feature_selection": "median", "weight_power": 0},
    {"name": "all_unweighted", "feature_selection": "none", "weight_power": 0}
  ]
}
```

Keep model hyperparameters fixed for the first feature/weight comparisons. Later
confirm promising candidates in a new run with `seeds: [42, 123, 2026]`. Summary
standard deviations are population standard deviations across those runs, not
CV fold standard deviations or confidence intervals. Report one-seed results as
one-seed results, even though the summary's standard deviation is zero.

## Artifacts and error analysis

```text
<run>/
    config.json                         Resolved configuration snapshot
    experiment_report.json              Status, versions, progress, failures
    data_report.json                    Training cleaning/counts/source SHA-256
    label_mapping.json
    validation_comparison.csv           One row per seed/candidate/decision
    validation_summary.csv              Mean/std across seeds
    validation_deltas.csv               Paired changes from first candidate's argmax
    selection.json                      Frozen validation winner
    selection_report.json               Unconstrained and FPR-guarded winners
    seed_42/
        splits.npz                      Original row IDs, targets, class order
        overlap.json                    Development/validation feature overlap
        <candidate>/
            model.joblib                Development-trained argmax bundle
            training.json               Timing, selected features, weights, rounds
            bias.json                   OOF-tuned offsets and tuning history
            oof/
                probabilities.npy       Memory-mapped float32 [rows, classes]
                rows.npz                Matching original row IDs, y, fold IDs
                report.json             Completion status and fold provenance
                {argmax,bias}/          Training diagnostic reports
            validation/{argmax,bias}/
                metrics.json
                classification_report.csv
                confusion_matrix.csv    Actual labels in rows, predictions in columns
                confusions.csv          Off-diagonal counts and row fractions
                predictions.npz
                family_confusion_matrix.csv  34-class mode only
```

Read an OOF array only when its `report.json` says `complete`. Every row appears
in exactly one scoring fold. Column order is in the run label mapping, and raw
row IDs allow alignment across models/tasks. Partial files are retained for
diagnosis; neither checkpoints nor partially populated directories implement
automatic resume. A candidate must complete every declared seed to be eligible.

Metrics include Macro-F1, weighted-F1, accuracy, balanced accuracy, Benign false
positive rate, and attack-to-Benign rate. The 34-class report additionally splits
within-family and between-family error counts and reports family Macro-F1 after
mapping predicted subtypes into families.

Latency is the median of repeated CPU batch predictions, including feature
transformation, probability prediction, and the selected decision rule. Warm-up
and CSV loading are excluded. Per-sample batch throughput is **not** measured
single-flow latency. The batch size, repetitions, device, and each timing are
recorded. Development timing separates OOF, bias optimization, and model fits.

Overlap audits count validation/test feature vectors also present in training,
after float32 parsing, using feature hashes with labels excluded. They are
diagnostics, not an automatic deduplication step or proof of capture independence.
Use a separately declared grouped/capture/time protocol for a stronger
generalization claim when the source metadata permits it.

## Read-only comparison report

Render saved metrics without loading models or touching either raw dataset:

```powershell
.\.venv\Scripts\python.exe -B src/report_multiclass_experiment.py --run-dir outputs/experiments/8class_v1
```

This writes `report/REPORT.md`, comparison CSVs, and standalone PNG/SVG plots of
Macro-F1 against Benign FPR and CPU inference cost. It also exports precision,
recall, F1, and support for every class and seed, mean/std class summaries, and
two target-class precision/recall operating-point plots. These connect the
evaluated argmax/bias decisions, not a full threshold sweep. To inspect completed results
while a run continues, add `--allow-partial`; outputs then live in
`provisional_report/` and carry an explicit provisional label. Run the command
again to refresh the snapshot. It does not alter training or frozen selections.
`training_costs.csv` distinguishes the development fit, OOF fitting and total
elapsed OOF work, and bias optimization. Only one candidate is used for a model
prediction; the six variants are an experiment comparison, not an ensemble.

## Paired post-training timing

For the reference and selected model, measure CPU inference after training with
a full validation warm-up and five rotating-order repeats:

```powershell
.\.venv\Scripts\python.exe -B src/benchmark_multiclass_experiment.py --run-dir outputs/experiments/8class_v1 --seed 42 --repeats 5
```

This rechecks saved predictions exactly and writes raw timings/ranges in
`report/paired_benchmark_seed_42/`. It reads training-source validation rows only.
It does not retrain or access the test set. Refresh the read-only report to add
the timing table. Interpret in-run timings cautiously when cold-start effects
or hardware scheduling cause differences from the warmed paired measurement.

## Class-level overlap audit

For a class-level check of a saved development/validation split, use:

```powershell
.\.venv\Scripts\python.exe -B src/audit_multiclass_overlap.py --run-dir outputs/experiments/8class_v1 --seed 42
```

It verifies the original training-source SHA-256, scans that source in bounded
chunks, and confirms the recomputed total against the workflow's saved audit.
Outputs in `overlap_audit/seed_42/` include overlap by task class and by original
subtype, plus a validation mask aligned to the saved row IDs. Features are hashed
without labels, so identical vectors carrying conflicting labels still share a
group. Invalid source rows omitted from the saved split remain omitted from its
counts. The command never reads the test CSV or trains a model. Existing audit
outputs are protected; choose `--output-dir` for a separate repeated audit.

Filtering existing validation predictions to vectors absent from development is
an exploratory diagnostic. It does not replace grouped retraining and must not
be presented as independent final-test performance or silently replace the
declared model-selection cohort.

## Recall-constrained retuning from cached OOF predictions

After a completed six-candidate comparison, preserve reference recall for named
attack classes while bounding Benign false alarms. Limits are taken from each
seed's reference **OOF argmax** predictions and applied to all candidates:

```powershell
.\.venv\Scripts\python.exe -B src/retune_multiclass_recall.py --run-dir outputs/experiments/8class_v1 --output-dir outputs/experiments/8class_v1/recall_constraints_all_targets_v1 --protect WebBased Recon BruteForce --export-selected
```

This command validates saved OOF row/label/fold alignment, searches a declared
fixed bias grid from zero and the earlier OOF bias, and prioritizes feasibility
before Macro-F1. Once a search path is feasible it cannot accept a violating
move. Failure to find a feasible grid point is not proof of model infeasibility.
The largest class is the fixed zero-bias anchor; all other classes may be tuned.

All biases and the candidate with the best mean feasible OOF Macro-F1 are
persisted and hashed **before** validation features or labels are loaded.
Validation assesses transfer and compares recalls/FPR against the same seed's
validation reference; OOF feasibility does not guarantee that check will pass.
Its metrics are never fed back into this search or its candidate selection.
The existing validation cohort has already been used in earlier experiments,
so this is not a new final test or an independent estimate of OOF tuning scores.

The default protected classes are WebBased and BruteForce. Use exact family or
raw label names with `--protect`; the same machinery supports 34-class caches.
No recall constraint is silently imposed on classes omitted from that list.
Compare every class in `per_class_comparison.csv` to detect shifted errors.
All training is reused, predictions from reloaded models must reproduce the
original validation argmax outputs, and the original test CSV remains unread.

Outputs include `REPORT.md`, OOF feasibility diagnostics, all original and new
validation operating points, confusion matrices, and `selection.json`. Optional
`benchmark_models/seed_<seed>/model.joblib` exports automatically apply the bias
through the existing prediction API. They remain development-fitted research
bundles, with their original transformation and class mapping.

**The parent run's `selection.json` is unchanged.** Its existing `evaluate-test`
command still selects the earlier validation winner, not the constrained child
run's choice. Final test evaluation must wait for baseline reconciliation,
policy review, multiseed confirmation, and an explicitly frozen final protocol.

Regenerate a completed report without loading models/data:

```powershell
.\.venv\Scripts\python.exe -B src/report_recall_tuning.py --run-dir outputs/experiments/8class_v1/recall_constraints_all_targets_v1
```

## Auditing the paper baseline

The [baseline audit](MULTICLASS_BASELINE_AUDIT.md) records the public versioned
source, checksums, and material protocol differences. Source inspection parses
Python syntax and never imports or executes downloaded author scripts:

```powershell
.\.venv\Scripts\python.exe -B src/audit_paper_baseline.py --run-dir outputs/experiments/8class_v1
.\.venv\Scripts\python.exe -B src/prepare_paper_training.py
```

The second command requires the downloaded version-3 `datasets.tar.xz` and its
published record metadata. It verifies the archive's published size/MD5, computes
SHA-256, and only extracts the explicitly named multiclass training CSVs. It
checks their numeric features and label inventory without parsing test CSVs.
These subsets already contain 23 selected/scaled features: they support paper
fidelity work, not a new claim of fold-local feature selection or an all-46
feature ablation. Keep paper fidelity and the stricter local protocol distinct.

## Final test evaluation

Inspect validation reports and freeze your experiment choice before invoking:

```powershell
.\.venv\Scripts\python.exe -B src/experiment_multiclass.py evaluate-test --run-dir outputs/experiments/8class_v1
```

This verifies the frozen configuration/selection and unchanged training source,
trains the selected candidate on all configured training rows for every declared
seed, and saves every final model **before opening the test CSV**. It then
evaluates those models with the previously tuned biases. No seed or decision is
selected using test performance. Results and deployable bundles are written to
`test_evaluation/seed_<seed>/`, with `test_summary.csv` and a status report above.
Existing final-evaluation outputs are never overwritten.

Keep controlled run directories under `outputs/experiments/`, separate from the
legacy dashboard's flat `models/` report format. Run directories can be relocated;
the frozen configuration retains its original output path as provenance, while
`evaluate-test --run-dir` locates reports and writes final outputs beside them.

Final bundles work with the existing prediction-only CLI and apply their stored
bias automatically. `predict_proba` exposes the estimator's probabilities before
bias adjustment; weighted-model outputs are not claimed to be calibrated natural
traffic probabilities. Raw-score SHAP explanations describe the underlying
classifier, not the postprocessing decision rule.

This protocol is a controlled local baseline, not an exact reproduction of the
paper's sampled data and split. Stacking, specialist routing, grouped splitting,
and hyperparameter search are deferred until these ablations identify the
remaining errors. None is silently enabled by this harness.

## Verification

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Tests cover matching raw rows across tasks, fold-local preprocessing, development
with a missing test CSV, both bias directions, normalized weight powers, frozen
selection checks, and persisted-model prediction agreement for both task sizes.
