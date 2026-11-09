# Project structure and artifact policy

## Code boundaries

The repository has three workflows. `run.py` is binary-only. The dashboard opens
on a project-wide Overview with the research journey; Navigation → Workflows
contains the binary / eight-class / 34-class task selector.
Keep command-line entry points thin and implementation code inside `src/ddos_iot`.

| Workflow | Entry points | Implementation |
|---|---|---|
| Binary Benign vs DDoS | `run.py`, `src/preprocess.py`, `src/train.py`, `src/compare_models.py`, `src/predict.py` | `preprocessing.py`, `modeling.py`, `explainability.py` |
| 34 original labels | `src/multiclass.py` | `multiclass.py` |
| Eight attack families | `src/multiclass_8.py` | `attack_families.py` plus the shared `multiclass.py` |
| Saved multiclass prediction | `src/predict_multiclass.py` | `multiclass_inference.py`; loads the stored transformation and estimator |
| Saved multiclass SHAP | `src/explain_multiclass.py` | `multiclass_shap.py` (computation/exports), `multiclass_shap_plots.py` (plots from arrays) |
| Dashboard | `app.py` | `dashboard.py` (routing and binary views), `dashboard_multiclass.py` (8/34 views) |
| Project Overview / research journey | Dashboard default **Navigation → Overview** | `dashboard_journey.py` (one page with Binary / 8-class / 34-class narratives, line charts and selected results), `journey_charts.py` (binary comparison and confirmed validation charts), `journey.py` (bounded report readers, recorded comparison order and task/phase/seed validation); no training, model deserialization or raw-data reading |
| Dashboard background training | `src/dashboard_worker.py` | `dashboard_runs.py` (JSON discovery, safe launch and status) |
| Controlled multiclass ablations | `src/experiment_multiclass.py` | `experiments/` (config, data, training, OOF, decisions, evaluation, storage, orchestration), shared `weighting.py` |
| Controlled validation reports | `src/report_multiclass_experiment.py` | `experiments/reporting.py`; reads metrics without loading models or datasets |
| Class-level overlap audit | `src/audit_multiclass_overlap.py` | `experiments/overlap_audit.py`; streams the training source and audits saved row identities |
| Paired post-training CPU timing | `src/benchmark_multiclass_experiment.py` | `experiments/benchmarking.py`; full validation warm-up and rotating-order measurement |
| Recall-constrained OOF retuning | `src/retune_multiclass_recall.py` | `experiments/recall_constraints.py` (pure search), `constrained.py` (cached-run orchestration) |
| Recall operating-point reports | `src/report_recall_tuning.py` | `experiments/constrained_reporting.py`; metrics-only reports |
| Paper baseline audit | `src/audit_paper_baseline.py` | `experiments/baseline_audit.py`; AST inspection of pinned author source without executing it |
| Published training subset preparation | `src/prepare_paper_training.py` | `experiments/reference_data.py`; archive checksum, training-only extraction, schema/count inventory |
| Staged published-data training | `src/study_multiclass.py` | `experiments/study_config.py`, `study_data.py`, `study_training.py`, `study.py` |
| Frozen finalist confirmation | `src/confirm_multiclass_study.py` | `experiments/study_confirmation.py`; fixed parameters across split seeds |
| Conditional specialist/stack | `src/specialize_multiclass_study.py`, `src/stack_multiclass_study.py` | `study_specialist.py`, `study_stacking.py`, serializable `study_models.py` |
| Single-XGBoost OOF biases | `src/bias_multiclass_study.py` | `study_bias.py`, shared immutable OOF cache, coordinate and `joint_bias.py` searches |
| Final hierarchical OOF decisions | `src/bias_multiclass_specialist.py`, `src/confirm_multiclass_specialist_bias.py` | `study_specialist_bias.py`; routed specialist OOF, frozen decisions and shared-probability confirmation |
| Advanced three-seed confirmation | `src/confirm_multiclass_advanced.py` | `study_advanced_confirmation.py`; frozen bias/specialist/stack choices |
| Exact-vector label collision audit | `src/audit_study_collisions.py` | `study_collisions.py`; exploratory training-only empirical ambiguity |
| Paper software-version reference | `src/train_published_reference.py`, `src/score_published_reference.py` | `paper_reference.py`; isolated Python/runtime and predeclared final reference |
| Sequential research execution | `src/run_multiclass_research.py` | `research_pipeline.py`; one-time execution, logs, complete-stage reuse, explicit final-test flag |
| Published-study final evaluation | `src/evaluate_multiclass_study.py` | `study_final.py`, `study_advanced_final.py`; all final models saved before once-only test access |
| Published-study reports/prediction | `src/report_multiclass_study.py`, `src/predict_multiclass_study.py` | `study_reporting.py`; published scaled-format contract |
| Cross-stage research comparison | `src/report_multiclass_research.py` | `research_reporting.py`; completed metrics, per-class seed summaries and F1/false-alarm trade-offs |
| Paired published-study CPU timing | `src/benchmark_multiclass_study.py` | `study_benchmark.py`; common validation cohort and rotating model order, with no fitting |
| Frozen-test paired uncertainty | `src/report_multiclass_uncertainty.py` | `study_uncertainty.py`; prespecified label-stratified paired bootstrap on saved test predictions |
| Joint final protocol | `src/finalize_multiclass_research.py` | `study_protocol.py`; freezes both task choices before either test and executes only the frozen protocol |
| Final research outcome | `docs/MULTICLASS_FINAL_RESULTS.md` | Official test metrics, per-class recall shortfalls, pinned paper reproduction, uncertainty, saved models and audited timing |

Common helpers: `artifacts.py` handles atomic writes; `paths.py` handles shared
directories; `utils.py` handles logging/JSON; `config.py` holds binary constants.
`dashboard_style.py` provides shared page headers and styling for every workflow;
`.streamlit/config.toml` defines the light theme and chart palette, including Docker.
The configurable multiclass grid example is `configs/multiclass_grid.json`.
Controlled feature/weight/bias experiments have separate configurations in
`configs/experiments/` and a [protocol and artifact guide](MULTICLASS_EXPERIMENTS.md).
Their selection module compares the highest Macro-F1 with the best candidate
meeting the reference Benign false-positive budget in every declared seed.
This new CLI workflow uses a training-only development stage and a separately
invoked frozen test evaluation. It is not launched by the existing dashboard.

Do not move/rename `MulticlassBundle` or `FeatureTransform` without a migration:
existing joblib artifacts refer to `ddos_iot.multiclass` when reloaded. The shared
engine is intentionally retained at that path rather than split into duplicate
eight-class and 34-class implementations.

## Data and result locations

```text
data/raw/                  Original datasets; never overwrite during experiments
data/processed/            Binary preprocessed train/test files
models/<run-name>/         Real training artifacts and experiment reports
outputs/predictions/       Prediction exports produced from saved models
outputs/experiments/       Controlled ablations, OOF arrays, validation/final models
outputs/baseline_audit_v1/ Pinned source metadata and paper/local protocol comparison
outputs/studies/           Published-data training studies and conditional models
configs/                   User-editable experiment configuration examples
tests/                     Maintained regression tests (keep these)
docs/                      Project documentation
tmp/                       Disposable experiments; ignored by version control
.cache/                    Runtime caches; not scientific results
.venv/                     Installed Python environment
```

Do not rename active run directories. Keep interrupted full-data runs: their CV
reports are useful evidence, even when they do not yet contain a final model.
Sample-count reports are data inventory, not disposable model smoke tests.

Previously tracked `models/**/model.joblib` bundles, companion JSON reports, and
`model_comparison.csv` summaries remain in Git. New model bundles stay local in
the website workflow. Multiclass bundles contain
the trained estimator, selected features, fitted scaler, and label encoder, so
they can be reused without training. Binary bundles contain their estimator and
feature columns and accept processed input through `src/predict.py`.

The small `confusion_matrix.csv` and `classification_report_{train,test}.csv`
evaluation reports are also tracked. **Model Comparison** shows model-level
score and timing tables, per-class precision/recall/F1/support, and a confusion
matrix as a heatmap and table. Matrix rows are actual labels and columns are
predicted labels; switch between raw counts and row percentages. Binary runs
also support selecting the train or test split using their saved JSON counts.
These views require neither training datasets nor model retraining.

Controlled CLI runs use `outputs/experiments/`, not the legacy flat `models/`
layout. Their reports and reusable bundles are retained locally; the website
publishes only the display reports listed in `configs/website_reports.json`.
Validation reports are explicitly named as
validation and are not presented as test scores in the existing dashboard.

Raw and processed datasets, prediction CSVs, training samples, plots, and SHAP
exports remain local and ignored. Recorded absolute paths in experiment reports
describe the original training computer; use the checked-out model path when
running predictions on another computer.

Published reference data stay under `data/raw/paper_v3/`, separate from the
existing local train/test files. Only the 8-class and 34-class training CSVs are
extracted by the preparation command. Their 23 selected/scaled columns cannot
replace the 46-feature input required by controlled feature-selection ablations.
Recall-retuning outputs are separate child runs; they never overwrite the
original configuration, validation reports, models, or frozen selection.
Published-data studies have a separate [protocol](MULTICLASS_TRAINING_STUDIES.md).
Their 23-feature identity transform consumes preselected/scaled values and must
not be mistaken for a raw-flow deployment scaler. Keep OOF arrays and screening
checkpoints local; retain frozen reports and reusable finalist models.

On Streamlit Community Cloud, zero raw CSV files and missing dataset/output
directories are expected until those files are supplied or generated on that
server. The saved binary models and comparison reports remain available. Use
the Binary **Predict** page with a processed CSV to reuse them. Disable
**Generate local SHAP** when the processed training/background dataset is absent.

## Multiclass persistence lifecycle

1. Read/prepare data and perform the requested CV/grid search.
2. Train a model with the chosen settings and perform the full-data refit.
3. Immediately save its estimator, selected columns, scaler, and label encoder
   together in `<run>/<model>/model.joblib`, using a same-directory temporary
   file and atomic replacement.
4. Record saved-model status, then generate test predictions, metrics, and plots.
5. Continue to the next requested model. A later failure does not remove earlier
   saved models. A failed evaluation can still leave a reusable trained bundle.

The default eight-class run directory is `models/multiclass_8_tuned`; the default
34-class directory is `models/multiclass_34_tuned`. Other runs can choose their own
`--output-path`. Training refuses to overwrite existing experiment outputs.

Use `src/predict_multiclass.py` to load an existing bundle with no model fitting.
This is inference reuse, not automatic resumption of interrupted grid search.
Existing binary models use the separate `src/predict.py` interface.

## Dashboard integration

Select **Classification problem** in the sidebar. Eight-class and 34-class views
share the same training and inference engine as the CLI; there is no separate
dashboard implementation of the machine-learning pipeline.

- **Overview:** experiments, saved bundles, recorded CV progress, failures and logs.
- **Run Pipeline:** explicit launch into a new `models/<run-name>` directory.
  Five-fold grid search is enabled by default. GPU is selected for XGBoost and
  CatBoost; LightGBM defaults to CPU to avoid the observed OpenCL failure.
- **Dataset:** stored class counts, feature gains, configuration, and the exact
  eight-family mapping. It does not rescan gigabytes of data on each page load.
- **Model Comparison:** metrics, classification reports, confusion matrices and
  per-model grid search results. Current defaults do not relabel old experiments.
- **Predict:** choose a saved bundle and raw CSV (local path or upload); stream
  CPU predictions without fitting or CV and download a uniquely named export.
- **SHAP:** generate bounded explanations from a saved model, then select a class
  and sampled row to inspect. `dashboard_multiclass_shap.py` isolates this UI.
  Existing artifacts can be viewed without loading or recomputing the model.

Dashboard-started training runs in a hidden background Python worker, independent
of Streamlit reruns. `dashboard_job.json` records its exit status and
`dashboard_training.log` captures output. Refresh reports to see new checkpoints.
CLI runs are also discovered from their existing JSON reports. A last-reported
`running` status is not proof that an external process is still alive (for example
after a forced termination). The dashboard neither stops nor resumes such jobs.

An explicit confirmation is required to launch training; no training occurs when
browsing pages or loading a saved model. Existing experiment directories are
rejected. Binary SHAP is unchanged. Both multiclass tasks use the separate
[multiclass SHAP workflow](MULTICLASS_SHAP.md), explicitly run after training;
it is never added inside cross-validation.

## Cleaning and tests

Keep regression test **source files**. They use automatically cleaned temporary
subdirectories under `tmp/`, so running tests should not fill `models/` with
smoke artifacts. Manual small experiments should also use `--output-path tmp/...`.

```powershell
python -B -m unittest discover -s tests -v
```

GPU integration tests are opt-in via `DDOS_TEST_GPU=1`; leave them off when a real
GPU training run is active. `-B` avoids creating Python bytecode cache files.

Remove only verified test outputs, not raw/processed data, real models, active
training directories, user prediction exports, or the Python environment.
Prefer the Windows Recycle Bin for recovery instead of permanent deletion.
