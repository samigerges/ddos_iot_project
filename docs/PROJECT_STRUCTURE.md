# Project structure and artifact policy

## Code boundaries

The repository has three workflows. `run.py` is binary-only; the dashboard has
an explicit binary / eight-class / 34-class task selector.
Keep command-line entry points thin and implementation code inside `src/ddos_iot`.

| Workflow | Entry points | Implementation |
|---|---|---|
| Binary Benign vs DDoS | `run.py`, `src/preprocess.py`, `src/train.py`, `src/compare_models.py`, `src/predict.py` | `preprocessing.py`, `modeling.py`, `explainability.py` |
| 34 original labels | `src/multiclass.py` | `multiclass.py` |
| Eight attack families | `src/multiclass_8.py` | `attack_families.py` plus the shared `multiclass.py` |
| Saved multiclass prediction | `src/predict_multiclass.py` | `multiclass_inference.py`; loads the stored transformation and estimator |
| Saved multiclass SHAP | `src/explain_multiclass.py` | `multiclass_shap.py` (computation/exports), `multiclass_shap_plots.py` (plots from arrays) |
| Dashboard | `app.py` | `dashboard.py` (routing and binary views), `dashboard_multiclass.py` (8/34 views) |
| Dashboard background training | `src/dashboard_worker.py` | `dashboard_runs.py` (JSON discovery, safe launch and status) |

Common helpers: `artifacts.py` handles atomic writes; `paths.py` handles shared
directories; `utils.py` handles logging/JSON; `config.py` holds binary constants.
The configurable multiclass grid example is `configs/multiclass_grid.json`.

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
