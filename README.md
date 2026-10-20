# IoT Intrusion Detection

Three separate CICIoT2023 workflows: binary Benign vs DDoS, 34 original labels,
and eight attack families. See [project structure](docs/PROJECT_STRUCTURE.md)
for entry points, shared modules, and artifact locations.

## Binary Pipeline (the default for run.py and the dashboard)

1. Keep only Benign and DDoS rows.
2. Map labels to `0 = Benign`, `1 = DDoS`.
3. Remove duplicates, missing values, and infinite values.
4. Drop useless columns such as constants, duplicate features, and obvious IDs.
5. Encode categorical features if any exist.
6. Use `data/raw/train/train.csv` as the training source and `data/raw/test/test.csv` as the test source.
7. Rebalance the training split only.
8. Select top features using train-only target correlation.
9. Train and compare supported classifiers.
10. Save processed data, model artifacts, predictions, metrics, and optional SHAP explanations.

## Code Structure

- `run.py`: one-command entry point for preprocessing plus model comparison.
- `app.py`: thin Streamlit dashboard entry point.
- `src/preprocess.py`: command-line entry point for preprocessing.
- `src/train.py`: command-line entry point for model training and evaluation.
- `src/compare_models.py`: command-line entry point for training and comparing multiple models.
- `src/predict.py`: command-line entry point for prediction.
- `src/multiclass.py`: 34-class training entry point (also accepts `--num-classes 8`).
- `src/multiclass_8.py`: eight-family training entry point.
- `src/predict_multiclass.py`: reuse a saved 8/34-class model without training.
- `src/explain_multiclass.py`: class-aware SHAP for saved 8/34-class models, without training.
- `src/ddos_iot/dashboard.py`: Streamlit dashboard implementation.
- `src/ddos_iot/dashboard_multiclass.py`: 8/34-class dashboard views.
- `src/ddos_iot/dashboard_runs.py`: experiment discovery and safe background training launches.
- `src/dashboard_worker.py`: background training worker with persistent status/logs.
- `src/ddos_iot/preprocessing.py`: reusable preprocessing functions.
- `src/ddos_iot/modeling.py`: reusable model, metric, plot, and prediction functions.
- `src/ddos_iot/multiclass.py`: shared multiclass training, bundles, and evaluation.
- `src/ddos_iot/attack_families.py`: explicit family mapping and paper-style sampling.
- `src/ddos_iot/multiclass_inference.py`: streaming prediction with saved multiclass bundles.
- `src/ddos_iot/multiclass_shap.py`: sampled native SHAP, optional permutation fallback and exports.
- `src/ddos_iot/multiclass_shap_plots.py`: class/sample plots from saved SHAP arrays.
- `src/ddos_iot/dashboard_multiclass_shap.py`: multiclass SHAP generation and viewing controls.
- `src/ddos_iot/artifacts.py`: atomic artifact writes.
- `src/ddos_iot/paths.py`: shared project paths and cache/output locations.
- `src/ddos_iot/config.py`: shared constants.
- `src/ddos_iot/utils.py`: logging, JSON, and small helper utilities.

## Setup

```powershell
pip install -r requirements.txt
```

The project uses `pandas`/`numpy` for preprocessing, `scikit-learn` for the shared training flow, and optional XGBoost/LightGBM estimators for comparison.

## Run Everything

Run preprocessing, train all requested models, compare them, and save the best-model report:

```powershell
python run.py
```

Run the full pipeline and also generate SHAP global/local explanations for the selected best model:

```powershell
python run.py --explain-shap
```

Useful options:

- `--train-input-path data/raw/train/train.csv`
- `--test-input-path data/raw/test/test.csv`
- `--processed-path data/processed`
- `--models-output-path models/requested_model_comparison`
- `--skip-preprocess` to reuse existing processed files
- `--shap-for-all-models` to explain every successful model instead of only the best model

## Dashboard

Run the local Streamlit dashboard:

```powershell
streamlit run app.py
```

Use **Classification problem** in the sidebar to switch between **Binary**, **8 classes**,
and **34 classes**. Binary keeps its preprocessing, comparison, prediction and SHAP pages.
Both multiclass tasks have Overview, Run Pipeline, Dataset, Model Comparison, SHAP, and Predict
pages, including stored class counts, CV progress, confusion matrices, and saved-model reuse.

Multiclass training runs in a background worker, so switching pages does not interrupt
it. Start is explicit, requires confirmation, and only accepts a new experiment directory.
Five-fold grid search is on by default; XGBoost/CatBoost default to GPU, with LightGBM
on CPU because of the previously observed OpenCL error. No job starts merely by opening
the dashboard. Refresh reports to read new progress. Existing CLI experiments are also
visible; statuses reflect the last saved checkpoint, not live process monitoring.

The multiclass **Predict** page loads an existing bundle and accepts a raw-feature CSV
upload or local path. It never retrains and writes a unique prediction export. Only load
trusted local joblib artifacts.

The multiclass **SHAP** page explains saved XGBoost, LightGBM and CatBoost models
without retraining. It includes overall/class-specific importance, beeswarm plots,
and a class/sample selector for local waterfalls. Native SHAP explains raw scores,
not probability changes. The default samples up to 200 valid rows from the first
50,000 CSV rows; set the scan limit to 0 for full-file sampling. See
[multiclass SHAP](docs/MULTICLASS_SHAP.md) for sampling, units, fallback and CLI use.

## Preprocess

Use the raw train CSV for training and the raw test CSV for evaluation:

```powershell
python src/preprocess.py --train-input-path data/raw/train/train.csv --test-input-path data/raw/test/test.csv --output-path data/processed --top-features 25 --rebalance undersample
```

Outputs:

- `train.csv`
- `test.csv`
- `selected_features.json`
- `feature_scores.csv`
- `preprocessing_report.json`

## Train

```powershell
python src/train.py --processed-path data/processed --output-path models/lightgbm --model lightgbm
```

The default uses `--n-jobs 1` for Windows compatibility. You can raise it on machines where multiprocessing works well.

To add SHAP global and local explanations for one trained model:

```powershell
python src/train.py --processed-path data/processed --output-path models/lightgbm --model lightgbm --explain-shap
```

Available models:

- `random_forest`
- `xgboost`
- `hist_gradient_boosting`
- `lightgbm`
- `naive_bayes`

## Compare Models

Train multiple models on the full processed dataset and rank them by a selected metric.
The comparison includes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix counts, training time, and test inference time.

```powershell
python src/compare_models.py --processed-path data/processed --output-path models/requested_model_comparison --models random_forest xgboost hist_gradient_boosting lightgbm naive_bayes --select-metric f1
```

Add `--explain-shap` to generate SHAP explanations for the selected best model. Add `--shap-for-all-models` only when you want SHAP for every successful model.

```powershell
python src/compare_models.py --processed-path data/processed --output-path models/requested_model_comparison --models random_forest xgboost hist_gradient_boosting lightgbm naive_bayes --select-metric f1 --explain-shap
```

Outputs:

- `model_comparison.csv`
- `model_comparison_report.json`
- One subfolder per model, each containing its model, metrics, predictions, plots, and explainability artifacts.

Single-model training outputs:

- `model.joblib`
- `model_metadata.json`
- `metrics.json`
- `feature_importance.csv`
- `classification_report_train.csv`
- `classification_report_test.csv`
- `test_predictions.csv`
- `train_predictions.csv`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `score_distribution.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `shap/shap_global_importance.csv` when SHAP is enabled
- `shap/shap_values_sample.csv` when SHAP is enabled
- `shap/shap_summary_bar.png` when SHAP is enabled
- `shap/shap_beeswarm.png` when SHAP is enabled
- `shap/shap_local_top_features.csv` when SHAP is enabled
- `shap/shap_local_waterfall_*.png` when SHAP is enabled
- `shap/shap_report.json` when SHAP is enabled

## Predict

Use a trained model on a processed CSV with matching feature columns:

```powershell
python src/predict.py --model-path models/lightgbm/model.joblib --input-file data/processed/test.csv --output-file outputs/predictions/test_predictions.csv
```

## Save Once, Reuse Multiclass Models

Both multiclass entry points automatically save **each model as soon as its final
training/refit completes**, before test prediction and plots, without waiting for
the other models. The file is first written to a unique temporary file and then
atomically published as `model.joblib`; a failed write cannot leave a partial
model at that final path.

Each bundle contains the trained estimator, selected feature names, fitted
scaler, and label encoder. The default locations after successful training are:

```text
models/multiclass_34_tuned/{xgboost,lightgbm,catboost}/model.joblib
models/multiclass_8_tuned/{xgboost,lightgbm,catboost}/model.joblib
```

An explicit `--output-path` changes the run folder. A `model_save_report.json`
records when each model was saved and its evaluation status; the comparison
report also lists saved models, even if later evaluation fails. No extra save
command is needed. Existing training outputs are protected against accidental
overwrite: rerunning a training command does **not** silently retrain or reuse
them, and requires a new output directory.

For inference, run the prediction-only command instead. This example uses the
already saved full-data 34-class XGBoost model:

```powershell
python src/predict_multiclass.py --model-path models/multiclass_34/xgboost/model.joblib --input-file data/raw/test/test.csv --output-file outputs/predictions/multiclass_34_predictions.csv
```

For an eight-class model, change `--model-path` to its saved bundle. Input is raw
numeric flow features, **not binary processed data**; any label column is ignored.
The command applies the saved preprocessing, emits `predicted_label` in input row
order, and never runs training, grid search, or CV. Prediction uses CPU and streams
the CSV in bounded chunks. Choose a fresh output file for each export.

Only load trusted local `.joblib` files: their pickle-based format can execute
code. Retain this project's compatible environment when loading saved bundles.
The existing `MulticlassBundle`/`FeatureTransform` module paths remain unchanged
so previous project models can still be loaded.

CV progress reports are **not resumable training checkpoints**. An interruption
before final training completes may leave reports but no final model. Code changes
do not alter an already-running Python process; the earlier save behavior remains
in effect for a run started before this update.

## 8-Class Attack-Family Classification (Paper Protocol)

`src/multiclass_8.py` trains **XGBoost, LightGBM, and CatBoost** using the exact
eight names in the paper's Fig. 7. The mapping covers all 34 original labels;
unknown labels raise an error instead of silently becoming benign.

| Output family | Original CICIoT2023 labels |
|---|---|
| Benign | BenignTraffic |
| BruteForce | DictionaryBruteForce |
| DDoS | All 12 explicitly listed DDoS variants |
| DoS | DoS-HTTP_Flood, DoS-SYN_Flood, DoS-TCP_Flood, DoS-UDP_Flood |
| Mirai | Mirai-greeth_flood, Mirai-greip_flood, Mirai-udpplain |
| Recon | Recon-HostDiscovery, Recon-OSScan, Recon-PingSweep, Recon-PortScan, VulnerabilityScan |
| Spoofing | DNS_Spoofing, MITM-ArpSpoofing |
| WebBased | Backdoor_Malware, BrowserHijacking, CommandInjection, SqlInjection, Uploading_Attack, XSS |

Membership follows the paper's Table 6 and the
[official CICIoT2023 taxonomy](https://www.unb.ca/cic/datasets/iotdataset-2023.html).
In particular, Backdoor_Malware is **WebBased**, not a ninth malware class;
VulnerabilityScan is **Recon**.

Run after any other GPU training job finishes:

```powershell
python src/multiclass_8.py --device gpu --lightgbm-device cpu --n-jobs -1
```

This is a **new experiment** saved to `models/multiclass_8_tuned`. It does not
change the binary pipeline, the default 34-class task, raw CSV files, or previous
model artifacts. Use a new `--output-path` when repeating a completed/interrupted
run; checkpoint reports do not implement training resume.

Eight-class defaults follow the paper's described protocol:

1. Use **only `data/raw/train/train.csv` as the source pool**, preserving the
   existing raw test file. To use another/larger raw collection, explicitly pass
   `--source-input-path path/to/raw1.csv path/to/raw2.csv`. These inputs are raw
   sources, not designated train/test splits. Duplicate source paths are rejected.
2. Map labels to the eight families **before** counting or sampling. Remove rows
   with missing labels or nonfinite numerical features; retain duplicates already
   present in the source.
3. Target 5,000,000 flows with at least 1,000 per family, using the paper's equation
   `quota = max(1000, floor(valid_family_count / valid_total * 5000000))` and
   chunk-wise random sampling without replacement until quotas fill. No SMOTE,
   duplication, or synthetic oversampling is used. Impossible quotas fail clearly.
   The paper does not specify exact-total reconciliation after rounding/minimum
   constraints: we preserve its equation and report the actual sampled total,
   which can differ slightly from five million.
4. Create a fresh **80/20 stratified train/test split** after sampling. The
   separately configured inner validation split is used for early stopping only.
5. Fit a CPU LightGBM gain selector on up to 200,000 training rows, considering
   **all 46 features** and retaining gain **strictly above their median**. Fit
   scaling on the training split. The feature count is not forced to 23.
6. As in the paper, `--cv-preprocessing train` fits feature selection/scaling
   **once before CV**, sharing them across models/folds. The test set is excluded,
   but CV scores can be optimistic because selection has seen their labels.
   Use `--cv-preprocessing fold` for stricter preprocessing fitted inside each
   fold. The existing 34-class task still defaults to `fold`.
7. Use inverse-frequency weights `N / (8 * family_count)`, **grid search with
   stratified five-fold CV enabled by default**, mean Macro F1 selection, 30-round
   early stopping, and full-training refits. Uses the same configurable grid as
   the 34-class mode; the paper does not disclose its exact grid/winning settings.
8. Save eight-class predictions/reports, an **8 x 8 confusion matrix**, feature
   gains, model bundles, CV results, and CPU-only final inference latency.
   XGBoost uses GPU for fitting and CV predictions; CatBoost uses GPU for fitting
   and CPU for multiclass predictions. The command above keeps LightGBM on CPU
   because of the earlier OpenCL failure.

Inspect valid source-family counts and planned quotas without training (this
checks all numeric features, so it scans the source CSV):

```powershell
python src/multiclass_8.py --counts-only
```

To keep the existing raw train/test partition instead of constructing the
paper-style split, explicitly choose `--data-protocol existing`. Optional
`--max-train-per-class` / `--max-test-per-class` then cap **families**, not raw
subtypes. In paper mode use `--sampling-target` / `--min-samples-per-class` instead:

```powershell
python src/multiclass_8.py --device gpu --lightgbm-device cpu --sampling-target 8000 --min-samples-per-class 100 --n-estimators 10 --output-path tmp/multiclass_8_smoke
```

Smoke-test metrics are not full-data results. Our available source pool, seeds,
library versions, chunk size (250,000 by default; `--chunk-size 2000000` matches
the paper), and hardware differ from the authors'. Matching class names and the
stated protocol **does not guarantee their reported scores**. The new 8-class
split is not the old 34-class test set; compare results only with that distinction
made explicit.

Audit artifacts include `raw_to_family_mapping.json`, `label_mapping.json`,
`class_counts.csv`, `sampling_report.json` (paper mode), and `data_report.json`.
`--counts-only` writes `class_counts_raw.csv` and `source_inventory.json` in paper
mode. No derived training/test CSVs are written over the originals.

## 34-Class Classification: XGBoost, LightGBM, CatBoost

The separate `src/multiclass.py` entry point implements the three multiclass models
compared in the paper. It reads the **raw** train/test CSVs and preserves all 34
original labels, including BenignTraffic. Binary preprocessing and the dashboard
continue to use their existing pipeline; do not pass binary processed files here.

Install dependencies (including CatBoost):

```powershell
python -m pip install -r requirements.txt
```

Inspect raw sample counts without training:

```powershell
python src/multiclass.py --counts-only
```

Tune all three models with **grid search and five-fold stratified CV enabled by
default**, then train on all valid training rows. This command uses GPU for
XGBoost and CatBoost and CPU for LightGBM (useful after an OpenCL split failure):

```powershell
python src/multiclass.py --models xgboost lightgbm catboost --device gpu --lightgbm-device cpu --max-train-per-class 0 --n-jobs -1
```

Use `--device gpu` without the override to request GPU for all three. XGBoost uses
CUDA, CatBoost uses CUDA, and LightGBM uses OpenCL with double-precision histogram
accumulation and 63 bins. Double precision may be slower on consumer GPUs; it
does not guarantee that every LightGBM GPU failure is resolved. Catchable model
errors are reported, and other requested models continue. A native GPU crash
can terminate Python; completed models remain saved. XGBoost's effective
device is checked so a CPU fallback cannot be reported as GPU training.

`--device gpu` applies to **every grid-search fold, early-stopping fit, and
refit**, not only the final model. XGBoost also uses CUDA for CV and validation
predictions; its actual prediction device is checked. CatBoost's installed GPU
evaluator does not support multiclass predictions, so its CV predictions remain
on CPU even though its training uses GPU. LightGBM prediction, the gain selector,
CSV loading, scaling, and metric calculation also remain on CPU. Final test
prediction deliberately uses CPU to preserve the paper-style latency benchmark.
Startup logs and grid-search reports identify training and scoring devices.

For a new GPU run after an interrupted run, keep the previous results and choose
a fresh output directory (the reports are checkpoints, not resumable models):

```powershell
python src/multiclass.py --device gpu --lightgbm-device cpu --cv-folds 5 --max-train-per-class 0 --n-jobs -1 --output-path models/multiclass_34_tuned_gpu
```

This retains all three models, full-data tuning, the same grid, and five folds.
It is GPU-accelerated, not GPU-only, and still performs 60 CV evaluations with
two training passes per evaluation by default. GPU use alone does not make
full-data grid search a short run.

Run a smaller **sampled grid-search smoke test**, retaining all classes and all
five folds, with a lower round limit:

```powershell
python src/multiclass.py --device gpu --lightgbm-device cpu --max-train-per-class 100 --max-test-per-class 30 --n-estimators 10 --output-path tmp/multiclass_34_grid_sample
```

Each sample scans the entire source file and uses random per-class reservoirs;
it does not take the first rows. Test sampling changes the class distribution,
so sampled metrics are not full-test results and cannot be compared directly
with the paper. Both sample caps default to zero (all valid rows).

Training behavior:

- Validates exactly 34 training labels and matching test labels, with one shared
  encoder. Removes missing labels and rows with nonfinite numeric features;
  retains duplicate flows. Raw, valid, selected, fitting, and validation class
  counts are saved before any model training.
- Reserves a stratified 20% of training rows for early stopping (30 rounds).
  Never uses the test file for feature selection, early stopping, or model choice.
- Trains a CPU LightGBM selector on up to 200,000 fitting rows using **all 46
  original features**, including constants and zero-gain features. Keeps exactly
  those whose gain is **strictly greater than the median of all 46 gains**.
  No constant/identifier removal occurs before this calculation. Ties can yield
  fewer than 23 selected features; the count is not forced to 23. A missing
  46-feature schema or an empty selection raises an explicit error.
- Standardizes the selected features with training-only statistics. The final
  transformation is shared by all three models. `--feature-selection none`
  explicitly disables selection and keeps every input feature. This replaces
  the previous `--top-features` option.
- Uses inverse-frequency sample weights `N / (34 * class_count)` calculated
  separately for each fitting partition. `--class-weight none` disables weights.
- Uses validation to choose the number of trees, then by default refits each
  model on **all selected training rows**. `--no-refit-full` skips this second fit.
- Runs **exhaustive grid search with stratified five-fold CV by default**.
  Every candidate uses the same folds and is scored by unweighted Macro F1 on
  the held-out CV fold. Each fold fits its own feature selector and scaler using
  its inner fitting partition; an inner validation split controls early stopping.
  With `--refit-full` (default), each candidate is refitted on the entire outer
  training fold before scoring. The outer CV scoring fold never controls early
  stopping. Fold preprocessing is reused across candidates to avoid repeated work.
- Selects the candidate with the highest five-fold mean Macro F1 separately for
  each model, then uses those settings for the final early-stopping fit and full
  training refit. Failed candidates are excluded, and failure details are saved.
  `--cv-folds 0` explicitly skips both CV and grid search for a fast untuned run.
- Chooses the best model by CV mean Macro F1 when CV is enabled, otherwise by
  validation Macro F1; reports held-out test accuracy,
  balanced accuracy, Macro F1, weighted F1, and all 34 classes' precision/recall/F1.
  Metric values are decimals, e.g. `0.8876` means `88.76%`.
- Measures batched CPU model prediction after a warm-up (excluding preprocessing
  and disk I/O). Training time separately records early-stopping fit and full
  refit; grid-search and shared preprocessing times are recorded separately.
  `total_training_seconds` includes search plus both final fits, excluding shared
  preprocessing. Search time includes each fold's preprocessing and scoring.

The default Cartesian grid for each model is:

| Parameter | Candidate values |
|---|---|
| `learning_rate` | 0.05, 0.1 |
| `max_depth` | 4, 6 |
| `n_estimators` | 500 maximum rounds, with early stopping |

This is **4 candidates x 5 folds = 20 CV evaluations per model**, or 60 for all
three models. With default fold refitting, each evaluation contains two training
passes; each model then performs its final early-stopping fit and full refit.
Full-data tuning can take much longer than a single training run. Candidates and
models run sequentially to avoid multiple GPU fits competing for memory. Use
`--n-jobs` to control threads within each estimator.

Without a custom grid, `--learning-rate R` searches `[R, min(2R, 1)]`,
`--max-depth D` searches `[max(1, D-2), D]`, and `--n-estimators N` sets the maximum
rounds for every candidate. Duplicate values are removed. To supply explicit
per-model search spaces, edit `configs/multiclass_grid.json` and run:

```powershell
python src/multiclass.py --device gpu --lightgbm-device cpu --param-grid configs/multiclass_grid.json --output-path models/multiclass_34_custom_grid
```

The JSON supports `learning_rate`, `max_depth`, and `n_estimators`, each as a
nonempty list. It uses the same names for all models; `max_depth` and
`n_estimators` map to CatBoost's `depth` and `iterations`. For a model listed in
the JSON, unspecified parameters remain at their CLI values; models omitted
from the JSON use the default grid. To also tune tree counts, add, for example,
`"n_estimators": [200, 500]` to a model's grid. Explicit grid values override the
corresponding CLI values. The supplied grids are configurable starting points,
**not the paper's undisclosed grids or optimized settings**.

This implementation preserves your existing train/test split and all available
training rows instead of reproducing the paper's five-million-row proportional
sampling with a minimum of 1,000 per class. Matching the model families, search
method, and selection rule does not guarantee matching the paper's scores.
The best CV score is used for tuning, so final generalization is assessed on the
untouched test set. The final early-stopping validation score is diagnostic, not
an independent estimate after CV has used the training dataset.

Outputs in `models/multiclass_34_tuned/` (previous model folders are preserved):

- `class_counts.csv`, `data_report.json`, `label_mapping.json`, and `parameter_grids.json`
- `selected_features.json`, `feature_selection_scores.csv` (all 46 gains), and
  `feature_selection_report.json` (median threshold, selection rule, and actual count)
- `model_comparison.csv` and `model_comparison_report.json` (checkpointed per model)
- Per-model `grid_search_results.csv` and `grid_search_report.json`, checkpointed
  after each candidate/fold, including fold scores, mean/std, ranks, failed
  candidates, best parameters, and each fold's feature selection details
- Per-model `model.joblib`, `metrics.json`, `model_metadata.json`,
  `classification_report_test.csv`, `test_predictions.csv`, `confusion_matrix.csv`,
  and `confusion_matrix_normalized.png`

Choose a new `--output-path` for each run; existing training artifacts are protected
from overwrite. Counts-only output can coexist with the first training run.
If a model fails, successful artifacts are retained and the command exits with
an error summarizing the failed models. Ctrl+C interrupts training and preserves
completed model results.

The saved multiclass artifact includes its feature transformation and label
mapping. Load it from a Python process with `src` on its import path:

```python
import sys
sys.path.insert(0, "src")
import joblib
import pandas as pd

bundle = joblib.load("models/multiclass_34_tuned/xgboost/model.joblib")
raw_rows = pd.read_csv("data/raw/test/test.csv", nrows=100)
predicted_attack_names = bundle.predict(raw_rows)
```

The existing binary `src/predict.py` and dashboard do not load these multiclass
artifacts. Run regression tests with:

```powershell
python -m unittest discover -s tests -v
```
