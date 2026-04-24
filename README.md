# IoT DDoS Detection

Baseline project for detecting DDoS traffic in CICIOT-style IoT network-flow data.

## Current Pipeline

1. Keep only Benign and DDoS rows.
2. Map labels to `0 = Benign`, `1 = DDoS`.
3. Remove duplicates, missing values, and infinite values.
4. Drop useless columns such as constants, duplicate features, and obvious IDs.
5. Encode categorical features if any exist.
6. Create a stratified train/test split.
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
- `src/ddos_iot/dashboard.py`: Streamlit dashboard implementation.
- `src/ddos_iot/preprocessing.py`: reusable preprocessing functions.
- `src/ddos_iot/modeling.py`: reusable model, metric, plot, and prediction functions.
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

- `--input-path data/raw/train`
- `--processed-path data/processed`
- `--models-output-path models/requested_model_comparison`
- `--skip-preprocess` to reuse existing processed files
- `--shap-for-all-models` to explain every successful model instead of only the best model

## Dashboard

Run the local Streamlit dashboard:

```powershell
streamlit run app.py
```

The dashboard can run the pipeline, show processed dataset summaries, compare models, display SHAP explanations, and run predictions with trained model files.

## Preprocess

Use the currently visible raw CSV:

```powershell
python src/preprocess.py --input-path data/raw/train --output-path data/processed --top-features 25 --rebalance undersample
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


