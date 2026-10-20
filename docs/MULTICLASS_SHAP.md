# Multiclass SHAP: saved models, no retraining

Both eight-family and 34-label models support post-hoc explanations for the
existing **XGBoost, LightGBM and CatBoost** classifiers. Binary SHAP is unchanged.
SHAP is an explicit operation after training, not an extra step inside CV.

## Dashboard

1. Choose **8 classes** or **34 classes**, then select an experiment.
2. Open **SHAP**, choose a saved model and a raw-feature CSV.
3. Set sample/scan limits and click **Generate SHAP without retraining**.
4. View the overall summary, select a class for its importance/beeswarm plot,
   then select a sampled CSV row for a waterfall and contribution table.

Browsing a saved explanation only reads its numeric archive; changing the class
or sample does not reload the estimator, recompute SHAP, or train anything.
Generated outputs are stored in a new timestamped folder under
`models/<experiment>/<model>/shap/`. Existing explanations and models are preserved.
Only load trusted local joblib model files.

## Command line

```powershell
python src/explain_multiclass.py --model-path models/multiclass_34_tuned_gpu/xgboost/model.joblib --input-file data/raw/test/test.csv
python src/explain_multiclass.py --model-path models/multiclass_8_tuned/catboost/model.joblib --input-file data/raw/test/test.csv --class-label DDoS
```

Defaults: 200 explained rows, a scan of at most 50,000 CSV data rows, two CPU
threads, and five exported local waterfalls. The sample is uniform among valid
rows **within the scanned prefix**; it is not stratified or guaranteed to include
rare classes. These are sample-specific summaries, not population estimates.
Use `--max-scan-rows 0` to randomly sample across the full file in bounded-memory
chunks; this still needs to read the entire file. Label columns are ignored.
Invalid/nonfinite selected-feature rows are skipped and counted in the report.
The zero-based original CSV data-row positions are retained in every export.

The input is explicitly chosen explanation data, not automatically the original
held-out test set. In particular, the eight-class paper protocol created its own
split; using the existing raw test CSV here does not recreate that split or its
evaluation. No accuracy claims or training decisions are made from SHAP outputs.

## Meaning and correctness

The implementation uses native TreeSHAP contribution APIs for
[XGBoost](https://xgboost.readthedocs.io/en/stable/prediction.html),
[LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html#lightgbm.Booster.predict),
and [CatBoost](https://catboost.ai/docs/en/concepts/shap-values).
It keeps every output class, rather than selecting binary class 1. It verifies
the estimator's encoded class order, separates each class baseline from the
feature contributions, and checks:

`baseline[class] + sum(feature SHAP[class]) = model raw score[class]`

Native values are **raw scores/logits, not probability changes**. Softmax over all
class scores produces the class probabilities; that conversion is also checked
against the saved estimator. Early-stopped XGBoost explanations use the same
best-iteration tree range as prediction. Native baselines use the training tree
statistics, so no additional background CSV is required.

The saved feature selection/scaler is applied without fitting. Plot feature
values are shown in raw units; the archived transformed values record what the
estimator actually received. Overall importance averages absolute contributions
over the sampled rows and equally over output classes. A per-class summary
explains that class's score across **all sampled rows**, not only rows predicted
as or labeled with that class. Explanations describe associations, not causation.

## Optional slower fallback

Native SHAP is tried first (`--method auto`). If it fails, a capped permutation
fallback is allowed only when a separate `--background-file` is explicitly
provided. Use a training-only reference CSV, never the explained/test data. The
tool checks that the files differ; the user is responsible for reference-data
provenance and absence of overlapping rows. No background is inferred from the
paper source pool because that pool may also contain held-out rows.

```powershell
python src/explain_multiclass.py --model-path models/multiclass_34_tuned_gpu/xgboost/model.joblib --input-file data/raw/test/test.csv --background-file data/raw/train/train.csv --method permutation --fallback-size 20 --background-size 50
```

The [permutation explainer](https://shap.readthedocs.io/en/latest/generated/shap.explainers.Permutation.html)
explains all class **probabilities**, with default limits of 20 explained rows,
50 background rows and an evaluation budget of `2 × (2 × selected_features + 1)`
per explained row. The report records the
method, output space, actual sample, reference and native failure reason. Do not
compare probability-SHAP magnitudes directly with native raw-score SHAP values.
SHAP always uses CPU in this implementation; GPU training settings are unchanged.

## Outputs

- `shap_report.json`: completion/failure status, classes, sampling details,
  method/units, additivity error, model SHA-256 fingerprint, timing and versions.
- `shap_values.npz`: non-pickle arrays with shape **rows × features × classes**,
  baselines, raw scores or probabilities, class probabilities, feature names,
  raw/transformed feature values and CSV row indices.
- `shap_global_importance.csv`, `shap_class_importance.csv`: overall and per-class
  importance. `shap_local_top_features.csv` stores top contributions for all
  classes on the requested local rows (omitted when `--local-count 0`).
- `explained_samples.csv`: row indices, predicted labels and confidence.
- Summary bar, selected-class bar/beeswarm, and predicted-class local waterfall
  PNGs. `--class-label` selects the exported class summary; arrays retain all
  classes. The dashboard can display a waterfall for **any** sampled row/class.

`--no-plots` saves arrays/tables only; plots remain available in the dashboard.
`--output-path` must be a new directory. A failed calculation or plot export is
recorded as failed, never silently published as a completed explanation. Neither
failure nor success changes the saved model or its existing evaluation results.
