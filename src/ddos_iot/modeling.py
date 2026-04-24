"""Model training, evaluation, plotting, and prediction helpers."""

from __future__ import annotations

import logging
import os
import time
from copy import copy
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import matplotlib

from ddos_iot.paths import configure_runtime_environment

configure_runtime_environment()
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB

from ddos_iot.config import CLASS_NAMES, TARGET_COLUMN
from ddos_iot.explainability import save_shap_explanations
from ddos_iot.utils import predict_scores, write_json


LOGGER = logging.getLogger(__name__)


def load_processed_data(processed_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test CSV files."""

    train_path = processed_path / "train.csv"
    test_path = processed_path / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected processed files at {train_path} and {test_path}. Run src/preprocess.py first."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    validate_processed_splits(train, test)
    LOGGER.info("Loaded train=%s test=%s", train.shape, test.shape)
    return train, test


def validate_processed_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Validate processed train/test files before modeling."""

    if TARGET_COLUMN not in train.columns or TARGET_COLUMN not in test.columns:
        raise ValueError(f"Both train and test must contain {TARGET_COLUMN!r}.")

    train_features = [column for column in train.columns if column != TARGET_COLUMN]
    test_features = [column for column in test.columns if column != TARGET_COLUMN]
    if train_features != test_features:
        raise ValueError("Train and test feature columns do not match.")

    train_classes = set(train[TARGET_COLUMN].dropna().astype(int).unique())
    test_classes = set(test[TARGET_COLUMN].dropna().astype(int).unique())
    if train_classes != {0, 1} or test_classes != {0, 1}:
        raise ValueError(f"Expected binary classes {{0, 1}}. Got train={train_classes}, test={test_classes}.")


def split_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split processed data into features and target."""

    return data.drop(columns=[TARGET_COLUMN]), data[TARGET_COLUMN].astype(int)


def build_model(args: Any) -> Any:
    """Build a configured sklearn estimator."""

    if args.model == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )

    if args.model == "xgboost":
        return build_xgboost(args)

    if args.model == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=args.hgb_max_iter,
            learning_rate=args.hgb_learning_rate,
            l2_regularization=args.hgb_l2,
            random_state=args.random_state,
        )

    if args.model == "lightgbm":
        return build_lightgbm(args)

    if args.model == "naive_bayes":
        return GaussianNB()

    raise ValueError(f"Unsupported model: {args.model}")


def build_xgboost(args: Any) -> Any:
    """Build an XGBoost classifier, if xgboost is installed."""

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Run: pip install xgboost") from exc

    return XGBClassifier(
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbosity=0,
    )


def build_lightgbm(args: Any) -> Any:
    """Build a LightGBM classifier, if lightgbm is installed."""

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm") from exc

    return LGBMClassifier(
        n_estimators=args.lgbm_estimators,
        max_depth=args.lgbm_max_depth,
        num_leaves=args.lgbm_num_leaves,
        learning_rate=args.lgbm_learning_rate,
        subsample=args.lgbm_subsample,
        colsample_bytree=args.lgbm_colsample_bytree,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=-1,
    )


def train_model(model: Any, x_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Fit a sklearn model."""

    LOGGER.info("Training model on %s rows x %s features", len(x_train), x_train.shape[1])
    model.fit(x_train, y_train)
    return model


def timed_train_model(model: Any, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, float]:
    """Fit a model and return elapsed training time in seconds."""

    start = time.perf_counter()
    fitted_model = train_model(model, x_train, y_train)
    return fitted_model, time.perf_counter() - start


def evaluate_model(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    split_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate a fitted binary classifier."""

    start = time.perf_counter()
    scores = predict_scores(model, x)
    inference_time = time.perf_counter() - start
    predictions = (scores >= threshold).astype(int)
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "split": split_name,
        "threshold": threshold,
        "inference_rows": int(len(x)),
        "inference_time_seconds": float(inference_time),
        "inference_time_per_row_ms": safe_divide(inference_time * 1000, len(x)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": safe_metric(roc_auc_score, y, scores),
        "pr_auc": safe_metric(average_precision_score, y, scores),
        "false_positive_rate": safe_divide(fp, fp + tn),
        "false_negative_rate": safe_divide(fn, fn + tp),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }
    prediction_frame = pd.DataFrame(
        {"y_true": y.to_numpy(dtype=int), "y_score": scores, "y_pred": predictions}
    )
    return metrics, prediction_frame


def safe_metric(metric_func: Any, y_true: pd.Series, scores: np.ndarray) -> float | None:
    """Return None for metrics that cannot be computed for a split."""

    try:
        return float(metric_func(y_true, scores))
    except ValueError:
        return None


def safe_divide(numerator: float, denominator: float) -> float:
    """Return zero when a metric denominator is zero."""

    return 0.0 if denominator == 0 else float(numerator / denominator)


def extract_feature_importance(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    """Extract feature importance when the fitted estimator exposes it."""

    if hasattr(model, "feature_importances_"):
        frame = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        return frame.sort_values("importance", ascending=False, na_position="last")
    if hasattr(model, "coef_"):
        frame = pd.DataFrame({"feature": feature_columns, "coefficient": np.ravel(model.coef_)})
        frame["abs_coefficient"] = frame["coefficient"].abs()
        return frame.sort_values("abs_coefficient", ascending=False)
    return pd.DataFrame({"feature": feature_columns, "importance": np.nan})


def save_training_artifacts(
    model: Any,
    feature_columns: list[str],
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    train_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output_path: Path,
    args: Any,
) -> None:
    """Save model, metrics, predictions, and metadata."""

    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
            "class_names": CLASS_NAMES,
        },
        output_path / "model.joblib",
    )

    train_predictions.to_csv(output_path / "train_predictions.csv", index=False)
    test_predictions.to_csv(output_path / "test_predictions.csv", index=False)
    feature_importance.to_csv(output_path / "feature_importance.csv", index=False)
    pd.DataFrame(train_metrics["classification_report"]).T.to_csv(
        output_path / "classification_report_train.csv"
    )
    pd.DataFrame(test_metrics["classification_report"]).T.to_csv(
        output_path / "classification_report_test.csv"
    )
    write_json(output_path / "metrics.json", {"train": train_metrics, "test": test_metrics})
    write_json(
        output_path / "model_metadata.json",
        {
            "model": args.model,
            "target": "0=Benign, 1=DDoS",
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "training_args": vars(args),
        },
    )


def save_plots(y_test: pd.Series, test_predictions: pd.DataFrame, output_path: Path) -> None:
    """Save confusion matrix, score, ROC, and precision-recall plots."""

    output_path.mkdir(parents=True, exist_ok=True)
    y_pred = test_predictions["y_pred"]
    y_score = test_predictions["y_score"]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    save_confusion_matrix_plot(cm, output_path / "confusion_matrix.png", normalized=False)
    save_confusion_matrix_plot(cm, output_path / "confusion_matrix_normalized.png", normalized=True)
    save_score_distribution_plot(y_test, y_score, output_path / "score_distribution.png")
    save_roc_plot(y_test, y_score, output_path / "roc_curve.png")
    save_precision_recall_plot(y_test, y_score, output_path / "precision_recall_curve.png")


def maybe_save_shap_explanations(
    model: Any,
    model_name: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    output_path: Path,
    args: Any,
) -> dict[str, Any] | None:
    """Save SHAP outputs when requested without failing the whole training run."""

    if not getattr(args, "explain_shap", False):
        return None

    try:
        return save_shap_explanations(
            model=model,
            model_name=model_name,
            x_train=x_train,
            x_test=x_test,
            output_path=output_path,
            sample_size=getattr(args, "shap_sample_size", 1000),
            background_size=getattr(args, "shap_background_size", 200),
            local_count=getattr(args, "shap_local_count", 10),
            max_display=getattr(args, "shap_max_display", 15),
            model_agnostic_sample_size=getattr(args, "shap_model_agnostic_sample_size", 100),
            random_state=getattr(args, "random_state", 42),
        )
    except Exception as exc:
        shap_dir = output_path / "shap"
        shap_dir.mkdir(parents=True, exist_ok=True)
        write_json(shap_dir / "shap_error.json", {"model": model_name, "error": str(exc)})
        LOGGER.exception("Failed to save SHAP explanations for %s", model_name)
        return None


def save_confusion_matrix_plot(cm: np.ndarray, path: Path, normalized: bool) -> None:
    """Save one confusion matrix heatmap."""

    values = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalized else cm
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        values,
        annot=True,
        fmt=".3f" if normalized else "d",
        cmap="magma" if normalized else "Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Normalized Confusion Matrix" if normalized else "Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_score_distribution_plot(y_test: pd.Series, y_score: pd.Series, path: Path) -> None:
    """Save the predicted score distribution plot."""

    plt.figure(figsize=(7, 4))
    sns.histplot(
        data=pd.DataFrame({"score": y_score, "class": y_test.map({0: "Benign", 1: "DDoS"})}),
        x="score",
        hue="class",
        bins=40,
        stat="density",
        common_norm=False,
    )
    plt.title("Predicted DDoS Score Distribution")
    plt.xlabel("Predicted probability / score")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_roc_plot(y_test: pd.Series, y_score: pd.Series, path: Path) -> None:
    """Save ROC curve plot."""

    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_precision_recall_plot(y_test: pd.Series, y_score: pd.Series, path: Path) -> None:
    """Save precision-recall curve plot."""

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def load_model(model_path: Path) -> dict[str, Any]:
    """Load a saved model bundle."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    bundle = joblib.load(model_path)
    for key in ("model", "feature_columns"):
        if key not in bundle:
            raise ValueError(f"Model bundle is missing {key!r}.")
    return bundle


def load_features(input_path: Path, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load a processed CSV and align it to model feature columns."""

    data = pd.read_csv(input_path)
    missing = sorted(set(feature_columns) - set(data.columns))
    if missing:
        raise ValueError(f"Input data is missing required features: {missing}")
    target = data[TARGET_COLUMN].astype(int) if TARGET_COLUMN in data.columns else None
    return data[feature_columns], target


def build_predictions(scores: np.ndarray, threshold: float, target: pd.Series | None) -> pd.DataFrame:
    """Build prediction output rows."""

    output = pd.DataFrame({"y_score": scores, "y_pred": (scores >= threshold).astype(int)})
    if target is not None:
        output.insert(0, "y_true", target.to_numpy(dtype=int))
    return output


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, float | None] | None:
    """Evaluate predictions when ground truth is available."""

    if "y_true" not in predictions.columns:
        return None

    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    y_score = predictions["y_score"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_metric(roc_auc_score, y_true, y_score),
    }


def run_training(args: Any) -> None:
    """Run model training and evaluation from parsed CLI args."""

    train, test = load_processed_data(args.processed_path)
    x_train, y_train = split_xy(train)
    x_test, y_test = split_xy(test)
    model, train_time = timed_train_model(build_model(args), x_train, y_train)

    train_metrics, train_predictions = evaluate_model(model, x_train, y_train, args.threshold, "train")
    test_metrics, test_predictions = evaluate_model(model, x_test, y_test, args.threshold, "test")
    train_metrics["train_time_seconds"] = train_time
    test_metrics["train_time_seconds"] = train_time
    feature_importance = extract_feature_importance(model, list(x_train.columns))

    save_training_artifacts(
        model,
        list(x_train.columns),
        train_metrics,
        test_metrics,
        train_predictions,
        test_predictions,
        feature_importance,
        args.output_path,
        args,
    )
    if not args.no_plots:
        save_plots(y_test, test_predictions, args.output_path)
    maybe_save_shap_explanations(model, args.model, x_train, x_test, args.output_path, args)

    LOGGER.info(
        "Test metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%s pr_auc=%s",
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
        test_metrics["roc_auc"],
        test_metrics["pr_auc"],
    )
    LOGGER.info("Saved model artifacts to %s", args.output_path)


def run_model_comparison(args: Any) -> None:
    """Train multiple models and save a ranked comparison report."""

    train, test = load_processed_data(args.processed_path)
    x_train, y_train = split_xy(train)
    x_test, y_test = split_xy(test)
    args.output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    shap_candidates: dict[str, tuple[Any, Path]] = {}
    for model_name in args.models:
        model_args = copy(args)
        model_args.model = model_name
        model_output_path = args.output_path / model_name
        LOGGER.info("=== Training %s ===", model_name)

        try:
            model, train_time = timed_train_model(build_model(model_args), x_train, y_train)
            train_metrics, train_predictions = evaluate_model(
                model,
                x_train,
                y_train,
                args.threshold,
                "train",
            )
            test_metrics, test_predictions = evaluate_model(
                model,
                x_test,
                y_test,
                args.threshold,
                "test",
            )
            train_metrics["train_time_seconds"] = train_time
            test_metrics["train_time_seconds"] = train_time
            feature_importance = extract_feature_importance(model, list(x_train.columns))

            save_training_artifacts(
                model,
                list(x_train.columns),
                train_metrics,
                test_metrics,
                train_predictions,
                test_predictions,
                feature_importance,
                model_output_path,
                model_args,
            )
            if not args.no_plots:
                save_plots(y_test, test_predictions, model_output_path)
            if getattr(args, "explain_shap", False):
                if getattr(args, "shap_for_all_models", False):
                    maybe_save_shap_explanations(
                        model,
                        model_name,
                        x_train,
                        x_test,
                        model_output_path,
                        model_args,
                    )
                else:
                    shap_candidates[model_name] = (model, model_output_path)

            rows.append(comparison_row(model_name, "success", train_time, test_metrics, model_output_path))
        except Exception as exc:  # Keep the comparison running if one model fails.
            LOGGER.exception("Model %s failed", model_name)
            rows.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "error": str(exc),
                    "model_dir": str(model_output_path),
                }
            )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output_path / "model_comparison.csv", index=False)

    successful = comparison[comparison["status"] == "success"].copy()
    if successful.empty:
        write_json(args.output_path / "model_comparison_report.json", {"models": rows, "best_model": None})
        raise RuntimeError("All model comparison runs failed.")

    successful = successful.sort_values(
        by=[args.select_metric, "inference_time_seconds"],
        ascending=[False, True],
    )
    best_model = successful.iloc[0].to_dict()
    write_json(
        args.output_path / "model_comparison_report.json",
        {
            "selected_metric": args.select_metric,
            "best_model": best_model,
            "models": rows,
        },
    )
    if getattr(args, "explain_shap", False) and not getattr(args, "shap_for_all_models", False):
        best_model_name = str(best_model["model"])
        if best_model_name in shap_candidates:
            best_estimator, best_output_path = shap_candidates[best_model_name]
            maybe_save_shap_explanations(
                best_estimator,
                best_model_name,
                x_train,
                x_test,
                best_output_path,
                args,
            )
    LOGGER.info(
        "Best model: %s (%s=%.6f, inference_time=%.6fs)",
        best_model["model"],
        args.select_metric,
        best_model[args.select_metric],
        best_model["inference_time_seconds"],
    )
    LOGGER.info("Saved model comparison to %s", args.output_path)


def comparison_row(
    model_name: str,
    status: str,
    train_time: float,
    test_metrics: dict[str, Any],
    model_output_path: Path,
) -> dict[str, Any]:
    """Build one row for the model comparison table."""

    cm = test_metrics["confusion_matrix"]
    return {
        "model": model_name,
        "status": status,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
        "false_positive_rate": test_metrics["false_positive_rate"],
        "false_negative_rate": test_metrics["false_negative_rate"],
        "train_time_seconds": float(train_time),
        "inference_rows": test_metrics["inference_rows"],
        "inference_time_seconds": test_metrics["inference_time_seconds"],
        "inference_time_per_row_ms": test_metrics["inference_time_per_row_ms"],
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
        "tp": cm["tp"],
        "model_dir": str(model_output_path),
    }


def run_prediction(args: Any) -> None:
    """Run prediction from parsed CLI args."""

    bundle = load_model(args.model_path)
    features, target = load_features(args.input_file, list(bundle["feature_columns"]))
    predictions = build_predictions(predict_scores(bundle["model"], features), args.threshold, target)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)
    metrics = evaluate_predictions(predictions)
    if metrics is not None:
        write_json(args.output_file.with_suffix(".metrics.json"), metrics)
        LOGGER.info("Metrics: %s", metrics)

    LOGGER.info("Saved predictions to %s", args.output_file)
