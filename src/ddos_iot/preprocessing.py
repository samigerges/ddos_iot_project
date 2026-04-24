"""Preprocessing logic for Benign vs DDoS classification."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ddos_iot.config import RebalanceStrategy, SUPPORTED_EXTENSIONS, TARGET_COLUMN, UNKNOWN_CATEGORY
from ddos_iot.utils import class_distribution, write_json


LOGGER = logging.getLogger(__name__)


def find_dataset_files(input_path: Path) -> list[Path]:
    """Find raw CSV files under a file or directory path."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {input_path.suffix}")
        return [input_path]

    files = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No CSV files found under {input_path}")
    return files


def load_dataset(files: list[Path], max_rows_per_file: int | None) -> pd.DataFrame:
    """Load raw CSV files and ensure they share one schema."""

    frames: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None
    for file_path in files:
        LOGGER.info("Loading %s", file_path)
        frame = pd.read_csv(file_path, nrows=max_rows_per_file)
        if expected_columns is None:
            expected_columns = list(frame.columns)
        elif list(frame.columns) != expected_columns:
            raise ValueError(f"Schema mismatch in {file_path}")
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    LOGGER.info("Loaded %s rows and %s columns", f"{len(data):,}", len(data.columns))
    return data


def infer_label_column(data: pd.DataFrame, explicit_label: str | None = None) -> str:
    """Infer the label column, or validate an explicit one."""

    if explicit_label:
        if explicit_label not in data.columns:
            raise ValueError(f"Label column {explicit_label!r} was not found.")
        return explicit_label

    lowered = {column.lower().strip(): column for column in data.columns}
    for candidate in ("label", "class", "target", "category"):
        if candidate in lowered:
            return lowered[candidate]

    text_columns = [
        column
        for column in data.columns
        if pd.api.types.is_object_dtype(data[column]) or pd.api.types.is_string_dtype(data[column])
    ]
    if len(text_columns) == 1:
        return text_columns[0]

    raise ValueError("Could not infer label column. Pass --label-column explicitly.")


def keep_benign_and_ddos(data: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only Benign and DDoS labels, then create binary target values."""

    label_text = data[label_column].astype("string").str.strip()
    label_lower = label_text.str.lower()
    benign_mask = label_lower.str.contains("benign|normal", regex=True, na=False)
    ddos_mask = label_lower.str.contains("ddos", regex=False, na=False)
    keep_mask = benign_mask | ddos_mask

    filtered = data.loc[keep_mask].copy()
    if filtered.empty:
        raise ValueError("No Benign or DDoS rows were found.")

    filtered[TARGET_COLUMN] = np.where(benign_mask.loc[filtered.index], 0, 1).astype("int8")
    class_counts = class_distribution(filtered[TARGET_COLUMN])
    if set(class_counts) != {"0", "1"}:
        raise ValueError(f"Expected both Benign and DDoS classes after filtering. Got {class_counts}.")

    return filtered, {
        "rows_before_filter": len(data),
        "rows_after_filter": len(filtered),
        "raw_label_counts_kept": label_text.loc[keep_mask].value_counts().to_dict(),
        "binary_class_counts": class_counts,
    }


def clean_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove duplicate rows, rows with missing values, and infinite values."""

    before_rows = len(data)
    cleaned = data.replace([np.inf, -np.inf], np.nan)

    missing_rows = int(cleaned.isna().any(axis=1).sum())
    if missing_rows:
        cleaned = cleaned.dropna(axis=0).copy()

    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates(ignore_index=True) if duplicate_rows else cleaned.reset_index(drop=True)

    return cleaned, {
        "rows_before_cleaning": before_rows,
        "rows_after_cleaning": len(cleaned),
        "removed_missing_or_inf_rows": missing_rows,
        "removed_duplicate_rows": duplicate_rows,
    }


def drop_useless_columns(data: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop raw label, identifier-like, constant, and duplicate feature columns."""

    features = data.drop(columns=[label_column], errors="ignore")
    target = features.pop(TARGET_COLUMN)

    id_columns = detect_identifier_columns(features)
    features = features.drop(columns=id_columns, errors="ignore")

    constant_columns = [
        column for column in features.columns if features[column].nunique(dropna=False) <= 1
    ]
    features = features.drop(columns=constant_columns, errors="ignore")

    duplicate_map = detect_duplicate_columns(features)
    features = features.drop(columns=list(duplicate_map), errors="ignore")

    cleaned = features.copy()
    cleaned[TARGET_COLUMN] = target.to_numpy()
    return cleaned, {
        "dropped_label_column": label_column,
        "identifier_like_columns": id_columns,
        "constant_columns": constant_columns,
        "duplicate_columns": duplicate_map,
    }


def detect_identifier_columns(features: pd.DataFrame) -> list[str]:
    """Detect obvious IDs, raw addresses, and timestamps by column name."""

    id_columns: list[str] = []
    for column in features.columns:
        tokens = {token for token in re.split(r"[^a-z0-9]+", column.lower()) if token}
        is_identifier = (
            "id" in tokens
            or "uuid" in tokens
            or "timestamp" in tokens
            or "datetime" in tokens
            or "date" in tokens
            or "mac" in tokens
            or ("ip" in tokens and tokens & {"src", "source", "dst", "dest", "destination"})
        )
        if is_identifier:
            id_columns.append(column)
    return id_columns


def detect_duplicate_columns(features: pd.DataFrame) -> dict[str, str]:
    """Find exactly duplicated feature columns as dropped_column -> kept_column."""

    seen: dict[int, str] = {}
    duplicates: dict[str, str] = {}
    for column in features.columns:
        column_hash = int(pd.util.hash_pandas_object(features[column], index=False).sum())
        kept_column = seen.get(column_hash)
        if kept_column is not None and features[column].equals(features[kept_column]):
            duplicates[column] = kept_column
        else:
            seen[column_hash] = column
    return duplicates


def detect_column_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Detect numeric and categorical columns."""

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in features.columns:
        series = features[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue

        numeric_version = pd.to_numeric(series, errors="coerce")
        failed_values = int((numeric_version.isna() & series.notna()).sum())
        if failed_values == 0:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns


def stratified_train_test_split(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified train/test split without sklearn."""

    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for class_value, group in data.groupby(TARGET_COLUMN, sort=True):
        shuffled = group.sample(frac=1.0, random_state=random_state + int(class_value))
        test_count = max(1, int(round(len(shuffled) * test_size)))
        if test_count >= len(shuffled):
            raise ValueError(f"Class {class_value} is too small for a stratified split.")
        test_parts.append(shuffled.iloc[:test_count])
        train_parts.append(shuffled.iloc[test_count:])

    train = pd.concat(train_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=random_state + 1).reset_index(drop=True)
    return train, test


def fit_transform_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit numeric/categorical preprocessing on train and transform both splits."""

    train_features = train.drop(columns=[TARGET_COLUMN])
    test_features = test.drop(columns=[TARGET_COLUMN])
    numeric_columns, categorical_columns = detect_column_types(train_features)

    medians = {
        column: float(pd.to_numeric(train_features[column], errors="coerce").median())
        for column in numeric_columns
    }
    categories = {
        column: sorted(train_features[column].astype("string").astype(str).unique().tolist())
        for column in categorical_columns
    }

    x_train = transform_features(train_features, numeric_columns, medians, categories)
    x_test = transform_features(test_features, numeric_columns, medians, categories)
    x_train[TARGET_COLUMN] = train[TARGET_COLUMN].to_numpy(dtype="int8")
    x_test[TARGET_COLUMN] = test[TARGET_COLUMN].to_numpy(dtype="int8")

    return x_train, x_test, {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categorical_levels": categories,
        "numeric_missing_fill": medians,
    }


def transform_features(
    features: pd.DataFrame,
    numeric_columns: list[str],
    medians: dict[str, float],
    categories: dict[str, list[str]],
) -> pd.DataFrame:
    """Apply numeric conversion and train-fitted one-hot encoding."""

    parts: list[pd.DataFrame] = []
    if numeric_columns:
        numeric = pd.DataFrame(index=features.index)
        for column in numeric_columns:
            values = pd.to_numeric(features[column], errors="coerce")
            numeric[column] = values.fillna(medians[column]).astype("float32")
        parts.append(numeric)

    for column, levels in categories.items():
        values = features[column].astype("string").astype(str)
        values = values.where(values.isin(levels), UNKNOWN_CATEGORY)
        levels_with_unknown = levels + [UNKNOWN_CATEGORY]
        encoded = pd.DataFrame(
            {f"{column}__{level}": (values == level).astype("int8") for level in levels_with_unknown},
            index=features.index,
        )
        parts.append(encoded)

    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=features.index)


def rebalance_training(
    train: pd.DataFrame,
    strategy: RebalanceStrategy,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebalance only the training split."""

    before = class_distribution(train[TARGET_COLUMN])
    if strategy == "none":
        return train, {"strategy": strategy, "before": before, "after": before}

    groups = list(train.groupby(TARGET_COLUMN, sort=True))
    class_sizes = [len(group) for _, group in groups]
    target_size = min(class_sizes) if strategy == "undersample" else max(class_sizes)

    sampled_parts = [
        group.sample(
            n=target_size,
            replace=len(group) < target_size,
            random_state=random_state + offset,
        )
        for offset, (_, group) in enumerate(groups)
    ]
    balanced = (
        pd.concat(sampled_parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    return balanced, {
        "strategy": strategy,
        "before": before,
        "after": class_distribution(balanced[TARGET_COLUMN]),
    }


def select_top_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select top K features by absolute train-set correlation with target."""

    feature_columns = [column for column in train.columns if column != TARGET_COLUMN]
    y = train[TARGET_COLUMN].astype("float64")
    scores = []
    for column in feature_columns:
        x = pd.to_numeric(train[column], errors="coerce").fillna(0).astype("float64")
        score = 0.0 if x.nunique(dropna=False) <= 1 else abs(float(x.corr(y)))
        scores.append({"feature": column, "score": 0.0 if np.isnan(score) else score})

    score_frame = pd.DataFrame(scores).sort_values(
        by=["score", "feature"],
        ascending=[False, True],
        ignore_index=True,
    )
    selected = score_frame["feature"].head(top_k).tolist() if top_k > 0 else feature_columns
    return train[selected + [TARGET_COLUMN]], test[selected + [TARGET_COLUMN]], score_frame


def save_preprocessed_outputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_scores: pd.DataFrame,
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save processed splits, selected features, scores, and report."""

    output_path.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path / "train.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)
    feature_scores.to_csv(output_path / "feature_scores.csv", index=False)
    write_json(output_path / "selected_features.json", [c for c in train.columns if c != TARGET_COLUMN])
    write_json(output_path / "preprocessing_report.json", report)


def run_preprocessing(args: Any) -> None:
    """Run the complete preprocessing workflow from parsed CLI args."""

    files = find_dataset_files(args.input_path)
    data = load_dataset(files, args.max_rows_per_file)
    label_column = infer_label_column(data, args.label_column)
    LOGGER.info("Using label column: %s", label_column)

    data, filter_report = keep_benign_and_ddos(data, label_column)
    data, cleaning_report = clean_rows(data)
    data, drop_report = drop_useless_columns(data, label_column)

    train, test = stratified_train_test_split(data, args.test_size, args.random_state)
    train, test, encoding_report = fit_transform_features(train, test)
    train, rebalance_report = rebalance_training(train, args.rebalance, args.random_state)
    train, test, feature_scores = select_top_features(train, test, args.top_features)

    report = {
        "input_files": [str(path) for path in files],
        "label_column": label_column,
        "filter": filter_report,
        "cleaning": cleaning_report,
        "dropped_columns": drop_report,
        "encoding": encoding_report,
        "split": {
            "test_size": args.test_size,
            "train_rows_after_rebalance_and_feature_selection": len(train),
            "test_rows": len(test),
            "train_class_counts": class_distribution(train[TARGET_COLUMN]),
            "test_class_counts": class_distribution(test[TARGET_COLUMN]),
        },
        "rebalance": rebalance_report,
        "top_features": [column for column in train.columns if column != TARGET_COLUMN],
    }
    save_preprocessed_outputs(train, test, feature_scores, report, args.output_path)
    LOGGER.info("Saved processed files to %s", args.output_path)

