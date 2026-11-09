"""Validated experiment specifications, independent of model execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any

from ddos_iot.paths import PROJECT_ROOT


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_selection: str = "none"
    weight_power: float = 1.0
    model: str = "xgboost"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", self.name):
            raise ValueError("Candidate names must be simple directory names.")
        if self.feature_selection not in ("median", "none"):
            raise ValueError("feature_selection must be median or none.")
        if self.model not in ("xgboost", "lightgbm", "catboost"):
            raise ValueError("Unsupported candidate model.")
        if isinstance(self.weight_power, bool) or not math.isfinite(self.weight_power) or self.weight_power < 0:
            raise ValueError("weight_power must be finite and nonnegative.")


def default_candidates() -> tuple[Candidate, ...]:
    return (
        Candidate("median_balanced", "median", 1.0),
        Candidate("all_balanced", "none", 1.0),
        Candidate("median_unweighted", "median", 0.0),
        Candidate("all_unweighted", "none", 0.0),
        Candidate("all_power_050", "none", 0.5),
        Candidate("all_power_075", "none", 0.75),
    )


@dataclass(frozen=True)
class ExperimentConfig:
    num_classes: int = 8
    train_input_path: Path = PROJECT_ROOT / "data/raw/train/train.csv"
    test_input_path: Path = PROJECT_ROOT / "data/raw/test/test.csv"
    output_path: Path = PROJECT_ROOT / "outputs/experiments/8class_v1"
    label_column: str = "label"
    candidates: tuple[Candidate, ...] = field(default_factory=default_candidates)
    seeds: tuple[int, ...] = (42,)
    data_seed: int = 42
    folds: int = 5
    validation_size: float = 0.2
    early_stopping_size: float = 0.2
    n_estimators: int = 500
    learning_rate: float = 0.1
    max_depth: int = 6
    early_stopping_rounds: int = 30
    selector_sample_size: int = 200_000
    chunk_size: int = 250_000
    max_train_per_raw_class: int = 0
    n_jobs: int = 2
    device: str = "cpu"
    gpu_id: int = 0
    bias_grid: tuple[float, ...] = (-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2.)
    bias_passes: int = 2
    bias_f1_cutoff: float = 0.95
    latency_repeats: int = 3
    prediction_batch_size: int = 100_000
    selection_policy: str = "reference_fpr"
    reference_candidate: str | None = None
    max_benign_fpr_increase: float = 0.0

    def __post_init__(self) -> None:
        if self.num_classes not in (8, 34):
            raise ValueError("num_classes must be 8 or 34.")
        if not self.candidates or len({c.name.lower() for c in self.candidates}) != len(self.candidates):
            raise ValueError("Candidate names must be nonempty and unique (case insensitive).")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Provide distinct seeds.")
        for name in ("data_seed", "gpu_id", "max_train_per_raw_class"):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or (name == "data_seed" and value >= 2**32):
                raise ValueError(f"{name} must be a nonnegative integer (data_seed < 2**32).")
        if any(type(s) is not int or not 0 <= s < 2**32 - 1000 for s in self.seeds):
            raise ValueError("Seeds must be integers in [0, 2**32 - 1000).")
        for name in ("folds", "n_estimators", "max_depth", "early_stopping_rounds", "selector_sample_size",
                     "chunk_size", "bias_passes", "latency_repeats", "prediction_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.folds < 2 or self.selector_sample_size < self.num_classes:
            raise ValueError("Need at least two folds and one selector row per class.")
        if self.n_jobs != -1 and (type(self.n_jobs) is not int or self.n_jobs <= 0):
            raise ValueError("n_jobs must be -1 or positive.")
        for name in ("validation_size", "early_stopping_size", "learning_rate"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie strictly between zero and one.")
        if not 0 < self.bias_f1_cutoff <= 1 or not math.isfinite(self.bias_f1_cutoff):
            raise ValueError("bias_f1_cutoff must be in (0, 1].")
        if not self.bias_grid or any(isinstance(v, bool) or not math.isfinite(v) for v in self.bias_grid):
            raise ValueError("bias_grid must contain finite values.")
        if self.device not in ("cpu", "gpu"):
            raise ValueError("device must be cpu or gpu.")
        if self.selection_policy not in ("reference_fpr", "macro_f1"):
            raise ValueError("selection_policy must be reference_fpr or macro_f1.")
        if self.reference_name not in {c.name for c in self.candidates}:
            raise ValueError("reference_candidate must name a declared candidate.")
        if isinstance(self.max_benign_fpr_increase, bool) or not math.isfinite(self.max_benign_fpr_increase) or not 0 <= self.max_benign_fpr_increase <= 1:
            raise ValueError("max_benign_fpr_increase must be a finite fraction in [0, 1].")
        if self.train_input_path.resolve() == self.test_input_path.resolve():
            raise ValueError("Train and test inputs must be different files.")
        if self.output_path.resolve() in (self.train_input_path.resolve(), self.test_input_path.resolve()):
            raise ValueError("Output directory must not be an input file.")

    @property
    def reference_name(self) -> str:
        return self.candidates[0].name if self.reference_candidate is None else self.reference_candidate

    @property
    def selection_metric(self) -> str:
        if self.selection_policy == "reference_fpr":
            return "Mean validation Macro-F1 subject to the reference Benign FPR limit in every seed"
        return "Mean validation Macro-F1 across all declared seeds"

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("train_input_path", "test_input_path", "output_path"):
            result[name] = str(getattr(self, name).resolve())
        return result


def config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    values = dict(payload)
    values.setdefault("output_path", f"outputs/experiments/{values.get('num_classes', 8)}class_v1")
    for name in ("train_input_path", "test_input_path", "output_path"):
        if name in values:
            path = Path(values[name])
            values[name] = path if path.is_absolute() else PROJECT_ROOT / path
    if "candidates" in values:
        values["candidates"] = tuple(Candidate(**c) for c in values["candidates"])
    for name in ("seeds", "bias_grid"):
        if name in values:
            values[name] = tuple(values[name])
    return ExperimentConfig(**values)


def load_config(path: Path) -> ExperimentConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment configuration must be a JSON object.")
    return config_from_dict(payload)


def experiment_plan(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "class_count": config.num_classes,
        "candidates": [asdict(c) for c in config.candidates],
        "seeds": list(config.seeds), "folds": config.folds,
        "development_model_fits": len(config.candidates) * len(config.seeds) * (config.folds + 1) * 2,
        "test_file_read": False,
        "selection": config.selection_metric,
        "selection_policy": config.selection_policy,
        "reference_candidate": config.reference_name,
        "reference_decision": "argmax",
        "max_benign_fpr_increase": config.max_benign_fpr_increase,
        "output_path": str(config.output_path),
    }
