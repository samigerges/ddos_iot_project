"""Explicit settings for staged training studies on published selected features."""

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import re

from ddos_iot.paths import PROJECT_ROOT


@dataclass(frozen=True)
class TrainingTrial:
    name: str
    model: str = "xgboost"
    depth: int = 10
    rounds: int = 500
    learning_rate: float = .1
    early_stopping: bool = False
    min_child_weight: float = 1.
    reg_lambda: float = 1.
    weight_power: float = 1.

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", self.name):
            raise ValueError("Use a simple trial directory name.")
        if self.model not in ("xgboost", "catboost"):
            raise ValueError("Unsupported study model.")
        if type(self.depth) is not int or not 1 <= self.depth <= 16 or type(self.rounds) is not int or self.rounds < 1:
            raise ValueError("Invalid depth/rounds.")
        for value in (self.learning_rate, self.min_child_weight, self.reg_lambda, self.weight_power):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("Trial parameters must be finite and nonnegative.")
        if not 0 < self.learning_rate <= 1 or type(self.early_stopping) is not bool:
            raise ValueError("Invalid learning rate/early-stopping flag.")


def capacity_trials() -> tuple[TrainingTrial, ...]:
    return (TrainingTrial("paper_fixed"), TrainingTrial("depth6_fixed", depth=6),
            TrainingTrial("depth6_early", depth=6, early_stopping=True),
            TrainingTrial("depth8_early", depth=8, early_stopping=True),
            TrainingTrial("depth10_early", early_stopping=True),
            TrainingTrial("depth10_regularized", early_stopping=True, min_child_weight=5., reg_lambda=5.))


@dataclass(frozen=True)
class TrainingStudyConfig:
    num_classes: int = 8
    output_path: Path = PROJECT_ROOT / "outputs/studies/8class_v1"
    subset_directory: Path = PROJECT_ROOT / "data/raw/paper_v3/training_subsets"
    seeds: tuple[int, ...] = (42,)
    validation_size: float = .2
    tuning_size_within_development: float = .25
    early_stopping_rounds: int = 30
    capacity: tuple[TrainingTrial, ...] = field(default_factory=capacity_trials)
    weight_powers: tuple[float, ...] = (.75, .5)
    catboost: bool = True
    full_reference_fit: bool = True
    device: str = "gpu"
    n_jobs: int = 2
    batch_size: int = 100_000
    latency_repeats: int = 3
    protected_families: tuple[str, ...] = ("WebBased", "Recon", "BruteForce")

    def __post_init__(self):
        if self.num_classes not in (8, 34) or self.device not in ("cpu", "gpu"):
            raise ValueError("Invalid task/device.")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds):
            raise ValueError("Use distinct nonnegative seeds below 2**32.")
        if not self.capacity or self.capacity[0] != TrainingTrial("paper_fixed"):
            raise ValueError("The first trial must be the pinned depth-10, 500-round weighted reference.")
        if len({t.name.lower() for t in self.capacity}) != len(self.capacity):
            raise ValueError("Trial names must be unique.")
        if any(t.model != "xgboost" or t.weight_power != 1. for t in self.capacity):
            raise ValueError("Capacity ablations must keep XGBoost and inverse weights fixed.")
        for value in (self.validation_size, self.tuning_size_within_development):
            if not math.isfinite(value) or not 0 < value < .5:
                raise ValueError("Split sizes must be fractions between zero and .5.")
        for name in ("early_stopping_rounds", "n_jobs", "batch_size", "latency_repeats"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if len(set(self.weight_powers)) != len(self.weight_powers) or any(not math.isfinite(v) or not 0 <= v < 1 for v in self.weight_powers):
            raise ValueError("Weight powers must be distinct fractions below one.")
        if not self.protected_families or set(self.protected_families) - {"WebBased", "Recon", "BruteForce"}:
            raise ValueError("Declare protected target families.")
        if type(self.catboost) is not bool or type(self.full_reference_fit) is not bool:
            raise ValueError("Stage flags must be Boolean.")
        if self.subset_directory.resolve() == self.output_path.resolve():
            raise ValueError("Study output cannot overwrite source data.")

    @property
    def source_path(self):
        return self.subset_directory / f"{self.num_classes}class_train_23features.csv"

    def as_dict(self):
        values = asdict(self)
        values.update(output_path=str(self.output_path.resolve()), subset_directory=str(self.subset_directory.resolve()))
        return values


def load_study_config(path: Path) -> TrainingStudyConfig:
    values = json.loads(path.read_text(encoding="utf-8"))
    for key in ("output_path", "subset_directory"):
        if key in values:
            p = Path(values[key]); values[key] = p if p.is_absolute() else PROJECT_ROOT / p
    if "capacity" in values:
        values["capacity"] = tuple(TrainingTrial(**t) for t in values["capacity"])
    for key in ("seeds", "weight_powers", "protected_families"):
        if key in values:
            values[key] = tuple(values[key])
    return TrainingStudyConfig(**values)


def weight_trials(best: TrainingTrial, powers: tuple[float, ...]) -> tuple[TrainingTrial, ...]:
    return tuple(replace(best, name=f"weight_{round(p * 100):03d}", weight_power=p) for p in powers)
