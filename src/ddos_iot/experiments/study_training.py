"""GPU-aware fits with explicit identity preprocessing for published features."""

from dataclasses import asdict, replace
import gc
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

from ddos_iot.multiclass import FeatureTransform, MulticlassBundle, predict_probabilities
from ddos_iot.weighting import balanced_class_weights
from .study_config import TrainingStudyConfig, TrainingTrial

LOGGER = logging.getLogger(__name__)


class StudyProgress(TrainingCallback):
    def __init__(self, name):
        self.name = name

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % 100 == 0:
            LOGGER.info("%s: completed %s boosting rounds", self.name, epoch + 1)
        return False


def fit_trial(x: pd.DataFrame, y: np.ndarray, encoder, trial: TrainingTrial,
              config: TrainingStudyConfig, seed: int, *, tuning=None):
    if trial.early_stopping and tuning is None:
        raise ValueError("Early stopping requires a training-only tuning partition.")
    started = time.perf_counter()
    weights = balanced_class_weights(y, config.num_classes, trial.weight_power)
    if trial.model == "xgboost":
        model = XGBClassifier(n_estimators=trial.rounds, max_depth=trial.depth,
                              learning_rate=trial.learning_rate, objective="multi:softprob",
                              num_class=config.num_classes, eval_metric="mlogloss", tree_method="hist",
                              device="cuda:0" if config.device == "gpu" else "cpu", n_jobs=config.n_jobs,
                              max_bin=256, subsample=1., colsample_bytree=1.,
                              min_child_weight=trial.min_child_weight, reg_lambda=trial.reg_lambda,
                              random_state=seed, callbacks=[StudyProgress(trial.name)],
                              **({"early_stopping_rounds": config.early_stopping_rounds} if trial.early_stopping else {}))
        kwargs = {"sample_weight": weights[y], "verbose": False}
        if trial.early_stopping:
            kwargs.update(eval_set=[tuning], sample_weight_eval_set=[weights[tuning[1]]])
        model.fit(x, y, **kwargs)
        rounds = int(model.best_iteration) + 1 if trial.early_stopping else trial.rounds
        actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if config.device == "gpu" and not actual.startswith("cuda"):
            raise RuntimeError("XGBoost silently fell back from GPU.")
        model.set_params(callbacks=None)
    else:
        from catboost import CatBoostClassifier, Pool
        model = CatBoostClassifier(iterations=trial.rounds, depth=trial.depth, learning_rate=trial.learning_rate,
                                   loss_function="MultiClass", task_type="GPU" if config.device == "gpu" else "CPU",
                                   devices="0" if config.device == "gpu" else None,
                                   thread_count=config.n_jobs, random_seed=seed, border_count=128,
                                   allow_writing_files=False, gpu_ram_part=.7)
        kwargs = {"sample_weight": weights[y], "verbose": 100}
        if trial.early_stopping:
            kwargs.update(eval_set=Pool(tuning[0], tuning[1], weight=weights[tuning[1]]),
                          early_stopping_rounds=config.early_stopping_rounds, use_best_model=True)
        model.fit(x, y, **kwargs)
        rounds = int(model.tree_count_)
    # Published CSVs are already transformed. Preserve the bundle contract
    # without fitting another scaler or claiming new feature selection.
    identity = StandardScaler(with_mean=False, with_std=False).fit(x.iloc[:1].to_numpy(dtype=np.float32))
    features = FeatureTransform(list(x.columns), identity, {}, None, "published_23_identity")
    bundle = MulticlassBundle(trial.model, model, features, encoder)
    return bundle, {"trial": asdict(trial), "rounds": rounds, "training_rows": len(y),
                    "training_seconds": time.perf_counter() - started,
                    "class_weights": weights, "seed": seed, "device": config.device,
                    "tuning_rows": len(tuning[1]) if tuning is not None else 0,
                    "preprocessing": "Identity; features selected/scaled upstream by authors on their full training split.",
                    "test_file_read": False}


def trial_probabilities(bundle, x, config):
    return predict_probabilities(bundle.name, bundle.model, x, config.batch_size,
                                 device=config.device if bundle.name == "xgboost" else "cpu")


def refit_trial(x, y, encoder, trial, rounds, config, seed):
    return fit_trial(x, y, encoder, replace(trial, rounds=rounds, early_stopping=False), config, seed)
