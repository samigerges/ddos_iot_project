"""Serializable research ensembles consuming the published scaled feature format."""

from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from .decisions import predict_with_bias, validate_probabilities


def log_probability_features(probabilities):
    values = [np.log(np.maximum(validate_probabilities(p), 1e-12)) for p in probabilities]
    if not values or len({len(p) for p in values}) != 1:
        raise ValueError("Base predictions must have matching nonempty cohorts.")
    return np.concatenate(values, axis=1)


def route_specialist(base_probabilities, specialist_probabilities, group_ids):
    """Replace group probabilities on routed rows, preserving their total mass."""
    base = validate_probabilities(base_probabilities)
    ids = np.asarray(group_ids, dtype=int)
    if len(ids) < 2 or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= base.shape[1]):
        raise ValueError("Invalid specialist class mapping.")
    routed = np.isin(base.argmax(axis=1), ids)
    local = validate_probabilities(specialist_probabilities)
    if local.shape != (int(routed.sum()), len(ids)):
        raise ValueError("Specialist predictions do not align with routed rows/classes.")
    result = base.copy()
    rows = np.flatnonzero(routed)
    mass = base[np.ix_(rows, ids)].sum(axis=1, keepdims=True)
    result[np.ix_(rows, ids)] = mass * local
    return result, routed


@dataclass
class SpecialistBundle:
    base: object
    specialist: object
    group_ids: np.ndarray
    decision_bias: np.ndarray | None = None

    @property
    def encoder(self):
        return self.base.encoder

    def predict_proba(self, x):
        p = self.base.predict_proba(x)
        routed = np.isin(p.argmax(axis=1), self.group_ids)
        local = self.specialist.predict_proba(self.base.features.transform(x.iloc[np.flatnonzero(routed)])) if routed.any() else np.empty((0, len(self.group_ids)))
        return route_specialist(p, local, self.group_ids)[0]

    def predict(self, x):
        p = self.predict_proba(x)
        ids = p.argmax(axis=1) if self.decision_bias is None else predict_with_bias(p, self.decision_bias)
        return self.encoder.inverse_transform(ids)


@dataclass
class StackBundle:
    bases: tuple
    meta: object
    decision_bias: np.ndarray | None = None
    n_jobs: int = 2

    @property
    def encoder(self):
        return self.bases[0].encoder

    def predict_proba(self, x):
        features = log_probability_features([b.predict_proba(x) for b in self.bases])
        with threadpool_limits(limits=self.n_jobs):
            return self.meta.predict_proba(features)

    def predict(self, x):
        p = self.predict_proba(x)
        ids = p.argmax(axis=1) if self.decision_bias is None else predict_with_bias(p, self.decision_bias)
        return self.encoder.inverse_transform(ids)
