"""Feature poisoning: corrupt the telemetry of selected TRAINING nodes.

Two clearly defined strategies:

noise      Random telemetry corruption. A share `ratio` of the TRAINING nodes
           (chosen uniformly) get bounded multiplicative noise,
           x' = x * (1 + u), u ~ Uniform(-noise_scale, +noise_scale),
           on the chosen features.

camouflage Targeted evasion by compromised devices. A share `ratio` of the
           MALICIOUS TRAINING nodes move their reported behaviour towards the
           benign average, x' = (1 - a) * x + a * mu_benign, with a =
           camouflage_strength and mu_benign computed from benign TRAINING
           nodes only. The malicious class therefore looks more benign in the
           training data.

Only training rows change. Validation and test features are untouched, and the
clean input array is never modified. Derived features (traffic volume, anomaly
score, failure ratio) are recomputed so each poisoned row stays internally
consistent.

The two strategies measure ratio against different groups (all training nodes
vs. malicious training nodes); the audit record states which one was used.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from app.data.generator import _REF_LOG_SD, _REF_LOG_TRAFFIC

STRATEGIES = ("noise", "camouflage")
BASE_FEATURES = (
    "in_packet_rate", "out_packet_rate", "active_connections",
    "failed_connections", "avg_response_ms", "activity_variability",
)


@dataclass(frozen=True)
class FeaturePoisonConfig:
    ratio: float
    strategy: str = "camouflage"
    features: Tuple[str, ...] = BASE_FEATURES
    noise_scale: float = 0.5
    camouflage_strength: float = 0.7
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 0.5:
            raise ValueError("feature poisoning ratio must be between 0 and 0.5.")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}.")
        if not self.features or any(f not in BASE_FEATURES for f in self.features):
            raise ValueError(f"features must be a non-empty subset of {BASE_FEATURES}.")
        if not 0.0 < self.noise_scale < 1.0:
            raise ValueError("noise_scale must satisfy 0 < noise_scale < 1.")
        if not 0.0 < self.camouflage_strength <= 1.0:
            raise ValueError("camouflage_strength must satisfy 0 < strength <= 1.")


def _recompute_derived(X: np.ndarray, old: np.ndarray, names: list, rows: np.ndarray) -> None:
    c = {n: names.index(n) for n in names}
    old_io = old[rows, c["in_packet_rate"]] + old[rows, c["out_packet_rate"]]
    new_io = X[rows, c["in_packet_rate"]] + X[rows, c["out_packet_rate"]]
    if "traffic_volume" in c:
        X[rows, c["traffic_volume"]] = old[rows, c["traffic_volume"]] * new_io / old_io
        if "traffic_anomaly_score" in c:
            traffic = X[rows, c["traffic_volume"]]
            X[rows, c["traffic_anomaly_score"]] = np.clip(
                np.abs(np.log(traffic) - _REF_LOG_TRAFFIC) / _REF_LOG_SD, 0.0, 10.0)
    if "connection_failure_ratio" in c:
        a, f = X[rows, c["active_connections"]], X[rows, c["failed_connections"]]
        X[rows, c["connection_failure_ratio"]] = f / (a + f + 1e-9)


def poison_features(features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray,
                    feature_names, cfg: FeaturePoisonConfig):
    """Return (poisoned_features, info). `features` itself is not modified."""
    names = list(feature_names)
    cols = [names.index(f) for f in cfg.features]
    train_idx = np.asarray(train_idx)
    out = features.astype(np.float32).copy()

    if cfg.strategy == "noise":
        pool, basis = train_idx, "training_nodes"
    else:
        pool = train_idx[labels[train_idx] == 1]
        basis = "malicious_training_nodes"
        if len(pool) == 0:
            raise ValueError("camouflage needs at least one malicious training node.")
    n_sel = int(round(cfg.ratio * len(pool)))

    rng = np.random.default_rng(cfg.seed + 301)
    selected = np.sort(rng.choice(pool, size=n_sel, replace=False)).astype(np.int64)

    if n_sel:
        if cfg.strategy == "noise":
            u = rng.uniform(-cfg.noise_scale, cfg.noise_scale, size=(n_sel, len(cols)))
            out[np.ix_(selected, cols)] = features[np.ix_(selected, cols)] * (1.0 + u)
        else:
            benign_train = train_idx[labels[train_idx] == 0]
            mu = features[np.ix_(benign_train, cols)].mean(axis=0)
            a = cfg.camouflage_strength
            out[np.ix_(selected, cols)] = (1 - a) * features[np.ix_(selected, cols)] + a * mu
        _recompute_derived(out, features, names, selected)

    info = {
        "strategy": cfg.strategy,
        "requested_ratio": float(cfg.ratio),
        "ratio_basis": basis,
        "pool_size": int(len(pool)),
        "num_selected": int(n_sel),
        "num_nodes_modified": int(n_sel),
        "actual_ratio": float(n_sel / len(pool)) if len(pool) else 0.0,
        "features_modified": list(cfg.features),
        "derived_features_recomputed": [n for n in ("traffic_volume", "traffic_anomaly_score",
                                                    "connection_failure_ratio") if n in names],
        "noise_scale": cfg.noise_scale if cfg.strategy == "noise" else None,
        "camouflage_strength": cfg.camouflage_strength if cfg.strategy == "camouflage" else None,
        "affected_nodes": [int(i) for i in selected],
        "seed": int(cfg.seed),
    }
    return out, info
