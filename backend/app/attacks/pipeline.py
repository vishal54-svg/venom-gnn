"""Apply a poisoning condition to a clean GraphDataset, with a full audit trail.

attack_type
    none      clean training (the reference condition)
    feature   feature poisoning only
    label     label poisoning only
    combined  both, applied independently from the clean data (each exactly once,
              with separate random streams; overlap between the two selections
              is recorded)

The clean dataset is never modified, so any condition can be repeated.
Validation / test features and ALL true labels stay clean.
"""
from dataclasses import asdict, dataclass, replace
from typing import Optional, Tuple

import numpy as np

from app.attacks.feature_poisoning import (
    BASE_FEATURES, STRATEGIES, FeaturePoisonConfig, poison_features,
)
from app.attacks.label_poisoning import poison_labels
from app.data.dataset import GraphDataset
from app.graph.features import NEIGHBORHOOD_FEATURE_NAMES, neighborhood_features
from app.graph.prepare import PreparedData, prepare_data

ATTACK_TYPES = ("none", "feature", "label", "combined")


@dataclass(frozen=True)
class PoisonConfig:
    attack_type: str = "none"
    ratio: float = 0.0
    feature_strategy: str = "camouflage"
    features: Tuple[str, ...] = BASE_FEATURES
    noise_scale: float = 0.5
    camouflage_strength: float = 0.7
    seed: int = 42

    def __post_init__(self) -> None:
        if self.attack_type not in ATTACK_TYPES:
            raise ValueError(f"attack_type must be one of {ATTACK_TYPES}.")
        if not 0.0 <= self.ratio <= 0.5:
            raise ValueError("ratio must be between 0 and 0.5.")
        if self.feature_strategy not in STRATEGIES:
            raise ValueError(f"feature_strategy must be one of {STRATEGIES}.")
        FeaturePoisonConfig(self.ratio, self.feature_strategy, self.features,
                            self.noise_scale, self.camouflage_strength, self.seed)


@dataclass
class PoisonedData:
    clean: GraphDataset          # untouched reference dataset
    features: np.ndarray         # (N, F) features after poisoning (train rows only may differ)
    train_labels: np.ndarray     # (N,) labels the model trains on (train rows only may differ)
    labels: np.ndarray           # (N,) TRUE labels - used for validation / test evaluation
    audit: dict
    config: PoisonConfig

    def to_network_dataset(self):
        """Poisoned features + TRUE labels, for the existing model code."""
        return replace(self.clean.to_network_dataset(), features=self.features,
                       labels=self.labels)


def apply_poisoning(ds: GraphDataset, cfg: Optional[PoisonConfig] = None) -> PoisonedData:
    cfg = cfg or PoisonConfig()
    features = ds.features.copy()
    train_labels = ds.labels.copy()
    feature_info = label_info = None

    run_feature = cfg.attack_type in ("feature", "combined")
    run_label = cfg.attack_type in ("label", "combined")
    ratio = cfg.ratio if cfg.attack_type != "none" else 0.0

    if run_feature:
        fcfg = FeaturePoisonConfig(ratio, cfg.feature_strategy, cfg.features,
                                   cfg.noise_scale, cfg.camouflage_strength, cfg.seed)
        features, feature_info = poison_features(ds.features, ds.labels, ds.train_idx,
                                                 ds.feature_names, fcfg)
    if run_label:
        train_labels, label_info = poison_labels(ds.labels, ds.train_idx, ratio, cfg.seed)

    overlap = None
    if feature_info and label_info:
        overlap = len(set(feature_info["affected_nodes"]) & set(label_info["affected_nodes"]))

    audit = {
        "attack_type": cfg.attack_type,
        "requested_ratio": float(ratio),
        "feature_poisoning": feature_info,
        "label_poisoning": label_info,
        "overlap_between_selections": overlap,
        "seed": int(cfg.seed),
        "poison_config": asdict(cfg),
        "dataset_config": ds.metadata.get("config"),
        "num_training_nodes": int(len(ds.train_idx)),
        "validation_and_test_features_modified": False,
        "test_labels_modified": False,
        "applied": True,
    }
    return PoisonedData(ds, features, train_labels, ds.labels.copy(), audit, cfg)


def to_prepared(pd: PoisonedData, include_structural: bool = False,
                include_neighborhood: bool = False) -> PreparedData:
    """Model-ready arrays: scaler fitted on (poisoned) TRAINING rows only,
    training labels = poisoned labels, evaluation labels = true labels."""
    ds = pd.clean
    X, names = pd.features, tuple(ds.feature_names)
    if include_neighborhood:
        nb = neighborhood_features(ds.graph, X, names)
        X, names = np.concatenate([X, nb], axis=1).astype(np.float32), names + NEIGHBORHOOD_FEATURE_NAMES
    nds = replace(ds.to_network_dataset(), features=X, labels=pd.labels, feature_names=names)
    prep = prepare_data(nds, ds.split, label_noise=0.0, include_structural=include_structural,
                        seed=pd.config.seed)
    flipped = np.flatnonzero(pd.train_labels != pd.labels).astype(np.int64)
    return replace(prep, y_train=pd.train_labels.copy(), flipped_nodes=flipped)
