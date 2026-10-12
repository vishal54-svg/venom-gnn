"""Phase 4: leakage-safe preparation of model inputs.

* Normalisation statistics (mean/std) come from TRAINING nodes only.
* Label poisoning is applied to TRAINING labels only.
* Evaluation always uses the true labels (`y_true`).

Note: structural features (degree, clustering) are computed on the whole
graph, which is the standard transductive setting for node classification.
"""
from dataclasses import dataclass

import numpy as np

from app.attacks.poisoning import apply_label_poisoning
from app.data.splits import Split
from app.data.synthetic import NetworkDataset
from app.graph.builder import build_graph_arrays, standardize


@dataclass
class PreparedData:
    x: np.ndarray            # (N, F) float32, normalised with train statistics
    edge_index: np.ndarray   # (2, 2E) int64
    y_true: np.ndarray       # (N,) true labels - use for evaluation
    y_train: np.ndarray      # (N,) labels for training (noisy on train nodes only)
    split: Split
    feature_names: tuple
    mean: np.ndarray
    std: np.ndarray
    flipped_nodes: np.ndarray  # train nodes whose label was flipped


def prepare_data(ds: NetworkDataset, split: Split, label_noise: float = 0.0,
                 include_structural: bool = True, seed: int = 42) -> PreparedData:
    arr = build_graph_arrays(ds, include_structural=include_structural, normalize=False)

    mean = arr.x[split.train_idx].mean(axis=0)
    std = arr.x[split.train_idx].std(axis=0)
    x, mean, std = standardize(arr.x, mean, std)

    y_train = arr.y.copy()
    flipped = np.array([], dtype=np.int64)
    if label_noise > 0:
        y_train, flipped = apply_label_poisoning(
            arr.y, label_noise, seed=seed, candidates=split.train_idx
        )

    return PreparedData(x, arr.edge_index, arr.y.copy(), y_train, split,
                        arr.feature_names, mean, std, flipped)
