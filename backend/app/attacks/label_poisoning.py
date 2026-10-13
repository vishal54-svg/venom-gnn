"""Label poisoning: flip the labels of randomly chosen TRAINING nodes.

Only `train_idx` rows can change. Validation and test labels, and the true
labels used for evaluation, are never touched.
"""
import numpy as np


def poison_labels(labels: np.ndarray, train_idx: np.ndarray, ratio: float, seed: int = 42):
    """Return (train_labels, info).

    ratio : share of TRAINING nodes whose label is flipped (0 <= ratio <= 0.5)
    train_labels : copy of `labels` with the selected training labels flipped
    """
    if not 0.0 <= ratio <= 0.5:
        raise ValueError("label poisoning ratio must be between 0 and 0.5.")
    labels = np.asarray(labels)
    train_idx = np.asarray(train_idx)
    n_train = len(train_idx)
    n_flip = int(round(ratio * n_train))

    rng = np.random.default_rng(seed + 401)
    selected = np.sort(rng.choice(train_idx, size=n_flip, replace=False)).astype(np.int64)

    train_labels = labels.copy()
    train_labels[selected] = 1 - train_labels[selected]

    info = {
        "requested_ratio": float(ratio),
        "ratio_basis": "training_nodes",
        "num_training_nodes": int(n_train),
        "num_selected": int(n_flip),
        "num_labels_corrupted": int((train_labels != labels).sum()),
        "actual_ratio": float(n_flip / n_train) if n_train else 0.0,
        "benign_to_malicious": int((labels[selected] == 0).sum()),
        "malicious_to_benign": int((labels[selected] == 1).sum()),
        "affected_nodes": [int(i) for i in selected],
        "seed": int(seed),
    }
    return train_labels, info
