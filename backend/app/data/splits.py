"""Phase 4: stratified train / validation / test node splits."""
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

MIN_PER_CLASS = 10


@dataclass
class Split:
    """Disjoint node-id arrays (sorted, int64)."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_splits(y: np.ndarray, train_frac: float = 0.6, val_frac: float = 0.2,
                seed: int = 42) -> Split:
    """Stratified split so train, val and test each contain attackers.

    The remaining fraction (1 - train_frac - val_frac) is the test set.
    """
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Need train_frac > 0, val_frac > 0 and train+val < 1.")
    y = np.asarray(y)
    counts = np.bincount(y, minlength=2)
    if counts.min() < MIN_PER_CLASS:
        raise ValueError(
            f"Each class needs at least {MIN_PER_CLASS} nodes to split safely "
            f"(got benign={counts[0]}, poisoned={counts[1]})."
        )
    idx = np.arange(len(y))
    train_idx, rest = train_test_split(
        idx, train_size=train_frac, stratify=y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        rest, train_size=val_frac / (1.0 - train_frac), stratify=y[rest],
        random_state=seed,
    )
    return Split(np.sort(train_idx).astype(np.int64),
                 np.sort(val_idx).astype(np.int64),
                 np.sort(test_idx).astype(np.int64))
