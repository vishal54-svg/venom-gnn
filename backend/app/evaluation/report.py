"""Shared evaluation helper: turn predicted probabilities into val/test metrics."""
import numpy as np

from app.evaluation.metrics import compute_metrics, select_threshold
from app.graph.prepare import PreparedData


def evaluate_probabilities(prep: PreparedData, proba: np.ndarray) -> dict:
    """Pick the threshold on validation nodes, report val and test metrics.

    Always scored against the TRUE labels, whatever label noise was used
    for training.
    """
    va, te = prep.split.val_idx, prep.split.test_idx
    thr = select_threshold(prep.y_true[va], proba[va])
    return {
        "threshold": thr,
        "val": compute_metrics(prep.y_true[va], proba[va], thr),
        "test": compute_metrics(prep.y_true[te], proba[te], thr),
    }
