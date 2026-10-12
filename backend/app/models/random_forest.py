"""Phase 4: Random Forest baseline (uses node features only, no graph message passing)."""
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.data.splits import Split, make_splits
from app.data.synthetic import NetworkDataset
from app.evaluation.metrics import compute_metrics, select_threshold
from app.graph.prepare import PreparedData, prepare_data


def train_random_forest(prep: PreparedData, n_estimators: int = 100,
                        max_depth: Optional[int] = 12, seed: int = 42
                        ) -> RandomForestClassifier:
    """Fit on TRAIN nodes using the (possibly noisy) training labels."""
    tr = prep.split.train_idx
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight="balanced", random_state=seed, n_jobs=1,
    )
    clf.fit(prep.x[tr], prep.y_train[tr])
    return clf


def evaluate_random_forest(clf: RandomForestClassifier, prep: PreparedData) -> dict:
    """Choose the threshold on validation nodes, report val and test metrics."""
    proba = clf.predict_proba(prep.x)[:, 1]
    va, te = prep.split.val_idx, prep.split.test_idx
    thr = select_threshold(prep.y_true[va], proba[va])
    return {
        "threshold": thr,
        "val": compute_metrics(prep.y_true[va], proba[va], thr),
        "test": compute_metrics(prep.y_true[te], proba[te], thr),
    }


def run_baseline(ds: NetworkDataset, split: Optional[Split] = None,
                 label_noise: float = 0.0, include_structural: bool = True,
                 seed: int = 42) -> dict:
    """Prepare data, train a Random Forest and evaluate it on the test nodes."""
    split = split or make_splits(ds.labels, seed=seed)
    prep = prepare_data(ds, split, label_noise, include_structural, seed)
    clf = train_random_forest(prep, seed=seed)
    result = evaluate_random_forest(clf, prep)
    result["feature_names"] = prep.feature_names
    result["num_flipped_train_labels"] = int(len(prep.flipped_nodes))
    return result
