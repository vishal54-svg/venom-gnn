"""One entry point to train + evaluate any supported model on a PoisonedData condition.

random_forest    baseline: each device's own 9 behavioural features, no graph.
gcn              2-layer GCN: same 9 features, plus message passing over the edges.
rf_neighborhood  ablation: Random Forest given neighbour-averaged features. It shows
                 whether a simple hand-made aggregation captures what the GCN learns.

All models share the same split, the same (poisoned) training data, a scaler fitted on
training rows only, and are scored on the same CLEAN test nodes.

inference_seconds = time to produce test-node predictions. The Random Forest scores
only the test rows; the GCN has to process the whole graph in one forward pass.
"""
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.attacks.pipeline import PoisonedData, to_prepared
from app.evaluation.detailed_metrics import detailed_metrics
from app.evaluation.metrics import select_threshold

MODEL_NAMES = ("random_forest", "gcn", "rf_neighborhood")
MODEL_INFO = {
    "random_forest": {"label": "Random Forest", "uses_graph": False,
                      "description": "Baseline on per-device features only."},
    "gcn": {"label": "GCN", "uses_graph": True,
            "description": "Graph Convolutional Network with message passing."},
    "rf_neighborhood": {"label": "RF + neighbourhood features", "uses_graph": True,
                        "description": "Random Forest given neighbour-averaged features (ablation)."},
}
RF_DEFAULTS = {"n_estimators": 100, "max_depth": 12}
GCN_DEFAULTS = {"hidden": 32, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4,
                "epochs": 200, "patience": 30}


@dataclass
class ModelRun:
    name: str
    probabilities: np.ndarray   # (N,) P(malicious) for every node
    threshold: float            # chosen on validation nodes
    val: dict
    test: dict
    train_seconds: float
    inference_seconds: float
    config: dict
    history: Optional[dict]


def run_model(name: str, poisoned: PoisonedData, seed: int = 42,
              rf_kwargs: Optional[dict] = None, gcn_kwargs: Optional[dict] = None) -> ModelRun:
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")
    prep = to_prepared(poisoned, include_structural=False,
                       include_neighborhood=(name == "rf_neighborhood"))
    va, te = prep.split.val_idx, prep.split.test_idx
    history = None

    if name == "gcn":
        from app.models.gcn import predict_proba, train_gcn  # lazy: needs torch
        cfg = {**GCN_DEFAULTS, **(gcn_kwargs or {})}
        t0 = time.perf_counter()
        model, info = train_gcn(prep, seed=seed, **cfg)
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        proba = predict_proba(model, prep)
        infer_s = time.perf_counter() - t0
        history = {"train_loss": info["history"]["train_loss"], "val_loss": info["history"]["val_loss"],
                   "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"]}
    else:
        from app.models.random_forest import train_random_forest
        cfg = {**RF_DEFAULTS, **(rf_kwargs or {})}
        t0 = time.perf_counter()
        clf = train_random_forest(prep, seed=seed, **cfg)
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        clf.predict_proba(prep.x[te])
        infer_s = time.perf_counter() - t0
        proba = clf.predict_proba(prep.x)[:, 1]

    thr = select_threshold(prep.y_true[va], proba[va])
    config = {"model": name, **cfg, "seed": seed, "num_input_features": int(prep.x.shape[1]),
              "feature_names": list(prep.feature_names), "uses_graph_edges": name == "gcn"}
    return ModelRun(name, proba, thr,
                    detailed_metrics(prep.y_true[va], proba[va], thr),
                    detailed_metrics(prep.y_true[te], proba[te], thr),
                    float(train_s), float(infer_s), config, history)
