"""Run Random Forest / GCN experiments under controlled conditions and store the results.

The experiment matrix follows the project specification:

    E1  none                    0 %
    E2-E5   feature poisoning   5 / 10 / 20 / 30 %
    E6-E9   label poisoning     5 / 10 / 20 / 30 %
    E10-E12 combined (optional) 10 / 20 / 30 %

For every seed ONE dataset is generated and every condition is applied to that same
clean dataset, so all models and conditions share the same graph, split and clean test
nodes. A seed changes the network, the malicious devices, the split, the poisoning
choices and the model initialisation.
"""
import hashlib
import time
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

import numpy as np

from app.attacks.pipeline import PoisonConfig, PoisonedData, apply_poisoning
from app.data.dataset import GraphDataset
from app.data.generator import DatasetConfig, generate_graph_dataset
from app.models.model_factory import MODEL_NAMES, run_model
from app.storage.db import save_experiment

RATIOS = (0.05, 0.10, 0.20, 0.30)


def build_conditions(include_combined: bool = False, feature_strategy: str = "camouflage") -> list:
    """Return [(matrix_label, PoisonConfig)] for E1..E9 (+ E10..E12 when combined)."""
    conds = [("E1", PoisonConfig("none", 0.0))]
    n = 2
    for attack in ("feature", "label"):
        for r in RATIOS:
            conds.append((f"E{n}", PoisonConfig(attack, r, feature_strategy)))
            n += 1
    if include_combined:
        for r in (0.10, 0.20, 0.30):
            conds.append((f"E{n}", PoisonConfig("combined", r, feature_strategy)))
            n += 1
    return conds


def split_fingerprint(ds: GraphDataset) -> str:
    """Identifies the clean test set + labels, so only comparable runs are compared."""
    h = hashlib.sha1()
    h.update(ds.test_idx.tobytes())
    h.update(ds.labels.tobytes())
    return h.hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_experiment(poisoned: PoisonedData, model: str, seed: int, matrix_label: Optional[str] = None,
                   batch_id: Optional[str] = None, rf_kwargs: Optional[dict] = None,
                   gcn_kwargs: Optional[dict] = None) -> dict:
    """Train + evaluate one model on one poisoned condition. Returns the record to store.

    A runtime failure is returned as a record with status 'failed' and the error text
    (no metrics are invented). An unknown model name raises ValueError immediately.
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model}'. Choose from {MODEL_NAMES}.")
    ds, audit = poisoned.clean, poisoned.audit
    f_info, l_info = audit["feature_poisoning"], audit["label_poisoning"]
    a_feat = f_info["actual_ratio"] if f_info else None
    a_label = l_info["actual_ratio"] if l_info else None
    record = {
        "experiment_id": uuid.uuid4().hex[:12], "batch_id": batch_id, "matrix_label": matrix_label,
        "created_at": _now(), "model": model, "attack_type": audit["attack_type"],
        "feature_strategy": f_info["strategy"] if f_info else None,
        "requested_ratio": audit["requested_ratio"],
        "actual_ratio_feature": a_feat, "actual_ratio_label": a_label,
        "actual_ratio": a_label if a_label is not None else (a_feat if a_feat is not None else 0.0),
        "seed": seed, "split_fingerprint": split_fingerprint(ds),
        "dataset_config": asdict(ds.config), "poison_audit": audit,
    }
    try:
        t0 = time.perf_counter()
        run = run_model(model, poisoned, seed, rf_kwargs, gcn_kwargs)
    except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", metrics=None,
                      model_config=None, history=None, predictions=None,
                      train_seconds=None, inference_seconds=None)
        return record
    record.update(
        status="completed", error=None, model_config=run.config, history=run.history,
        metrics={"threshold": run.threshold, "validation": run.val, "test": run.test},
        train_seconds=run.train_seconds, inference_seconds=run.inference_seconds,
        predictions=np.round(run.probabilities, 4).tolist(),
        total_seconds=round(time.perf_counter() - t0, 3),
    )
    return record


def run_single_experiment(dataset_cfg: DatasetConfig, poison_cfg: PoisonConfig, model: str,
                          save: bool = True, db_path=None, **kwargs) -> dict:
    """Generate data, apply one condition, run one model (used by the API later)."""
    ds = generate_graph_dataset(dataset_cfg)
    rec = run_experiment(apply_poisoning(ds, poison_cfg), model, dataset_cfg.seed, **kwargs)
    if save:
        save_experiment(rec, db_path)
    return rec


def run_matrix(models: Iterable[str] = ("random_forest", "gcn"), seeds: Iterable[int] = range(5),
               num_nodes: int = 300, include_combined: bool = False,
               feature_strategy: str = "camouflage", dataset_kwargs: Optional[dict] = None,
               rf_kwargs: Optional[dict] = None, gcn_kwargs: Optional[dict] = None,
               save: bool = True, db_path=None,
               progress: Optional[Callable[[str], None]] = None) -> list:
    """Run every (seed, condition, model). Results are stored as they finish."""
    models, seeds = tuple(models), tuple(seeds)
    for m in models:
        if m not in MODEL_NAMES:
            raise ValueError(f"Unknown model '{m}'. Choose from {MODEL_NAMES}.")
    conditions = build_conditions(include_combined, feature_strategy)
    batch_id = uuid.uuid4().hex[:10]
    records = []
    for seed in seeds:
        ds = generate_graph_dataset(DatasetConfig(num_nodes=num_nodes, seed=seed,
                                                  **(dataset_kwargs or {})))
        for label, cond in conditions:
            poisoned = apply_poisoning(ds, replace(cond, seed=seed))
            for model in models:
                rec = run_experiment(poisoned, model, seed, label, batch_id, rf_kwargs, gcn_kwargs)
                if save:
                    save_experiment(rec, db_path)
                records.append(rec)
                if progress:
                    t = rec["metrics"]["test"] if rec["status"] == "completed" else None
                    progress(f"seed={seed} {label:<4}{cond.attack_type:<9}{cond.ratio:>5.0%} "
                             f"{model:<16}" + (f"F1={t['f1']:.2f} AUC={t['roc_auc']:.2f}" if t
                                               else f"FAILED: {rec['error']}"))
    return records
