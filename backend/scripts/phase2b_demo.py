"""Spec-aligned Phase 2 demo: generate a dataset and check how separable it is.

Run from the backend folder:  python scripts/phase2b_demo.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import roc_auc_score  # noqa: E402

from app.data.generator import NODE_FEATURES, DatasetConfig, generate_graph_dataset  # noqa: E402
from app.models.random_forest import run_baseline  # noqa: E402

ds = generate_graph_dataset(DatasetConfig(seed=42))
m = ds.metadata
g = m["graph"]
print(f"Dataset: {g['num_nodes']} nodes, {g['num_edges']} edges, "
      f"avg degree {g['average_degree']:.2f}, density {g['density']:.3f}, "
      f"components {g['num_connected_components']}")
print(f"Ground truth (simulation metadata): benign={m['num_benign']}  malicious={m['num_malicious']}")
print(f"Split sizes: {m['split_sizes']}")

print("\nSingle-feature ROC-AUC for telling malicious from benign (0.5 = no signal):")
for i, name in enumerate(NODE_FEATURES):
    auc = roc_auc_score(ds.labels, ds.features[:, i])
    print(f"  {name:<26}{auc:.2f}")
deg = np.array([ds.graph.degree(i) for i in range(len(ds.labels))])
print(f"  {'degree (structure only)':<26}{roc_auc_score(ds.labels, deg):.2f}   <- should be near 0.5")

print("\nQuick Random Forest sanity check on the test nodes:")
nds = ds.to_network_dataset()
for structural in (False, True):
    r = run_baseline(nds, ds.split, include_structural=structural, seed=42)["test"]
    tag = "behaviour+graph features" if structural else "behaviour features only"
    print(f"  {tag:<26} F1={r['f1']:.2f}  ROC-AUC={r['roc_auc']:.2f}")
print("\nPhase 2b demo OK")
