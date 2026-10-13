"""Phase 5 demo: Random Forest vs GCN, averaged over a few seeds.

Run from the backend folder:  python scripts/phase5_demo.py
(Takes about a minute on a laptop CPU.)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.attacks.poisoning import (  # noqa: E402
    MODES, SELECTIONS, AttackConfig, inject_feature_poisoning,
)
from app.core.config import TopologyConfig  # noqa: E402
from app.data.splits import make_splits  # noqa: E402
from app.data.synthetic import generate_dataset  # noqa: E402
from app.models.gcn import run_gcn  # noqa: E402
from app.models.random_forest import run_baseline  # noqa: E402

SEEDS = (0, 1, 2)
MODELS = (
    ("RF  behaviour", lambda ds, sp, s: run_baseline(ds, sp, include_structural=False, seed=s)),
    ("RF  +graph",    lambda ds, sp, s: run_baseline(ds, sp, include_structural=True, seed=s)),
    ("GCN behaviour", lambda ds, sp, s: run_gcn(ds, sp, include_structural=False, seed=s)),
    ("GCN +graph",    lambda ds, sp, s: run_gcn(ds, sp, include_structural=True, seed=s)),
)

print(f"Test-set results, mean over seeds {SEEDS} (500 nodes, 10% attackers)")
print("Each cell: F1 / ROC-AUC\n")
print(f"{'attack':<20}" + "".join(f"{name:>16}" for name, _ in MODELS))

for mode in MODES:
    for selection in SELECTIONS:
        scores = {name: [] for name, _ in MODELS}
        for seed in SEEDS:
            clean = generate_dataset(TopologyConfig(num_nodes=500, seed=seed))
            ds = inject_feature_poisoning(
                clean, AttackConfig(0.10, selection, mode, seed=seed)).dataset
            split = make_splits(ds.labels, seed=seed)
            for name, fn in MODELS:
                t = fn(ds, split, seed)["test"]
                scores[name].append((t["f1"], t["roc_auc"]))
        row = f"{mode + '/' + selection:<20}"
        for name, _ in MODELS:
            f1, auc = np.mean(scores[name], axis=0)
            row += f"{f1:>9.2f} /{auc:5.2f}"
        print(row)
print("\nPhase 5 demo OK")
