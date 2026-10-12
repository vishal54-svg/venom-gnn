"""Phase 3 demo: inject poisoning and show how separable the attackers are.

Run from the backend folder:  python scripts/phase3_demo.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.attacks.poisoning import (  # noqa: E402
    MODES, SELECTIONS, AttackConfig, apply_label_poisoning, inject_feature_poisoning,
)
from app.core.config import TopologyConfig  # noqa: E402
from app.data.synthetic import generate_dataset  # noqa: E402
from app.graph.builder import build_graph_arrays  # noqa: E402

clean = generate_dataset(TopologyConfig(num_nodes=500, topology="barabasi_albert", seed=42))
print(f"Clean network: {clean.graph.number_of_nodes()} nodes, "
      f"{clean.graph.number_of_edges()} edges")

for mode in MODES:
    for selection in SELECTIONS:
        res = inject_feature_poisoning(
            clean, AttackConfig(poison_fraction=0.10, selection=selection, mode=mode, seed=42)
        )
        arr = build_graph_arrays(res.dataset)
        bad = arr.y == 1
        # Separation per feature: gap between group means in standard deviations.
        gap = np.abs(arr.x[bad].mean(axis=0) - arr.x[~bad].mean(axis=0))
        top = np.argsort(gap)[::-1][:3]
        deg = np.mean([clean.graph.degree(int(i)) for i in res.poisoned_nodes])
        print(f"\n[{mode} / {selection}] attackers={bad.sum()}  avg_degree={deg:.1f}")
        print("  most separable features: " + ", ".join(
            f"{arr.feature_names[i]} ({gap[i]:.2f} sd)" for i in top))

noisy, flipped = apply_label_poisoning(res.dataset.labels, 0.05, seed=42)
print(f"\nLabel poisoning demo: flipped {len(flipped)} of {len(noisy)} labels")
print("\nPhase 3 demo OK")
