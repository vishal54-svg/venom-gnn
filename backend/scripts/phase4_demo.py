"""Phase 4 demo: Random Forest baseline across attack settings.

Run from the backend folder:  python scripts/phase4_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.attacks.poisoning import (  # noqa: E402
    MODES, SELECTIONS, AttackConfig, inject_feature_poisoning,
)
from app.core.config import TopologyConfig  # noqa: E402
from app.data.splits import make_splits  # noqa: E402
from app.data.synthetic import generate_dataset  # noqa: E402
from app.models.random_forest import run_baseline  # noqa: E402

clean = generate_dataset(TopologyConfig(num_nodes=500, topology="barabasi_albert", seed=42))
HEAD = f"{'attack':<20}{'features':<14}{'precision':>10}{'recall':>9}{'f1':>7}{'roc_auc':>9}"

print("Random Forest baseline (test nodes only, 500-node network, 10% attackers)\n")
print(HEAD)
for mode in MODES:
    for selection in SELECTIONS:
        ds = inject_feature_poisoning(
            clean, AttackConfig(0.10, selection, mode, seed=42)).dataset
        split = make_splits(ds.labels, seed=42)
        for structural in (False, True):
            r = run_baseline(ds, split, include_structural=structural, seed=42)["test"]
            tag = f"{mode}/{selection}"
            feats = "behaviour+graph" if structural else "behaviour"
            print(f"{tag:<20}{feats:<14}{r['precision']:>10.2f}{r['recall']:>9.2f}"
                  f"{r['f1']:>7.2f}{r['roc_auc']:>9.2f}")

print("\nEffect of label poisoning on training labels (subtle / random):")
print(f"{'label noise':<14}{'flipped':>8}{'precision':>10}{'recall':>9}{'f1':>7}{'roc_auc':>9}")
ds = inject_feature_poisoning(clean, AttackConfig(0.10, "random", "subtle", seed=42)).dataset
split = make_splits(ds.labels, seed=42)
for noise in (0.0, 0.1, 0.2, 0.3):
    out = run_baseline(ds, split, label_noise=noise, seed=42)
    r = out["test"]
    print(f"{noise:<14.0%}{out['num_flipped_train_labels']:>8}{r['precision']:>10.2f}"
          f"{r['recall']:>9.2f}{r['f1']:>7.2f}{r['roc_auc']:>9.2f}")
print("\nPhase 4 demo OK")
