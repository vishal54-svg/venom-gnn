"""Spec-aligned poisoning demo: RF trained under each condition, evaluated on CLEAN test nodes.

Run from the backend folder:  python scripts/phase3b_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.attacks.pipeline import PoisonConfig, apply_poisoning, to_prepared  # noqa: E402
from app.data.generator import DatasetConfig, generate_graph_dataset  # noqa: E402
from app.evaluation.report import evaluate_probabilities  # noqa: E402
from app.models.random_forest import train_random_forest  # noqa: E402

ds = generate_graph_dataset(DatasetConfig(seed=42))
print(f"{ds.metadata['num_malicious']} malicious / {ds.metadata['num_benign']} benign devices; "
      f"{len(ds.train_idx)} train, {len(ds.val_idx)} val, {len(ds.test_idx)} test\n")

CASES = [("none", 0.0, "-")]
CASES += [("feature", r, "camouflage") for r in (0.05, 0.10, 0.20, 0.30)]
CASES += [("label", r, "-") for r in (0.05, 0.10, 0.20, 0.30)]
CASES += [("combined", 0.20, "camouflage")]

print(f"{'attack':<10}{'ratio':>6}  {'applied (a / b)':<22}{'F1':>6}{'precision':>10}{'recall':>8}{'AUC':>6}")
for attack, ratio, strategy in CASES:
    cfg = PoisonConfig(attack, ratio, strategy if strategy != "-" else "camouflage", seed=42)
    p = apply_poisoning(ds, cfg)
    prep = to_prepared(p)
    clf = train_random_forest(prep, seed=42)
    t = evaluate_probabilities(prep, clf.predict_proba(prep.x)[:, 1])["test"]
    f, l = p.audit["feature_poisoning"], p.audit["label_poisoning"]
    applied = []
    if f:
        applied.append(f"{f['num_nodes_modified']} feat rows")
    if l:
        applied.append(f"{l['num_labels_corrupted']} labels")
    print(f"{attack:<10}{ratio:>6.0%}  {' + '.join(applied) or 'nothing':<22}"
          f"{t['f1']:>6.2f}{t['precision']:>10.2f}{t['recall']:>8.2f}{t['roc_auc']:>6.2f}")
print("\nSingle seed, ~9 malicious test nodes: treat small differences as noise.")
print("Phase 3b demo OK")
