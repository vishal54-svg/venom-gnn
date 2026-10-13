"""Phase 6: run the Random Forest vs GCN experiments and save results.

Run from the backend folder, for example:

    python scripts/run_experiments.py --preset quick        # ~1 min check
    python scripts/run_experiments.py --preset standard     # full study
    python scripts/run_experiments.py --preset standard --no-gcn   # RF only

Outputs go to ../results (results.csv, summary.csv, comparison.csv, summary.md).
Those files are git-ignored by design: they are generated, not source code.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.experiments import (  # noqa: E402
    MODEL_NAMES, build_settings, run_experiments, write_outputs,
)

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--preset", choices=("quick", "standard"), default="quick")
parser.add_argument("--seeds", type=int, default=None,
                    help="number of seeds (default: 2 for quick, 5 for standard)")
parser.add_argument("--nodes", type=int, default=None, help="network size override")
parser.add_argument("--no-gcn", action="store_true", help="skip GCN models (no torch needed)")
parser.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "results"))
args = parser.parse_args()

n_seeds = args.seeds or (2 if args.preset == "quick" else 5)
seeds = tuple(range(n_seeds))
models = tuple(m for m in MODEL_NAMES if not (args.no_gcn and m.startswith("gcn")))
settings = build_settings(args.preset, args.nodes)

print(f"Preset={args.preset}  settings={len(settings)}  seeds={n_seeds}  models={len(models)}"
      f"  -> {len(settings) * n_seeds * len(models)} runs")
t0 = time.perf_counter()
rows = run_experiments(settings, seeds, models, progress=print)
paths = write_outputs(rows, args.out)
print(f"\nFinished {len(rows)} runs in {time.perf_counter() - t0:.0f}s")
for name, p in paths.items():
    print(f"  {name:<10} {p}")
print("\nPhase 6 run OK")
