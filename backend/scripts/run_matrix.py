"""Run the E1-E9 experiment matrix and save results, charts and a report.

Examples (from the backend folder):
    python scripts/run_matrix.py --seeds 2 --no-gcn        # quick RF-only check
    python scripts/run_matrix.py --seeds 5                 # full study (RF + GCN)
    python scripts/run_matrix.py --seeds 5 --ablation --combined

Outputs go to ../results: SQLite database, results.json, summary.csv, report.md, PNG charts.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.analysis import (  # noqa: E402
    degradation, markdown_report, paired_comparison, summarize,
)
from app.evaluation.charts import make_all_charts  # noqa: E402
from app.evaluation.experiment_runner import run_matrix  # noqa: E402
from app.storage.db import DEFAULT_DB, export_json  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--seeds", type=int, default=3, help="number of repeated seeds (default 3)")
p.add_argument("--nodes", type=int, default=300)
p.add_argument("--no-gcn", action="store_true", help="skip the GCN (no torch needed)")
p.add_argument("--ablation", action="store_true", help="also run RF + neighbourhood features")
p.add_argument("--combined", action="store_true", help="also run combined poisoning (E10-E12)")
p.add_argument("--strategy", choices=("camouflage", "noise"), default="camouflage")
p.add_argument("--out", default=str(DEFAULT_DB.parent))
args = p.parse_args()

out = Path(args.out)
db = out / DEFAULT_DB.name
models = ["random_forest"] + ([] if args.no_gcn else ["gcn"]) + (["rf_neighborhood"] if args.ablation else [])
t0 = time.perf_counter()
records = run_matrix(models, range(args.seeds), args.nodes, args.combined, args.strategy,
                     db_path=db, progress=print)
failed = [r for r in records if r["status"] != "completed"]

summary = summarize(records)
degr = degradation(summary)
comps = [paired_comparison(records, "gcn", "random_forest")] if "gcn" in models else []
if "rf_neighborhood" in models and "gcn" in models:
    comps.append(paired_comparison(records, "gcn", "rf_neighborhood"))
import pandas as pd  # noqa: E402
comparison = pd.concat([c for c in comps if not c.empty], ignore_index=True) if comps else None

out.mkdir(parents=True, exist_ok=True)
summary.to_csv(out / "summary.csv", index=False)
(out / "report.md").write_text(markdown_report(summary, comparison, degr), encoding="utf-8")
n_json = export_json(out / "results.json", db)
charts = make_all_charts(summary, out)

print(f"\nFinished {len(records)} runs ({len(failed)} failed) in {time.perf_counter() - t0:.0f}s")
for r in failed[:5]:
    print("  FAILED:", r["model"], r["matrix_label"], r["error"])
print(f"  database : {db}\n  report   : {out / 'report.md'}\n  json     : {out / 'results.json'} ({n_json} records)")
for c in charts:
    print("  chart    :", c)
print("\nPhase 4b run OK")
