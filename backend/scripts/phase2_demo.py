"""Phase 2 demo: generate a synthetic network and print a summary.

Run from the backend folder:  python scripts/phase2_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import TOPOLOGIES, TopologyConfig  # noqa: E402
from app.data.synthetic import generate_dataset  # noqa: E402
from app.graph.builder import build_graph_arrays  # noqa: E402

for topo in TOPOLOGIES:
    ds = generate_dataset(TopologyConfig(num_nodes=500, topology=topo, seed=42))
    arr = build_graph_arrays(ds)
    n, e = ds.graph.number_of_nodes(), ds.graph.number_of_edges()
    print(f"\n[{topo}]")
    print(f"  nodes={n}  edges={e}  avg_degree={2 * e / n:.2f}")
    print(f"  x={arr.x.shape} {arr.x.dtype}  edge_index={arr.edge_index.shape}")
    print(f"  memory: {(arr.x.nbytes + arr.edge_index.nbytes) / 1024:.1f} KB")
print("\nPhase 2 demo OK")
