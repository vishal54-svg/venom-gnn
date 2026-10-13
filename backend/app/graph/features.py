"""Graph statistics and optional neighbourhood-activity features."""
import networkx as nx
import numpy as np

NEIGHBORHOOD_FEATURE_NAMES = ("neighbor_mean_traffic", "neighbor_mean_failure_ratio")


def graph_statistics(G: nx.Graph) -> dict:
    """Basic structural summary (no labels involved)."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    degrees = np.array([d for _, d in G.degree()], dtype=int)
    comps = list(nx.connected_components(G))
    hist = np.bincount(degrees) if n else np.array([], dtype=int)
    return {
        "num_nodes": n,
        "num_edges": m,
        "average_degree": float(2 * m / n) if n else 0.0,
        "max_degree": int(degrees.max()) if n else 0,
        "density": float(nx.density(G)),
        "num_connected_components": len(comps),
        "largest_component_size": max((len(c) for c in comps), default=0),
        "degree_histogram": [[int(d), int(c)] for d, c in enumerate(hist) if c > 0],
    }


def neighborhood_features(G: nx.Graph, features: np.ndarray, feature_names) -> np.ndarray:
    """Mean traffic and mean failure ratio over each node's neighbours. (N, 2) float32.

    These use OTHER nodes' behaviour, i.e. graph relationships. They are kept
    separate from the intrinsic features so the Random Forest baseline can be run
    with and without them as a clearly labelled ablation.
    """
    names = list(feature_names)
    ti, ri = names.index("traffic_volume"), names.index("connection_failure_ratio")
    n = G.number_of_nodes()
    out = np.empty((n, 2), dtype=np.float32)
    for i in range(n):
        nbrs = list(G.neighbors(i))
        if nbrs:
            out[i, 0] = features[nbrs, ti].mean()
            out[i, 1] = features[nbrs, ri].mean()
        else:  # isolated node: fall back to its own values
            out[i, 0], out[i, 1] = features[i, ti], features[i, ri]
    return out
