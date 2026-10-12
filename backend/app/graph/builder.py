"""Graph construction: turn a NetworkDataset into model-ready arrays."""
from dataclasses import dataclass

import networkx as nx
import numpy as np

from app.data.synthetic import NetworkDataset

STRUCTURAL_FEATURE_NAMES = ("degree", "clustering", "avg_neighbor_degree")


@dataclass
class GraphArrays:
    """Plain NumPy arrays describing a graph (no torch needed)."""

    x: np.ndarray           # (N, F) float32 node features
    edge_index: np.ndarray  # (2, 2E) int64, both directions
    y: np.ndarray           # (N,) int64 labels
    feature_names: tuple
    mean: np.ndarray        # (F,) normalisation mean
    std: np.ndarray         # (F,) normalisation std


def _check_node_ids(G: nx.Graph) -> None:
    if sorted(G.nodes()) != list(range(G.number_of_nodes())):
        raise ValueError("Graph nodes must be labelled 0..N-1.")


def compute_structural_features(G: nx.Graph) -> np.ndarray:
    """Degree, clustering coefficient, avg neighbour degree. (N, 3) float32."""
    _check_node_ids(G)
    n = G.number_of_nodes()
    clustering = nx.clustering(G)
    avg_nbr = nx.average_neighbor_degree(G)
    out = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        out[i] = (G.degree(i), clustering[i], avg_nbr[i])
    return out


def edge_index_from_graph(G: nx.Graph) -> np.ndarray:
    """Directed (2, 2E) edge list containing both directions of every edge."""
    _check_node_ids(G)
    edges = np.array(
        sorted((min(u, v), max(u, v)) for u, v in G.edges()), dtype=np.int64
    ).reshape(-1, 2)
    return np.concatenate([edges.T, edges[:, ::-1].T], axis=1)


def standardize(X: np.ndarray, mean=None, std=None):
    """Z-score features. Pass mean/std (e.g. from training nodes) to reuse them."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Xn = ((X - mean) / std).astype(np.float32)
    return Xn, np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def build_graph_arrays(
    ds: NetworkDataset, include_structural: bool = True, normalize: bool = True
) -> GraphArrays:
    """Combine behavioural + structural features and the edge list."""
    X = ds.features
    names = tuple(ds.feature_names)
    if include_structural:
        X = np.concatenate([X, compute_structural_features(ds.graph)], axis=1)
        names = names + STRUCTURAL_FEATURE_NAMES
    X = X.astype(np.float32)

    if normalize:
        X, mean, std = standardize(X)
    else:
        mean = np.zeros(X.shape[1], dtype=np.float32)
        std = np.ones(X.shape[1], dtype=np.float32)

    return GraphArrays(
        x=X,
        edge_index=edge_index_from_graph(ds.graph),
        y=ds.labels.copy(),
        feature_names=names,
        mean=mean,
        std=std,
    )


def to_pyg_data(arrays: GraphArrays):
    """Convert to a torch_geometric Data object (imports torch lazily)."""
    import torch
    from torch_geometric.data import Data

    return Data(
        x=torch.from_numpy(arrays.x),
        edge_index=torch.from_numpy(arrays.edge_index),
        y=torch.from_numpy(arrays.y),
    )
