"""Data contract for one synthetic decentralized-network dataset.

A GraphDataset bundles everything a model or experiment needs, and `validate()`
checks that it is internally consistent (aligned shapes, valid edges and labels,
disjoint splits, no label leaking into the features).
"""
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from app.data.splits import Split
from app.data.synthetic import NetworkDataset
from app.graph.builder import edge_index_from_graph


@dataclass
class GraphDataset:
    graph: nx.Graph
    features: np.ndarray        # (N, F) float32 node features (labels NOT included)
    labels: np.ndarray          # (N,) int64: 0 = benign, 1 = malicious
    feature_names: tuple
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    config: Any
    metadata: dict = field(default_factory=dict)

    @property
    def node_ids(self) -> np.ndarray:
        return np.arange(self.graph.number_of_nodes())

    @property
    def edge_index(self) -> np.ndarray:
        """(2, 2E) int64, both directions of every undirected edge."""
        return edge_index_from_graph(self.graph)

    @property
    def split(self) -> Split:
        return Split(self.train_idx, self.val_idx, self.test_idx)

    def to_network_dataset(self) -> NetworkDataset:
        """Adapter for the existing model / experiment code."""
        return NetworkDataset(self.graph, self.features, self.labels,
                              tuple(self.feature_names), self.config)

    def validate(self) -> None:
        n = self.graph.number_of_nodes()
        if sorted(self.graph.nodes()) != list(range(n)):
            raise ValueError("Graph nodes must be labelled 0..N-1.")
        if self.features.ndim != 2 or self.features.shape != (n, len(self.feature_names)):
            raise ValueError("features must have shape (N, number of feature names).")
        if not np.isfinite(self.features).all():
            raise ValueError("features contain NaN or infinite values.")
        if self.labels.shape != (n,) or not np.isin(self.labels, (0, 1)).all():
            raise ValueError("labels must be a length-N array containing only 0 and 1.")
        if nx.number_of_selfloops(self.graph) > 0:
            raise ValueError("Graph contains self-loops.")
        ei = self.edge_index
        if ei.size and (ei.min() < 0 or ei.max() >= n):
            raise ValueError("An edge references a node that does not exist.")
        for j in range(self.features.shape[1]):  # label must not be a feature
            if np.array_equal(self.features[:, j], self.labels.astype(self.features.dtype)):
                raise ValueError(f"Feature '{self.feature_names[j]}' is identical to the labels.")
        parts = [self.train_idx, self.val_idx, self.test_idx]
        everything = np.concatenate(parts)
        if len(everything) != n or len(np.unique(everything)) != n:
            raise ValueError("train/val/test indices must be disjoint and cover every node.")
        if everything.min() < 0 or everything.max() >= n:
            raise ValueError("A split index is out of range.")
