"""Synthetic decentralized-network data: topology + benign node behaviour.

Phase 2 produces a clean (all-benign) network. Poisoning attacks are injected
in a later phase, which will modify `features` and `labels`.
"""
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np

from app.core.config import TopologyConfig

# Behavioural features simulated for every peer.
FEATURE_NAMES = (
    "traffic_rate",      # messages per second
    "msg_size_mean",     # average message size (bytes)
    "latency_ms",        # average response latency
    "uptime_ratio",      # fraction of time online, in (0, 1)
    "tx_success_rate",   # fraction of valid/accepted transactions, in (0, 1)
    "update_frequency",  # model/state updates shared per minute
    "reputation_score",  # peer reputation, in (0, 1)
)


@dataclass
class NetworkDataset:
    """A synthetic network and its node-level data."""

    graph: nx.Graph
    features: np.ndarray  # (N, F) float32 behavioural features
    labels: np.ndarray    # (N,) int64; 0 = benign, 1 = poisoned
    feature_names: tuple
    config: TopologyConfig


def _connect_components(G: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    """Join disconnected components with one random edge each."""
    comps = [sorted(c) for c in nx.connected_components(G)]
    for a, b in zip(comps[:-1], comps[1:]):
        G.add_edge(int(rng.choice(a)), int(rng.choice(b)))
    return G


def generate_topology(cfg: TopologyConfig) -> nx.Graph:
    """Build a connected, undirected graph with nodes labelled 0..N-1."""
    n, k, seed = cfg.num_nodes, cfg.avg_degree, cfg.seed

    if cfg.topology == "barabasi_albert":
        G = nx.barabasi_albert_graph(n, max(1, k // 2), seed=seed)
    elif cfg.topology == "erdos_renyi":
        G = nx.fast_gnp_random_graph(n, k / (n - 1), seed=seed)
        G = _connect_components(G, np.random.default_rng(seed + 1))
    else:  # watts_strogatz
        even_k = max(2, k - (k % 2))
        G = nx.connected_watts_strogatz_graph(
            n, even_k, cfg.rewire_prob, tries=100, seed=seed
        )
    return G


def generate_node_features(G: nx.Graph, rng: np.random.Generator) -> np.ndarray:
    """Simulate benign behaviour for each peer. Returns (N, F) float32."""
    n = G.number_of_nodes()
    degree = np.array([G.degree(i) for i in range(n)], dtype=np.float64)

    traffic = rng.lognormal(mean=3.0, sigma=0.5, size=n)
    traffic *= 1.0 + 0.15 * np.log1p(degree)  # better-connected peers are busier
    msg_size = rng.lognormal(mean=6.0, sigma=0.4, size=n)
    latency = rng.gamma(shape=4.0, scale=15.0, size=n)
    uptime = rng.beta(20.0, 1.5, size=n)
    tx_success = rng.beta(18.0, 1.2, size=n)
    update_freq = rng.lognormal(mean=1.0, sigma=0.4, size=n)
    reputation = rng.beta(8.0, 2.0, size=n)

    X = np.column_stack(
        [traffic, msg_size, latency, uptime, tx_success, update_freq, reputation]
    )
    return X.astype(np.float32)


def generate_dataset(cfg: Optional[TopologyConfig] = None) -> NetworkDataset:
    """Generate a full benign synthetic network from a config."""
    cfg = cfg or TopologyConfig()
    G = generate_topology(cfg)
    rng = np.random.default_rng(cfg.seed)
    features = generate_node_features(G, rng)
    labels = np.zeros(cfg.num_nodes, dtype=np.int64)
    return NetworkDataset(G, features, labels, FEATURE_NAMES, cfg)
