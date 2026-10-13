"""Phase 2 (spec-aligned): synthetic decentralized-network generator.

Simulation assumptions (all configurable, all documented here):

* Every device has a benign behaviour profile: packet rates, connections,
  failed connections, response time, traffic volume and burstiness.
* A configurable fraction of devices is MALICIOUS (class imbalance is a
  parameter). Malicious devices send more traffic, open many connections, fail
  many connections, respond slower and are burstier. A share of them is
  "stealthy" and deviates only a little, so the two classes overlap.
* Malicious devices are placed uniformly at random by default, so graph
  structure does not reveal the label by accident. `malicious_placement=
  "clustered"` makes colluding groups an explicit experimental variable.
* Contagion (`neighbor_effect`): benign devices next to malicious ones see a few
  more failed connections and slower responses. This is the relational signal a
  graph model could exploit; set it to 0 to switch it off.
* `traffic_anomaly_score` is derived from the device's own traffic against a
  fixed nominal baseline. It never uses the label.

All of this is simulated data. Results describe behaviour under these
assumptions, not real network security.
"""
from dataclasses import asdict, dataclass
from collections import deque
from typing import Optional

import networkx as nx
import numpy as np

from app.core.config import TopologyConfig
from app.data.dataset import GraphDataset
from app.data.splits import make_splits
from app.data.synthetic import generate_topology
from app.graph.features import graph_statistics

NODE_FEATURES = (
    "in_packet_rate",            # packets / s received
    "out_packet_rate",           # packets / s sent
    "active_connections",        # open links to other devices
    "failed_connections",        # failed connection attempts per window
    "avg_response_ms",           # average response time
    "traffic_volume",            # KB / s (in + out)
    "traffic_anomaly_score",     # deviation from nominal traffic (0 - 10)
    "connection_failure_ratio",  # failed / (active + failed)
    "activity_variability",      # burstiness of packet rate (coefficient of variation)
)
PLACEMENTS = ("random", "clustered")
MAX_NODES = 5000  # keeps experiments comfortable on an 8 GB laptop

_REF_LOG_TRAFFIC = float(np.log(80.0))  # nominal benign traffic (assumed baseline)
_REF_LOG_SD = 0.4


@dataclass(frozen=True)
class DatasetConfig:
    """Settings for one synthetic dataset.

    malicious_fraction : share of malicious devices (class imbalance)
    separation         : how far malicious behaviour departs from benign (1.0 = default)
    stealth_fraction   : share of malicious devices that deviate only ~40 %
    neighbor_effect    : contagion strength on benign neighbours of malicious devices
    """

    num_nodes: int = 300
    topology: str = "barabasi_albert"
    avg_degree: int = 6
    rewire_prob: float = 0.1
    malicious_fraction: float = 0.15
    malicious_placement: str = "random"
    separation: float = 1.0
    stealth_fraction: float = 0.3
    neighbor_effect: float = 0.5
    train_frac: float = 0.6
    val_frac: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        if not 10 <= self.num_nodes <= MAX_NODES:
            raise ValueError(f"num_nodes must be between 10 and {MAX_NODES}.")
        TopologyConfig(self.num_nodes, self.topology, self.avg_degree,
                       self.rewire_prob, self.seed)  # validates topology settings
        if not 0.02 <= self.malicious_fraction <= 0.5:
            raise ValueError("malicious_fraction must be between 0.02 and 0.5.")
        if self.malicious_placement not in PLACEMENTS:
            raise ValueError(f"malicious_placement must be one of {PLACEMENTS}.")
        if not 0.0 < self.separation <= 5.0:
            raise ValueError("separation must satisfy 0 < separation <= 5.")
        if not 0.0 <= self.stealth_fraction <= 1.0:
            raise ValueError("stealth_fraction must be between 0 and 1.")
        if not 0.0 <= self.neighbor_effect <= 5.0:
            raise ValueError("neighbor_effect must be between 0 and 5.")
        if self.train_frac <= 0 or self.val_frac <= 0 or self.train_frac + self.val_frac >= 1:
            raise ValueError("Need train_frac > 0, val_frac > 0 and train + val < 1.")


def _choose_malicious(G: nx.Graph, k: int, placement: str,
                      rng: np.random.Generator) -> np.ndarray:
    n = G.number_of_nodes()
    if placement == "random":
        return rng.choice(n, size=k, replace=False).astype(np.int64)
    start = int(rng.integers(n))  # clustered: grow one connected group
    seen, in_set, queue = [start], {start}, deque([start])
    while queue and len(seen) < k:
        u = queue.popleft()
        for v in rng.permutation(sorted(G.neighbors(u))):
            v = int(v)
            if v not in in_set:
                in_set.add(v)
                seen.append(v)
                queue.append(v)
                if len(seen) == k:
                    break
    if len(seen) < k:  # disconnected graph fallback
        rest = [i for i in range(n) if i not in in_set]
        seen += [int(x) for x in rng.choice(rest, size=k - len(seen), replace=False)]
    return np.array(seen, dtype=np.int64)


def _generate_features(G: nx.Graph, labels: np.ndarray, cfg: DatasetConfig,
                       rng: np.random.Generator) -> np.ndarray:
    n = len(labels)
    deg = np.array([G.degree(i) for i in range(n)], dtype=float)
    mal = labels == 1
    mal_nbrs = np.array([labels[list(G.neighbors(i))].sum() for i in range(n)], dtype=float)

    # --- benign profile for every device (same random draws for all nodes) ---
    activity = rng.lognormal(0.0, 0.35, n)
    in_rate = 50.0 * activity * (1 + 0.1 * np.log1p(deg)) * rng.lognormal(0, 0.15, n)
    out_rate = in_rate * rng.lognormal(0, 0.2, n)
    active = rng.poisson(np.maximum(deg * rng.uniform(0.6, 1.0, n), 0.5)).astype(float)
    failed_lam = 0.5 + 0.05 * active
    resp = rng.gamma(5.0, 12.0, n)
    msg_kb = rng.lognormal(np.log(0.8), 0.2, n)
    variab = rng.lognormal(np.log(0.25), 0.3, n)

    # --- malicious behaviour (stealthy devices deviate less) ---
    stealthy = rng.random(n) < cfg.stealth_fraction
    jitter = rng.uniform(0.6, 1.4, n)
    k = np.where(mal, cfg.separation * np.where(stealthy, 0.4, 1.0), 0.0)
    out_rate = out_rate * (1 + 1.2 * k * jitter)
    in_rate = in_rate * (1 + 0.3 * k)
    active = active + rng.poisson(3.0 * k)
    failed_lam = failed_lam + 6.0 * k * jitter
    resp = resp * (1 + 0.5 * k)
    variab = variab * (1 + 1.0 * k)

    # --- contagion on benign neighbours of malicious devices ---
    victim = (~mal).astype(float)
    failed_lam = failed_lam + cfg.neighbor_effect * mal_nbrs * victim
    resp = resp * (1 + 0.05 * cfg.neighbor_effect * mal_nbrs * victim)

    failed = rng.poisson(failed_lam).astype(float)
    traffic = (in_rate + out_rate) * msg_kb
    ratio = failed / (active + failed + 1e-9)
    anomaly = np.clip(np.abs(np.log(traffic) - _REF_LOG_TRAFFIC) / _REF_LOG_SD, 0.0, 10.0)

    X = np.column_stack([in_rate, out_rate, active, failed, resp, traffic,
                         anomaly, ratio, variab])
    return X.astype(np.float32)


def generate_graph_dataset(cfg: Optional[DatasetConfig] = None) -> GraphDataset:
    """Generate a reproducible, validated synthetic dataset."""
    cfg = cfg or DatasetConfig()
    G = generate_topology(TopologyConfig(cfg.num_nodes, cfg.topology, cfg.avg_degree,
                                         cfg.rewire_prob, cfg.seed))
    n = cfg.num_nodes
    k = int(round(cfg.malicious_fraction * n))
    if k < 10 or n - k < 10:
        raise ValueError("Need at least 10 benign and 10 malicious devices "
                         "(increase num_nodes or adjust malicious_fraction).")

    malicious = _choose_malicious(G, k, cfg.malicious_placement,
                                  np.random.default_rng(cfg.seed + 11))
    labels = np.zeros(n, dtype=np.int64)
    labels[malicious] = 1
    features = _generate_features(G, labels, cfg, np.random.default_rng(cfg.seed + 22))

    split = make_splits(labels, cfg.train_frac, cfg.val_frac, seed=cfg.seed)
    metadata = {
        "config": asdict(cfg),
        "num_malicious": int(labels.sum()),
        "num_benign": int(n - labels.sum()),
        "actual_malicious_fraction": float(labels.mean()),
        "graph": graph_statistics(G),
        "split_sizes": {"train": len(split.train_idx), "val": len(split.val_idx),
                        "test": len(split.test_idx)},
        "simulated": True,
        "note": "Synthetic data: results describe the simulation, not real networks.",
    }
    ds = GraphDataset(G, features, labels, NODE_FEATURES, split.train_idx,
                      split.val_idx, split.test_idx, cfg, metadata)
    ds.validate()
    return ds
