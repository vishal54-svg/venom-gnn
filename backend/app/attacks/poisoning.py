"""Phase 3: poisoning-attack simulation on a clean synthetic network.

Two kinds of poisoning are simulated:

* Feature poisoning  - attacker peers change their observable behaviour
  (traffic, latency, update rate, success rate, reputation). The ground-truth
  label of an attacker is 1 (poisoned), everyone else is 0 (benign).
* Label poisoning    - training labels are flipped to simulate noisy or
  corrupted annotations. This is a separate function so that a later phase can
  apply it to the TRAINING nodes only, while evaluation always uses the true
  labels.

Everything here is a simulation on synthetic data; it does not touch real
networks.
"""
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from app.data.synthetic import FEATURE_NAMES, NetworkDataset

SELECTIONS = ("random", "hub", "clustered")
MODES = ("subtle", "obvious")

# feature -> (kind, subtle_value, obvious_value)
#   "mul": value is a multiplier       (1.3 means +30 %)
#   "add": value is an additive shift  (-0.06 means 6 percentage points lower)
# Reputation goes UP slightly in subtle mode: a stealthy attacker tries to look
# trustworthy, while an obvious attacker is simply bad.
EFFECTS = {
    "traffic_rate":     ("mul", 1.3, 2.5),
    "msg_size_mean":    ("mul", 1.1, 1.8),
    "latency_ms":       ("mul", 1.25, 2.0),
    "uptime_ratio":     ("add", -0.02, -0.15),
    "tx_success_rate":  ("add", -0.06, -0.35),
    "update_frequency": ("mul", 1.5, 3.0),
    "reputation_score": ("add", 0.03, -0.25),
}
BOUNDED = ("uptime_ratio", "tx_success_rate", "reputation_score")  # kept in (0, 1)


@dataclass(frozen=True)
class AttackConfig:
    """Settings for one feature-poisoning attack.

    poison_fraction : fraction of peers that are attackers (0 < f <= 0.5)
    selection       : random    - uniformly random peers
                      hub       - the best-connected peers (most damaging)
                      clustered - one connected group around a random peer
                                  (a colluding / Sybil-style cluster)
    mode            : subtle (hard to spot) or obvious (large deviations)
    seed            : random seed for reproducibility
    """

    poison_fraction: float = 0.10
    selection: str = "random"
    mode: str = "subtle"
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.poison_fraction <= 0.5:
            raise ValueError("poison_fraction must satisfy 0 < f <= 0.5.")
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}.")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}.")


@dataclass
class AttackResult:
    """Output of an attack. The input dataset is never modified."""

    dataset: NetworkDataset       # poisoned features; labels = ground truth
    poisoned_nodes: np.ndarray    # sorted int64 ids of attacker peers
    clean_features: np.ndarray    # untouched copy of the original features
    config: AttackConfig


def select_attackers(G, cfg: AttackConfig, rng: np.random.Generator) -> np.ndarray:
    """Choose which peers are attackers. Returns sorted int64 node ids."""
    n = G.number_of_nodes()
    k = max(1, int(round(cfg.poison_fraction * n)))

    if cfg.selection == "random":
        chosen = rng.choice(n, size=k, replace=False)
    elif cfg.selection == "hub":
        degree = np.array([G.degree(i) for i in range(n)])
        order = np.lexsort((rng.random(n), -degree))  # high degree first, random ties
        chosen = order[:k]
    else:  # clustered: breadth-first growth from a random start peer
        start = int(rng.integers(n))
        seen = [start]
        in_set = {start}
        queue = deque([start])
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
        if len(seen) < k:  # only possible on a disconnected graph
            rest = [i for i in range(n) if i not in in_set]
            extra = rng.choice(rest, size=k - len(seen), replace=False)
            seen.extend(int(x) for x in extra)
        chosen = np.array(seen)

    return np.sort(chosen.astype(np.int64))


def _poison_features(X: np.ndarray, nodes: np.ndarray, mode: str,
                     rng: np.random.Generator) -> np.ndarray:
    """Apply the behaviour shift to the attacker rows of X (in place)."""
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}
    value_idx = 1 if mode == "subtle" else 2
    for name, spec in EFFECTS.items():
        kind, amount = spec[0], spec[value_idx]
        j = col[name]
        strength = rng.uniform(0.7, 1.3, size=len(nodes))  # attackers differ a bit
        if kind == "mul":
            X[nodes, j] = X[nodes, j] * (1.0 + (amount - 1.0) * strength)
        else:
            X[nodes, j] = X[nodes, j] + amount * strength
        if name in BOUNDED:
            X[nodes, j] = np.clip(X[nodes, j], 0.001, 0.999)
    return X


def inject_feature_poisoning(ds: NetworkDataset, cfg: Optional[AttackConfig] = None
                             ) -> AttackResult:
    """Return a poisoned copy of `ds` with ground-truth labels set."""
    cfg = cfg or AttackConfig()
    rng = np.random.default_rng(cfg.seed + 1000)  # separate stream from Phase 2

    nodes = select_attackers(ds.graph, cfg, rng)
    features = ds.features.astype(np.float32).copy()
    _poison_features(features, nodes, cfg.mode, rng)

    labels = np.zeros(ds.graph.number_of_nodes(), dtype=np.int64)
    labels[nodes] = 1

    poisoned = replace(ds, features=features.astype(np.float32), labels=labels)
    return AttackResult(poisoned, nodes, ds.features.copy(), cfg)


def apply_label_poisoning(labels: np.ndarray, fraction: float, seed: int = 42,
                          candidates: Optional[np.ndarray] = None):
    """Flip a fraction of labels (0 <-> 1) to simulate corrupted annotations.

    labels     : (N,) int array of true labels (not modified)
    fraction   : share of the candidate nodes to flip, in [0, 1]
    candidates : node ids eligible for flipping (default: all nodes). A later
                 phase will pass the TRAINING node ids only.

    Returns (noisy_labels, flipped_node_ids).
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1.")
    pool = np.arange(len(labels)) if candidates is None else np.asarray(candidates)
    k = int(round(fraction * len(pool)))
    rng = np.random.default_rng(seed + 2000)
    flipped = np.sort(rng.choice(pool, size=k, replace=False)).astype(np.int64)
    noisy = labels.copy()
    noisy[flipped] = 1 - noisy[flipped]
    return noisy, flipped
