"""Configuration for synthetic network generation."""
from dataclasses import dataclass

TOPOLOGIES = ("barabasi_albert", "erdos_renyi", "watts_strogatz")


@dataclass(frozen=True)
class TopologyConfig:
    """Settings for one synthetic decentralized network.

    num_nodes   : number of peers (keep <= ~5000 on an 8 GB machine)
    topology    : barabasi_albert (scale-free), erdos_renyi (random),
                  watts_strogatz (small-world)
    avg_degree  : target average number of neighbours per peer
    rewire_prob : rewiring probability (watts_strogatz only)
    seed        : random seed, so every run is reproducible
    """

    num_nodes: int = 500
    topology: str = "barabasi_albert"
    avg_degree: int = 6
    rewire_prob: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.topology not in TOPOLOGIES:
            raise ValueError(
                f"Unknown topology '{self.topology}'. Choose from {TOPOLOGIES}."
            )
        if self.num_nodes < 10:
            raise ValueError("num_nodes must be at least 10.")
        if not 2 <= self.avg_degree < self.num_nodes:
            raise ValueError("avg_degree must satisfy 2 <= avg_degree < num_nodes.")
        if not 0.0 <= self.rewire_prob <= 1.0:
            raise ValueError("rewire_prob must be between 0 and 1.")
