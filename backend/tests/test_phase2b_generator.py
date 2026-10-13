"""Spec-aligned Phase 2 tests: dataset generation, contract and graph construction."""
import networkx as nx
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from app.data.dataset import GraphDataset
from app.data.generator import NODE_FEATURES, PLACEMENTS, DatasetConfig, generate_graph_dataset
from app.graph.features import (
    NEIGHBORHOOD_FEATURE_NAMES, graph_statistics, neighborhood_features,
)

COL = {n: i for i, n in enumerate(NODE_FEATURES)}


# ---------- defaults, shapes, validity ----------
def test_default_dataset_matches_spec_basics():
    ds = generate_graph_dataset()
    assert ds.graph.number_of_nodes() == 300
    assert ds.features.shape == (300, len(NODE_FEATURES))
    assert ds.features.dtype == np.float32
    assert ds.labels.dtype == np.int64 and set(np.unique(ds.labels)) <= {0, 1}
    assert np.isfinite(ds.features).all()
    assert ds.edge_index.shape[0] == 2
    assert nx.is_connected(ds.graph)
    assert len(ds.node_ids) == 300
    ds.validate()


@pytest.mark.parametrize("topology", ["barabasi_albert", "erdos_renyi", "watts_strogatz"])
def test_every_topology_is_valid(topology):
    ds = generate_graph_dataset(DatasetConfig(topology=topology, seed=3))
    assert ds.edge_index.min() >= 0 and ds.edge_index.max() < 300
    assert nx.is_connected(ds.graph)


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_placements_valid(placement):
    ds = generate_graph_dataset(DatasetConfig(malicious_placement=placement))
    assert ds.labels.sum() == 45
    ds.validate()


def test_feature_value_ranges():
    X = generate_graph_dataset(DatasetConfig(seed=5)).features
    assert (X >= 0).all()
    ratio = X[:, COL["connection_failure_ratio"]]
    assert (ratio >= 0).all() and (ratio <= 1).all()
    anomaly = X[:, COL["traffic_anomaly_score"]]
    assert (anomaly >= 0).all() and (anomaly <= 10).all()


# ---------- class imbalance + reproducibility ----------
@pytest.mark.parametrize("frac,expected", [(0.05, 15), (0.15, 45), (0.30, 90)])
def test_class_imbalance_is_configurable(frac, expected):
    ds = generate_graph_dataset(DatasetConfig(malicious_fraction=frac))
    assert ds.labels.sum() == expected
    assert ds.metadata["actual_malicious_fraction"] == pytest.approx(frac, abs=0.005)


def test_same_seed_reproducible_different_seed_differs():
    a, b, c = (generate_graph_dataset(DatasetConfig(seed=s)) for s in (1, 1, 2))
    assert sorted(a.graph.edges()) == sorted(b.graph.edges())
    assert np.array_equal(a.features, b.features) and np.array_equal(a.labels, b.labels)
    assert np.array_equal(a.test_idx, b.test_idx)
    assert not np.array_equal(a.features, c.features)


# ---------- splits ----------
def test_splits_disjoint_stratified_and_complete():
    ds = generate_graph_dataset()
    parts = [ds.train_idx, ds.val_idx, ds.test_idx]
    assert sum(len(p) for p in parts) == 300
    assert len(set(np.concatenate(parts).tolist())) == 300
    for p in parts:
        assert ds.labels[p].mean() == pytest.approx(0.15, abs=0.03)


# ---------- label / leakage safety ----------
def test_labels_are_not_a_feature_column():
    ds = generate_graph_dataset()
    for j in range(ds.features.shape[1]):
        assert not np.array_equal(ds.features[:, j], ds.labels.astype(np.float32))
    assert "label" not in " ".join(NODE_FEATURES).lower()


def test_validate_catches_broken_datasets():
    ds = generate_graph_dataset()
    bad = GraphDataset(ds.graph, ds.features, ds.labels.copy(), ds.feature_names,
                       ds.train_idx, ds.val_idx, ds.test_idx, ds.config)
    bad.labels[0] = 2
    with pytest.raises(ValueError):
        bad.validate()
    bad = GraphDataset(ds.graph, ds.features.copy(), ds.labels, ds.feature_names,
                       ds.train_idx, ds.val_idx, ds.test_idx, ds.config)
    bad.features[0, 0] = np.nan
    with pytest.raises(ValueError):
        bad.validate()
    bad = GraphDataset(ds.graph, ds.features[:, :5], ds.labels, ds.feature_names,
                       ds.train_idx, ds.val_idx, ds.test_idx, ds.config)
    with pytest.raises(ValueError):
        bad.validate()
    leaked = GraphDataset(ds.graph, np.column_stack([ds.features, ds.labels]).astype(np.float32),
                          ds.labels, ds.feature_names + ("leak",), ds.train_idx, ds.val_idx,
                          ds.test_idx, ds.config)
    with pytest.raises(ValueError):
        leaked.validate()
    overlap = GraphDataset(ds.graph, ds.features, ds.labels, ds.feature_names,
                           ds.train_idx, ds.train_idx, ds.test_idx, ds.config)
    with pytest.raises(ValueError):
        overlap.validate()


def test_config_validation():
    for kwargs in ({"num_nodes": 5}, {"num_nodes": 99999}, {"malicious_fraction": 0.9},
                   {"malicious_fraction": 0.0}, {"topology": "star"},
                   {"malicious_placement": "everywhere"}, {"separation": 0.0},
                   {"stealth_fraction": 1.5}, {"neighbor_effect": -1.0},
                   {"train_frac": 0.9, "val_frac": 0.2}):
        with pytest.raises(ValueError):
            DatasetConfig(**kwargs)
    with pytest.raises(ValueError):  # fewer than 10 malicious devices
        generate_graph_dataset(DatasetConfig(num_nodes=100, malicious_fraction=0.05))


# ---------- structural leakage ----------
def test_random_placement_does_not_leak_through_degree():
    aucs = []
    for seed in (1, 2, 3):
        ds = generate_graph_dataset(DatasetConfig(num_nodes=600, seed=seed))
        degree = np.array([ds.graph.degree(i) for i in range(600)])
        aucs.append(roc_auc_score(ds.labels, degree))
    assert 0.38 < float(np.mean(aucs)) < 0.62


def test_behaviour_separates_classes_but_not_perfectly():
    ds = generate_graph_dataset(DatasetConfig(num_nodes=600, seed=4))
    auc = roc_auc_score(ds.labels, ds.features[:, COL["out_packet_rate"]])
    assert 0.7 < auc < 0.99


def test_neighbor_effect_creates_relational_signal():
    def benign_failed(effect):
        ds = generate_graph_dataset(DatasetConfig(num_nodes=1500, malicious_fraction=0.2,
                                                  neighbor_effect=effect, seed=6))
        benign = ds.labels == 0
        near = np.array([ds.labels[list(ds.graph.neighbors(i))].sum() > 0
                         for i in range(1500)])
        f = ds.features[:, COL["failed_connections"]]
        return f[benign & near].mean() - f[benign & ~near].mean()

    assert benign_failed(2.0) > benign_failed(0.0) + 0.5


# ---------- graph statistics + neighbourhood features ----------
def test_graph_statistics():
    ds = generate_graph_dataset()
    s = ds.metadata["graph"]
    assert s == graph_statistics(ds.graph)
    assert s["num_nodes"] == 300 and s["num_edges"] == ds.graph.number_of_edges()
    assert s["average_degree"] == pytest.approx(2 * s["num_edges"] / 300)
    assert s["num_connected_components"] == 1 and s["largest_component_size"] == 300
    assert 0 < s["density"] < 1
    assert sum(c for _, c in s["degree_histogram"]) == 300


def test_neighborhood_features_alignment():
    ds = generate_graph_dataset()
    nb = neighborhood_features(ds.graph, ds.features, ds.feature_names)
    assert nb.shape == (300, len(NEIGHBORHOOD_FEATURE_NAMES)) and nb.dtype == np.float32
    i = 7
    nbrs = list(ds.graph.neighbors(i))
    assert nb[i, 0] == pytest.approx(ds.features[nbrs, COL["traffic_volume"]].mean(), rel=1e-5)
    assert nb[i, 1] == pytest.approx(
        ds.features[nbrs, COL["connection_failure_ratio"]].mean(), rel=1e-5)


# ---------- compatibility with the existing model code ----------
def test_adapter_works_with_existing_pipeline():
    from app.graph.builder import build_graph_arrays
    ds = generate_graph_dataset()
    nds = ds.to_network_dataset()
    arr = build_graph_arrays(nds)
    assert arr.x.shape == (300, len(NODE_FEATURES) + 3)
    assert arr.edge_index.shape == ds.edge_index.shape
    assert np.array_equal(arr.y, ds.labels)
    assert ds.split.train_idx is ds.train_idx
