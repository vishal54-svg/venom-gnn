"""Phase 2 tests: synthetic data generation and graph construction."""
import networkx as nx
import numpy as np
import pytest

from app.core.config import TOPOLOGIES, TopologyConfig
from app.data.synthetic import FEATURE_NAMES, generate_dataset
from app.graph.builder import (
    STRUCTURAL_FEATURE_NAMES,
    build_graph_arrays,
    compute_structural_features,
    edge_index_from_graph,
    standardize,
)

N = 80


def make(topology="barabasi_albert", seed=7, n=N):
    return generate_dataset(TopologyConfig(num_nodes=n, topology=topology, seed=seed))


# ---------- config ----------
def test_config_rejects_unknown_topology():
    with pytest.raises(ValueError):
        TopologyConfig(topology="ring_of_fire")


def test_config_rejects_tiny_network():
    with pytest.raises(ValueError):
        TopologyConfig(num_nodes=5)


def test_config_rejects_bad_degree_and_rewire():
    with pytest.raises(ValueError):
        TopologyConfig(num_nodes=20, avg_degree=1)
    with pytest.raises(ValueError):
        TopologyConfig(rewire_prob=1.5)


# ---------- topology + features ----------
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_topology_is_connected_and_labelled(topology):
    ds = make(topology)
    G = ds.graph
    assert G.number_of_nodes() == N
    assert sorted(G.nodes()) == list(range(N))
    assert nx.is_connected(G)
    assert nx.number_of_selfloops(G) == 0


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_features_shape_dtype_and_ranges(topology):
    ds = make(topology)
    assert ds.features.shape == (N, len(FEATURE_NAMES))
    assert ds.features.dtype == np.float32
    assert np.isfinite(ds.features).all()
    col = {name: ds.features[:, i] for i, name in enumerate(FEATURE_NAMES)}
    for name in ("uptime_ratio", "tx_success_rate", "reputation_score"):
        assert (col[name] > 0).all() and (col[name] < 1).all()
    for name in ("traffic_rate", "msg_size_mean", "latency_ms", "update_frequency"):
        assert (col[name] > 0).all()


def test_labels_start_all_benign():
    ds = make()
    assert ds.labels.shape == (N,)
    assert ds.labels.dtype == np.int64
    assert ds.labels.sum() == 0


def test_same_seed_is_reproducible_and_different_seed_differs():
    a, b, c = make(seed=1), make(seed=1), make(seed=2)
    assert sorted(a.graph.edges()) == sorted(b.graph.edges())
    assert np.array_equal(a.features, b.features)
    assert not np.array_equal(a.features, c.features)


# ---------- graph construction ----------
def test_structural_features():
    ds = make()
    S = compute_structural_features(ds.graph)
    assert S.shape == (N, 3) and S.dtype == np.float32
    degrees = np.array([ds.graph.degree(i) for i in range(N)])
    assert np.array_equal(S[:, 0], degrees)
    assert ((S[:, 1] >= 0) & (S[:, 1] <= 1)).all()


def test_edge_index_is_symmetric_without_self_loops():
    ds = make("watts_strogatz")
    ei = edge_index_from_graph(ds.graph)
    assert ei.dtype == np.int64
    assert ei.shape == (2, 2 * ds.graph.number_of_edges())
    pairs = set(map(tuple, ei.T.tolist()))
    assert all((v, u) in pairs for u, v in pairs)
    assert (ei[0] != ei[1]).all()
    assert ei.min() >= 0 and ei.max() < N


def test_standardize_gives_zero_mean_unit_std():
    X = np.random.default_rng(0).normal(5, 3, size=(200, 4)).astype(np.float32)
    Xn, mean, std = standardize(X)
    assert np.allclose(Xn.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(Xn.std(axis=0), 1, atol=1e-4)
    Xn2, _, _ = standardize(X, mean, std)  # reusing stats gives the same result
    assert np.allclose(Xn, Xn2)


def test_standardize_handles_constant_column():
    X = np.ones((10, 2), dtype=np.float32)
    Xn, _, _ = standardize(X)
    assert np.isfinite(Xn).all()


def test_build_graph_arrays_consistent():
    ds = make()
    arr = build_graph_arrays(ds)
    expected_f = len(FEATURE_NAMES) + len(STRUCTURAL_FEATURE_NAMES)
    assert arr.x.shape == (N, expected_f)
    assert len(arr.feature_names) == expected_f
    assert arr.y.shape == (N,)
    assert np.isfinite(arr.x).all()
    assert arr.x.dtype == np.float32


def test_build_graph_arrays_without_structural_or_normalisation():
    ds = make()
    arr = build_graph_arrays(ds, include_structural=False, normalize=False)
    assert arr.x.shape == (N, len(FEATURE_NAMES))
    assert np.array_equal(arr.x, ds.features)


def test_larger_network_stays_small_in_memory():
    ds = make("barabasi_albert", n=2000)
    arr = build_graph_arrays(ds)
    assert arr.x.shape[0] == 2000
    assert (arr.x.nbytes + arr.edge_index.nbytes) < 2 * 1024 * 1024  # < 2 MB


# ---------- PyTorch Geometric (skipped if not installed) ----------
def test_to_pyg_data():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from app.graph.builder import to_pyg_data

    arr = build_graph_arrays(make())
    data = to_pyg_data(arr)
    assert data.num_nodes == N
    assert data.x.shape == arr.x.shape
    assert data.edge_index.shape == arr.edge_index.shape
    assert data.is_undirected()
    assert data.y.dtype.is_floating_point is False
