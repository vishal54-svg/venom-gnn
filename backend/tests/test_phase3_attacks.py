"""Phase 3 tests: feature poisoning and label poisoning."""
import networkx as nx
import numpy as np
import pytest

from app.attacks.poisoning import (
    MODES,
    SELECTIONS,
    AttackConfig,
    apply_label_poisoning,
    inject_feature_poisoning,
)
from app.core.config import TopologyConfig
from app.data.synthetic import FEATURE_NAMES, generate_dataset
from app.graph.builder import build_graph_arrays

N = 200
COL = {name: i for i, name in enumerate(FEATURE_NAMES)}


@pytest.fixture(scope="module")
def clean():
    return generate_dataset(TopologyConfig(num_nodes=N, seed=3))


# ---------- config ----------
def test_config_validation():
    with pytest.raises(ValueError):
        AttackConfig(poison_fraction=0.0)
    with pytest.raises(ValueError):
        AttackConfig(poison_fraction=0.9)
    with pytest.raises(ValueError):
        AttackConfig(selection="everyone")
    with pytest.raises(ValueError):
        AttackConfig(mode="loud")


# ---------- feature poisoning ----------
@pytest.mark.parametrize("selection", SELECTIONS)
@pytest.mark.parametrize("mode", MODES)
def test_poison_count_labels_and_benign_untouched(clean, selection, mode):
    cfg = AttackConfig(poison_fraction=0.1, selection=selection, mode=mode, seed=5)
    res = inject_feature_poisoning(clean, cfg)
    nodes = res.poisoned_nodes
    assert len(nodes) == 20 and len(set(nodes.tolist())) == 20
    assert np.array_equal(nodes, np.sort(nodes))
    assert res.dataset.labels.sum() == 20
    assert (res.dataset.labels[nodes] == 1).all()

    benign = np.setdiff1d(np.arange(N), nodes)
    assert np.array_equal(res.dataset.features[benign], clean.features[benign])
    assert not np.allclose(res.dataset.features[nodes], clean.features[nodes])


def test_input_dataset_is_not_modified():
    ds = generate_dataset(TopologyConfig(num_nodes=N, seed=3))
    feats, labels = ds.features.copy(), ds.labels.copy()
    inject_feature_poisoning(ds, AttackConfig(mode="obvious"))
    assert np.array_equal(ds.features, feats)
    assert np.array_equal(ds.labels, labels)


def test_reproducible_and_seed_changes_result(clean):
    a = inject_feature_poisoning(clean, AttackConfig(seed=1))
    b = inject_feature_poisoning(clean, AttackConfig(seed=1))
    c = inject_feature_poisoning(clean, AttackConfig(seed=2))
    assert np.array_equal(a.poisoned_nodes, b.poisoned_nodes)
    assert np.array_equal(a.dataset.features, b.dataset.features)
    assert not np.array_equal(a.poisoned_nodes, c.poisoned_nodes)


def test_hub_selection_targets_highest_degree(clean):
    res = inject_feature_poisoning(clean, AttackConfig(selection="hub"))
    deg = np.array([clean.graph.degree(i) for i in range(N)])
    chosen = np.zeros(N, dtype=bool)
    chosen[res.poisoned_nodes] = True
    assert deg[chosen].min() >= deg[~chosen].max()


def test_clustered_selection_is_a_connected_group(clean):
    res = inject_feature_poisoning(clean, AttackConfig(selection="clustered"))
    sub = clean.graph.subgraph(res.poisoned_nodes.tolist())
    assert nx.is_connected(sub)


def test_obvious_shifts_more_than_subtle(clean):
    sub = inject_feature_poisoning(clean, AttackConfig(mode="subtle", seed=9))
    obv = inject_feature_poisoning(clean, AttackConfig(mode="obvious", seed=9))
    nodes = sub.poisoned_nodes
    assert np.array_equal(nodes, obv.poisoned_nodes)  # same attackers, same seed
    for name in ("traffic_rate", "latency_ms", "update_frequency"):
        j = COL[name]
        base = clean.features[nodes, j]
        assert (obv.dataset.features[nodes, j] / base).mean() > \
               (sub.dataset.features[nodes, j] / base).mean() > 1.0
    j = COL["tx_success_rate"]
    assert obv.dataset.features[nodes, j].mean() < sub.dataset.features[nodes, j].mean() \
           < clean.features[nodes, j].mean()


def test_stealth_reputation_rises_when_subtle(clean):
    res = inject_feature_poisoning(clean, AttackConfig(mode="subtle"))
    j = COL["reputation_score"]
    n = res.poisoned_nodes
    assert res.dataset.features[n, j].mean() > clean.features[n, j].mean()


@pytest.mark.parametrize("mode", MODES)
def test_features_stay_valid(clean, mode):
    res = inject_feature_poisoning(clean, AttackConfig(mode=mode, poison_fraction=0.5))
    X = res.dataset.features
    assert X.dtype == np.float32 and np.isfinite(X).all()
    for name in ("uptime_ratio", "tx_success_rate", "reputation_score"):
        assert (X[:, COL[name]] > 0).all() and (X[:, COL[name]] < 1).all()
    for name in ("traffic_rate", "msg_size_mean", "latency_ms", "update_frequency"):
        assert (X[:, COL[name]] > 0).all()


def test_clean_features_copy_and_graph_pipeline(clean):
    res = inject_feature_poisoning(clean, AttackConfig())
    assert np.array_equal(res.clean_features, clean.features)
    arr = build_graph_arrays(res.dataset)
    assert arr.x.shape[0] == N and arr.y.sum() == len(res.poisoned_nodes)


# ---------- label poisoning ----------
def test_label_flip_count_and_original_unchanged():
    labels = np.array([0] * 80 + [1] * 20, dtype=np.int64)
    original = labels.copy()
    noisy, flipped = apply_label_poisoning(labels, 0.1, seed=4)
    assert np.array_equal(labels, original)
    assert len(flipped) == 10
    assert (noisy != labels).sum() == 10
    assert (noisy[flipped] == 1 - labels[flipped]).all()


def test_label_flip_respects_candidates():
    labels = np.zeros(100, dtype=np.int64)
    candidates = np.arange(0, 40)
    noisy, flipped = apply_label_poisoning(labels, 0.5, seed=1, candidates=candidates)
    assert len(flipped) == 20
    assert np.isin(flipped, candidates).all()
    assert (noisy[40:] == 0).all()


def test_label_flip_edge_cases():
    labels = np.zeros(50, dtype=np.int64)
    noisy, flipped = apply_label_poisoning(labels, 0.0)
    assert len(flipped) == 0 and np.array_equal(noisy, labels)
    with pytest.raises(ValueError):
        apply_label_poisoning(labels, 1.5)
    a = apply_label_poisoning(labels, 0.2, seed=3)[1]
    b = apply_label_poisoning(labels, 0.2, seed=3)[1]
    assert np.array_equal(a, b)
