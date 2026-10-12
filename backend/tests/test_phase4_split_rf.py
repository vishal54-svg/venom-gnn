"""Phase 4 tests: leakage-safe splits, data preparation, metrics, Random Forest."""
import numpy as np
import pytest

from app.attacks.poisoning import AttackConfig, inject_feature_poisoning
from app.core.config import TopologyConfig
from app.data.splits import make_splits
from app.data.synthetic import generate_dataset
from app.evaluation.metrics import compute_metrics, select_threshold
from app.graph.prepare import prepare_data
from app.models.random_forest import run_baseline, train_random_forest

N = 400


def poisoned(mode="obvious", selection="random", seed=11):
    clean = generate_dataset(TopologyConfig(num_nodes=N, seed=seed))
    return inject_feature_poisoning(
        clean, AttackConfig(poison_fraction=0.1, selection=selection, mode=mode, seed=seed)
    ).dataset


# ---------- splits ----------
def test_splits_are_disjoint_cover_everything_and_stratified():
    ds = poisoned()
    s = make_splits(ds.labels, 0.6, 0.2, seed=1)
    all_idx = np.concatenate([s.train_idx, s.val_idx, s.test_idx])
    assert len(all_idx) == N and len(set(all_idx.tolist())) == N
    assert len(s.train_idx) == 240 and len(s.val_idx) == 80 and len(s.test_idx) == 80
    for part in (s.train_idx, s.val_idx, s.test_idx):
        assert ds.labels[part].mean() == pytest.approx(0.1, abs=0.02)


def test_splits_reproducible_and_validated():
    y = poisoned().labels
    a, b = make_splits(y, seed=3), make_splits(y, seed=3)
    assert np.array_equal(a.train_idx, b.train_idx)
    assert not np.array_equal(a.train_idx, make_splits(y, seed=4).train_idx)
    with pytest.raises(ValueError):
        make_splits(y, 0.8, 0.3)
    with pytest.raises(ValueError):  # too few attackers to split
        make_splits(np.array([0] * 95 + [1] * 5))


# ---------- leakage-safe preparation ----------
def test_normalisation_uses_train_nodes_only():
    ds = poisoned()
    s = make_splits(ds.labels, seed=2)
    prep = prepare_data(ds, s, include_structural=True)
    assert np.allclose(prep.x[s.train_idx].mean(axis=0), 0, atol=1e-4)
    assert np.allclose(prep.x[s.train_idx].std(axis=0), 1, atol=1e-3)
    # statistics are NOT the whole-graph ones
    assert not np.allclose(prep.x.mean(axis=0), 0, atol=1e-6)


def test_label_noise_touches_train_nodes_only():
    ds = poisoned()
    s = make_splits(ds.labels, seed=2)
    prep = prepare_data(ds, s, label_noise=0.2, seed=2)
    assert np.array_equal(prep.y_true, ds.labels)
    assert (prep.y_train != prep.y_true).sum() == len(prep.flipped_nodes) == 48
    assert np.isin(prep.flipped_nodes, s.train_idx).all()
    assert np.array_equal(prep.y_train[s.val_idx], prep.y_true[s.val_idx])
    assert np.array_equal(prep.y_train[s.test_idx], prep.y_true[s.test_idx])


def test_no_noise_means_identical_labels():
    ds = poisoned()
    prep = prepare_data(ds, make_splits(ds.labels))
    assert np.array_equal(prep.y_train, prep.y_true) and len(prep.flipped_nodes) == 0


def test_feature_set_switch():
    ds = poisoned()
    s = make_splits(ds.labels)
    assert prepare_data(ds, s, include_structural=False).x.shape[1] == 7
    assert prepare_data(ds, s, include_structural=True).x.shape[1] == 10


# ---------- metrics ----------
def test_metrics_perfect_and_all_negative_predictions():
    y = np.array([0, 0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.1, 0.9, 0.8]))
    assert m["precision"] == m["recall"] == m["f1"] == m["roc_auc"] == 1.0
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 3)
    z = compute_metrics(y, np.zeros(5))
    assert z["recall"] == 0.0 and z["f1"] == 0.0 and z["accuracy"] == 0.6


def test_metrics_single_class_does_not_crash():
    m = compute_metrics(np.zeros(5, dtype=int), np.full(5, 0.3))
    assert np.isnan(m["roc_auc"]) and m["tn"] == 5


def test_select_threshold_prefers_best_f1():
    y = np.array([0] * 8 + [1] * 4)
    p = np.array([0.1] * 8 + [0.3, 0.35, 0.4, 0.45])  # positives all below 0.5
    t = select_threshold(y, p)
    assert t <= 0.3
    assert compute_metrics(y, p, t)["f1"] > compute_metrics(y, p, 0.5)["f1"]


# ---------- Random Forest ----------
def test_random_forest_detects_obvious_attacks():
    ds = poisoned("obvious")
    out = run_baseline(ds, seed=5)
    assert out["test"]["roc_auc"] > 0.95
    assert out["test"]["recall"] > 0.7
    assert len(out["feature_names"]) == 10


def test_random_forest_handles_subtle_attacks_and_is_reproducible():
    ds = poisoned("subtle")
    a, b = run_baseline(ds, seed=5), run_baseline(ds, seed=5)
    assert a["test"]["roc_auc"] > 0.7
    assert a["test"] == b["test"]


def test_random_forest_with_label_noise_runs_and_reports_flips():
    out = run_baseline(poisoned("obvious"), label_noise=0.2, seed=5)
    assert out["num_flipped_train_labels"] == 48
    assert np.isfinite(out["test"]["roc_auc"])


def test_random_forest_probabilities_valid():
    ds = poisoned()
    s = make_splits(ds.labels)
    prep = prepare_data(ds, s)
    clf = train_random_forest(prep, n_estimators=20)
    p = clf.predict_proba(prep.x)[:, 1]
    assert p.shape == (N,) and (p >= 0).all() and (p <= 1).all()
