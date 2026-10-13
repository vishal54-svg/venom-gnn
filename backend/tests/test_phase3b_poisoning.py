"""Spec-aligned poisoning tests: training-only feature / label / combined poisoning."""
import numpy as np
import pytest

from app.attacks.feature_poisoning import BASE_FEATURES, FeaturePoisonConfig, poison_features
from app.attacks.label_poisoning import poison_labels
from app.attacks.pipeline import ATTACK_TYPES, PoisonConfig, apply_poisoning, to_prepared
from app.data.generator import NODE_FEATURES, DatasetConfig, generate_graph_dataset

COL = {n: i for i, n in enumerate(NODE_FEATURES)}


@pytest.fixture(scope="module")
def ds():
    return generate_graph_dataset(DatasetConfig(num_nodes=400, malicious_fraction=0.2, seed=8))


def rows_not_train(ds):
    return np.concatenate([ds.val_idx, ds.test_idx])


# ---------- zero poisoning leaves everything unchanged ----------
@pytest.mark.parametrize("attack", ATTACK_TYPES)
def test_zero_ratio_changes_nothing(ds, attack):
    p = apply_poisoning(ds, PoisonConfig(attack, 0.0))
    assert np.array_equal(p.features, ds.features)
    assert np.array_equal(p.train_labels, ds.labels)
    assert p.audit["requested_ratio"] == 0.0


def test_none_ignores_ratio_and_records_zero(ds):
    p = apply_poisoning(ds, PoisonConfig("none", 0.3))
    assert np.array_equal(p.features, ds.features) and np.array_equal(p.train_labels, ds.labels)
    assert p.audit["requested_ratio"] == 0.0


# ---------- label poisoning ----------
@pytest.mark.parametrize("ratio", [0.05, 0.1, 0.2, 0.3])
def test_label_poisoning_counts_and_scope(ds, ratio):
    p = apply_poisoning(ds, PoisonConfig("label", ratio, seed=3))
    n_train = len(ds.train_idx)
    expected = int(round(ratio * n_train))
    changed = np.flatnonzero(p.train_labels != ds.labels)
    assert len(changed) == expected
    assert np.isin(changed, ds.train_idx).all()
    assert np.array_equal(p.train_labels[rows_not_train(ds)], ds.labels[rows_not_train(ds)])
    assert np.array_equal(p.labels, ds.labels)  # true labels untouched
    info = p.audit["label_poisoning"]
    assert info["num_labels_corrupted"] == expected
    assert info["actual_ratio"] == pytest.approx(expected / n_train)
    assert info["benign_to_malicious"] + info["malicious_to_benign"] == expected
    assert np.array_equal(np.array(info["affected_nodes"]), changed)
    assert np.array_equal(p.features, ds.features)  # label attack leaves features alone


def test_label_poisoning_function_validates(ds):
    with pytest.raises(ValueError):
        poison_labels(ds.labels, ds.train_idx, 0.9)
    with pytest.raises(ValueError):
        poison_labels(ds.labels, ds.train_idx, -0.1)


# ---------- feature poisoning ----------
@pytest.mark.parametrize("strategy", ["noise", "camouflage"])
def test_feature_poisoning_only_touches_selected_training_rows(ds, strategy):
    before = ds.features.copy()
    p = apply_poisoning(ds, PoisonConfig("feature", 0.3, strategy, seed=4))
    changed = np.flatnonzero((p.features != ds.features).any(axis=1))
    info = p.audit["feature_poisoning"]
    assert np.array_equal(changed, np.array(info["affected_nodes"]))
    assert np.isin(changed, ds.train_idx).all()
    assert np.array_equal(p.features[rows_not_train(ds)], ds.features[rows_not_train(ds)])
    assert np.array_equal(ds.features, before)            # clean reference not contaminated
    assert np.array_equal(p.train_labels, ds.labels)      # feature attack leaves labels alone
    unmodified = [c for c in NODE_FEATURES
                  if c not in info["features_modified"] + info["derived_features_recomputed"]]
    for name in unmodified:  # features outside the attack scope stay identical
        assert np.array_equal(p.features[:, COL[name]], ds.features[:, COL[name]])


def test_noise_selects_from_all_training_nodes_with_bounded_change(ds):
    cfg = FeaturePoisonConfig(0.3, "noise", ("out_packet_rate",), noise_scale=0.4, seed=2)
    out, info = poison_features(ds.features, ds.labels, ds.train_idx, ds.feature_names, cfg)
    assert info["ratio_basis"] == "training_nodes"
    assert info["num_selected"] == int(round(0.3 * len(ds.train_idx)))
    rows = np.array(info["affected_nodes"])
    ratio = out[rows, COL["out_packet_rate"]] / ds.features[rows, COL["out_packet_rate"]]
    assert (ratio >= 0.6 - 1e-5).all() and (ratio <= 1.4 + 1e-5).all()


def test_camouflage_targets_malicious_training_nodes_only(ds):
    p = apply_poisoning(ds, PoisonConfig("feature", 0.5, "camouflage", seed=6))
    info = p.audit["feature_poisoning"]
    rows = np.array(info["affected_nodes"])
    assert info["ratio_basis"] == "malicious_training_nodes"
    assert (ds.labels[rows] == 1).all() and np.isin(rows, ds.train_idx).all()
    n_mal_train = int((ds.labels[ds.train_idx] == 1).sum())
    assert info["num_selected"] == int(round(0.5 * n_mal_train))
    # selected malicious nodes moved towards the benign training mean
    benign_train = ds.train_idx[ds.labels[ds.train_idx] == 0]
    j = COL["out_packet_rate"]
    mu = ds.features[benign_train, j].mean()
    assert np.abs(p.features[rows, j] - mu).mean() < np.abs(ds.features[rows, j] - mu).mean()


def test_camouflage_uses_training_statistics_only(ds):
    """Changing test-node features must not change what camouflage does."""
    cfg = PoisonConfig("feature", 0.5, "camouflage", seed=6)
    a = apply_poisoning(ds, cfg)
    from dataclasses import replace
    tampered = replace(ds, features=ds.features.copy())
    tampered.features[ds.test_idx] *= 3.0
    b = apply_poisoning(tampered, cfg)
    rows = np.array(a.audit["feature_poisoning"]["affected_nodes"])
    assert np.array_equal(a.features[rows], b.features[rows])


def test_derived_features_stay_consistent(ds):
    p = apply_poisoning(ds, PoisonConfig("feature", 0.5, "camouflage", seed=1))
    X = p.features
    a, f = X[:, COL["active_connections"]], X[:, COL["failed_connections"]]
    assert np.allclose(X[:, COL["connection_failure_ratio"]], f / (a + f + 1e-9), atol=1e-5)
    anomaly = X[:, COL["traffic_anomaly_score"]]
    assert (anomaly >= 0).all() and (anomaly <= 10).all()
    assert np.isfinite(X).all() and (X >= 0).all()


# ---------- combined ----------
def test_combined_applies_both_independently(ds):
    p = apply_poisoning(ds, PoisonConfig("combined", 0.2, "noise", seed=5))
    f, l = p.audit["feature_poisoning"], p.audit["label_poisoning"]
    assert f["num_selected"] > 0 and l["num_labels_corrupted"] > 0
    assert not np.array_equal(p.features, ds.features)
    assert not np.array_equal(p.train_labels, ds.labels)
    overlap = len(set(f["affected_nodes"]) & set(l["affected_nodes"]))
    assert p.audit["overlap_between_selections"] == overlap
    # result equals running each attack separately on the clean data (no double application)
    only_f = apply_poisoning(ds, PoisonConfig("feature", 0.2, "noise", seed=5))
    only_l = apply_poisoning(ds, PoisonConfig("label", 0.2, seed=5))
    assert np.array_equal(p.features, only_f.features)
    assert np.array_equal(p.train_labels, only_l.train_labels)


# ---------- reproducibility, audit ----------
def test_same_seed_reproducible_different_seed_differs(ds):
    cfg = PoisonConfig("combined", 0.2, seed=1)
    a, b = apply_poisoning(ds, cfg), apply_poisoning(ds, cfg)
    c = apply_poisoning(ds, PoisonConfig("combined", 0.2, seed=2))
    assert np.array_equal(a.features, b.features) and np.array_equal(a.train_labels, b.train_labels)
    assert not np.array_equal(a.train_labels, c.train_labels)


def test_audit_contains_required_fields(ds):
    p = apply_poisoning(ds, PoisonConfig("combined", 0.1, seed=9))
    for key in ("attack_type", "requested_ratio", "feature_poisoning", "label_poisoning",
                "seed", "dataset_config", "poison_config", "num_training_nodes"):
        assert key in p.audit
    f = p.audit["feature_poisoning"]
    for key in ("requested_ratio", "actual_ratio", "num_nodes_modified", "features_modified",
                "ratio_basis", "seed", "strategy"):
        assert key in f
    assert p.audit["dataset_config"]["seed"] == 8
    assert p.audit["test_labels_modified"] is False


# ---------- validation ----------
def test_invalid_configs_rejected():
    for kwargs in ({"attack_type": "magic"}, {"ratio": -0.1}, {"ratio": 0.9},
                   {"feature_strategy": "zap"}, {"features": ("traffic_volume",)},
                   {"features": ()}, {"noise_scale": 0.0}, {"camouflage_strength": 0.0}):
        with pytest.raises(ValueError):
            PoisonConfig(**kwargs)


# ---------- integration with the model pipeline ----------
def test_to_prepared_is_leakage_safe(ds):
    p = apply_poisoning(ds, PoisonConfig("combined", 0.2, "camouflage", seed=3))
    prep = to_prepared(p)
    tr, va, te = ds.train_idx, ds.val_idx, ds.test_idx
    assert np.array_equal(prep.y_true, ds.labels)
    assert np.array_equal(prep.y_train, p.train_labels)
    assert np.array_equal(prep.y_train[va], ds.labels[va]) and np.array_equal(prep.y_train[te], ds.labels[te])
    assert np.allclose(prep.x[tr].mean(axis=0), 0, atol=1e-4)  # scaler fitted on training rows
    assert prep.x.shape[1] == len(NODE_FEATURES)
    assert np.isin(prep.flipped_nodes, tr).all()


def test_to_prepared_feature_groups(ds):
    p = apply_poisoning(ds, PoisonConfig("none"))
    assert to_prepared(p).x.shape[1] == 9
    assert to_prepared(p, include_structural=True).x.shape[1] == 12
    assert to_prepared(p, include_neighborhood=True).x.shape[1] == 11
    assert to_prepared(p, True, True).x.shape[1] == 14


def test_random_forest_runs_on_poisoned_data(ds):
    from app.evaluation.report import evaluate_probabilities
    from app.models.random_forest import train_random_forest
    p = apply_poisoning(ds, PoisonConfig("combined", 0.2, seed=2))
    prep = to_prepared(p)
    clf = train_random_forest(prep, n_estimators=30)
    out = evaluate_probabilities(prep, clf.predict_proba(prep.x)[:, 1])
    assert np.isfinite(out["test"]["roc_auc"])
