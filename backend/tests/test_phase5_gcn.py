"""Phase 5 tests: GCN model, training loop, leakage safety."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from app.attacks.poisoning import AttackConfig, inject_feature_poisoning  # noqa: E402
from app.core.config import TopologyConfig  # noqa: E402
from app.data.splits import make_splits  # noqa: E402
from app.data.synthetic import generate_dataset  # noqa: E402
from app.graph.prepare import prepare_data  # noqa: E402
from app.models.gcn import GCN, predict_proba, run_gcn, train_gcn  # noqa: E402

N = 300


def poisoned(mode="obvious", selection="random", seed=21):
    clean = generate_dataset(TopologyConfig(num_nodes=N, seed=seed))
    return inject_feature_poisoning(
        clean, AttackConfig(poison_fraction=0.1, selection=selection, mode=mode, seed=seed)
    ).dataset


def prep_for(ds, **kw):
    return prepare_data(ds, make_splits(ds.labels, seed=2), **kw)


def test_forward_shape_and_eval_is_deterministic():
    prep = prep_for(poisoned())
    model = GCN(prep.x.shape[1], hidden=16)
    model.eval()
    x, ei = torch.from_numpy(prep.x), torch.from_numpy(prep.edge_index)
    out1, out2 = model(x, ei), model(x, ei)
    assert out1.shape == (N, 2)
    assert torch.equal(out1, out2)


def test_training_reduces_loss_and_records_history():
    prep = prep_for(poisoned())
    model, info = train_gcn(prep, epochs=60, patience=60, seed=1)
    h = info["history"]["train_loss"]
    assert len(h) == info["epochs_run"] == 60
    assert h[-1] < h[0]
    assert 1 <= info["best_epoch"] <= 60


def test_early_stopping_stops_before_max_epochs():
    prep = prep_for(poisoned())
    _, info = train_gcn(prep, epochs=500, patience=5, seed=1)
    assert info["epochs_run"] < 500
    assert info["epochs_run"] >= info["best_epoch"]


def test_probabilities_valid_and_reproducible():
    prep = prep_for(poisoned())
    m1, _ = train_gcn(prep, epochs=40, seed=7)
    m2, _ = train_gcn(prep, epochs=40, seed=7)
    p1, p2 = predict_proba(m1, prep), predict_proba(m2, prep)
    assert p1.shape == (N,) and (p1 >= 0).all() and (p1 <= 1).all()
    assert np.allclose(p1, p2, atol=1e-5)


def test_invalid_hyperparameters_rejected():
    prep = prep_for(poisoned())
    with pytest.raises(ValueError):
        train_gcn(prep, epochs=0)
    with pytest.raises(ValueError):
        train_gcn(prep, hidden=0)


def test_val_and_test_training_labels_are_never_used():
    ds = poisoned()
    a = prep_for(ds)
    b = prep_for(ds)
    other = np.concatenate([b.split.val_idx, b.split.test_idx])
    b.y_train[other] = 1 - b.y_train[other]  # corrupt non-train training labels
    pa = predict_proba(train_gcn(a, epochs=40, seed=3)[0], a)
    pb = predict_proba(train_gcn(b, epochs=40, seed=3)[0], b)
    assert np.allclose(pa, pb, atol=1e-5)


def test_test_labels_do_not_influence_training():
    ds = poisoned()
    a = prep_for(ds)
    b = prep_for(ds)
    te = b.split.test_idx
    b.y_true[te] = 1 - b.y_true[te]  # corrupt TEST ground truth only
    pa = predict_proba(train_gcn(a, epochs=40, seed=3)[0], a)
    pb = predict_proba(train_gcn(b, epochs=40, seed=3)[0], b)
    assert np.allclose(pa, pb, atol=1e-5)


def test_gcn_detects_obvious_attacks():
    out = run_gcn(poisoned("obvious"), seed=5)
    assert out["test"]["roc_auc"] > 0.85
    assert out["test"]["recall"] > 0.5
    assert len(out["feature_names"]) == 10


def test_gcn_beats_chance_on_subtle_attacks():
    out = run_gcn(poisoned("subtle"), seed=5)
    assert out["test"]["roc_auc"] > 0.6


def test_gcn_with_label_noise_and_without_structural_features():
    out = run_gcn(poisoned("obvious"), label_noise=0.2, include_structural=False, seed=5)
    assert out["num_flipped_train_labels"] == int(round(0.2 * len(make_splits(
        poisoned("obvious").labels, seed=5).train_idx)))
    assert len(out["feature_names"]) == 7
    assert np.isfinite(out["test"]["roc_auc"])
