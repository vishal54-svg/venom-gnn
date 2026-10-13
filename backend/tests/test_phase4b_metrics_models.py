"""Detailed metrics and the model factory (Random Forest / GCN / ablation)."""
import numpy as np
import pytest

from app.attacks.pipeline import PoisonConfig, apply_poisoning
from app.data.generator import DatasetConfig, generate_graph_dataset
from app.evaluation.detailed_metrics import detailed_metrics
from app.models.model_factory import MODEL_INFO, MODEL_NAMES, run_model


@pytest.fixture(scope="module")
def poisoned():
    ds = generate_graph_dataset(DatasetConfig(num_nodes=200, seed=4))
    return apply_poisoning(ds, PoisonConfig("label", 0.1, seed=4))


# ---------- detailed metrics ----------
def test_detailed_metrics_confusion_matrix_and_per_class():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.9, 0.1, 0.8, 0.3])   # one FP, one FN
    m = detailed_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == [[3, 1], [1, 1]]  # [[TN, FP], [FN, TP]]
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["malicious"]["support"] == 2
    assert m["per_class"]["benign"]["support"] == 4
    assert m["per_class"]["benign"]["recall"] == pytest.approx(0.75)
    assert m["per_class"]["malicious"]["precision"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)


def test_detailed_metrics_handles_undefined_cases():
    m = detailed_metrics(np.array([0, 0, 1, 1]), np.zeros(4))  # nothing predicted malicious
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    single = detailed_metrics(np.zeros(5, dtype=int), np.full(5, 0.2))
    assert np.isnan(single["roc_auc"]) and single["per_class"]["malicious"]["support"] == 0


# ---------- model factory ----------
def test_model_registry():
    assert set(MODEL_NAMES) == set(MODEL_INFO) == {"random_forest", "gcn", "rf_neighborhood"}
    with pytest.raises(ValueError):
        run_model("transformer", None)


def test_random_forest_run_outputs(poisoned):
    run = run_model("random_forest", poisoned, seed=1, rf_kwargs={"n_estimators": 30})
    n = poisoned.clean.graph.number_of_nodes()
    assert run.probabilities.shape == (n,)
    assert (run.probabilities >= 0).all() and (run.probabilities <= 1).all()
    assert run.train_seconds > 0 and run.inference_seconds > 0
    assert run.config["n_estimators"] == 30 and run.config["num_input_features"] == 9
    assert run.config["uses_graph_edges"] is False and run.history is None
    for split in (run.val, run.test):
        assert {"confusion_matrix", "per_class", "f1", "accuracy"} <= set(split)
    assert sum(map(sum, run.test["confusion_matrix"])) == len(poisoned.clean.test_idx)
    assert sum(map(sum, run.val["confusion_matrix"])) == len(poisoned.clean.val_idx)


def test_neighbourhood_ablation_uses_extra_features(poisoned):
    run = run_model("rf_neighborhood", poisoned, seed=1, rf_kwargs={"n_estimators": 30})
    assert run.config["num_input_features"] == 11
    assert run.config["feature_names"][-2:] == ["neighbor_mean_traffic", "neighbor_mean_failure_ratio"]


def test_random_forest_is_reproducible(poisoned):
    a = run_model("random_forest", poisoned, seed=2, rf_kwargs={"n_estimators": 30})
    b = run_model("random_forest", poisoned, seed=2, rf_kwargs={"n_estimators": 30})
    assert np.array_equal(a.probabilities, b.probabilities) and a.test == b.test


def test_gcn_run_outputs(poisoned):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    run = run_model("gcn", poisoned, seed=1, gcn_kwargs={"epochs": 25, "patience": 10})
    assert run.probabilities.shape == (poisoned.clean.graph.number_of_nodes(),)
    assert run.config["uses_graph_edges"] is True and run.config["hidden"] == 32
    assert run.history["epochs_run"] <= 25 and run.history["best_epoch"] >= 1
    assert run.train_seconds > 0 and run.inference_seconds > 0
