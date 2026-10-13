"""Experiment runner, SQLite storage, analysis and charts."""
import json

import numpy as np
import pytest

from app.attacks.pipeline import PoisonConfig, apply_poisoning
from app.data.generator import DatasetConfig, generate_graph_dataset
from app.evaluation import experiment_runner as er
from app.evaluation.analysis import (
    degradation, markdown_report, paired_comparison, records_to_frame, summarize,
)
from app.evaluation.charts import make_all_charts
from app.storage import db

RF = {"n_estimators": 20}


@pytest.fixture(scope="module")
def matrix_records(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "t.db"
    recs = er.run_matrix(("random_forest", "rf_neighborhood"), seeds=(0, 1), num_nodes=200,
                         rf_kwargs=RF, db_path=path)
    return recs, path


# ---------- matrix definition ----------
def test_matrix_matches_spec():
    conds = er.build_conditions()
    assert [c[0] for c in conds] == [f"E{i}" for i in range(1, 10)]
    assert conds[0][1].attack_type == "none"
    assert [c[1].ratio for c in conds[1:5]] == [0.05, 0.10, 0.20, 0.30]
    assert {c[1].attack_type for c in conds[1:5]} == {"feature"}
    assert {c[1].attack_type for c in conds[5:9]} == {"label"}
    combined = er.build_conditions(include_combined=True)
    assert len(combined) == 12 and combined[-1][1].attack_type == "combined"


# ---------- runner ----------
def test_run_matrix_counts_status_and_fields(matrix_records):
    recs, _ = matrix_records
    assert len(recs) == 9 * 2 * 2
    assert all(r["status"] == "completed" for r in recs)
    r = recs[0]
    for key in ("experiment_id", "created_at", "model", "attack_type", "requested_ratio",
                "actual_ratio", "seed", "dataset_config", "poison_audit", "model_config",
                "metrics", "train_seconds", "inference_seconds", "predictions", "split_fingerprint"):
        assert r[key] is not None or key in ("train_seconds",)
    assert len(r["predictions"]) == 200
    assert len({x["experiment_id"] for x in recs}) == len(recs)


def test_same_seed_shares_dataset_across_models_and_conditions(matrix_records):
    recs, _ = matrix_records
    for seed in (0, 1):
        prints = {r["split_fingerprint"] for r in recs if r["seed"] == seed}
        assert len(prints) == 1
    assert len({r["split_fingerprint"] for r in recs}) == 2  # different seeds differ


def test_actual_ratios_recorded(matrix_records):
    recs, _ = matrix_records
    none = [r for r in recs if r["attack_type"] == "none"][0]
    assert none["requested_ratio"] == 0.0 and none["actual_ratio"] == 0.0
    label = [r for r in recs if r["attack_type"] == "label" and r["requested_ratio"] == 0.2][0]
    assert label["actual_ratio"] == pytest.approx(0.2, abs=0.01)
    assert label["poison_audit"]["label_poisoning"]["num_labels_corrupted"] > 0


def test_unknown_model_raises_but_runtime_failure_is_recorded(monkeypatch):
    ds = generate_graph_dataset(DatasetConfig(num_nodes=200, seed=1))
    p = apply_poisoning(ds, PoisonConfig("none"))
    with pytest.raises(ValueError):
        er.run_experiment(p, "oracle", 1)

    def boom(*a, **k):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(er, "run_model", boom)
    rec = er.run_experiment(p, "random_forest", 1)
    assert rec["status"] == "failed" and "out of memory" in rec["error"]
    assert rec["metrics"] is None and rec["predictions"] is None  # nothing invented


def test_run_single_experiment_saves(tmp_path):
    path = tmp_path / "single.db"
    rec = er.run_single_experiment(DatasetConfig(num_nodes=200, seed=2),
                                   PoisonConfig("feature", 0.2, "noise", seed=2),
                                   "random_forest", db_path=path, rf_kwargs=RF)
    stored = db.get_experiment(rec["experiment_id"], path)
    assert stored["status"] == "completed" and stored["poison_audit"]["attack_type"] == "feature"


# ---------- storage ----------
def test_storage_roundtrip_and_filters(matrix_records):
    recs, path = matrix_records
    rec = recs[5]
    got = db.get_experiment(rec["experiment_id"], path)
    assert got["model"] == rec["model"] and got["seed"] == rec["seed"]
    assert got["metrics"]["test"]["f1"] == pytest.approx(rec["metrics"]["test"]["f1"])
    assert got["poison_audit"]["seed"] == rec["poison_audit"]["seed"]
    assert db.get_experiment("missing", path) is None

    rf_label = db.list_experiments(path, model="random_forest", attack_type="label", limit=1000)
    assert len(rf_label) == 4 * 2 and all("predictions" not in r for r in rf_label)
    assert len(db.list_experiments(path, ratio=0.3, limit=1000)) == 2 * 2 * 2
    assert len(db.list_experiments(path, limit=5)) == 5
    latest = db.latest_experiment(path)
    assert latest["status"] == "completed" and len(latest["predictions"]) == 200


def test_storage_nan_becomes_null_and_json_export_is_valid(tmp_path):
    path = tmp_path / "n.db"
    rec = {"experiment_id": "abc", "created_at": "2026-01-01T00:00:00+00:00", "status": "completed",
           "model": "random_forest", "attack_type": "none", "requested_ratio": 0.0,
           "metrics": {"test": {"roc_auc": float("nan"), "f1": np.float64(0.5)}},
           "predictions": [0.1, 0.2]}
    db.save_experiment(rec, path)
    got = db.get_experiment("abc", path)
    assert got["metrics"]["test"]["roc_auc"] is None and got["metrics"]["test"]["f1"] == 0.5
    out = tmp_path / "x" / "export.json"
    assert db.export_json(out, path) == 1
    data = json.loads(out.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert data[0]["experiment_id"] == "abc" and "predictions" not in data[0]


def test_failed_records_are_stored_with_error(tmp_path):
    path = tmp_path / "f.db"
    db.save_experiment({"experiment_id": "bad", "created_at": "2026-01-01T00:00:00+00:00",
                        "status": "failed", "error": "RuntimeError: boom", "model": "gcn",
                        "attack_type": "label", "requested_ratio": 0.1}, path)
    got = db.get_experiment("bad", path)
    assert got["status"] == "failed" and got["metrics"] is None
    assert db.latest_experiment(path) is None  # nothing completed
    assert len(db.list_experiments(path, status="failed")) == 1


# ---------- analysis + charts ----------
def test_summarize_degradation_and_comparison(matrix_records):
    recs, _ = matrix_records
    frame = records_to_frame(recs)
    assert len(frame) == len(recs)
    s = summarize(recs)
    assert len(s) == 9 * 2 and (s["n"] == 2).all() and (s["f1_std"] >= 0).all()
    d = degradation(s)
    assert len(d) == 8 * 2 and set(d["attack_type"].astype(str)) == {"feature", "label"}
    clean = s[s["attack_type"] == "none"].set_index("model")["f1_mean"]
    row = d.iloc[0]
    expected = clean[row.model] - s[(s.attack_type == row.attack_type) &
                                    (s.requested_ratio == row.requested_ratio) &
                                    (s.model == row.model)]["f1_mean"].iloc[0]
    assert row.f1_drop == pytest.approx(expected)
    c = paired_comparison(recs, "rf_neighborhood", "random_forest")
    c2 = paired_comparison(recs, "random_forest", "rf_neighborhood")
    assert len(c) == 9 and np.allclose(c["mean_diff"], -c2["mean_diff"])
    assert ((c["wins"] >= 0) & (c["wins"] <= c["n"])).all()


def test_failed_runs_are_excluded_from_analysis(matrix_records):
    recs, _ = matrix_records
    broken = [dict(r, status="failed", metrics=None) for r in recs[:4]]
    assert len(records_to_frame(recs + broken)) == len(recs)


def test_report_and_charts(matrix_records, tmp_path):
    recs, _ = matrix_records
    s = summarize(recs)
    text = markdown_report(s, paired_comparison(recs, "rf_neighborhood", "random_forest"), degradation(s))
    assert "# Venom-GNN experiment report" in text and "Paired comparison" in text
    assert "random_forest" in text and "label" in text
    paths = make_all_charts(s, tmp_path / "charts")
    assert len(paths) == 4 and all(p.exists() and p.stat().st_size > 1000 for p in paths)
    assert make_all_charts(summarize([]), tmp_path / "none") == []


def test_gcn_in_matrix_when_torch_available(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    recs = er.run_matrix(("random_forest", "gcn"), seeds=(0,), num_nodes=200, rf_kwargs=RF,
                         gcn_kwargs={"epochs": 20, "patience": 10}, save=False)
    assert {r["model"] for r in recs} == {"random_forest", "gcn"}
    assert all(r["status"] == "completed" for r in recs)
