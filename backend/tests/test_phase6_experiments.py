"""Phase 6 tests: experiment grid, runner, aggregation, outputs."""
import csv

import numpy as np
import pytest

from app.attacks.poisoning import AttackConfig
from app.core.config import TopologyConfig
from app.evaluation.experiments import (
    MODEL_NAMES, SWEEP_LABEL, Setting, build_settings, compare_models,
    run_experiments, summarize, write_outputs,
)

RF_ONLY = ("rf_behaviour", "rf_graph")
SMALL = [("attack", Setting(num_nodes=150, mode="obvious")),
         ("label_noise", Setting(num_nodes=150, label_noise=0.2))]


# ---------- grid ----------
@pytest.mark.parametrize("preset", ["quick", "standard"])
def test_presets_are_valid(preset):
    settings = build_settings(preset)
    assert settings
    for sweep, s in settings:
        assert sweep in SWEEP_LABEL
        TopologyConfig(num_nodes=s.num_nodes, topology=s.topology)
        AttackConfig(s.poison_fraction, s.selection, s.mode)
        assert 0.0 <= s.label_noise <= 1.0
    assert len(build_settings("standard")) == 16


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        build_settings("everything")


# ---------- runner ----------
def test_run_experiments_row_count_and_fields():
    rows = run_experiments(SMALL, seeds=(0, 1), models=RF_ONLY)
    assert len(rows) == len(SMALL) * 2 * len(RF_ONLY)
    for key in ("sweep", "setting", "seed", "model", "f1", "roc_auc", "precision",
                "recall", "pr_auc", "seconds", "topology", "mode", "selection",
                "poison_fraction", "label_noise", "num_nodes"):
        assert key in rows[0]
    assert all(np.isfinite(r["f1"]) and np.isfinite(r["roc_auc"]) for r in rows)


def test_run_experiments_is_reproducible():
    a = run_experiments(SMALL[:1], seeds=(0,), models=RF_ONLY)
    b = run_experiments(SMALL[:1], seeds=(0,), models=RF_ONLY)
    assert [r["f1"] for r in a] == [r["f1"] for r in b]


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        run_experiments(SMALL[:1], seeds=(0,), models=("magic_forest",))


def test_progress_callback_called_per_setting():
    seen = []
    run_experiments(SMALL, seeds=(0,), models=("rf_graph",), progress=seen.append)
    assert len(seen) == len(SMALL)


# ---------- aggregation ----------
def test_summarize_groups_and_counts():
    rows = run_experiments(SMALL, seeds=(0, 1, 2), models=RF_ONLY)
    s = summarize(rows)
    assert len(s) == len(SMALL) * len(RF_ONLY)
    assert (s["n"] == 3).all()
    assert (s["f1_std"] >= 0).all() and s["f1_std"].notna().all()


def test_summarize_single_seed_has_zero_std():
    s = summarize(run_experiments(SMALL[:1], seeds=(0,), models=RF_ONLY))
    assert (s["f1_std"] == 0).all()


def test_compare_models_paired_difference():
    rows = run_experiments(SMALL, seeds=(0, 1), models=RF_ONLY)
    c = compare_models(rows, "rf_graph", "rf_behaviour", "f1")
    assert len(c) == len(SMALL)
    assert (c["n"] == 2).all()
    assert ((c["wins"] >= 0) & (c["wins"] <= c["n"])).all()
    # antisymmetry: swapping the models flips the sign of the mean difference
    c2 = compare_models(rows, "rf_behaviour", "rf_graph", "f1")
    assert np.allclose(c["mean_diff"].to_numpy(), -c2["mean_diff"].to_numpy())


# ---------- outputs ----------
def test_write_outputs_creates_files(tmp_path):
    rows = run_experiments(SMALL, seeds=(0, 1), models=RF_ONLY)
    paths = write_outputs(rows, tmp_path / "nested" / "results")
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    with open(paths["results"], newline="") as f:
        assert len(list(csv.DictReader(f))) == len(rows)
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "# Venom-GNN experiment summary" in text
    assert "## attack" in text and "## label_noise" in text
    assert "rf_graph" in text


# ---------- GCN path (skipped when torch is missing) ----------
def test_gcn_models_run_through_runner():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    rows = run_experiments(SMALL[:1], seeds=(0,), models=MODEL_NAMES,
                           gcn_kwargs={"epochs": 20, "patience": 10})
    assert {r["model"] for r in rows} == set(MODEL_NAMES)
    assert all(np.isfinite(r["roc_auc"]) for r in rows)
