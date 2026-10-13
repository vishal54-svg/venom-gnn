"""Phase 6: experiment runner - Random Forest vs GCN across attack settings.

One-factor-at-a-time sweeps around a default setting (BA network, subtle
attack, random attackers, 10 % poisoned, clean training labels):

    attack       mode x selection
    topology     barabasi_albert / erdos_renyi / watts_strogatz
    poison_rate  5 % / 10 % / 20 % attackers
    label_noise  0 % / 10 % / 20 % / 30 % flipped training labels

Every setting is repeated over several seeds; a seed changes the network, the
attacker choice, the split and the model initialisation. Results are reported
as mean and spread over seeds.
"""
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from app.attacks.poisoning import AttackConfig, inject_feature_poisoning
from app.core.config import TopologyConfig
from app.data.splits import make_splits
from app.data.synthetic import generate_dataset
from app.models.random_forest import run_baseline

MODEL_NAMES = ("rf_behaviour", "rf_graph", "gcn_behaviour", "gcn_graph")
METRICS = ("f1", "precision", "recall", "roc_auc", "pr_auc")


@dataclass(frozen=True)
class Setting:
    topology: str = "barabasi_albert"
    mode: str = "subtle"
    selection: str = "random"
    poison_fraction: float = 0.10
    label_noise: float = 0.0
    num_nodes: int = 500


SWEEP_LABEL = {
    "attack": lambda s: f"{s.mode}/{s.selection}",
    "topology": lambda s: s.topology,
    "poison_rate": lambda s: f"{s.poison_fraction:.0%}",
    "label_noise": lambda s: f"{s.label_noise:.0%}",
}


def build_settings(preset: str = "standard", num_nodes: Optional[int] = None):
    """Return a list of (sweep_name, Setting)."""
    if preset not in ("quick", "standard"):
        raise ValueError("preset must be 'quick' or 'standard'.")
    n = num_nodes or (200 if preset == "quick" else 500)
    base = Setting(num_nodes=n)
    if preset == "quick":
        return (
            [("attack", replace(base, mode=m)) for m in ("subtle", "obvious")]
            + [("label_noise", replace(base, label_noise=x)) for x in (0.0, 0.2)]
        )
    out = [("attack", replace(base, mode=m, selection=s))
           for m in ("subtle", "obvious") for s in ("random", "hub", "clustered")]
    out += [("topology", replace(base, topology=t))
            for t in ("barabasi_albert", "erdos_renyi", "watts_strogatz")]
    out += [("poison_rate", replace(base, poison_fraction=f)) for f in (0.05, 0.10, 0.20)]
    out += [("label_noise", replace(base, label_noise=x)) for x in (0.0, 0.1, 0.2, 0.3)]
    return out


def build_dataset(setting: Setting, seed: int):
    """Synthetic network + poisoning + stratified split for one (setting, seed)."""
    clean = generate_dataset(TopologyConfig(
        num_nodes=setting.num_nodes, topology=setting.topology, seed=seed))
    ds = inject_feature_poisoning(clean, AttackConfig(
        setting.poison_fraction, setting.selection, setting.mode, seed=seed)).dataset
    return ds, make_splits(ds.labels, seed=seed)


def run_model(name: str, ds, split, label_noise: float, seed: int,
              gcn_kwargs: Optional[dict] = None) -> dict:
    """Train + evaluate one named model; returns its run_* result dict."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")
    structural = name.endswith("_graph")
    if name.startswith("rf"):
        return run_baseline(ds, split, label_noise=label_noise,
                            include_structural=structural, seed=seed)
    from app.models.gcn import run_gcn  # lazy: torch is only needed for the GCN
    return run_gcn(ds, split, label_noise=label_noise, include_structural=structural,
                   seed=seed, **(gcn_kwargs or {}))


def run_experiments(settings: Iterable, seeds: Iterable[int],
                    models: Iterable[str] = MODEL_NAMES,
                    gcn_kwargs: Optional[dict] = None,
                    progress: Optional[Callable[[str], None]] = None) -> list:
    """Run every (setting, seed, model). Returns a flat list of row dicts."""
    models, seeds = tuple(models), tuple(seeds)
    for m in models:
        if m not in MODEL_NAMES:
            raise ValueError(f"Unknown model '{m}'. Choose from {MODEL_NAMES}.")
    rows = []
    for sweep, setting in settings:
        label = SWEEP_LABEL[sweep](setting)
        if progress:
            progress(f"[{sweep}] {label}")
        for seed in seeds:
            ds, split = build_dataset(setting, seed)
            for name in models:
                t0 = time.perf_counter()
                res = run_model(name, ds, split, setting.label_noise, seed, gcn_kwargs)
                row = {"sweep": sweep, "setting": label, **asdict(setting),
                       "seed": seed, "model": name,
                       "seconds": round(time.perf_counter() - t0, 3)}
                row.update({k: res["test"][k] for k in METRICS})
                rows.append(row)
    return rows


def summarize(rows: list) -> pd.DataFrame:
    """Mean and std over seeds for every (sweep, setting, model)."""
    df = pd.DataFrame(rows)
    agg = df.groupby(["sweep", "setting", "model"], sort=False).agg(
        n=("seed", "count"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_mean=("roc_auc", "mean"), auc_std=("roc_auc", "std"),
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
        seconds_mean=("seconds", "mean"),
    ).reset_index()
    return agg.fillna({"f1_std": 0.0, "auc_std": 0.0})


def compare_models(rows: list, model_a: str, model_b: str, metric: str = "f1"
                   ) -> pd.DataFrame:
    """Paired difference (a - b) per setting, matched by seed.

    wins = number of seeds where model_a beat model_b strictly.
    """
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["sweep", "setting", "seed"], columns="model",
                          values=metric, sort=False)
    wide = wide.dropna(subset=[model_a, model_b])
    wide["diff"] = wide[model_a] - wide[model_b]
    wide["win"] = (wide["diff"] > 0).astype(int)
    out = wide.groupby(["sweep", "setting"], sort=False).agg(
        n=("diff", "count"), mean_diff=("diff", "mean"),
        std_diff=("diff", "std"), wins=("win", "sum")).reset_index()
    out.insert(2, "comparison", f"{model_a} - {model_b} ({metric})")
    return out.fillna({"std_diff": 0.0})


def summary_markdown(summary: pd.DataFrame, comparison: Optional[pd.DataFrame] = None
                     ) -> str:
    """Readable per-sweep tables: cell = F1 mean +/- std (ROC-AUC mean)."""
    lines = ["# Venom-GNN experiment summary", "",
             "Test-node results, mean over seeds. Cell = F1 +/- std (ROC-AUC).",
             "Synthetic data: simulated experiments, not real-world performance.", ""]
    models = [m for m in MODEL_NAMES if m in set(summary["model"])]
    for sweep in summary["sweep"].unique():
        part = summary[summary["sweep"] == sweep]
        lines += [f"## {sweep}", "", "| setting | " + " | ".join(models) + " |",
                  "|---|" + "---|" * len(models)]
        for setting in part["setting"].unique():
            cells = []
            for m in models:
                r = part[(part["setting"] == setting) & (part["model"] == m)]
                cells.append("-" if r.empty else
                             f"{r.f1_mean.iloc[0]:.2f} +/- {r.f1_std.iloc[0]:.2f} "
                             f"({r.auc_mean.iloc[0]:.2f})")
            lines.append(f"| {setting} | " + " | ".join(cells) + " |")
        n = int(part["n"].iloc[0])
        lines += ["", f"_n = {n} seeds per cell._", ""]
    if comparison is not None and not comparison.empty:
        lines += ["## Paired comparison", "",
                  "| sweep | setting | comparison | mean diff | std | wins / n |",
                  "|---|---|---|---|---|---|"]
        for _, r in comparison.iterrows():
            lines.append(f"| {r['sweep']} | {r['setting']} | {r['comparison']} | "
                         f"{r['mean_diff']:+.3f} | {r['std_diff']:.3f} | "
                         f"{int(r['wins'])}/{int(r['n'])} |")
        lines.append("")
    return "\n".join(lines)


def write_outputs(rows: list, out_dir) -> dict:
    """Write results.csv, summary.csv, comparison.csv and summary.md."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    summary = summarize(rows)
    present = set(df["model"])
    parts = [compare_models(rows, a, b) for a, b in
             (("gcn_behaviour", "rf_behaviour"), ("gcn_graph", "rf_graph"))
             if a in present and b in present]
    comparison = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    paths = {"results": out / "results.csv", "summary": out / "summary.csv",
             "comparison": out / "comparison.csv", "markdown": out / "summary.md"}
    df.to_csv(paths["results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    paths["markdown"].write_text(summary_markdown(summary, comparison), encoding="utf-8")
    return paths
