"""Aggregate stored experiment records: mean/std over seeds, degradation, paired comparison."""
from typing import Optional

import numpy as np
import pandas as pd

METRIC_COLS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
ATTACK_ORDER = ["none", "feature", "label", "combined"]
KEY = ["attack_type", "feature_strategy", "requested_ratio"]


def records_to_frame(records: list) -> pd.DataFrame:
    """One row per COMPLETED experiment, test metrics flattened."""
    rows = []
    for r in records:
        if r.get("status") != "completed" or not r.get("metrics"):
            continue
        t = r["metrics"]["test"]
        rows.append({
            "experiment_id": r["experiment_id"], "matrix_label": r.get("matrix_label"),
            "model": r["model"], "attack_type": r["attack_type"],
            "feature_strategy": r.get("feature_strategy") or "-",
            "requested_ratio": r["requested_ratio"], "actual_ratio": r.get("actual_ratio"),
            "seed": r["seed"], **{m: t.get(m) for m in METRIC_COLS},
            "train_seconds": r.get("train_seconds"), "inference_seconds": r.get("inference_seconds"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["attack_type"] = pd.Categorical(df["attack_type"], ATTACK_ORDER, ordered=True)
    return df


def summarize(records: list) -> pd.DataFrame:
    """Mean and std over seeds for every (attack, strategy, ratio, model)."""
    df = records_to_frame(records)
    if df.empty:
        return df
    g = df.groupby(KEY + ["model"], observed=True)
    agg = g.agg(n=("seed", "count"), **{f"{m}_mean": (m, "mean") for m in METRIC_COLS},
                **{f"{m}_std": (m, "std") for m in ("f1", "roc_auc", "recall", "precision", "accuracy")},
                train_seconds_mean=("train_seconds", "mean"),
                inference_seconds_mean=("inference_seconds", "mean")).reset_index()
    std_cols = [c for c in agg.columns if c.endswith("_std")]
    agg[std_cols] = agg[std_cols].fillna(0.0)
    return agg.sort_values(["attack_type", "feature_strategy", "requested_ratio", "model"]
                           ).reset_index(drop=True)


def degradation(summary: pd.DataFrame) -> pd.DataFrame:
    """Drop in each metric relative to the same model's clean (attack 'none') result.

    drop > 0 means the poisoned condition is worse than clean.
    """
    if summary.empty:
        return summary
    clean = summary[summary["attack_type"] == "none"].set_index("model")
    out = summary[summary["attack_type"] != "none"].copy()
    for m in ("f1", "accuracy", "precision", "recall", "roc_auc"):
        base = out["model"].map(clean[f"{m}_mean"])
        out[f"{m}_drop"] = base - out[f"{m}_mean"]
    base_f1 = out["model"].map(clean["f1_mean"])
    out["f1_relative_drop"] = np.where(base_f1 > 0, out["f1_drop"] / base_f1, np.nan)
    keep = KEY + ["model", "n"] + [c for c in out.columns if c.endswith("_drop") or c.endswith("relative_drop")]
    return out[keep].reset_index(drop=True)


def paired_comparison(records: list, model_a: str = "gcn", model_b: str = "random_forest",
                      metric: str = "f1") -> pd.DataFrame:
    """Per-condition mean difference (a - b), matched by seed; wins = seeds where a > b."""
    df = records_to_frame(records)
    if df.empty:
        return df
    wide = df.pivot_table(index=KEY + ["seed"], columns="model", values=metric, observed=True)
    if model_a not in wide or model_b not in wide:
        return pd.DataFrame()
    wide = wide.dropna(subset=[model_a, model_b])
    wide["diff"] = wide[model_a] - wide[model_b]
    wide["win"] = (wide["diff"] > 0).astype(int)
    out = wide.groupby(KEY, observed=True).agg(n=("diff", "count"), mean_diff=("diff", "mean"),
                                               std_diff=("diff", "std"), wins=("win", "sum")).reset_index()
    out["comparison"] = f"{model_a} - {model_b} ({metric})"
    out["std_diff"] = out["std_diff"].fillna(0.0)
    return out.sort_values(["attack_type", "feature_strategy", "requested_ratio"]).reset_index(drop=True)


def markdown_report(summary: pd.DataFrame, comparison: Optional[pd.DataFrame] = None,
                    degr: Optional[pd.DataFrame] = None) -> str:
    models = list(dict.fromkeys(summary["model"])) if not summary.empty else []
    lines = ["# Venom-GNN experiment report", "",
             "All numbers come from experiments that were actually run. Test-node results on a",
             "clean held-out test set, mean +/- std over seeds. Synthetic data: simulated",
             "experiments, not evidence of real-world security performance.", "",
             "## F1-score (ROC-AUC)", "", "| attack | strategy | ratio | " + " | ".join(models) + " |",
             "|---|---|---|" + "---|" * len(models)]
    for key, part in summary.groupby(KEY, observed=True, sort=False):
        cells = []
        for m in models:
            r = part[part["model"] == m]
            cells.append("Not run" if r.empty else
                         f"{r.f1_mean.iloc[0]:.2f} +/- {r.f1_std.iloc[0]:.2f} ({r.roc_auc_mean.iloc[0]:.2f})")
        lines.append(f"| {key[0]} | {key[1]} | {key[2]:.0%} | " + " | ".join(cells) + " |")
    if degr is not None and not degr.empty:
        lines += ["", "## F1 drop relative to the clean condition (positive = worse)", "",
                  "| attack | strategy | ratio | model | F1 drop | ROC-AUC drop |", "|---|---|---|---|---|---|"]
        for _, r in degr.iterrows():
            lines.append(f"| {r.attack_type} | {r.feature_strategy} | {r.requested_ratio:.0%} | {r.model} | "
                         f"{r.f1_drop:+.3f} | {r.roc_auc_drop:+.3f} |")
    if comparison is not None and not comparison.empty:
        lines += ["", "## Paired comparison", "", "| attack | strategy | ratio | comparison | mean diff | std | wins / n |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in comparison.iterrows():
            lines.append(f"| {r.attack_type} | {r.feature_strategy} | {r.requested_ratio:.0%} | {r.comparison} | "
                         f"{r.mean_diff:+.3f} | {r.std_diff:.3f} | {int(r.wins)}/{int(r.n)} |")
    n = int(summary["n"].max()) if not summary.empty else 0
    lines += ["", f"_Up to {n} seeds per cell. With small test sets, differences smaller than the "
              "std are not reliable._", ""]
    return "\n".join(lines)
