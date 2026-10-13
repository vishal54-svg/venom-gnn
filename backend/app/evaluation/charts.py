"""Matplotlib charts of recorded results (PNG files). Only recorded results are plotted."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

MODEL_COLORS = {"random_forest": "#1f77b4", "gcn": "#d62728", "rf_neighborhood": "#2ca02c"}
MODEL_LABELS = {"random_forest": "Random Forest", "gcn": "GCN", "rf_neighborhood": "RF + neighbourhood"}


def plot_metric_vs_ratio(summary, metric: str, out_path) -> Path:
    """One panel per attack type; x = poisoning ratio (0 % = clean), y = metric mean +/- std."""
    attacks = [a for a in summary["attack_type"].cat.categories
               if a != "none" and (summary["attack_type"] == a).any()]
    fig, axes = plt.subplots(1, max(1, len(attacks)), figsize=(5.2 * max(1, len(attacks)), 4.2),
                             sharey=True, squeeze=False)
    clean = summary[summary["attack_type"] == "none"]
    for ax, attack in zip(axes[0], attacks):
        part = summary[summary["attack_type"] == attack]
        for model in dict.fromkeys(summary["model"]):
            p = part[part["model"] == model].sort_values("requested_ratio")
            c = clean[clean["model"] == model]
            xs = ([0.0] if not c.empty else []) + [r * 100 for r in p["requested_ratio"]]
            ys = ([c[f"{metric}_mean"].iloc[0]] if not c.empty else []) + list(p[f"{metric}_mean"])
            errs = ([c[f"{metric}_std"].iloc[0]] if not c.empty and f"{metric}_std" in c else [0.0] * len(c)) \
                + (list(p[f"{metric}_std"]) if f"{metric}_std" in p else [0.0] * len(p))
            ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, linewidth=1.8,
                        color=MODEL_COLORS.get(model), label=MODEL_LABELS.get(model, model))
        ax.set_title(f"{attack} poisoning")
        ax.set_xlabel("poisoning ratio (%)")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel(metric.replace("_", " "))
    axes[0][0].legend()
    fig.suptitle(f"{metric.replace('_', ' ')} vs poisoning ratio (mean +/- std over seeds)")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def make_all_charts(summary, out_dir) -> list:
    """f1, accuracy, precision and recall versus ratio. Returns the written file paths."""
    if summary is None or summary.empty:
        return []
    return [plot_metric_vs_ratio(summary, m, Path(out_dir) / f"{m}_vs_ratio.png")
            for m in ("f1", "accuracy", "precision", "recall")]
