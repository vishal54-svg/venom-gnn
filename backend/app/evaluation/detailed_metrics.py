"""Full metric set for one evaluation: scores, confusion matrix, per-class report."""
import numpy as np
from sklearn.metrics import classification_report

from app.evaluation.metrics import compute_metrics

CLASS_NAMES = ("benign", "malicious")


def detailed_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """compute_metrics() plus a confusion matrix and a per-class report.

    confusion_matrix = [[TN, FP], [FN, TP]]   (rows = true class, columns = predicted)
    Undefined scores (e.g. precision with no predicted positives) are reported as 0;
    ROC-AUC / PR-AUC are NaN when only one class is present.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    out = compute_metrics(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)
    rep = classification_report(y_true, y_pred, labels=[0, 1], target_names=list(CLASS_NAMES),
                                output_dict=True, zero_division=0)
    out["confusion_matrix"] = [[out["tn"], out["fp"]], [out["fn"], out["tp"]]]
    out["per_class"] = {
        name: {"precision": float(rep[name]["precision"]), "recall": float(rep[name]["recall"]),
               "f1": float(rep[name]["f1-score"]), "support": int(rep[name]["support"])}
        for name in CLASS_NAMES
    }
    return out
