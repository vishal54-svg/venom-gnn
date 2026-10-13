"""Phase 5: a small two-layer GCN for poisoned-peer detection (CPU, full-batch).

The GCN sees the same normalised features and the same train/val/test split as
the Random Forest. Its extra ingredient is message passing: each peer's
prediction also depends on its neighbours (two hops).

Training uses the (possibly noisy) training labels of TRAIN nodes only.
Early stopping and the decision threshold use the validation nodes' true labels
(for both this model and the Random Forest). Test nodes are never used.
"""
import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from app.data.splits import Split, make_splits
from app.data.synthetic import NetworkDataset
from app.evaluation.report import evaluate_probabilities
from app.graph.prepare import PreparedData, prepare_data


class GCN(nn.Module):
    """Two GCN layers -> 2 logits per node (benign / poisoned)."""

    def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, 2)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


def train_gcn(prep: PreparedData, hidden: int = 32, dropout: float = 0.5,
              lr: float = 0.01, weight_decay: float = 5e-4, epochs: int = 200,
              patience: int = 30, seed: int = 42):
    """Train with class-weighted cross-entropy. Returns (best_model, info)."""
    if epochs < 1 or hidden < 1 or patience < 1:
        raise ValueError("epochs, hidden and patience must be at least 1.")
    torch.manual_seed(seed)

    x = torch.from_numpy(prep.x)
    edge_index = torch.from_numpy(prep.edge_index)
    y_train = torch.from_numpy(prep.y_train)
    y_true = torch.from_numpy(prep.y_true)
    tr = torch.from_numpy(prep.split.train_idx)
    va = torch.from_numpy(prep.split.val_idx)

    counts = torch.bincount(y_train[tr], minlength=2).float()
    weights = counts.sum() / (2.0 * counts.clamp(min=1.0))  # rare class counts more

    model = GCN(x.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, edge_index)[tr], y_train[tr], weight=weights)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(
                model(x, edge_index)[va], y_true[va], weight=weights
            ).item()
        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(val_loss)

        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:  # None only if val loss was never finite
        model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "epochs_run": len(history["train_loss"]),
            "history": history}
    return model, info


def predict_proba(model: GCN, prep: PreparedData) -> np.ndarray:
    """Probability that each node is poisoned, shape (N,)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(prep.x), torch.from_numpy(prep.edge_index))
        return F.softmax(logits, dim=1)[:, 1].numpy()


def run_gcn(ds: NetworkDataset, split: Optional[Split] = None,
            label_noise: float = 0.0, include_structural: bool = True,
            seed: int = 42, **train_kwargs) -> dict:
    """Prepare data, train a GCN and evaluate it. Mirrors random_forest.run_baseline."""
    split = split or make_splits(ds.labels, seed=seed)
    prep = prepare_data(ds, split, label_noise, include_structural, seed)
    model, info = train_gcn(prep, seed=seed, **train_kwargs)
    result = evaluate_probabilities(prep, predict_proba(model, prep))
    result.update(
        feature_names=prep.feature_names,
        num_flipped_train_labels=int(len(prep.flipped_nodes)),
        best_epoch=info["best_epoch"],
        epochs_run=info["epochs_run"],
    )
    return result
