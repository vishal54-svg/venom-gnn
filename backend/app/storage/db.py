"""SQLite storage for experiment records (standard library only).

Only records produced by real experiment runs are written. NaN values become NULL/None
so exported JSON is always valid. The database file is generated data: keep it out of Git.
"""
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Optional

DEFAULT_DB = Path(__file__).resolve().parents[3] / "results" / "venom_gnn.db"

JSON_COLUMNS = ("dataset_config", "poison_audit", "model_config", "metrics", "history", "predictions")
SCALAR_COLUMNS = ("experiment_id", "batch_id", "matrix_label", "created_at", "status", "error",
                  "model", "attack_type", "feature_strategy", "requested_ratio", "actual_ratio",
                  "actual_ratio_feature", "actual_ratio_label", "seed", "split_fingerprint",
                  "train_seconds", "inference_seconds")

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    batch_id TEXT, matrix_label TEXT, created_at TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
    model TEXT NOT NULL, attack_type TEXT NOT NULL, feature_strategy TEXT,
    requested_ratio REAL, actual_ratio REAL, actual_ratio_feature REAL, actual_ratio_label REAL,
    seed INTEGER, split_fingerprint TEXT, train_seconds REAL, inference_seconds REAL,
    dataset_config TEXT, poison_audit TEXT, model_config TEXT, metrics TEXT, history TEXT,
    predictions TEXT
);
CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_exp_filter ON experiments(model, attack_type, requested_ratio);
"""


def _clean(obj):
    """Recursively turn NaN / inf into None and numpy scalars into Python numbers."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def connect(path=None) -> sqlite3.Connection:
    p = Path(path) if path else DEFAULT_DB
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_experiment(record: dict, path=None) -> str:
    row = {c: _clean(record.get(c)) for c in SCALAR_COLUMNS}
    for c in JSON_COLUMNS:
        value = record.get(c)
        row[c] = None if value is None else json.dumps(_clean(value), allow_nan=False)
    cols = list(row)
    sql = (f"INSERT OR REPLACE INTO experiments ({', '.join(cols)}) "
           f"VALUES ({', '.join('?' for _ in cols)})")
    with closing(connect(path)) as conn, conn:
        conn.execute(sql, [row[c] for c in cols])
    return row["experiment_id"]


def _row_to_dict(row: sqlite3.Row, include_predictions: bool) -> dict:
    d = dict(row)
    for c in JSON_COLUMNS:
        if d.get(c) is not None:
            d[c] = json.loads(d[c])
    if not include_predictions:
        d.pop("predictions", None)
    return d


def get_experiment(experiment_id: str, path=None, include_predictions: bool = True) -> Optional[dict]:
    with closing(connect(path)) as conn:
        row = conn.execute("SELECT * FROM experiments WHERE experiment_id = ?",
                           (experiment_id,)).fetchone()
    return _row_to_dict(row, include_predictions) if row else None


def list_experiments(path=None, model=None, attack_type=None, ratio=None, status=None,
                     batch_id=None, limit: int = 100, include_predictions: bool = False) -> list:
    where, args = [], []
    for col, val in (("model", model), ("attack_type", attack_type), ("requested_ratio", ratio),
                     ("status", status), ("batch_id", batch_id)):
        if val is not None:
            where.append(f"{col} = ?")
            args.append(val)
    sql = "SELECT * FROM experiments"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
    args.append(int(limit))
    with closing(connect(path)) as conn:
        rows = conn.execute(sql, args).fetchall()
    return [_row_to_dict(r, include_predictions) for r in rows]


def latest_experiment(path=None, status: str = "completed") -> Optional[dict]:
    found = list_experiments(path, status=status, limit=1, include_predictions=True)
    return found[0] if found else None


def export_json(out_path, db_path=None, **filters) -> int:
    """Write matching experiments (without per-node predictions) to a JSON file."""
    records = list_experiments(db_path, limit=1_000_000, **filters)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(records), indent=2, allow_nan=False), encoding="utf-8")
    return len(records)
