"""One schema for every benchmark row, plus helpers to write rows and report numbers."""
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "experiment", "dataset", "mode", "n_base", "dim", "index", "build_params", "variant", "group",
    "search_param", "search_value", "split", "n_queries", "k",
    "recall_at_10", "recall_p10", "frac_recall_below_0.8",
    "qps_median", "qps_runs", "latency_ms_mean", "latency_ms_p50", "latency_ms_p99",
    "build_time_s", "train_time_s", "add_time_s", "index_bytes", "bytes_per_vector", "extra_ram_bytes",
    "dist_comps_per_query", "max_list_size", "list_imbalance", "pq_rel_error",
    "search_threads", "build_threads", "repeats", "notes", "timestamp",
]


def recall_summary(per_query_recall: np.ndarray) -> dict:
    r = np.asarray(per_query_recall)
    return {
        "recall_at_10": float(r.mean()),
        "recall_p10": float(np.percentile(r, 10)),
        "frac_recall_below_0.8": float((r < 0.8).mean()),
    }


def timing_summary(qps_runs, latency_s) -> dict:
    lat_ms = np.asarray(latency_s) * 1e3
    return {
        "qps_median": float(np.median(qps_runs)),
        "qps_runs": ";".join(f"{q:.2f}" for q in qps_runs),
        "latency_ms_mean": float(lat_ms.mean()),
        "latency_ms_p50": float(np.percentile(lat_ms, 50)),
        "latency_ms_p99": float(np.percentile(lat_ms, 99)),
    }


class RowWriter:
    """Collects rows for one script and writes them to a CSV in the work directory."""

    def __init__(self, path: Path, **defaults):
        self.path = Path(path)
        self.defaults = defaults
        self.rows = []

    def add(self, **fields) -> None:
        row = dict(self.defaults)
        row.update(fields)
        row.setdefault("timestamp", _dt.datetime.now().isoformat(timespec="seconds"))
        unknown = set(row) - set(COLUMNS)
        if unknown:
            raise KeyError(f"unknown result columns: {sorted(unknown)}")
        self.rows.append(row)
        self.flush()

    def flush(self) -> None:
        pd.DataFrame(self.rows, columns=COLUMNS).to_csv(self.path, index=False)


def save_numbers(path: Path, numbers: dict) -> None:
    """Merge key -> value pairs used in the report text into a JSON file."""
    path = Path(path)
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(numbers)
    path.write_text(json.dumps(current, indent=1, sort_keys=True))
