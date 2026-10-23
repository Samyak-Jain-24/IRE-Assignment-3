"""Shared helpers for analysis scripts: plot style, tables, Pareto frontier, operating points."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed categorical order (the identity of an index never changes colour between figures);
# markers give a second, colour-independent encoding for print / colour-blind readers.
STYLE = {
    "HNSW": {"color": "#2a78d6", "marker": "o"},
    "IVF-Flat": {"color": "#eb6834", "marker": "s"},
    "IVF-PQ": {"color": "#1baf7a", "marker": "^"},
    "Exact": {"color": "#52514e", "marker": "*"},
}
FAMILIES = ["HNSW", "IVF-Flat", "IVF-PQ"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDINAL_BLUE = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # light -> dark (ordinal ramp)


def setup_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 9.5,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "lines.markersize": 4.5, "legend.frameon": False, "figure.facecolor": "white",
    })


def pareto_mask(recall: np.ndarray, qps: np.ndarray) -> np.ndarray:
    """True for points not dominated by another point with recall >= and qps >= (one strictly)."""
    order = np.lexsort((-qps, -recall))          # recall descending, then qps descending
    keep = np.zeros(len(recall), dtype=bool)
    best_qps = -np.inf
    for i in order:
        if qps[i] > best_qps:
            keep[i] = True
            best_qps = qps[i]
    return keep


def select_operating_point(tune: pd.DataFrame, target: float):
    """Fastest configuration (by tuning QPS) whose tuning recall reaches the target, else None."""
    ok = tune[tune["recall_at_10"] >= target]
    if ok.empty:
        return None
    return ok.loc[ok["qps_median"].idxmax()]


def paired(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (index, build, search value) with tuning and test columns side by side."""
    keys = ["dataset", "index", "build_params", "search_value"]
    t = df[df["split"] == "tune"].set_index(keys)
    s = df[df["split"] == "test"].set_index(keys)
    out = s.join(t[["recall_at_10", "qps_median"]], rsuffix="_tune")
    return out.reset_index()


def write_table(df: pd.DataFrame, path_stem: Path) -> None:
    """Write a table as CSV, Markdown and an HTML fragment (used by the report builder)."""
    df.to_csv(f"{path_stem}.csv", index=False)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    Path(f"{path_stem}.md").write_text("\n".join(lines) + "\n")
    Path(f"{path_stem}.html").write_text(df.to_html(index=False, escape=False, border=0, classes="tbl"))


def fmt_int(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}"
