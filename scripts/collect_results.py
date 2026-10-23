"""Merge every benchmark row into results/results.csv and record the machine / library versions.

Usage: python scripts/collect_results.py [--quick]
"""
import argparse
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import psutil

from annbench import config
from annbench.report_utils import write_table
from annbench.results import COLUMNS, save_numbers

PARTS = ["part1_sift.csv", "part1_glove.csv", "part2_sift.csv", "part2_glove.csv", "part3_glove.csv", "part4_drift.csv"]


def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def environment() -> dict:
    import faiss, h5py, matplotlib, numpy, threadpoolctl
    from threadpoolctl import threadpool_info
    blas = ", ".join(sorted({f"{i.get('internal_api')} {i.get('version') or ''}".strip()
                             for i in threadpool_info() if i.get("user_api") == "blas"}))
    return {
        "CPU": cpu_name(),
        "Cores / threads": f"{psutil.cpu_count(logical=False)} / {psutil.cpu_count(logical=True)}",
        "RAM": f"{psutil.virtual_memory().total / 2**30:.1f} GiB",
        "OS": platform.platform(),
        "Python": platform.python_version(),
        "faiss-cpu": faiss.__version__,
        "numpy": numpy.__version__,
        "h5py": h5py.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "BLAS libraries loaded": blas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    P = config.paths(args.quick)
    frames = [pd.read_csv(P["work"] / f) for f in PARTS if (P["work"] / f).exists()]
    missing = [f for f in PARTS if not (P["work"] / f).exists()]
    if missing:
        print("[collect] warning: missing", missing)
    df = pd.concat(frames, ignore_index=True)[COLUMNS]
    df.to_csv(P["results"] / "results.csv", index=False)
    print(f"[collect] {len(df)} rows -> {P['results'] / 'results.csv'}")

    env = environment()
    write_table(pd.DataFrame({"Item": list(env), "Value": list(env.values())}), P["tables"] / "environment")
    save_numbers(P["tables"] / "report_numbers.json", {f"env_{k}": v for k, v in env.items()} |
                 {"n_result_rows": len(df)})


if __name__ == "__main__":
    main()
