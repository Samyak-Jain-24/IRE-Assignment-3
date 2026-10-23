"""Download SIFT-128 and GloVe-100 from ann-benchmarks into the data directory if missing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annbench import config
from annbench.data import DATASETS, ensure_downloaded

if __name__ == "__main__":
    data_dir = config.data_dir()
    for name in DATASETS:
        print(f"{name}: {ensure_downloaded(name, data_dir)}")
