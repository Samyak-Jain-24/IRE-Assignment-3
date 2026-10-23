"""Package <rollnumber>_A3.zip with exactly the required deliverables.

Contents (at the zip root): report.pdf, code/ (including .git), results/results.csv,
results/figures/, ai_log/. Datasets, caches, index files and Python bytecode are left out.

Usage: python scripts/make_submission.py <rollnumber>
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annbench import config

SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".hdf5", ".faissindex", ".index", ".faiss"}


def add_tree(zf, root: Path, base: Path):
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(base)
        if any(part in SKIP_DIRS for part in rel.parts) or p.suffix in SKIP_SUFFIXES or p.is_dir():
            continue
        zf.write(p, rel.as_posix())


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/make_submission.py <rollnumber>")
    root = config.ROOT_DIR
    required = [root / "report.pdf", root / "results" / "results.csv", root / "results" / "figures",
                root / "ai_log" / "ai_statement.md"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    out = root / f"{sys.argv[1]}_A3.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(root / "report.pdf", "report.pdf")
        add_tree(zf, config.CODE_DIR, root)
        zf.write(root / "results" / "results.csv", "results/results.csv")
        add_tree(zf, root / "results" / "figures", root)
        add_tree(zf, root / "ai_log", root)
    with zipfile.ZipFile(out) as zf:
        tops = sorted({n.split("/")[0] for n in zf.namelist()})
        print(f"wrote {out} ({len(zf.namelist())} files, {out.stat().st_size / 2**20:.1f} MiB); top level: {tops}")


if __name__ == "__main__":
    main()
