#!/usr/bin/env bash
# Regenerate every number, figure and table of the report.
#   bash run_all.sh          full run (SIFT-1M, GloVe-1.18M; about 2-3 hours on a 4-core laptop)
#   bash run_all.sh --quick  first 100K base vectors, 1000+1000 queries (under 15 minutes)
set -euo pipefail
cd "$(dirname "$0")"

QUICK=""
SUFFIX=""
if [[ "${1:-}" == "--quick" ]]; then
    QUICK="--quick"
    SUFFIX="_quick"
elif [[ -n "${1:-}" ]]; then
    echo "usage: bash run_all.sh [--quick]" >&2
    exit 2
fi

PY="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
WORK="${ANN_WORK_DIR:-../work${SUFFIX}}"
mkdir -p "$WORK"
LOG="$WORK/run_all.log"
exec > >(tee "$LOG") 2>&1

step() { echo; echo "=== [$(date '+%H:%M:%S')] $*"; }
START=$(date +%s)

step "unit tests"
"$PY" -m pytest -q

step "data (downloads the ann-benchmarks files if they are missing)"
"$PY" scripts/download_data.py

for ds in sift glove; do
    step "Part 1: exact baseline on $ds"
    "$PY" scripts/part1_exact.py --dataset "$ds" $QUICK
    step "Part 2: index benchmark on $ds"
    "$PY" scripts/part2_benchmark.py --dataset "$ds" $QUICK
done

step "Parts 1-2: figures, tables, operating points"
"$PY" scripts/analyze_benchmarks.py $QUICK

step "Part 3: query difficulty on GloVe"
"$PY" scripts/part3_difficulty.py $QUICK

step "Part 4: drift experiment on synthetic data"
"$PY" scripts/part4_drift.py $QUICK

step "results.csv and environment"
"$PY" scripts/collect_results.py $QUICK

step "report"
"$PY" scripts/build_report.py $QUICK

step "done in $(( ($(date +%s) - START) / 60 )) min"
