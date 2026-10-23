# Benchmarking approximate nearest neighbour search (CS4.406 Assignment 3)

This repository benchmarks HNSW, IVF-Flat and IVF-PQ (all from FAISS) against exact search on
SIFT-1M and GloVe-100. It also contains a query-difficulty experiment on GloVe (Part 3) and a
data-drift experiment on seeded synthetic data (Part 4). The recall function, timing code,
analysis and plots are all written here. The ann-benchmarks framework is **not** used, only its
data files.

## Quick start

```bash
pip install -r requirements.txt   # Python 3.10+ (results were produced with 3.13.5)
bash run_all.sh --quick           # small subset, under 15 minutes on a laptop
bash run_all.sh                   # full run, regenerates every number in report.pdf (~2-3 h)
```

On Windows, run the commands from Git Bash. `PYTHON=/path/to/python bash run_all.sh` selects a
specific interpreter.

* **Data.** The scripts expect `sift-128-euclidean.hdf5` and `glove-100-angular.hdf5` in `../data`
  (next to this `code/` folder). If they are missing, `run_all.sh` downloads them from
  `http://ann-benchmarks.com/` (about 1 GB). Set `ANN_DATA_DIR` to use another folder.
* **Full run** (`bash run_all.sh`): every base vector (1,000,000 SIFT / 1,183,514 GloVe) and all
  10,000 queries. Outputs go to `../results/` (`results.csv`, `figures/`, `tables/`) and
  `../report.pdf`. Caches (ground truth, per-query results, logs) go to `../work/`.
* **Quick run** (`bash run_all.sh --quick`): the first 100,000 base vectors of each dataset.
  Queries 0-999 are used for tuning and 5000-5999 for testing, with smaller `nlist` values and
  a 50K + 50K synthetic set for Part 4. It runs the same pipeline end to end and writes to
  `../results_quick/`, `../work_quick/` and `../results_quick/report_quick.pdf` (the report text is
  written for the full run, so in quick mode only the tables, figures and inline numbers change).

`run_all.sh` runs the unit tests, then each part, then collects `results.csv`, then builds the
report. The log is saved to `../work[_quick]/run_all.log`. Individual steps can be run on
their own:

| Step | Command | Produces |
|---|---|---|
| tests | `python -m pytest -q` | recall@10 unit tests, generator determinism tests |
| Part 1 | `python scripts/part1_exact.py --dataset sift` (and `glove`) | ground truth, agreement with the stored neighbours, exact timings |
| Part 2 | `python scripts/part2_benchmark.py --dataset sift` (and `glove`) | all index builds and sweeps |
| Parts 1-2 analysis | `python scripts/analyze_benchmarks.py` | recall-QPS plots, QPS at recall 0.90/0.95, build cost tables |
| Part 3 | `python scripts/part3_difficulty.py` | relative-contrast bins on GloVe |
| Part 4 | `python scripts/part4_drift.py` | drift experiment on synthetic data |
| collect | `python scripts/collect_results.py` | `results/results.csv`, environment table |
| report | `python scripts/build_report.py` | fills `report/report.html`, prints `report.pdf` with headless Edge/Chrome |

Add `--quick` to any step to work on the quick subset.

`python scripts/make_submission.py <rollnumber>` packages `../<rollnumber>_A3.zip` with exactly
`report.pdf`, `code/` (including `.git`), `results/results.csv`, `results/figures/` and `ai_log/`.

## Method in brief

* **Distances.** GloVe vectors (and queries) are L2-normalised, and every index uses Euclidean
  distance. On unit vectors this ranks neighbours exactly like the angular distance.
* **Query split.** Queries 0-4999 are used for every choice: the sweep stopping point, which
  configurations form the trade-off curve, and the setting used for recall 0.90 / 0.95.
  Queries 5000-9999 are used only for the reported numbers.
* **Ground truth.** FAISS `IndexFlatL2` returns the top 120 neighbours. Their distances are then
  recomputed in float64 and re-sorted, so exact ties stay exact (SIFT is integer-valued). The
  top 100 are kept. Part 1 checks them against the neighbours stored in each file.
* **recall@10** (`annbench/metrics.py`). A returned id is correct if its true distance is no
  larger than the distance of the true 10th neighbour (with a 1e-6 relative tolerance). A tie
  at position 10 is therefore not a mistake. Duplicates count once and `-1` ids are misses.
  See `tests/test_recall.py`.
* **Timing** (`annbench/timing.py`). Queries are searched one per `search()` call on a single
  thread. Both FAISS's OpenMP threads and its bundled OpenBLAS pthreads are limited with
  `threadpoolctl`, because OpenBLAS ignores `omp_set_num_threads`. Each measurement is a full
  pass over the 5000 test queries after a 200-query warm-up, repeated 3 times, and the median
  QPS is reported. The garbage collector is off while timing. Builds use all 8 hardware threads.
* **Keeping timings honest on a laptop** (`annbench/memory.py`). Each timing script pins its
  working set (`SetProcessWorkingSetSizeEx` with a hard minimum on Windows), so the OS cannot
  page the index out while it is being timed. Every timed pass records `page_faults_per_query`
  in `results.csv`, which should be about 0. Each script also times a fixed CPU workload first
  (`calibration_cpu` rows, with the power state), so CPU-speed drift between parts is visible.
  With `ANN_REQUIRE_AC=1` set (as for the submitted run: `ANN_REQUIRE_AC=1 bash run_all.sh`),
  every timed pass waits while the laptop is on battery and is repeated if power was lost during
  it. On battery this machine's CPU runs at about half speed.
* **Exact single-query timing** uses the first 1000 test queries. Brute force costs the same for
  every query, and timing all 5000 three times would take about 15 minutes per dataset.
* **Index size** is the size of the index written with `faiss.write_index`. IVF-PQ's
  precomputed distance tables live in RAM but are not serialised, so they are reported
  separately (`extra_ram_bytes`).
* **IVF** coarse quantizers are trained with k-means (20 iterations, at most 256 points per
  centroid, seed 1234). IVF-Flat and IVF-PQ with the same `nlist` share one quantizer, and its
  training time is charged to both.

## Layout

```
annbench/        library code: data loading, recall, timing, index builders, Part 4 generator
scripts/         one script per part, plus analysis, results collection and report building
tests/           pytest unit tests
report/          report.html template (placeholders are filled from the generated results)
hypothesis.md    Part 3 and Part 4 hypotheses (committed before those experiments were run)
run_all.sh       one-command reproduction
```

`results/results.csv` has one row per benchmark run, using the schema in `annbench/results.py`.
`experiment` says which part a row belongs to (`part1_exact`, `part1_verify`,
`part2_benchmark`, `part3_difficulty`, `part4_drift`). Part 2 has one row per (index, build
setting, search value, split). Part 3 has one row per (index, RC bin). Part 4 has one row per
(delta, variant, index, search value, split).

## Part 4 data generator

`annbench/synth.py` generates a 64-dimensional mixture of 50 Gaussian clusters, each lying in
its own random 8-dimensional subspace, plus 0.05 isotropic noise. `make_drift_data(delta, ...)`
draws 200K old vectors, 200K new vectors and 2000 queries with a fixed seed, then shifts the new
vectors and the queries by `delta * R * u`. The same seed always gives the same data, and every
`delta` reuses the same draws; `tests/test_synth.py` checks both.
