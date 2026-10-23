# Hypotheses

Each hypothesis below was written and committed before the code for that experiment
was run (see `git log -- hypothesis.md`).

## Part 3: Query difficulty on GloVe-100

**Set-up.** For every query, compute its relative contrast at k = 10,
`RC = mean distance from the query to all base vectors / distance to its 10th nearest neighbour`
(Euclidean distance on the normalised vectors). A low RC means the 10 true neighbours are
barely closer than a typical point, so the query is hard. The quartiles of RC on the tuning
half (queries 0-4999) set the edges of four bins (Q1 = hardest ... Q4 = easiest), which are
then applied to the test half (queries 5000-9999). Each index (HNSW, IVF-Flat, IVF-PQ) is run
at the operating point chosen in Part 2 to reach overall recall 0.95 on the tuning half (or
at its highest-recall setting if it cannot reach 0.95). Recall@10 is then compared across bins.

**Hypothesis.**

1. For all three indexes, recall@10 falls steadily from the easiest bin to the hardest. At the
   setting that gives about 0.95 on average, I expect the hardest quartile to reach clearly
   less than the average (around 0.85 or lower) and the easiest quartile to reach 0.99 or
   more. So the headline "0.95" hides a large group of queries that are served much worse.
2. Why: with low RC there are many points at almost the same distance as the 10th neighbour.
   * In IVF the true neighbours of a hard query are spread over many Voronoi cells, and a
     fixed `nprobe` covers only part of them.
   * In HNSW the greedy search sees many candidates with nearly the same distance and has no
     strong distance gradient to follow, so a fixed `efSearch` queue gets filled with
     near-ties and true neighbours are dropped.
   * In IVF-PQ the quantisation error is about as large as the small gap between the 10th
     neighbour and its competitors, so neighbours are mis-ranked even inside probed lists.
     I expect IVF-PQ to show the largest gap between bins, then IVF-Flat, then HNSW.
3. Hard queries need much more search effort: to bring the hardest bin up to 0.95 I expect
   `efSearch` / `nprobe` to be at least 4x larger than what the easiest bin needs.
4. For HNSW, per-query latency should also be higher in the hard bins (the search visits more
   nodes before its queue settles). For IVF-Flat it should be roughly the same in every bin,
   because a fixed `nprobe` scans about the same number of vectors whatever the query.

## Part 4: Distribution drift after the index is trained (synthetic data)

**Set-up.** A seeded generator (`annbench/synth.py`) builds a 64-dimensional mixture of 50
low-rank Gaussian clusters. Each cluster spans a random 8-dimensional subspace and has a little
isotropic noise added. The index is built and trained on 200K "old" vectors. Then 200K "new"
vectors are added, taken from the same mixture but moved by `delta * R * u`. Here `u` is a fixed
random unit vector and `R` is the RMS radius of the old data. Queries (1000 for tuning and
1000 for testing) come from the moved distribution, and the ground truth is computed over
all 400K vectors. The only thing that changes is the size of the shift,
`delta in {0 (control), 0.25, 0.5, 1, 2}`. The same random draws are reused for every delta
(common random numbers). Indexes: HNSW (M=16), IVF-Flat (nlist=1024) and IVF-PQ (nlist=1024,
16 bytes/vector). The search parameter of each index is fixed at the smallest value that gives
recall 0.95 on the control's tuning queries (for IVF-PQ, the smallest value that reaches 99%
of its maximum recall if it cannot reach 0.95), and this value is kept for every delta.
"Stale" indexes keep the quantisers trained on the old data. "Rebuilt" indexes are trained
again on all 400K vectors.

**Hypothesis.**

1. **IVF-Flat (stale): speed breaks, recall does not.** The coarse centroids only cover
   the old region, so as delta grows the new vectors fall into the few cells on the edge
   facing `u` and those lists become huge. The queries are in the new region too, so their
   nearest cells are exactly those huge lists. Recall at a fixed `nprobe` should therefore stay
   near the control (or even rise), but each query scans far more vectors. I expect QPS at
   delta = 2 to be more than 5x lower than the control, and the largest inverted list to grow
   more than 20x.
2. **IVF-PQ (stale): recall breaks.** The PQ codebooks are trained on residuals of old vectors
   from their centroids. New vectors are far from every centroid, so their residuals are much
   larger than anything the codebooks have seen, and the encoding error becomes larger than
   the distances between neighbours. I expect recall at the fixed setting to fall sharply with
   delta (below 0.5 at delta = 2), with reconstruction error growing about with delta squared.
   Speed should fall as well, for the same list-imbalance reason as IVF-Flat.
3. **HNSW: robust.** HNSW has no trained part. New vectors are linked into the graph on
   insertion, and the search simply follows edges into the new region. I expect recall and QPS
   at a fixed `efSearch` to stay within a few percent of the control for every delta.
4. **Recovery.** Training IVF-Flat and IVF-PQ again on all the data should bring both recall
   and QPS back to near-control levels, because the centroids and codebooks then cover the new
   region. For HNSW, building again with a different insertion order should change nothing.
