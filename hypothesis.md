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
