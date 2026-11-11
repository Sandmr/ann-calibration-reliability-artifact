# Stage 2 results summary

The smoke test produced the following observed pattern on SIFT1M (10k gallery, 100 test queries):

1. Exact Flat and HNSW had Recall@100 = 1.00 and identical empirical coverage of 0.85 under the shared threshold.
2. HNSW reduced p50 single-query latency from 0.166 ms to 0.042 ms in this run, at the cost of a larger serialized index and longer build time than Flat.
3. IVFFlat had Recall@100 = 0.94, with candidate omission rate 0.06 and coverage 0.81. The omission component is therefore observable separately from threshold loss (0.13).
4. The decomposition invariant held for every query and every index.

These observations are feasibility evidence only. They do not yet demonstrate scale generality, calibration transfer, cross-workload robustness, or an ICDE-level systems contribution.
