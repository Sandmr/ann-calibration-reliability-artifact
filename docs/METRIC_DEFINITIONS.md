# Stage 2 metric definitions

- **Target**: the exact 1-nearest neighbor in the currently selected gallery, recomputed with Faiss `IndexFlatL2`.
- **Candidate omission**: the exact target is absent from the ANN top-100 candidate list.
- **Threshold exclusion**: the exact target is present in ANN top-100 but its squared-L2 distance exceeds the calibration threshold.
- **Missed**: the target is absent from the final prediction set.
- **Total miss rate**: `1 - empirical_coverage`.
- **Candidate omission rate**: fraction of test queries for which the exact target is absent from ANN top-100.
- **Threshold exclusion rate**: fraction of test queries for which the exact target is present in ANN top-100 but excluded by the threshold.
- **Decomposition identity**: `total_miss_rate = candidate_omission_rate + threshold_exclusion_rate`; the two loss events are mutually exclusive by construction.
- **Coverage**: fraction of test queries whose exact target is in the final set.
- **Coverage shortfall**: `max(0, target_coverage - empirical coverage)`.
- **Set efficiency**: mean prediction-set size and size divided by current gallery size.
- **Recall@1 / Recall@100**: exact target recovered at rank 1 / anywhere in the retrieved top-100.
- **Latency**: one-query wall-clock search latency, reported as p50 and p95.

For SIFT1M, the semantic target is explicitly **exact nearest-neighbor identity in the selected Euclidean gallery**, not a semantic relevance judgment. The calibration queries are disjoint from the test queries.
