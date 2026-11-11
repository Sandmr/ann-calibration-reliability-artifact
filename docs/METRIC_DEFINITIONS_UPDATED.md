# Updated Stage 3 metric definitions

All rates below are computed on the same test-query set.

- `empirical_coverage`: fraction of test queries whose exact target is in the returned prediction set.
- `target_coverage_shortfall`: `max(0, target_coverage - empirical_coverage)`.
- `total_miss_rate`: `1 - empirical_coverage`.
- `candidate_omission_rate`: fraction of queries where the exact target is absent from ANN top-K.
- `threshold_exclusion_rate`: fraction of queries where the exact target is present in ANN top-K but its score exceeds the threshold.
- `total_miss_rate = candidate_omission_rate + threshold_exclusion_rate` must hold exactly up to floating-point representation.
- `mean_set_size`: mean number of retrieved candidates whose distance is at or below the threshold.
- `normalized_set_size`: mean set size divided by the current gallery size.
- `TargetRecall@1`: fraction of queries whose exact Top-1 target is the first returned candidate.
- `TargetRecall@100`: fraction of queries whose exact Top-1 target is present anywhere in the returned top-100.
- `NeighborRecall@1`: overlap of exact Top-1 and ANN Top-1 candidate IDs, divided by 1.
- `NeighborRecall@100`: overlap count between exact Top-100 and ANN Top-100 candidate IDs, divided by 100.

The historical fields `recall_at_1` and `recall_at_100` in Stage 2/3 files are TargetRecall fields, not NeighborRecall fields.

The target is the exact nearest-neighbor identity in the current gallery, recomputed separately for every gallery size. The semantic target is not inferred from SIFT descriptors.

For exact-calibrated evaluation, the threshold is the finite-sample higher-order quantile of exact target distances on calibration queries. For ANN-calibrated evaluation, omitted calibration targets receive an infinite nonconformity score; this is recorded explicitly and is never replaced by a test label.
