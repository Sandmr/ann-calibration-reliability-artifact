# Stage 4 Recall-definition audit

## Finding

The existing Stage 2/3 fields named `recall_at_1` and `recall_at_100` are **TargetRecall@K**, not the standard candidate-set overlap recall.

The implementation checks whether the exact current-gallery Top-1 target ID appears in the ANN result:

- `recall_at_1 = (target position == 0)`;
- `recall_at_100 = (target ID appears anywhere in ANN top-100)`.

Therefore:

`TargetRecall@100 = 1 - candidate_omission_rate`

when the target is defined as the exact current-gallery Top-1 ID and the candidate list is ANN Top-100.

## Corrected names

Stage 4 outputs use explicit fields:

- `target_recall_at_1`
- `target_recall_at_100`
- `neighbor_recall_at_1`
- `neighbor_recall_at_100`

`NeighborRecall@K` is the fraction of the exact current-gallery Top-K candidate IDs appearing in the ANN Top-K, divided by K. It is a candidate-set overlap metric and is not interchangeable with TargetRecall.

Historical Stage 2/3 CSV and JSON files are preserved unchanged. Their legacy `recall_at_*` fields are reinterpreted as TargetRecall in the accompanying documentation; they are not relabeled in place.

## Audit result

The target-recall identity was rechecked on Stage 3 query-level records. It holds for every ANN row. NeighborRecall is newly computed from the exact Top-K and ANN Top-K ID sets in Stage 4.
