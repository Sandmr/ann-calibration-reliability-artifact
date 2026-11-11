# Frozen experiment results

## E1–E10 mapping

| ID | Experiment | Frozen evidence | Interpretation boundary |
|---|---|---|---|
| E1 | Exact Flat, HNSW, IVFFlat comparison | Stage 2 corrected smoke; Stage 4 cost/results CSV | Descriptive on declared CPU/runtime |
| E2 | TargetRecall / NeighborRecall audit | `RECALL_DEFINITION_AUDIT.md`, Stage 4 CSV | TargetRecall is not candidate overlap |
| E3 | Omission / threshold-exclusion decomposition | query-level Stage 4 CSV | Identity holds on all recorded rows |
| E4 | Exact-to-ANN transfer | Stage 4 calibration CSVs | Coverage gain accompanies larger sets |
| E5 | ANN-specific calibration / efficiency | Stage 4 calibration CSVs and score distributions | Not an algorithm improvement claim |
| E6 | Fixed mean-size budget 20 | `FIXED_BUDGET_MEAN_COMPARISON.csv` | Diagnostic; not a complete paired calibration-method comparison |
| E7 | HNSW efSearch / IVFFlat nprobe sweep | Stage 3 sweep CSV | Exploratory configuration response |
| E8 | 10k/50k/100k gallery scale | Stage 3 scale CSV | Exact target identity changes by scale |
| E9 | SIFT1M / GloVe25 workload comparison | Stage 4 per-workload CSVs | GloVe25 is not cross-modal |
| E10 | Latency/build/index/RSS cost | Stage 4 system-cost CSV | Runtime-specific descriptive measurements |

## Status

The strongest reproducible conclusion is that calibration source changes the operating threshold and prediction-set size. The fixed mean-size diagnostic does not establish a calibration-specific advantage after budget control. This negative boundary is part of the frozen evidence and must remain visible.
