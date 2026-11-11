# Constraint satisfaction analysis

## Main outcome

| Workload | Strategy | Selected configuration | Calibration feasible | Test coverage | Mean test set size | p95 latency (ms) | Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| SIFT1M | proposed constraints | IVFFlat nprobe=8 | yes | 0.896 | 48.534 | 0.02321 | Meets calibration constraints; test shortfall 0.004 |
| SIFT1M | fixed baseline | HNSW ef16 | yes | 0.886 | 37.224 | 0.02351 | Smaller sets but lower test coverage |
| SIFT1M | NeighborRecall-only | Exact Flat | no | 0.868 | 37.152 | 0.21924 | Violates latency budget |
| SIFT1M | latency-only | IVFFlat nprobe=1 | no | 0.498 | 100.000 | 0.01230 | Violates coverage and set-size constraints |
| GloVe25 | proposed constraints | HNSW ef16 | yes | 0.926 | 93.946 | 0.01951 | Same configuration as fixed baseline |
| GloVe25 | fixed baseline | HNSW ef16 | yes | 0.926 | 93.946 | 0.01951 | No selection gain |
| GloVe25 | NeighborRecall-only | Exact Flat | no | 0.896 | 70.586 | 0.18015 | Violates latency budget |
| GloVe25 | latency-only | HNSW ef8 | no | 0.860 | 100.000 | 0.01481 | Violates coverage constraint |

The SIFT1M proposed choice improves point coverage over fixed HNSW by 0.010 but increases mean set size by 11.31; the paired bootstrap 95% CI for coverage difference is [-0.016, 0.036], while the set-size difference CI is [10.082, 12.590]. GloVe25 selects exactly the fixed baseline. These results do not establish a general advantage.
