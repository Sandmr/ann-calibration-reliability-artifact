# Stage 2 smoke-test configuration

| Factor | Value |
|---|---|
| Dataset | SIFT1M official HDF5 endpoint |
| Gallery | first 10,000 vectors from `train` |
| Queries | first 200 vectors from `test` |
| Calibration/test | queries 0–99 / 100–199 |
| Target | exact 1NN recomputed on the 10k gallery |
| Top-K | 100 |
| Target coverage | 0.90 |
| Threads | Faiss/OpenMP 1 |
| Seed | 20271111 |
| Faiss | 1.8.0 (1.16.0 HNSW crashed on this Windows runtime; 1.8.0 passed the isolated HNSW probe) |
| Indexes | exact Flat, HNSW (M=16, efConstruction=100, efSearch=32), IVFFlat (nlist=100, nprobe=8) |
| Distance | squared Euclidean distance |

No model training or learned encoder is used. IVFFlat clustering is the only index-training operation.
