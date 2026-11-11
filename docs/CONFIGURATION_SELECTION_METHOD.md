# Calibration-driven ANN configuration selection

## Research question

Can calibration queries select an ANN index/search configuration that satisfies a pre-specified coverage target, prediction-set budget, and p95 latency budget on independent test queries?

## Candidate set

The selector uses only the already validated 10k-gallery configurations: Exact Flat; HNSW `efSearch` ∈ {8, 16, 32, 64}; and IVFFlat `nprobe` ∈ {1, 4, 8, 16, 32}. No new encoder or learned model is used.

## Protocol

- SIFT1M and GloVe25, 10,000-vector gallery, 1,000 fixed queries.
- First 500 queries are calibration; last 500 are test.
- Exact Top-1 target is computed independently from Exact Flat.
- For each candidate, the calibration threshold is the finite-sample 0.90 quantile of the candidate target score; an omitted target receives an infinite score and makes the candidate infeasible when omission is too high.
- Pre-specified budgets: target coverage 0.90; mean calibration set-size budget 50 (SIFT1M) or 120 (GloVe25); p95 query latency budget 0.10 ms.
- Proposed rule: retain candidates satisfying all three calibration constraints, then minimize p95 latency, followed by test-independent mean set size and serialized index size as deterministic tie-breakers.
- Baselines: fixed HNSW ef16; NeighborRecall@100-only selection; latency-only selection. All use the same candidate set and calibration budget. Infeasible baseline outcomes remain infeasible.

The selector does not inspect test labels, test coverage, or test set size. The repeated-split attempt was stopped after an execution-time failure and produced no result; the reported uncertainty is paired bootstrap over the completed 500-query test set.
