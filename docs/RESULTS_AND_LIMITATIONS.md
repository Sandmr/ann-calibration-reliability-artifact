# Results and limitations

## What the data support

- Candidate omission and threshold exclusion are empirically separable and sum to the total miss rate.
- Exact-calibrated and ANN-calibrated thresholds produce different coverage/set-size profiles on identical ANN searches.
- Gallery scale changes both the exact target identity and the ANN operating regime; it cannot be interpreted as a single-factor recall experiment.

## What the data do not support

- A universal calibration-transfer theorem.
- Generalization beyond the two measured public vector workloads.
- Claims about workloads or operating regimes not included in the experiments.
- Statistical independence of query-level rows as a substitute for independent datasets.

All confidence intervals in the Stage 3 report are exploratory query-level bootstrap intervals. They are not a replacement for multi-workload replication or multiplicity-controlled confirmatory analysis.
