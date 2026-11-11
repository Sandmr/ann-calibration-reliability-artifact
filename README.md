# ICDE 2027 EAB Artifact

This directory contains the public-facing reproduction materials for the ICDE 2027 EAB study of calibration transfer, reliability metrics, and system cost in approximate vector search. It is self-contained and does not contain the authors' manuscript paths, private metadata, access tokens, or the protected CVPR projects.

## Scope and reproducibility status

- CPU-only Faiss experiments; no deep-learning model or model weights.
- Public ANN-Benchmarks HDF5 workloads: SIFT1M (`sift-128-euclidean.hdf5`) and GloVe25 (`glove-25-angular.hdf5`).
- Exact targets are recomputed on the selected gallery; benchmark neighbor labels are not used as the experimental ground truth.
- The frozen CSV/JSON outputs and paper figures are included for inspection.
- The smoke-test command below was verified against the public SIFT1M file. Full Stage 3/4 and configuration-selection reruns remain explicitly unverified and are provided as reproducible commands.

## Environment

Python 3.11 is the validated interpreter family. Create a fresh environment outside the original experiment tree:

```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The minimal dependency lock uses `faiss-cpu==1.8.0`, NumPy 1.26.4, SciPy 1.11.4, pandas 2.1.4, h5py 3.10.0, psutil 5.9.8, matplotlib 3.8.4, seaborn 0.13.2, and pytest 8.3.5. `usearch` and `pyarrow` are not imported by the reproduction scripts and are omitted from the public runtime lock. The experiment scripts set Faiss/OpenMP to one thread where applicable.

## Data

Download and verify both public files:

```text
python scripts/download_data.py --output data --dataset all
```

The script checks SHA-256 hashes. No dataset file is redistributed in this artifact because the HDF5 files are large; the source URLs and checksums are recorded in `docs/DEPENDENCIES_AND_DATA.md` and the download script.

## Reproduction commands

All commands are run from this directory. Paths are relative and can be changed with command-line arguments.

### Table I / smoke validation

```text
python scripts/smoke_test.py --data data/sift-128-euclidean.hdf5 --output results/reproduction_smoke
```

### Stage 3: calibration transfer, parameter sweep, and gallery scale

```text
python scripts/stage3_experiments.py --data data/sift-128-euclidean.hdf5 --output results/reproduction_stage3
```

### Stage 4: 500/500 calibration/test split

```text
python scripts/stage4_experiments.py --data data/sift-128-euclidean.hdf5 --workload sift1m --output results/reproduction_stage4/sift1m
python scripts/stage4_experiments.py --data data/glove-25-angular.hdf5 --workload glove25 --output results/reproduction_stage4/glove25
```

### Configuration-selection experiment

```text
python scripts/configuration_selection_experiment.py --sift data/sift-128-euclidean.hdf5 --glove data/glove-25-angular.hdf5 --output results/reproduction_configuration_selection
```

The paired bootstrap command consumes the configuration-selection query-level CSV and baseline CSV produced above. The exact filenames are printed by the experiment script.

### Figures 1--4 and LaTeX tables

```text
python scripts/generate_figures.py --experiment-root . --paper-root .
```

This writes vector PDF/PNG figures and LaTeX tables under `figures/` and `tables/`. The included frozen figures are the versions used for the manuscript; regenerated figures should be compared visually and numerically rather than assumed byte-identical.

## Mapping to paper artifacts

| Paper item | Reproduction source |
|---|---|
| Table I | `docs/EXPERIMENT_CONFIGS.md`, Stage 2 smoke outputs |
| Table II | Stage 4 transfer CSVs and `results/BASELINE_COMPARISON.csv` |
| Table III | `results/CALIBRATION_TRANSFER_EXTENDED.csv`, `results/FIXED_BUDGET_COMPARISON.csv` |
| Table IV | `results/CONFIGURATION_SELECTION_RESULTS.csv`, `results/BASELINE_COMPARISON.csv` |
| Figure 1 | `figures/figure1_failure_decomposition.*` and `scripts/generate_figures.py` |
| Figure 2 | Stage 4 calibration-transfer CSVs |
| Figure 3 | Stage 4 target/neighbor-recall CSVs |
| Figure 4 | Stage 3 `GALLERY_SCALE_RESULTS.csv` |

## Integrity and privacy notes

The artifact excludes `.venv`, compiled Python caches, Faiss index binaries, local HDF5 data, manuscript files, CVPR paths, author names, university identifiers, and credentials. The supplied scripts accept paths as arguments and do not contain hard-coded Windows user directories.
