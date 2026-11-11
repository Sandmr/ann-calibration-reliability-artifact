# ICDE 2027 EAB Artifact Final Audit

Audit date: 2026-11-11  
Audit scope: the contents of this directory before adding this report (75 files)

## 1. Scan inventory

The audit scanned all Markdown, Python, JSON, text/configuration, CSV, PDF, and PNG files recursively. The 75-file pre-report inventory was:

- Root: `README.md`, `requirements.txt`.
- Configuration: `configs/reproduction_config.json`.
- Documentation (11): `CONFIGURATION_SELECTION_METHOD.md`, `CONSTRAINT_SATISFACTION_ANALYSIS.md`, `DEPENDENCIES_AND_DATA.md`, `EXPERIMENT_CONFIGS.md`, `FIGURE_TABLE_INVENTORY.md`, `FROZEN_EXPERIMENT_RESULTS.md`, `METRIC_DEFINITIONS_UPDATED.md`, `METRIC_DEFINITIONS.md`, `RECALL_DEFINITION_AUDIT.md`, `RESULTS_AND_LIMITATIONS.md`, `RESULTS_SUMMARY.md`.
- Scripts (10): `configuration_selection_experiment.py`, `configuration_selection_repeats.py`, `create_freeze_manifest.py`, `download_data.py`, `fixed_budget_analysis.py`, `generate_figures.py`, `paired_bootstrap_selection.py`, `smoke_test.py`, `stage3_experiments.py`, `stage4_experiments.py`.
- Figures (8): four PDF files and four PNG files named `figure1_failure_decomposition`, `figure2_calibration_transfer`, `figure3_recall_comparison`, and `figure4_gallery_scale_cost`.
- Results: all files under `results/**/*.csv` and `results/**/*.json`; no serialized index binaries remain.

After the audit, this report is the only additional file. The public package therefore contains 76 files including this audit report.

## 2. AI-dialogue and internal-process scan

### Findings

No conversational prompts, ChatGPT/Codex transcripts, system instructions, hidden prompts, generated-task templates, or model-internal work logs were found in the source, documentation, configuration, or result files.

One internal readiness file, `ARTIFACT_READINESS_REPORT.md`, was present before the audit. It described the assembly process rather than the experiment itself. It was removed; its scientifically relevant unverified-status information is retained in `README.md` and this report.

The following non-reproducibility wording was corrected in place:

- `docs/FIGURE_TABLE_INVENTORY.md`: replaced the old manuscript-project path with the artifact-relative `figures/` path.
- `docs/DEPENDENCIES_AND_DATA.md`: added the GloVe25 URL and checksum, which were missing from the public provenance note.
- `docs/RESULTS_AND_LIMITATIONS.md`: replaced the internal “ICDE Research Track readiness” statement with a workload-generalization boundary.
- `README.md`: replaced “not re-executed during artifact assembly” with an externally meaningful “remain explicitly unverified” statement.
- `requirements.txt`: removed `usearch` and `pyarrow`, which are not imported by the reproduction scripts and were unstable/unnecessary in the validated Windows environment.

No experiment logic, frozen CSV/JSON value, seed, metric definition, or reported numerical result was changed.

## 3. Anonymity and privacy audit

| Surface | Result | Evidence |
|---|---|---|
| Names, affiliations, emails | PASS | No personal identity strings found in text/config/source scans |
| Absolute workstation paths | PASS | No drive-specific workstation path, home-directory path, or repository-specific absolute path |
| Git identity | PASS | No `.git` directory, commit metadata, or author configuration included |
| Credentials | PASS | No access tokens, API keys, bearer strings, or private endpoints |
| PDF metadata | PASS | Figures expose only Matplotlib creator/producer metadata; no Author, Title, or Subject fields |
| PNG metadata | PASS | All four PNG files have empty EXIF dictionaries |
| External identity links | PASS | Only public ANN-Benchmarks dataset URLs are retained |

The recorded `platform` fields in frozen JSON identify a generic Windows runtime and do not contain a user name or machine name.

## 4. Command and environment audit

All ten Python scripts accepted `--help` successfully in the validated Python 3.11/Faiss 1.8.0 environment. The README commands use relative artifact paths and provide the required input/output arguments.

Verified execution evidence retained in the artifact:

- `results/reproduction_smoke/summary.json` and `query_level_results.csv` were produced by an actual SIFT1M CPU smoke run.
- The figure-generation script was executed successfully and generated all four figure pairs in a temporary validation directory; those temporary duplicates were removed after inspection.

Not fully verified in this audit:

- A clean-environment full Stage 3 rerun.
- A clean-environment full Stage 4 rerun for both SIFT1M and GloVe25.
- A clean-environment configuration-selection rerun and its paired bootstrap comparison.
- A live network download in the artifact directory. The downloader is syntax-checked and the URLs/checksums are documented; the source files' checksums were independently recorded.

These items remain explicitly unverified. The artifact does not claim end-to-end reproduction of every frozen table value until those reruns are performed.

## 5. README coverage

`README.md` provides:

- CPU-only environment creation and pinned dependencies;
- public dataset URLs and checksum verification command;
- commands for smoke, Stage 3, Stage 4, configuration selection, and figure/table generation;
- a mapping from Table I--IV and Figure 1--4 to source CSVs and scripts;
- data-target definitions, calibration/test split, and omission of large local data/index binaries;
- explicit distinction between verified smoke/figure paths and unverified full reruns.

The commands are structurally complete. Full numerical equivalence remains pending the reruns listed above.

## 6. Recommended GitHub upload set

Upload the following complete set:

1. `README.md`, `requirements.txt`, and `configs/reproduction_config.json`.
2. All 11 files under `docs/` listed in Section 1.
3. All 10 Python files under `scripts/` listed in Section 1.
4. All eight files under `figures/` listed in Section 1.
5. Every `.csv` and `.json` file under `results/`.
6. `ARTIFACT_FINAL_AUDIT.md`.

Do not upload HDF5 datasets, `.faiss` indexes, virtual environments, Python caches, manuscript sources, CVPR materials, or temporary render directories. The removed `ARTIFACT_READINESS_REPORT.md` is intentionally not part of the public upload set.

## 7. Submission decision

**Conditionally ready for ICDE 2027 EAB Artifact submission.** The package is anonymous, self-contained at the source/configuration/result level, and contains real smoke and figure-generation verification. It is not yet appropriate to claim a fully reproduced artifact badge for every table and figure until the listed Stage 3, Stage 4, configuration-selection, and download checks are run in a clean environment.
