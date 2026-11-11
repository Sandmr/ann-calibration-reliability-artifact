"""Calibration-only threshold selection for a fixed mean set-size budget."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import faiss
import h5py
import numpy as np


CALIBRATION_SIZE = 500
QUERY_COUNT = 1000
TOP_K = 100
TARGET_COVERAGE = 0.90
FIXED_MEAN_BUDGET = 20.0


def choose_threshold_for_budget(calibration_distances: np.ndarray, budget: float) -> float:
    """Choose an observed distance using calibration queries only."""
    flattened = np.sort(np.asarray(calibration_distances, dtype=np.float64).ravel())
    target_count = max(1, min(int(round(budget * calibration_distances.shape[0])), len(flattened)))
    return float(flattened[target_count - 1])


def run(workload: str, data_path: Path, index_dir: Path, output_path: Path) -> None:
    with h5py.File(data_path, "r") as source:
        gallery = np.asarray(source["train"][:10_000], dtype=np.float32)
        queries = np.asarray(source["test"][:QUERY_COUNT], dtype=np.float32)
    exact = faiss.IndexFlatL2(gallery.shape[1])
    exact.add(gallery)
    _, exact_ids = exact.search(queries, TOP_K)
    targets = exact_ids[:, 0]
    rows = []
    for name, parameter in [("hnsw", 16), ("ivfflat", 8)]:
        index = faiss.read_index(str(index_dir / f"{workload}_{name}_{parameter}.faiss"))
        cal_dist, cal_ids = index.search(queries[:CALIBRATION_SIZE], TOP_K)
        test_dist, test_ids = index.search(queries[CALIBRATION_SIZE:], TOP_K)
        threshold = choose_threshold_for_budget(cal_dist, FIXED_MEAN_BUDGET)
        cal_mean_size = float(np.mean(np.sum(cal_dist <= threshold, axis=1)))
        test_sizes = np.sum(test_dist <= threshold, axis=1)
        misses = []
        omissions = []
        exclusions = []
        for distances, ids, target, query in zip(test_dist, test_ids, targets[CALIBRATION_SIZE:], queries[CALIBRATION_SIZE:], strict=True):
            pos = np.flatnonzero(ids == int(target))
            target_distance = float(np.sum((query - gallery[int(target)]) ** 2, dtype=np.float64))
            omitted = not bool(pos.size)
            excluded = bool(pos.size and target_distance > threshold)
            omissions.append(omitted)
            exclusions.append(excluded)
            misses.append(omitted or excluded)
        coverage = 1.0 - float(np.mean(misses))
        rows.append({
            "workload": workload,
            "index": name,
            "parameter": parameter,
            "fixed_mean_budget": FIXED_MEAN_BUDGET,
            "threshold_squared_l2": threshold,
            "calibration_mean_set_size": cal_mean_size,
            "test_mean_set_size": float(np.mean(test_sizes)),
            "empirical_coverage": coverage,
            "target_coverage_shortfall": max(0.0, TARGET_COVERAGE - coverage),
            "total_miss_rate": 1.0 - coverage,
            "candidate_omission_rate": float(np.mean(omissions)),
            "threshold_exclusion_rate": float(np.mean(exclusions)),
            "decomposition_error": (1.0 - coverage) - float(np.mean(omissions)) - float(np.mean(exclusions)),
        })
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.workload, args.data, args.index_dir, args.output)


if __name__ == "__main__":
    main()
