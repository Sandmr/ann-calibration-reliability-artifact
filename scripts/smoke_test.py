"""Minimal exact-vs-ANN set-valued retrieval smoke test.

This script intentionally uses only public SIFT1M vectors and CPU Faiss. It
recomputes exact neighbors on the selected 10k gallery; the benchmark-provided
neighbors are never used as ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import faiss
import h5py
import numpy as np
import psutil


SEED = 20271111
CALIBRATION_SIZE = 100
QUERY_COUNT = 200
GALLERY_SIZE = 10_000
TOP_K = 100
TARGET_COVERAGE = 0.90


def quantile_higher(scores: np.ndarray, coverage: float) -> float:
    """Finite-sample conformal quantile using the higher order statistic."""
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    rank = min(int(np.ceil((len(ordered) + 1) * coverage)) - 1, len(ordered) - 1)
    return float(ordered[rank])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def build_index(name: str, gallery: np.ndarray) -> tuple[Any, dict[str, Any]]:
    """Build one CPU Faiss index and return it with reproducibility metadata."""
    dimension = gallery.shape[1]
    start_rss = rss_bytes()
    started = time.perf_counter()
    if name == "exact_flat":
        index = faiss.IndexFlatL2(dimension)
        index.add(gallery)
        build_training_seconds = 0.0
    elif name == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_L2)
        # Faiss HNSW uses an internal RNG; set it explicitly for reproducible
        # graph construction across repeated smoke-test runs.
        index.hnsw.rng = faiss.RandomGenerator(SEED)
        index.hnsw.efConstruction = 100
        index.hnsw.efSearch = 32
        index.add(gallery)
        build_training_seconds = 0.0
    elif name == "ivfflat":
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_L2)
        train_started = time.perf_counter()
        index.train(gallery)
        build_training_seconds = time.perf_counter() - train_started
        index.nprobe = 8
        index.add(gallery)
    else:
        raise ValueError(f"Unknown index: {name}")
    return index, {
        "build_seconds": time.perf_counter() - started,
        "training_seconds": build_training_seconds,
        "rss_delta_bytes": max(0, rss_bytes() - start_rss),
    }


def run_queries(
    index: Any,
    queries: np.ndarray,
    gallery: np.ndarray,
    target_ids: np.ndarray,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Search one query at a time and compute the mutually exclusive loss split."""
    rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for query_id, (query, target_id) in enumerate(zip(queries, target_ids, strict=True)):
        started = time.perf_counter_ns()
        distances, ids = index.search(query.reshape(1, -1), TOP_K)
        latency_ms = (time.perf_counter_ns() - started) / 1e6
        latencies_ms.append(latency_ms)
        retrieved_ids = ids[0].astype(np.int64)
        retrieved_distances = distances[0].astype(np.float64)
        valid = retrieved_ids >= 0
        retrieved_ids = retrieved_ids[valid]
        retrieved_distances = retrieved_distances[valid]
        target_position = np.flatnonzero(retrieved_ids == int(target_id))
        target_in_ann = bool(target_position.size)
        target_distance = float(np.linalg.norm(query - gallery[int(target_id)]) ** 2)
        candidate_omission = not target_in_ann
        threshold_exclusion = bool(target_in_ann and target_distance > threshold)
        missed = candidate_omission or threshold_exclusion
        prediction_set_size = int(np.sum(retrieved_distances <= threshold))
        rows.append(
            {
                "query_id": query_id,
                "target_id": int(target_id),
                "target_distance": target_distance,
                "target_in_topk": target_in_ann,
                "candidate_omission": candidate_omission,
                "threshold_exclusion": threshold_exclusion,
                "missed": missed,
                "decomposition_holds": missed == (candidate_omission or threshold_exclusion),
                "prediction_set_size": prediction_set_size,
                "prediction_set_size_normalized": prediction_set_size / GALLERY_SIZE,
                "recall_at_1": bool(target_in_ann and target_position[0] == 0),
                "recall_at_100": target_in_ann,
                "latency_ms": latency_ms,
            }
        )
    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    return rows, {
        "latency_p50_ms": float(np.percentile(latency_array, 50)),
        "latency_p95_ms": float(np.percentile(latency_array, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    np.random.seed(SEED)
    os.environ["OMP_NUM_THREADS"] = "1"
    faiss.omp_set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data, "r") as source:
        gallery = np.asarray(source["train"][:GALLERY_SIZE], dtype=np.float32)
        queries = np.asarray(source["test"][:QUERY_COUNT], dtype=np.float32)
    if gallery.shape != (GALLERY_SIZE, 128) or queries.shape != (QUERY_COUNT, 128):
        raise RuntimeError(f"Unexpected shapes: gallery={gallery.shape}, queries={queries.shape}")

    exact_ground_truth = faiss.IndexFlatL2(gallery.shape[1])
    exact_ground_truth.add(gallery)
    exact_distances, exact_ids = exact_ground_truth.search(queries, TOP_K)
    calibration_distances = exact_distances[:CALIBRATION_SIZE, 0]
    threshold = quantile_higher(calibration_distances, TARGET_COVERAGE)
    target_ids = exact_ids[:, 0].astype(np.int64)
    exact_target_distances = exact_distances[:, 0].astype(np.float64)
    exact_test_coverage = float(np.mean(exact_target_distances[CALIBRATION_SIZE:] <= threshold))

    configs = ["exact_flat", "hnsw", "ivfflat"]
    summary: dict[str, Any] = {
        "seed": SEED,
        "cpu_only": True,
        "faiss_version": faiss.__version__,
        "platform": platform.platform(),
        "gallery_size": GALLERY_SIZE,
        "query_count": QUERY_COUNT,
        "calibration_count": CALIBRATION_SIZE,
        "test_count": QUERY_COUNT - CALIBRATION_SIZE,
        "top_k": TOP_K,
        "target_coverage": TARGET_COVERAGE,
        "threshold_squared_l2": threshold,
        "exact_test_coverage": exact_test_coverage,
        "source_sha256": sha256_file(args.data),
        "indices": {},
    }
    all_rows: list[dict[str, Any]] = []
    for name in configs:
        index, build_meta = build_index(name, gallery)
        index_path = args.output / f"{name}.faiss"
        faiss.write_index(index, str(index_path))
        rows, latency_meta = run_queries(
            index,
            queries[CALIBRATION_SIZE:],
            gallery,
            target_ids[CALIBRATION_SIZE:],
            threshold,
        )
        for row in rows:
            row["index"] = name
            row["calibration_test_leakage"] = False
            all_rows.append(row)
        missed = np.asarray([row["missed"] for row in rows], dtype=bool)
        omission = np.asarray([row["candidate_omission"] for row in rows], dtype=bool)
        threshold_loss = np.asarray([row["threshold_exclusion"] for row in rows], dtype=bool)
        empirical_coverage = float(1.0 - missed.mean())
        omission_rate = float(omission.mean())
        threshold_exclusion_rate = float(threshold_loss.mean())
        summary["indices"][name] = {
            **build_meta,
            **latency_meta,
            "serialized_index_bytes": index_path.stat().st_size,
            "coverage": empirical_coverage,
            "coverage_shortfall": float(max(0.0, TARGET_COVERAGE - empirical_coverage)),
            "total_miss_rate": float(1.0 - empirical_coverage),
            "recall_at_1": float(np.mean([row["recall_at_1"] for row in rows])),
            "recall_at_100": float(np.mean([row["recall_at_100"] for row in rows])),
            "candidate_omission_rate": omission_rate,
            "threshold_exclusion_rate": threshold_exclusion_rate,
            "threshold_loss_rate": threshold_exclusion_rate,
            "decomposition_error": float((1.0 - empirical_coverage) - omission_rate - threshold_exclusion_rate),
            "mean_set_size": float(np.mean([row["prediction_set_size"] for row in rows])),
            "mean_normalized_set_size": float(np.mean([row["prediction_set_size_normalized"] for row in rows])),
            "decomposition_holds_all": bool(np.all([row["decomposition_holds"] for row in rows])),
        }

    fieldnames = list(all_rows[0].keys())
    with (args.output / "query_level_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
