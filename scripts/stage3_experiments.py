"""Controlled Stage 3 calibration-transfer and ANN configuration experiments."""

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
CAL_SIZE = 100
QUERY_COUNT = 200
TOP_K = 100
TARGET_COVERAGE = 0.90
TIMING_REPEATS = 3


def qhigher(values: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    rank = min(int(np.ceil((len(ordered) + 1) * coverage)) - 1, len(ordered) - 1)
    return float(ordered[rank])


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def rss() -> int:
    return int(psutil.Process().memory_info().rss)


def build(name: str, gallery: np.ndarray, parameter: int) -> tuple[Any, dict[str, Any]]:
    d = gallery.shape[1]
    before = rss()
    started = time.perf_counter()
    train_seconds = 0.0
    if name == "exact_flat":
        index = faiss.IndexFlatL2(d)
        index.add(gallery)
    elif name == "hnsw":
        index = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_L2)
        index.hnsw.rng = faiss.RandomGenerator(SEED)
        index.hnsw.efConstruction = 100
        index.hnsw.efSearch = parameter
        index.add(gallery)
    elif name == "ivfflat":
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_L2)
        train_start = time.perf_counter()
        index.train(gallery)
        train_seconds = time.perf_counter() - train_start
        index.nprobe = parameter
        index.add(gallery)
    else:
        raise ValueError(name)
    return index, {
        "build_seconds": time.perf_counter() - started,
        "training_seconds": train_seconds,
        "rss_delta_bytes": max(0, rss() - before),
    }


def search_all(index: Any, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances, ids = index.search(queries, TOP_K)
    return distances.astype(np.float64), ids.astype(np.int64)


def timing(index: Any, queries: np.ndarray) -> tuple[float, float]:
    # Warmup is excluded from the reported timings.
    index.search(queries[:10], TOP_K)
    values: list[float] = []
    for query in queries:
        repeats: list[float] = []
        for _ in range(TIMING_REPEATS):
            start = time.perf_counter_ns()
            index.search(query.reshape(1, -1), TOP_K)
            repeats.append((time.perf_counter_ns() - start) / 1e6)
        values.append(float(np.median(repeats)))
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


def evaluate(
    index: Any,
    index_name: str,
    parameter: int,
    gallery: np.ndarray,
    queries: np.ndarray,
    target_ids: np.ndarray,
    threshold: float,
    calibration_condition: str,
    evaluation_condition: str,
    build_meta: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    distances, ids = search_all(index, queries[CAL_SIZE:])
    test_queries = queries[CAL_SIZE:]
    test_targets = target_ids[CAL_SIZE:]
    target_distances = np.asarray(
        [np.sum((q - gallery[int(t)]) ** 2, dtype=np.float64) for q, t in zip(test_queries, test_targets, strict=True)]
    )
    omission: list[bool] = []
    exclusion: list[bool] = []
    sizes: list[int] = []
    recall1: list[bool] = []
    recall100: list[bool] = []
    query_rows: list[dict[str, Any]] = []
    for i, (row_ids, row_dist, target, target_distance) in enumerate(
        zip(ids, distances, test_targets, target_distances, strict=True)
    ):
        valid = row_ids >= 0
        row_ids = row_ids[valid]
        row_dist = row_dist[valid]
        position = np.flatnonzero(row_ids == int(target))
        omitted = not bool(position.size)
        excluded = bool(position.size and target_distance > threshold)
        missed = omitted or excluded
        omission.append(omitted)
        exclusion.append(excluded)
        sizes.append(int(np.sum(row_dist <= threshold)))
        recall1.append(bool(position.size and position[0] == 0))
        recall100.append(bool(position.size))
        query_rows.append(
            {
                "gallery_size": len(gallery),
                "index": index_name,
                "parameter": parameter,
                "calibration_condition": calibration_condition,
                "evaluation_condition": evaluation_condition,
                "query_id": CAL_SIZE + i,
                "target_id": int(target),
                "candidate_omission": omitted,
                "threshold_exclusion": excluded,
                "missed": missed,
                "set_size": sizes[-1],
                "recall_at_1": recall1[-1],
                "recall_at_100": recall100[-1],
            }
        )
    p50, p95 = timing(index, test_queries)
    empirical = float(1.0 - np.mean(np.asarray(omission) | np.asarray(exclusion)))
    omission_rate = float(np.mean(omission))
    exclusion_rate = float(np.mean(exclusion))
    row = {
        "gallery_size": len(gallery),
        "index": index_name,
        "parameter": parameter,
        "calibration_condition": calibration_condition,
        "evaluation_condition": evaluation_condition,
        "threshold_squared_l2": threshold,
        "target_coverage": TARGET_COVERAGE,
        "empirical_coverage": empirical,
        "target_coverage_shortfall": max(0.0, TARGET_COVERAGE - empirical),
        "total_miss_rate": 1.0 - empirical,
        "candidate_omission_rate": omission_rate,
        "threshold_exclusion_rate": exclusion_rate,
        "decomposition_error": (1.0 - empirical) - omission_rate - exclusion_rate,
        "recall_at_1": float(np.mean(recall1)),
        "recall_at_100": float(np.mean(recall100)),
        "mean_set_size": float(np.mean(sizes)),
        "mean_normalized_set_size": float(np.mean(sizes) / len(gallery)),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        **build_meta,
        "decomposition_holds": abs((1.0 - empirical) - omission_rate - exclusion_rate) < 1e-12,
    }
    return {"summary": row, "queries": query_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    faiss.omp_set_num_threads(1)
    np.random.seed(SEED)
    with h5py.File(args.data, "r") as source:
        train = np.asarray(source["train"][:100_000], dtype=np.float32)
        queries = np.asarray(source["test"][:QUERY_COUNT], dtype=np.float32)
    exact_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "seed": SEED,
        "cpu_only": True,
        "faiss_version": faiss.__version__,
        "platform": platform.platform(),
        "data_sha256": sha256(args.data),
        "calibration_size": CAL_SIZE,
        "query_count": QUERY_COUNT,
        "top_k": TOP_K,
        "timing_repeats": TIMING_REPEATS,
        "gallery_sizes": [10_000, 50_000, 100_000],
    }

    for gallery_size in manifest["gallery_sizes"]:
        gallery = train[:gallery_size]
        exact_gt, exact_meta = build("exact_flat", gallery, 0)
        exact_dist, exact_ids = exact_gt.search(queries, 1)
        target_ids = exact_ids[:, 0].astype(np.int64)
        exact_target_dist = exact_dist[:, 0].astype(np.float64)
        exact_threshold = qhigher(exact_target_dist[:CAL_SIZE], TARGET_COVERAGE)
        if gallery_size == 10_000:
            configs = [("hnsw", p) for p in (8, 16, 32, 64)] + [("ivfflat", p) for p in (1, 4, 8, 16, 32)]
        else:
            configs = [("hnsw", 32), ("ivfflat", 8)]
        for name, parameter in [("exact_flat", 0)] + configs:
            index, meta = (exact_gt, exact_meta) if name == "exact_flat" else build(name, gallery, parameter)
            prefix = f"g{gallery_size}_{name}_{parameter}"
            faiss.write_index(index, str(args.output / f"{prefix}.faiss"))
            ann_dist, ann_ids = search_all(index, queries[:CAL_SIZE])
            ann_scores: list[float] = []
            for row_ids, row_dist, target in zip(ann_ids, ann_dist, target_ids[:CAL_SIZE], strict=True):
                pos = np.flatnonzero(row_ids == int(target))
                ann_scores.append(float(row_dist[pos[0]]) if pos.size else float("inf"))
            ann_threshold = qhigher(np.asarray(ann_scores), TARGET_COVERAGE)
            conditions: list[tuple[str, str, float]] = []
            if name == "exact_flat":
                conditions.append(("exact", "exact", exact_threshold))
            else:
                conditions.extend(
                    [
                        ("exact", "ann", exact_threshold),
                        ("ann", "ann", ann_threshold),
                    ]
                )
            for calibration, evaluation, threshold in conditions:
                result = evaluate(
                    index,
                    name,
                    parameter,
                    gallery,
                    queries,
                    target_ids,
                    threshold,
                    calibration,
                    evaluation,
                    {**meta, "serialized_index_bytes": (args.output / f"{prefix}.faiss").stat().st_size},
                    args.output,
                )
                result["summary"]["ann_calibration_threshold_squared_l2"] = ann_threshold
                result["summary"]["exact_calibration_threshold_squared_l2"] = exact_threshold
                query_rows.extend(result["queries"])
                if gallery_size == 10_000 and calibration == "exact" and evaluation == "ann":
                    sweep_rows.append(result["summary"])
                if gallery_size == 10_000:
                    exact_rows.append(result["summary"])
                if name in ("exact_flat", "hnsw", "ivfflat") and gallery_size in (10_000, 50_000, 100_000) and calibration == "exact" and evaluation in ("exact", "ann"):
                    scale_rows.append(result["summary"])
    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output / "CALIBRATION_TRANSFER_RESULTS.csv", exact_rows)
    write_csv(args.output / "INDEX_PARAMETER_SWEEP.csv", sweep_rows)
    write_csv(args.output / "GALLERY_SCALE_RESULTS.csv", scale_rows)
    write_csv(args.output / "QUERY_LEVEL_STAGE3.csv", query_rows)
    (args.output / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"calibration_rows": len(exact_rows), "sweep_rows": len(sweep_rows), "scale_rows": len(scale_rows), "query_rows": len(query_rows), **manifest}, indent=2))


if __name__ == "__main__":
    main()
