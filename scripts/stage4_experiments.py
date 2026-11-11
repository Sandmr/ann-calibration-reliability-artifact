"""Stage 4 target-vs-neighbor recall and calibration-transfer experiments."""

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
CALIBRATION_SIZE = 500
TEST_SIZE = 500
QUERY_COUNT = CALIBRATION_SIZE + TEST_SIZE
TOP_K = 100
TARGET_COVERAGE = 0.90
FIXED_BUDGET_CAP = 25
TIMING_REPEATS = 5


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
    dimension = gallery.shape[1]
    before = rss()
    started = time.perf_counter()
    train_seconds = 0.0
    if name == "exact_flat":
        index = faiss.IndexFlatL2(dimension)
        index.add(gallery)
    elif name == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_L2)
        index.hnsw.rng = faiss.RandomGenerator(SEED)
        index.hnsw.efConstruction = 100
        index.hnsw.efSearch = parameter
        index.add(gallery)
    elif name == "ivfflat":
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, 100, faiss.METRIC_L2)
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


def search(index: Any, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances, ids = index.search(queries, TOP_K)
    return distances.astype(np.float64), ids.astype(np.int64)


def latency(index: Any, queries: np.ndarray) -> tuple[float, float]:
    index.search(queries[:20], TOP_K)
    values: list[float] = []
    for query in queries:
        reps: list[float] = []
        for _ in range(TIMING_REPEATS):
            start = time.perf_counter_ns()
            index.search(query.reshape(1, -1), TOP_K)
            reps.append((time.perf_counter_ns() - start) / 1e6)
        values.append(float(np.median(reps)))
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float("inf")


def target_distances(queries: np.ndarray, gallery: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.sum((q - gallery[int(t)]) ** 2, dtype=np.float64) for q, t in zip(queries, targets, strict=True)],
        dtype=np.float64,
    )


def neighbor_recall(exact_ids: np.ndarray, ann_ids: np.ndarray, k: int) -> float:
    vals = []
    for exact_row, ann_row in zip(exact_ids[:, :k], ann_ids[:, :k], strict=True):
        vals.append(len(set(map(int, exact_row)) & set(map(int, ann_row))) / k)
    return float(np.mean(vals))


def evaluate(
    index: Any,
    name: str,
    parameter: int,
    gallery: np.ndarray,
    queries: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    calibration_condition: str,
    output_dir: Path,
    build_meta: dict[str, Any],
    exact_ids_test: np.ndarray,
    ann_ids_test: np.ndarray,
    ann_dist_test: np.ndarray,
    ann_neighbor_recall_1: float,
    ann_neighbor_recall_100: float,
    score_calibration: np.ndarray,
    score_test: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    test_targets = targets[CALIBRATION_SIZE:]
    test_queries = queries[CALIBRATION_SIZE:]
    target_d = target_distances(test_queries, gallery, test_targets)
    omission: list[bool] = []
    exclusion: list[bool] = []
    sizes: list[int] = []
    fixed_sizes: list[int] = []
    fixed_missed: list[bool] = []
    target_r1: list[bool] = []
    target_r100: list[bool] = []
    rows: list[dict[str, Any]] = []
    for i, (ids, distances, target, td) in enumerate(zip(ann_ids_test, ann_dist_test, test_targets, target_d, strict=True)):
        valid = ids >= 0
        ids = ids[valid]
        distances = distances[valid]
        pos = np.flatnonzero(ids == int(target))
        omitted = not bool(pos.size)
        excluded = bool(pos.size and td > threshold)
        missed = omitted or excluded
        set_size = int(np.sum(distances <= threshold))
        cap_distances = distances[:FIXED_BUDGET_CAP]
        fixed_size = int(np.sum(cap_distances <= threshold))
        fixed_miss = bool(not pos.size or int(pos[0]) >= FIXED_BUDGET_CAP or td > threshold)
        omission.append(omitted)
        exclusion.append(excluded)
        sizes.append(set_size)
        fixed_sizes.append(fixed_size)
        fixed_missed.append(fixed_miss)
        target_r1.append(bool(pos.size and pos[0] == 0))
        target_r100.append(bool(pos.size))
        rows.append({
            "query_id": CALIBRATION_SIZE + i,
            "index": name,
            "parameter": parameter,
            "calibration_condition": calibration_condition,
            "target_id": int(target),
            "candidate_omission": omitted,
            "threshold_exclusion": excluded,
            "missed": missed,
            "set_size": set_size,
            "fixed_budget_set_size": fixed_size,
            "fixed_budget_missed": fixed_miss,
            "score_test": float(score_test[i]),
        })
    empirical = float(1.0 - np.mean(np.asarray(omission) | np.asarray(exclusion)))
    fixed_coverage = float(1.0 - np.mean(fixed_missed))
    p50, p95 = latency(index, test_queries)
    summary = {
        "index": name,
        "parameter": parameter,
        "calibration_condition": calibration_condition,
        "threshold_squared_l2": threshold,
        "calibration_score_p50": float(np.percentile(score_calibration, 50)),
        "calibration_score_p95": finite_percentile(score_calibration, 95),
        "test_score_p50": finite_percentile(score_test, 50),
        "test_score_p95": finite_percentile(score_test, 95),
        "empirical_coverage": empirical,
        "target_coverage_shortfall": max(0.0, TARGET_COVERAGE - empirical),
        "total_miss_rate": 1.0 - empirical,
        "candidate_omission_rate": float(np.mean(omission)),
        "threshold_exclusion_rate": float(np.mean(exclusion)),
        "decomposition_error": (1.0 - empirical) - float(np.mean(omission)) - float(np.mean(exclusion)),
        "target_recall_at_1": float(np.mean(target_r1)),
        "target_recall_at_100": float(np.mean(target_r100)),
        "neighbor_recall_at_1": ann_neighbor_recall_1,
        "neighbor_recall_at_100": ann_neighbor_recall_100,
        "mean_set_size": float(np.mean(sizes)),
        "mean_normalized_set_size": float(np.mean(sizes) / len(gallery)),
        "fixed_budget_cap": FIXED_BUDGET_CAP,
        "fixed_budget_coverage": fixed_coverage,
        "fixed_budget_mean_set_size": float(np.mean(fixed_sizes)),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        **build_meta,
        "decomposition_holds": abs((1.0 - empirical) - float(np.mean(omission)) - float(np.mean(exclusion))) < 1e-12,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    faiss.omp_set_num_threads(1)
    np.random.seed(SEED)
    with h5py.File(args.data, "r") as source:
        gallery = np.asarray(source["train"][:10_000], dtype=np.float32)
        queries = np.asarray(source["test"][:QUERY_COUNT], dtype=np.float32)
    if len(queries) < QUERY_COUNT:
        raise RuntimeError(f"Need {QUERY_COUNT} queries, found {len(queries)}")

    exact_index, exact_meta = build("exact_flat", gallery, 0)
    exact_dist_all, exact_ids_all = exact_index.search(queries, TOP_K)
    exact_dist_all = exact_dist_all.astype(np.float64)
    exact_ids_all = exact_ids_all.astype(np.int64)
    targets = exact_ids_all[:, 0]
    exact_threshold = qhigher(exact_dist_all[:CALIBRATION_SIZE, 0], TARGET_COVERAGE)
    configs = [("exact_flat", 0), ("hnsw", 16), ("ivfflat", 8)]
    transfer: list[dict[str, Any]] = []
    fixed: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for name, parameter in configs:
        index, meta = (exact_index, exact_meta) if name == "exact_flat" else build(name, gallery, parameter)
        prefix = f"{args.workload}_{name}_{parameter}"
        index_path = args.output / f"{prefix}.faiss"
        faiss.write_index(index, str(index_path))
        ann_dist_all, ann_ids_all = search(index, queries)
        ann_target_scores = []
        for ids, distances, target in zip(ann_ids_all[:CALIBRATION_SIZE], ann_dist_all[:CALIBRATION_SIZE], targets[:CALIBRATION_SIZE], strict=True):
            pos = np.flatnonzero(ids == int(target))
            ann_target_scores.append(float(distances[pos[0]]) if pos.size else float("inf"))
        ann_target_scores = np.asarray(ann_target_scores, dtype=np.float64)
        ann_threshold = qhigher(ann_target_scores, TARGET_COVERAGE)
        exact_target_scores_test = exact_dist_all[CALIBRATION_SIZE:, 0]
        ann_target_scores_test = []
        for ids, distances, target in zip(ann_ids_all[CALIBRATION_SIZE:], ann_dist_all[CALIBRATION_SIZE:], targets[CALIBRATION_SIZE:], strict=True):
            pos = np.flatnonzero(ids == int(target))
            ann_target_scores_test.append(float(distances[pos[0]]) if pos.size else float("inf"))
        ann_target_scores_test = np.asarray(ann_target_scores_test, dtype=np.float64)
        nr1 = neighbor_recall(exact_ids_all[CALIBRATION_SIZE:], ann_ids_all[CALIBRATION_SIZE:], 1)
        nr100 = neighbor_recall(exact_ids_all[CALIBRATION_SIZE:], ann_ids_all[CALIBRATION_SIZE:], 100)
        conditions = [("exact", exact_threshold, exact_target_scores_test)] if name == "exact_flat" else [("exact", exact_threshold, ann_target_scores_test), ("ann", ann_threshold, ann_target_scores_test)]
        for condition, threshold, test_scores in conditions:
            cal_scores = exact_dist_all[:CALIBRATION_SIZE, 0] if condition == "exact" else ann_target_scores
            summary, rows = evaluate(index, name, parameter, gallery, queries, targets, threshold, condition, args.output, {**meta, "serialized_index_bytes": index_path.stat().st_size}, exact_ids_all[CALIBRATION_SIZE:], ann_ids_all[CALIBRATION_SIZE:], ann_dist_all[CALIBRATION_SIZE:], nr1, nr100, cal_scores, test_scores)
            summary["workload"] = args.workload
            transfer.append(summary)
            query_rows.extend([{**r, "workload": args.workload} for r in rows])
            if name != "exact_flat":
                fixed.append(summary)
        costs.append({"workload": args.workload, "index": name, "parameter": parameter, **meta, "serialized_index_bytes": index_path.stat().st_size, "timing_repeats": TIMING_REPEATS, "warmup_queries": 20})
        score_groups = [
            ("exact_calibration_target_score", exact_dist_all[:CALIBRATION_SIZE, 0]),
            ("ann_calibration_target_score", ann_target_scores),
            ("exact_test_target_score", exact_target_scores_test),
            ("ann_test_target_score", ann_target_scores_test),
        ]
        for score_type, values in score_groups:
            for qid, score in enumerate(values):
                score_rows.append({"workload": args.workload, "index": name, "score_type": score_type, "query_position": qid, "score": float(score), "score_is_infinite": bool(np.isinf(score))})

    def write(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with (args.output / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write("CALIBRATION_TRANSFER_EXTENDED.csv", transfer)
    write("FIXED_BUDGET_COMPARISON.csv", fixed)
    write("SYSTEM_COST_REPEATS.csv", costs)
    write("QUERY_LEVEL_STAGE4.csv", query_rows)
    write("CALIBRATION_SCORE_DISTRIBUTIONS.csv", score_rows)
    (args.output / "RUN_MANIFEST.json").write_text(json.dumps({"workload": args.workload, "data_sha256": sha256(args.data), "seed": SEED, "gallery_size": len(gallery), "calibration_size": CALIBRATION_SIZE, "test_size": TEST_SIZE, "target_coverage": TARGET_COVERAGE, "fixed_budget_cap": FIXED_BUDGET_CAP, "faiss_version": faiss.__version__, "cpu_only": True}, indent=2), encoding="utf-8")
    print(json.dumps({"workload": args.workload, "transfer_rows": len(transfer), "fixed_rows": len(fixed), "cost_rows": len(costs), "query_rows": len(query_rows)}, indent=2))


if __name__ == "__main__":
    main()
