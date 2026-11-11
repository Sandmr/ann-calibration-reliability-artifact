"""Calibration-driven ANN configuration selection on frozen workloads.

This is a targeted replay over the already validated Stage-3 configuration set.
It writes only new outputs and never modifies Stage-1--4 artifacts.
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
CAL_SIZE = 500
TEST_SIZE = 500
TOP_K = 100
TARGET_COVERAGE = 0.90
SET_BUDGETS = {"sift1m": 50.0, "glove25": 120.0}
LATENCY_BUDGET_MS = 0.10
TIMING_REPEATS = 5
CONFIGS = [("exact_flat", 0), ("hnsw", 8), ("hnsw", 16), ("hnsw", 32),
           ("hnsw", 64), ("ivfflat", 1), ("ivfflat", 4), ("ivfflat", 8),
           ("ivfflat", 16), ("ivfflat", 32)]


def qhigher(values: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    rank = min(int(np.ceil((len(ordered) + 1) * coverage)) - 1, len(ordered) - 1)
    return float(ordered[rank])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rss() -> int:
    return int(psutil.Process().memory_info().rss)


def build(name: str, gallery: np.ndarray, parameter: int) -> tuple[Any, dict[str, Any]]:
    d = gallery.shape[1]
    before = rss()
    start = time.perf_counter()
    training = 0.0
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
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 100, faiss.METRIC_L2)
        t0 = time.perf_counter()
        index.train(gallery)
        training = time.perf_counter() - t0
        index.nprobe = parameter
        index.add(gallery)
    else:
        raise ValueError(name)
    return index, {"build_seconds": time.perf_counter() - start,
                   "training_seconds": training,
                   "rss_delta_bytes": max(0, rss() - before)}


def measure_latency(index: Any, queries: np.ndarray) -> tuple[float, float]:
    index.search(queries[:20], TOP_K)
    vals = []
    for q in queries:
        ts = []
        for _ in range(TIMING_REPEATS):
            t0 = time.perf_counter_ns()
            index.search(q.reshape(1, -1), TOP_K)
            ts.append((time.perf_counter_ns() - t0) / 1e6)
        vals.append(float(np.median(ts)))
    return float(np.percentile(vals, 50)), float(np.percentile(vals, 95))


def neighbor_recall(exact_ids: np.ndarray, ids: np.ndarray) -> float:
    return float(np.mean([len(set(map(int, a)) & set(map(int, b))) / TOP_K
                          for a, b in zip(exact_ids, ids, strict=True)]))


def evaluate(index: Any, name: str, parameter: int, gallery: np.ndarray,
             queries: np.ndarray, targets: np.ndarray, exact_target_d: np.ndarray,
             exact_ids: np.ndarray, output_dir: Path, workload: str,
             meta: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    distances, ids = index.search(queries, TOP_K)
    distances = distances.astype(np.float64)
    ids = ids.astype(np.int64)
    cal_ids, test_ids = ids[:CAL_SIZE], ids[CAL_SIZE:]
    cal_dist, test_dist = distances[:CAL_SIZE], distances[CAL_SIZE:]
    cal_target = targets[:CAL_SIZE]
    test_target = targets[CAL_SIZE:]
    def target_scores(rows_i: np.ndarray, rows_d: np.ndarray, ts: np.ndarray) -> np.ndarray:
        out = []
        for ri, rd, t in zip(rows_i, rows_d, ts, strict=True):
            pos = np.flatnonzero(ri == int(t))
            out.append(float(rd[pos[0]]) if pos.size else float("inf"))
        return np.asarray(out, dtype=np.float64)
    cal_scores = target_scores(cal_ids, cal_dist, cal_target)
    threshold = qhigher(cal_scores, TARGET_COVERAGE)
    test_scores = target_scores(test_ids, test_dist, test_target)
    omissions = ~np.isfinite(test_scores)
    exclusions = (~omissions) & (test_scores > threshold)
    misses = omissions | exclusions
    sizes = np.sum(test_dist <= threshold, axis=1)
    cal_sizes = np.sum(cal_dist <= threshold, axis=1)
    cal_om = ~np.isfinite(cal_scores)
    cal_ex = (~cal_om) & (cal_scores > threshold)
    cal_coverage = float(1 - np.mean(cal_om | cal_ex))
    test_coverage = float(1 - np.mean(misses))
    p50, p95 = measure_latency(index, queries[CAL_SIZE:])
    row = {
        "workload": workload, "index": name, "parameter": parameter,
        "threshold_squared_l2": threshold, "calibration_coverage": cal_coverage,
        "test_coverage": test_coverage,
        "target_coverage_shortfall": max(0.0, TARGET_COVERAGE - test_coverage),
        "total_miss_rate": float(np.mean(misses)),
        "candidate_omission_rate": float(np.mean(omissions)),
        "threshold_exclusion_rate": float(np.mean(exclusions)),
        "decomposition_error": float(np.mean(misses) - np.mean(omissions) - np.mean(exclusions)),
        "target_recall_at_100": float(np.mean(np.isfinite(test_scores))),
        "neighbor_recall_at_100": neighbor_recall(exact_ids[CAL_SIZE:], test_ids),
        "calibration_mean_set_size": float(np.mean(cal_sizes)),
        "mean_set_size": float(np.mean(sizes)), "latency_p50_ms": p50,
        "latency_p95_ms": p95, "set_budget": SET_BUDGETS[workload],
        "latency_budget_ms": LATENCY_BUDGET_MS, **meta,
        "serialized_index_bytes": int((output_dir / f"{workload}_{name}_{parameter}.faiss").stat().st_size),
    }
    raw = []
    for j in range(TEST_SIZE):
        raw.append({"workload": workload, "index": name, "parameter": parameter,
                    "query_id": CAL_SIZE + j, "test_coverage": not bool(misses[j]),
                    "candidate_omission": bool(omissions[j]),
                    "threshold_exclusion": bool(exclusions[j]),
                    "set_size": int(sizes[j]), "target_score": float(test_scores[j])})
    return row, raw


def choose(rows: list[dict[str, Any]], workload: str) -> list[dict[str, Any]]:
    budget = SET_BUDGETS[workload]
    for r in rows:
        r["feasible"] = bool(r["calibration_coverage"] >= TARGET_COVERAGE and
                              r["calibration_mean_set_size"] <= budget and
                              r["latency_p95_ms"] <= LATENCY_BUDGET_MS)
    feasible = [r for r in rows if r["feasible"]]
    proposed = min(feasible, key=lambda r: (r["latency_p95_ms"], r["mean_set_size"], r["serialized_index_bytes"])) if feasible else None
    recall = max(rows, key=lambda r: (r["neighbor_recall_at_100"], r["latency_p95_ms"]))
    latency = min(rows, key=lambda r: (r["latency_p95_ms"], r["mean_set_size"]))
    fixed = next(r for r in rows if (r["index"], r["parameter"]) == ("hnsw", 16))
    selected = {"proposed_constraints": proposed, "neighbor_recall_only": recall,
                "latency_only": latency, "fixed_hnsw_ef16": fixed}
    out = []
    for strategy, chosen in selected.items():
        if chosen is None:
            out.append({"workload": workload, "strategy": strategy, "selection_feasible": False,
                        "selected_index": "NONE", "selected_parameter": "NONE"})
        else:
            out.append({"workload": workload, "strategy": strategy, "selection_feasible": bool(chosen["feasible"]),
                        "selected_index": chosen["index"], "selected_parameter": chosen["parameter"],
                        **{f"selected_{k}": v for k, v in chosen.items() if k not in {"workload", "index", "parameter"}}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sift", type=Path, required=True)
    ap.add_argument("--glove", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    faiss.omp_set_num_threads(1)
    np.random.seed(SEED)
    all_rows, selection_rows, raw_rows = [], [], []
    for workload, data_path in (("sift1m", args.sift), ("glove25", args.glove)):
        with h5py.File(data_path, "r") as h:
            gallery = np.asarray(h["train"][:10_000], dtype=np.float32)
            queries = np.asarray(h["test"][:CAL_SIZE + TEST_SIZE], dtype=np.float32)
        exact = faiss.IndexFlatL2(gallery.shape[1]); exact.add(gallery)
        exact_d, exact_ids = exact.search(queries, TOP_K)
        targets = exact_ids[:, 0].astype(np.int64)
        exact_target_d = exact_d[:, 0].astype(np.float64)
        rows = []
        for name, parameter in CONFIGS:
            index, meta = build(name, gallery, parameter)
            path = args.output / f"{workload}_{name}_{parameter}.faiss"
            faiss.write_index(index, str(path))
            row, raw = evaluate(index, name, parameter, gallery, queries, targets,
                                exact_target_d, exact_ids, args.output, workload, meta)
            rows.append(row); all_rows.append(row); raw_rows.extend(raw)
        selection_rows.extend(choose(rows, workload))
    def write(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    write(args.output / "CONFIGURATION_SELECTION_RESULTS.csv", all_rows)
    write(args.output / "BASELINE_COMPARISON.csv", selection_rows)
    write(args.output / "CONFIGURATION_SELECTION_QUERY_LEVEL.csv", raw_rows)
    (args.output / "CONFIGURATION_SELECTION_RUN.json").write_text(json.dumps({
        "seed": SEED, "faiss_version": faiss.__version__, "cpu_only": True,
        "calibration_size": CAL_SIZE, "test_size": TEST_SIZE, "top_k": TOP_K,
        "target_coverage": TARGET_COVERAGE, "set_budgets": SET_BUDGETS,
        "latency_budget_ms": LATENCY_BUDGET_MS, "configs": CONFIGS,
        "sift_sha256": sha256(args.sift), "glove_sha256": sha256(args.glove),
        "platform": platform.platform()}, indent=2), encoding="utf-8")
    print(json.dumps({"result_rows": len(all_rows), "baseline_rows": len(selection_rows), "query_rows": len(raw_rows)}, indent=2))


if __name__ == "__main__":
    main()
