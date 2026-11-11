"""Repeated calibration/test splits for the configuration-selection audit."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import faiss
import h5py
import numpy as np

SEED = 20271111
CAL_SIZE = 500
TOP_K = 100
TARGET = 0.90
SET_BUDGETS = {"sift1m": 50.0, "glove25": 120.0}
LATENCY_BUDGET = 0.10
CONFIGS = [("exact_flat", 0), ("hnsw", 8), ("hnsw", 16), ("hnsw", 32), ("hnsw", 64),
           ("ivfflat", 1), ("ivfflat", 4), ("ivfflat", 8), ("ivfflat", 16), ("ivfflat", 32)]


def qhigher(x: np.ndarray) -> float:
    y = np.sort(np.asarray(x, dtype=np.float64))
    return float(y[min(int(np.ceil((len(y) + 1) * TARGET)) - 1, len(y) - 1)])


def choose(metric: dict[tuple[str, int], dict], workload: str) -> dict[str, tuple[str, int] | None]:
    feasible = [k for k, v in metric.items() if v["calibration_coverage"] >= TARGET and
                v["calibration_mean_set_size"] <= SET_BUDGETS[workload] and
                v["latency_p95_ms"] <= LATENCY_BUDGET]
    proposed = min(feasible, key=lambda k: (metric[k]["latency_p95_ms"], metric[k]["mean_set_size"])) if feasible else None
    recall = max(metric, key=lambda k: (metric[k]["neighbor_recall"], metric[k]["latency_p95_ms"]))
    latency = min(metric, key=lambda k: (metric[k]["latency_p95_ms"], metric[k]["mean_set_size"]))
    return {"proposed_constraints": proposed, "neighbor_recall_only": recall,
            "latency_only": latency, "fixed_hnsw_ef16": ("hnsw", 16)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sift", type=Path, required=True)
    ap.add_argument("--glove", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    lat = {}
    import pandas as pd
    prior = pd.read_csv(args.metrics)
    for _, r in prior.iterrows(): lat[(r.workload, r["index"], int(r.parameter))] = float(r.latency_p95_ms)
    rng = np.random.default_rng(SEED)
    rows = []
    for workload, data_path in (("sift1m", args.sift), ("glove25", args.glove)):
        with h5py.File(data_path, "r") as h:
            gallery = np.asarray(h["train"][:10_000], dtype=np.float32)
            queries = np.asarray(h["test"][:1000], dtype=np.float32)
        exact = faiss.IndexFlatL2(gallery.shape[1]); exact.add(gallery)
        exact_d, exact_ids = exact.search(queries, TOP_K)
        targets = exact_ids[:, 0].astype(np.int64)
        outputs = {}
        for name, p in CONFIGS:
            idx = faiss.read_index(str(args.index_dir / f"{workload}_{name}_{p}.faiss"))
            d, ids = idx.search(queries, TOP_K)
            outputs[(name, p)] = (d.astype(float), ids.astype(int))
        for split in range(5):
            perm = rng.permutation(len(queries)); cal, test = perm[:CAL_SIZE], perm[CAL_SIZE:]
            metric = {}
            for name, p in CONFIGS:
                d, ids = outputs[(name, p)]
                def scores(ix):
                    vals=[]
                    for q in ix:
                        pos=np.flatnonzero(ids[q] == targets[q]); vals.append(float(d[q, pos[0]]) if pos.size else np.inf)
                    return np.asarray(vals)
                cs=scores(cal); ts=scores(test); thr=qhigher(cs)
                com=~np.isfinite(ts); tex=(~com)&(ts>thr); miss=com|tex
                ccom=~np.isfinite(cs); ctex=(~ccom)&(cs>thr)
                sizes=np.sum(d[test] <= thr, axis=1); csizes=np.sum(d[cal] <= thr, axis=1)
                nr=float(np.mean([len(set(map(int, exact_ids[q])) & set(map(int, ids[q]))) / TOP_K for q in test]))
                metric[(name,p)]={"calibration_coverage":float(1-np.mean(ccom|ctex)),"calibration_mean_set_size":float(np.mean(csizes)),"mean_set_size":float(np.mean(sizes)),"latency_p95_ms":lat[(workload,name,p)],"neighbor_recall":nr}
                metric[(name,p)].update({"test_coverage":float(1-np.mean(miss)),"omission":float(np.mean(com)),"threshold_exclusion":float(np.mean(tex))})
            selected=choose(metric, workload)
            for strategy, key in selected.items():
                base={"workload":workload,"split":split,"strategy":strategy,"feasible":key is not None}
                if key is None: base.update({"index":"NONE","parameter":"NONE"})
                else: base.update({"index":key[0],"parameter":key[1],**metric[key]})
                rows.append(base)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"rows={len(rows)}")


if __name__ == "__main__": main()
