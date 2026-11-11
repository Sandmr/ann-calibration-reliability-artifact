"""Paired bootstrap uncertainty from the completed selection test queries."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('--query', type=Path, required=True); ap.add_argument('--baseline', type=Path, required=True); ap.add_argument('--output', type=Path, required=True); args=ap.parse_args()
    import pandas as pd
    q=pd.read_csv(args.query); b=pd.read_csv(args.baseline)
    rng=np.random.default_rng(20271111); out=[]
    for workload in b.workload.unique():
        fixed=b[(b.workload==workload)&(b.strategy=='fixed_hnsw_ef16')].iloc[0]
        proposed=b[(b.workload==workload)&(b.strategy=='proposed_constraints')].iloc[0]
        for metric, col, direction in [('coverage','test_coverage',1),('set_size','set_size',1),('omission','candidate_omission',1),('threshold_exclusion','threshold_exclusion',1)]:
            a=q[(q.workload==workload)&(q['index']==proposed.selected_index)&(q.parameter==int(proposed.selected_parameter))].sort_values('query_id')
            z=q[(q.workload==workload)&(q['index']=='hnsw')&(q.parameter==16)].sort_values('query_id')
            av=a[col].astype(float).to_numpy(); zv=z[col].astype(float).to_numpy();
            if metric=='coverage': av=av; zv=zv
            dif=[]
            for _ in range(2000):
                ix=rng.integers(0,len(av),len(av)); dif.append(float(np.mean(av[ix])-np.mean(zv[ix])))
            out.append({'workload':workload,'comparison':'proposed_minus_fixed_hnsw16','metric':metric,'point_estimate':float(np.mean(av)-np.mean(zv))})
            out[-1].update({'ci95_low':float(np.percentile(dif,2.5)),'ci95_high':float(np.percentile(dif,97.5)),'n_queries':len(av),'bootstrap_reps':2000,'proposed_index':proposed.selected_index,'proposed_parameter':int(proposed.selected_parameter)})
    with args.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f'rows={len(out)}')

if __name__=='__main__': main()
