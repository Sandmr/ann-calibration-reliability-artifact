"""Generate frozen ICDE figures and tables from recorded CSV results only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


COLORS = {"Exact Flat": "#000000", "HNSW": "#0072B2", "IVFFlat": "#D55E00", "Exact-calibrated": "#009E73", "ANN-calibrated": "#CC79A7"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def figure1(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.2, 1.55, 1.7, 0.9, "Exact target\n(Top-1)", "#E6F2F8"),
        (2.55, 1.55, 1.8, 0.9, "ANN candidate\ngeneration", "#E8F5E9"),
        (5.0, 2.35, 1.8, 0.9, "Candidate\nomission", "#FDECEC"),
        (5.0, 0.75, 1.8, 0.9, "Threshold\nexclusion", "#FFF4D6"),
        (7.5, 1.55, 1.8, 0.9, "Prediction set\n+ coverage", "#EFE8F7"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", linewidth=1, edgecolor="#444444", facecolor=color))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")
    arrows = [((1.9, 2.0), (2.55, 2.0)), ((4.35, 2.0), (5.0, 2.8)), ((4.35, 2.0), (5.0, 1.2)), ((6.8, 2.8), (7.5, 2.0)), ((6.8, 1.2), (7.5, 2.0))]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#555555"})
    ax.text(5.9, 3.65, "coverage / set-size / cost", ha="center", va="center", fontsize=8)
    save(fig, out / "figure1_failure_decomposition")


def figure2(df: pd.DataFrame, out: Path) -> None:
    d = df[df["calibration_condition"].isin(["exact", "ann"]) & (df["index"] != "exact_flat")].copy()
    d["method"] = d["calibration_condition"].map({"exact": "Exact-calibrated", "ann": "ANN-calibrated"})
    d["label"] = d["workload"].str.upper() + "\n" + d["index"].str.upper()
    labels = list(d[d["calibration_condition"] == "exact"]["label"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
    x = np.arange(len(labels))
    width = 0.36
    for offset, method in [(-width / 2, "Exact-calibrated"), (width / 2, "ANN-calibrated")]:
        vals = [d[(d["label"] == label) & (d["method"] == method)]["empirical_coverage"].iloc[0] for label in labels]
        axes[0].bar(x + offset, vals, width, label=method, color=COLORS[method])
        vals2 = [d[(d["label"] == label) & (d["method"] == method)]["mean_set_size"].iloc[0] for label in labels]
        axes[1].bar(x + offset, vals2, width, label=method, color=COLORS[method])
    axes[0].axhline(0.9, ls="--", color="#666666", lw=0.8)
    axes[0].text(0.98, 0.905, "target 0.90", transform=axes[0].get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color="#555555")
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("Mean prediction-set size")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    axes[0].set_title("Coverage")
    axes[1].set_title("Efficiency cost")
    fig.tight_layout()
    save(fig, out / "figure2_calibration_transfer")


def figure3(df: pd.DataFrame, out: Path) -> None:
    d = df[df["calibration_condition"] == "exact"].copy()
    d = d[d["index"] != "exact_flat"]
    d["label"] = d["workload"].str.upper() + "\n" + d["index"].str.upper()
    labels = list(d["label"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.bar(x - 0.18, d["target_recall_at_100"], 0.36, label="TargetRecall@100", color="#0072B2")
    ax.bar(x + 0.18, d["neighbor_recall_at_100"], 0.36, label="NeighborRecall@100", color="#D55E00")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Target recall is not candidate-set overlap")
    ax.legend(frameon=False, ncol=2)
    save(fig, out / "figure3_recall_comparison")


def figure4(scale: pd.DataFrame, out: Path) -> None:
    d = scale[(scale["calibration_condition"] == "exact") & (scale["evaluation_condition"] == "ann")]
    d = d[d["index"].isin(["hnsw", "ivfflat"])].copy()
    d = d[((d["index"] == "hnsw") & (d["parameter"] == 32)) | ((d["index"] == "ivfflat") & (d["parameter"] == 8))]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for name, label, color in [("hnsw", "HNSW", COLORS["HNSW"]), ("ivfflat", "IVFFlat", COLORS["IVFFlat"])]:
        q = d[d["index"] == name].sort_values("gallery_size")
        axes[0].plot(q["gallery_size"], q["empirical_coverage"], marker="o", label=label, color=color)
        axes[1].plot(q["gallery_size"], q["latency_p50_ms"], marker="o", label=label, color=color)
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("p50 query latency (ms)")
    for ax in axes:
        ax.set_xlabel("Gallery size")
        ax.set_xscale("log")
        ax.legend(frameon=False)
    axes[0].set_title("Scale-dependent coverage")
    axes[1].set_title("Scale-dependent query cost")
    fig.tight_layout()
    save(fig, out / "figure4_gallery_scale_cost")


def tables(transfer: pd.DataFrame, scale: pd.DataFrame, out: Path) -> None:
    def latex_table(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
        cols = list(frame.columns)
        lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline"]
        lines.append(" & ".join(cols) + " \\\\")
        lines.append("\\hline")
        for _, row in frame.iterrows():
            values = []
            for value in row:
                values.append(f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value).replace("_", "\\_"))
            lines.append(" & ".join(values) + " \\\\")
        lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
        path.write_text("\n".join(lines), encoding="utf-8")

    protocol = pd.DataFrame([
        ["SIFT1M", 128, 1000, "10k", "HNSW ef=16; IVFFlat nprobe=8", "500/500"],
        ["GloVe25", 25, 1000, "10k", "HNSW ef=16; IVFFlat nprobe=8", "500/500"],
    ], columns=["Workload", "Dim.", "Queries", "Gallery", "Indexes", "Calibration/Test"])
    latex_table(protocol, out / "table1_protocol.tex", "Frozen workload and protocol configuration.", "tab:protocol")
    main = transfer[(transfer["calibration_condition"] == "exact")].copy()
    main = main[["workload", "index", "target_recall_at_100", "neighbor_recall_at_100", "empirical_coverage", "mean_set_size", "latency_p50_ms", "latency_p95_ms"]]
    latex_table(main, out / "table2_main_results.tex", "Main frozen ANN retrieval and reliability results.", "tab:main")
    tr = transfer[transfer["index"] != "exact_flat"][["workload", "index", "calibration_condition", "threshold_squared_l2", "empirical_coverage", "mean_set_size", "candidate_omission_rate", "threshold_exclusion_rate"]]
    latex_table(tr, out / "table3_transfer_results.tex", "Calibration transfer results; fixed mean-budget analysis is reported separately.", "tab:transfer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, required=True)
    args = parser.parse_args()
    style()
    out = args.paper_root / "figures"
    table_out = args.paper_root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table_out.mkdir(parents=True, exist_ok=True)
    transfer = pd.read_csv(args.experiment_root / "results" / "stage4" / "CALIBRATION_TRANSFER_EXTENDED.csv")
    scale = pd.read_csv(args.experiment_root / "results" / "stage3" / "GALLERY_SCALE_RESULTS.csv")
    figure1(out)
    figure2(transfer, out)
    figure3(transfer, out)
    figure4(scale, out)
    tables(transfer, scale, table_out)
    print("generated", sorted(p.name for p in out.glob("figure*")))


if __name__ == "__main__":
    main()
