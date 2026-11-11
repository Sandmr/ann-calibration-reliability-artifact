"""Regenerate the four manuscript figures from frozen CSV evidence only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


PASTEL = {
    "blue": "#8FBCE6",
    "pink": "#E8B4CB",
    "purple": "#B9A7D9",
    "green": "#A9D6B5",
}
COLORS = {"exact": PASTEL["blue"], "ann": PASTEL["pink"],
          "hnsw": PASTEL["purple"], "ivfflat": PASTEL["green"]}
FIGURE4_COLORS = {"hnsw": "#9A86C4", "ivfflat": "#86BE98"}


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.7,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def figure1(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.45))
    ax.set_xlim(0, 8.4)
    ax.set_ylim(0, 3.8)
    ax.axis("off")
    boxes = [
        (0.05, 1.45, 1.45, 0.78, "Exact target\n(Top-1)", "#DCECF8"),
        (1.95, 1.45, 1.55, 0.78, "ANN candidate\ngeneration", "#E2F0E5"),
        (3.95, 2.25, 1.55, 0.78, "Candidate\nomission", "#F1E7F5"),
        (3.95, 0.65, 1.55, 0.78, "Threshold\nexclusion", "#F7E5EE"),
        (6.0, 1.45, 1.55, 0.78, "Prediction set\n+ coverage", "#E8E0F3"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.035", linewidth=1,
                edgecolor="#6B6B6B", facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")
    arrows = [
        ((1.50, 1.84), (1.95, 1.84)),
        ((3.50, 1.84), (3.95, 2.64)),
        ((3.50, 1.84), (3.95, 1.04)),
        ((5.50, 2.64), (6.00, 1.84)),
        ((5.50, 1.04), (6.00, 1.84)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.05, "color": "#666666"})
    ax.text(4.78, 3.38, "coverage / set size / cost", ha="center", va="center", fontsize=8)
    save(fig, out / "figure1_failure_decomposition")


def figure2(transfer: pd.DataFrame, out: Path) -> None:
    d = transfer[transfer["calibration_condition"] == "exact"]
    d = d[d["index"] != "exact_flat"].copy()
    order = [("sift1m", "hnsw"), ("sift1m", "ivfflat"), ("glove25", "hnsw"), ("glove25", "ivfflat")]
    labels = ["SIFT-HNSW", "SIFT-IVF", "GloVe-HNSW", "GloVe-IVF"]
    rows = [d[(d["workload"] == w) & (d["index"] == i)].iloc[0] for w, i in order]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 2.75))
    width = 0.34
    ax.bar(x - width / 2, [r["target_recall_at_100"] for r in rows], width, label="TargetRecall@100", color=COLORS["exact"], edgecolor="#6B6B6B", linewidth=0.55, hatch="//")
    ax.bar(x + width / 2, [r["neighbor_recall_at_100"] for r in rows], width, label="NeighborRecall@100", color=COLORS["hnsw"], edgecolor="#6B6B6B", linewidth=0.55, hatch="\\\\")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Target recall and candidate-set overlap")
    ax.grid(axis="y", linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.7, columnspacing=1.0)
    save(fig, out / "figure2_recall_comparison")


def figure3(transfer: pd.DataFrame, out: Path) -> None:
    d = transfer[transfer["index"] != "exact_flat"].copy()
    order = [("sift1m", "hnsw"), ("sift1m", "ivfflat"), ("glove25", "hnsw"), ("glove25", "ivfflat")]
    labels = ["SIFT-HNSW", "SIFT-IVF", "GloVe-HNSW", "GloVe-IVF"]
    rows = [[d[(d["workload"] == w) & (d["index"] == i) & (d["calibration_condition"] == c)].iloc[0] for c in ("exact", "ann")] for w, i in order]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    for ax, field, xlabel in [(axes[0], "empirical_coverage", "Empirical coverage"), (axes[1], "mean_set_size", "Mean prediction-set size")]:
        exact = np.array([r[0][field] for r in rows], dtype=float)
        ann = np.array([r[1][field] for r in rows], dtype=float)
        ax.hlines(y, exact, ann, color="#A9A9A9", linewidth=1.0, zorder=1)
        ax.scatter(exact, y, s=34, color=COLORS["exact"], edgecolor="#5C7890", linewidth=0.45, label="Exact-calibrated", zorder=3)
        ax.scatter(ann, y, s=34, color=COLORS["ann"], edgecolor="#94657D", linewidth=0.45, label="ANN-calibrated", zorder=3)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", linestyle="-", zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_title("Coverage")
    axes[1].set_title("Set efficiency")
    axes[0].axvline(0.9, ls="--", color="#777777", lw=0.85)
    axes[0].text(0.9, 1.02, "target 0.90", transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="#666666")
    handles, labels_legend = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_legend, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.68, 1.01), handletextpad=0.35, columnspacing=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out / "figure3_calibration_transfer")


def figure4(scale: pd.DataFrame, out: Path) -> None:
    d = scale[(scale["calibration_condition"] == "exact") & (scale["evaluation_condition"] == "ann")]
    d = d[d["index"].isin(["hnsw", "ivfflat"])].copy()
    d = d[((d["index"] == "hnsw") & (d["parameter"] == 32)) | ((d["index"] == "ivfflat") & (d["parameter"] == 8))]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75))
    ticks = [10000, 50000, 100000]
    for name, label, color in [("hnsw", "HNSW", FIGURE4_COLORS["hnsw"]), ("ivfflat", "IVFFlat", FIGURE4_COLORS["ivfflat"])]:
        q = d[d["index"] == name].sort_values("gallery_size")
        axes[0].plot(q["gallery_size"], q["empirical_coverage"], marker="o", markersize=4.5, linewidth=1.6, label=label, color=color, markeredgecolor="#666666", markeredgewidth=0.35)
        axes[1].plot(q["gallery_size"], q["latency_p50_ms"], marker="o", markersize=4.5, linewidth=1.6, label=label, color=color, markeredgecolor="#666666", markeredgewidth=0.35)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ticks, ["10k", "50k", "100k"])
        ax.set_xlabel("Gallery size")
        ax.legend(frameon=False, loc="best")
        ax.grid(axis="both", linestyle="-", zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("p50 query latency (ms)")
    axes[0].set_title("Scale-dependent coverage")
    axes[1].set_title("Scale-dependent query cost")
    fig.tight_layout()
    save(fig, out / "figure4_gallery_scale_cost")


def tables(transfer: pd.DataFrame, out: Path) -> None:
    """Write the compact LaTeX tables derived from frozen CSV values."""
    out.mkdir(parents=True, exist_ok=True)

    def write_table(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
        cols = list(frame.columns)
        lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline"]
        lines.append(" & ".join(cols) + " \\\\")
        lines.append("\\hline")
        for _, row in frame.iterrows():
            values = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    values.append(f"{value:.3f}")
                else:
                    values.append(str(value).replace("_", "\\_"))
            lines.append(" & ".join(values) + " \\\\")
        lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
        path.write_text("\n".join(lines), encoding="utf-8")

    protocol = pd.DataFrame([
        ["SIFT1M", 128, 1000, "10k", "HNSW ef=16; IVFFlat nprobe=8", "500/500"],
        ["GloVe25", 25, 1000, "10k", "HNSW ef=16; IVFFlat nprobe=8", "500/500"],
    ], columns=["Workload", "Dim.", "Queries", "Gallery", "Indexes", "Calibration/Test"])
    write_table(protocol, out / "table1_protocol.tex", "Frozen workload and protocol configuration.", "tab:protocol")
    main = transfer[transfer["calibration_condition"] == "exact"].copy()
    main = main[["workload", "index", "target_recall_at_100", "neighbor_recall_at_100", "empirical_coverage", "mean_set_size", "latency_p50_ms", "latency_p95_ms"]]
    write_table(main, out / "table2_main_results.tex", "Main frozen ANN retrieval and reliability results.", "tab:main")
    tr = transfer[transfer["index"] != "exact_flat"][["workload", "index", "calibration_condition", "threshold_squared_l2", "empirical_coverage", "mean_set_size", "candidate_omission_rate", "threshold_exclusion_rate"]]
    write_table(tr, out / "table3_transfer_results.tex", "Calibration transfer results; fixed mean-budget analysis is reported separately.", "tab:transfer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, required=True)
    args = parser.parse_args()
    setup()
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
    tables(transfer, table_out)


if __name__ == "__main__":
    main()
