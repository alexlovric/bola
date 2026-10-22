#!/usr/bin/env python3
"""Plot current BOLA and third-party Criterion exports. Requires Matplotlib."""

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
LABELS = {
    "BOLA": "BOLA",
    "UpdatedBOLA": "BOLA",
    "MatrixMultiply": "matrixmultiply",
    "CBLAS": "OpenBLAS",
    "Lapack": "LAPACK + OpenBLAS",
}
STYLES = {
    "BOLA": ("#2563eb", "o", "-"),
    "matrixmultiply": ("#64748b", "s", "--"),
    "OpenBLAS": ("#0d9488", "^", "-"),
    "LAPACK + OpenBLAS": ("#c47b12", "D", "--"),
}


def load_results(paths):
    data = {}
    for path in paths:
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                label = LABELS.get(row["implementation"])
                if label is None or row["bench"] not in ("gemm", "getrf"):
                    continue
                if not row["case"].isdigit() or int(row["case"]) not in SIZES:
                    continue
                threads = int(row["threads"])
                if threads not in (4, 20):
                    continue
                key = (row["bench"], threads, int(row["case"]), label)
                seconds = float(row["mean_ms"]) / 1000
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError(f"Invalid timing in {path}: {key}")
                if key in data:
                    raise ValueError(f"Duplicate measurement: {key}")
                data[key] = seconds
    return data


def time_label(seconds, _position):
    if seconds < 1e-3:
        return f"{seconds * 1e6:g} µs"
    if seconds < 1:
        return f"{seconds * 1e3:g} ms"
    return f"{seconds:g} s"


def plot(data, output, preview):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "svg.hashsalt": "bola-benchmarks",
        "axes.labelcolor": "#475569",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.16, top=0.85, hspace=0.38, wspace=0.22)
    legend = {}
    for row, (bench, title, labels) in enumerate([
        ("gemm", "GEMM", ("BOLA", "matrixmultiply", "OpenBLAS")),
        ("getrf", "LU factorization", ("BOLA", "LAPACK + OpenBLAS")),
    ]):
        for col, threads in enumerate((4, 20)):
            ax = axes[row, col]
            for label in labels:
                values = [data[(bench, threads, size, label)] for size in SIZES]
                color, marker, linestyle = STYLES[label]
                line, = ax.plot(
                    SIZES, values, label=label, color=color, marker=marker,
                    linestyle=linestyle, linewidth=2.2 if label == "BOLA" else 1.6,
                    markersize=4.5, clip_on=False,
                )
                legend.setdefault(label, line)
            ax.set_title(f"{title} · {threads} threads", loc="left", fontsize=12, fontweight="bold", color="#0f172a", pad=12)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlim(28, 4600)
            ax.set_ylim(8e-7, 1)
            ax.xaxis.set_major_locator(FixedLocator(SIZES))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda size, _: str(int(size))))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_locator(FixedLocator([10.0 ** power for power in range(-6, 1)]))
            ax.yaxis.set_major_formatter(FuncFormatter(time_label))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("bottom", "left"):
                ax.spines[spine].set_color("#cbd5e1")
            ax.tick_params(length=0, pad=7)
            if row == 1:
                ax.set_xlabel("Square matrix size", labelpad=10)
            if col == 0:
                ax.set_ylabel("Mean time · log scale", labelpad=10)

    fig.suptitle("BOLA on Intel Core i5-14600K", x=0.085, y=0.975, ha="left", fontsize=19, fontweight="bold", color="#0f172a")
    fig.text(0.085, 0.925, "CachyOS Linux · FP64 · 32–4096 square matrices · lower is better", color="#475569", fontsize=11)
    fig.legend(legend.values(), legend.keys(), loc="lower center", bbox_to_anchor=(0.53, 0.05), ncol=4, frameon=False)
    fig.text(0.53, 0.018, "4 threads: separate P-cores. 20 threads: all logical CPUs. matrixmultiply is capped at 4 threads.", ha="center", color="#64748b", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={
        "Date": None,
        "Title": "BOLA GEMM and LU benchmark results",
        "Description": "Current BOLA versus third-party libraries; Criterion mean latency on CachyOS Linux, Intel Core i5-14600K, 2026-10-22. Both axes use logarithmic scales.",
    })
    if preview:
        fig.savefig(preview, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, default=[
        ROOT / "docs/results/criterion-packed-2026-10-22.csv",
        ROOT / "docs/results/criterion-small-2026-10-22.csv",
    ])
    parser.add_argument("--output", type=Path, default=ROOT / "assets/benchmarks.svg")
    parser.add_argument("--preview", type=Path, help="Also render a PNG for visual inspection")
    args = parser.parse_args()
    plot(load_results(args.results), args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
