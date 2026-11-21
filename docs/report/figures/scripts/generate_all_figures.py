"""
Generate publication-quality figures for the From-Scratch Explainer & Literature Report
on Differometor optimizations (START Hack Munich 2026).

Outputs both 300 DPI PNG (for visual inspection) and vector PDF (for LaTeX).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Global publication styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# Okabe-Ito colorblind-friendly palette
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_VERM = "#D55E00"
C_GREEN = "#009E73"
C_SKY = "#56B4E9"
C_PURPLE = "#CC79A7"
C_GREY = "#7F7F7F"
C_DARK = "#222222"

OUT_DIR = "/work/docs/report/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"Saved {name}.png and {name}.pdf")


# ---------------------------------------------------------------------------
# Figure 1: Sparsity Structure & Algorithmic Work Reduction
# ---------------------------------------------------------------------------
def make_fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.0, 1.35]})

    # Panel A: Matrix density (nonzero vs zero %)
    systems = ["Carrier\n(286 x 286)", "Signal\n(751 x 751)", "Noise N\n(751 x 751)"]
    nnz_counts = [750, 2700, 760]
    total_entries = [286**2, 751**2, 751**2]
    densities = [100.0 * nz / tot for nz, tot in zip(nnz_counts, total_entries)]
    zeros_pct = [100.0 - d for d in densities]

    x = np.arange(len(systems))
    w = 0.55
    bars_nz = ax1.bar(x, densities, width=w, color=C_BLUE, label="Nonzero entries (%)")
    ax1.bar(x, zeros_pct, bottom=densities, width=w, color="#E5E7EB", edgecolor="#CBD5E1", label="Zero entries (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(systems)
    ax1.set_ylabel("Matrix Entries (%)")
    ax1.set_ylim(0, 112)
    ax1.set_title("(a) System Matrix Sparsity (uifo_800_3000)", pad=8, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    for idx, (nz, d) in enumerate(zip(nnz_counts, densities)):
        ax1.text(
            idx, d + 4.5,
            f"~{nz:,} nnz\n({d:.2f}%)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C_BLUE
        )

    # Panel B: Log-scale work reduction across 3 metrics
    metrics = [
        "751x751 LU\nFactorizations",
        "VJP Matrix\nMultiplies",
        "Quantum Noise\nMatvec Ops",
    ]
    before_vals = [606, 56_964_101, 57_000_000]
    after_vals = [101, 272_700, 76_760]
    speedups = ["6.0x", "209x", "742x"]

    y = np.arange(len(metrics))
    h = 0.34
    ax2.barh(y + h/2, before_vals, height=h, color=C_VERM, label="Baseline (Dense / Uncached)")
    ax2.barh(y - h/2, after_vals, height=h, color=C_GREEN, label="Combined (Verdicts 1-4)")

    ax2.set_xscale("log")
    ax2.set_xlim(1e1, 3e9)
    ax2.set_yticks(y)
    ax2.set_yticklabels(metrics)
    ax2.invert_yaxis()
    ax2.set_xlabel("Operations / Factorizations per Forward + Backward Pass (log scale)")
    ax2.set_title("(b) Algorithmic Work Reduction", pad=8, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)

    for idx, (bv, av, sp) in enumerate(zip(before_vals, after_vals, speedups)):
        ax2.text(bv * 1.25, idx + h/2, f"{bv:,}", va="center", ha="left", fontsize=8, color=C_VERM)
        ax2.text(av * 1.25, idx - h/2, f"{av:,} ({sp} less)", va="center", ha="left", fontsize=8, fontweight="bold", color=C_GREEN)

    fig.tight_layout(w_pad=2.5)
    save_fig(fig, "fig1_matrix_sparsity_and_work")


# ---------------------------------------------------------------------------
# Figure 2: Benchmark Timing Comparison (Baseline vs Final Combined Solution)
# ---------------------------------------------------------------------------
def make_fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))

    # Exact numbers from benchmarks/cpu-baseline.json and results/combined-final.json
    # Warm runtime medians:
    # Baseline: forward=13.1435, sensitivity=0.1027, gradient=4.4106, total_median=17.6787
    # Combined (combined-final.json): forward=0.2636, sensitivity=0.00064, gradient=0.1494, total_median=0.4110
    configs = ["Baseline\n(Xeon 8468, 1 CPU)", "Final Combined\n(i7-1355U WSL2, 1 CPU)"]
    fwd = np.array([13.1435, 0.2636])
    grad = np.array([4.4106, 0.1494])
    sens = np.array([0.1027, 0.00064])

    x = np.arange(len(configs))
    w = 0.48
    ax1.bar(x, fwd, width=w, color=C_BLUE, label="Forward Simulation")
    ax1.bar(x, grad, bottom=fwd, width=w, color=C_ORANGE, label="Reverse-Mode Gradient")
    ax1.bar(x, sens, bottom=fwd + grad, width=w, color=C_PURPLE, label="Sensitivity & Loss")

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.set_ylabel("Median Warm Execution Time (s)")
    ax1.set_ylim(0, 21.0)
    ax1.set_title("(a) Warm Execution Time (43.0x Faster)", pad=8, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    totals_warm = [17.679, 0.411]
    ax1.text(0, totals_warm[0] + 0.45, "17.68 s\n(fwd 13.14 s | grad 4.41 s)", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=C_DARK)
    ax1.text(1, totals_warm[1] + 0.45, "0.411 s (43.0x)\n(fwd 0.264 s | grad 0.149 s)", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=C_GREEN)

    # Panel B: Full First-Run End-to-End breakdown
    # Baseline: setup=3.661, compile=22.918, first_exec=17.625, total=44.204
    # Combined (combined-final.json): setup=1.852, compile=1.516, first_exec=0.605, total=3.973
    setup = np.array([3.661, 1.852])
    comp = np.array([22.918, 1.516])
    exec1 = np.array([17.625, 0.605])

    ax2.bar(x, setup, width=w, color=C_GREY, label="Imports & Setup")
    ax2.bar(x, comp, bottom=setup, width=w, color=C_SKY, label="XLA Trace / Lower / Compile")
    ax2.bar(x, exec1, bottom=setup + comp, width=w, color=C_GREEN, label="First-Run Execute")

    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.set_ylabel("First-Run End-to-End Time (s)")
    ax2.set_ylim(0, 52.0)
    ax2.set_title("(b) First-Run End-to-End Breakdown (11.1x Faster)", pad=8, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    ax2.text(0, 44.204 + 0.8, "44.20 s total\n(40% exec / 52% compile)", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=C_DARK)
    ax2.text(1, 3.973 + 0.8, "3.97 s total (11.1x)\n(1.52 s compile / 0.61 s exec)", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=C_BLUE)

    fig.tight_layout(w_pad=2.5)
    save_fig(fig, "fig2_benchmark_comparison")


# ---------------------------------------------------------------------------
# Figure 3: 10-Year Eidothea Literature Landscape (2016-2026)
# ---------------------------------------------------------------------------
def make_fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4), gridspec_kw={"width_ratios": [1.05, 1.25]})

    # Panel A: 10-year publication trajectory from eidothea_literature_verdicts.md
    years = np.arange(2016, 2027)
    counts = [56, 48, 61, 66, 76, 79, 80, 77, 105, 144, 275]

    bars = ax1.bar(years, counts, color=C_SKY, edgecolor=C_BLUE, linewidth=0.8)
    bars[-1].set_color(C_BLUE)
    bars[-2].set_color(C_BLUE)
    ax1.plot(years, counts, color=C_DARK, marker="o", markersize=4, linewidth=1.3)

    ax1.set_xticks(years)
    ax1.set_xticklabels([str(y) for y in years], rotation=35, ha="right")
    ax1.set_ylabel("Unique Papers in Catalog")
    ax1.set_ylim(0, 325)
    ax1.set_title("(a) 10-Year Eidothea Scan (1,067 Papers)", pad=8, fontweight="bold")

    # Key milestone annotations
    milestones = [
        (2016, 56, "Basker\nSparse LU"),
        (2018, 61, "JAX"),
        (2021, 79, "JAXOpt\nImplicit AD"),
        (2023, 77, "Urania /\nUIFO"),
        (2024, 105, "Linear AD &\nExpr. Growth"),
        (2025, 144, "Differometor"),
        (2026, 275, "RapidDOC\n& JAX-AMG"),
    ]
    for yr, val, txt in milestones:
        offset = 12 if yr != 2026 else 10
        ax1.annotate(
            txt, xy=(yr, val), xytext=(yr, val + offset),
            ha="center", va="bottom", fontsize=7.2, color=C_DARK,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6)
        )

    # Panel B: 13 Subfields grouped by 4 taxonomy clusters
    subfields = [
        ("jax_xla (D)", 187, C_PURPLE),
        ("quantum_noise (A)", 174, C_BLUE),
        ("autodiff_opt (D)", 165, C_PURPLE),
        ("sideband_mod (A)", 155, C_BLUE),
        ("gw_detector (A)", 144, C_BLUE),
        ("ifo_sim (A)", 107, C_BLUE),
        ("low_rank_update (B)", 99, C_GREEN),
        ("freq_sweep (B)", 83, C_GREEN),
        ("implicit_diff (C)", 78, C_ORANGE),
        ("sparse_lu (B)", 52, C_GREEN),
        ("batched_sparse (B)", 15, C_GREEN),
        ("diff_optics (C)", 12, C_ORANGE),
        ("adj_quantum_cov (C)", 7, C_ORANGE),
    ]
    labels = [s[0] for s in subfields]
    vals = [s[1] for s in subfields]
    cols = [s[2] for s in subfields]

    y = np.arange(len(labels))
    ax2.barh(y, vals, color=cols, height=0.68)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Papers per Subfield")
    ax2.set_xlim(0, 220)
    ax2.set_title("(b) Subfields by Research Cluster", pad=8, fontweight="bold")

    for idx, v in enumerate(vals):
        ax2.text(v + 3, idx, str(v), va="center", ha="left", fontsize=7.8, color=C_DARK)

    # Legend for clusters
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=C_BLUE, label="A: Interferometry"),
        Patch(facecolor=C_GREEN, label="B: Sparse Solvers"),
        Patch(facecolor=C_ORANGE, label="C: Autodiff & Adjoint"),
        Patch(facecolor=C_PURPLE, label="D: JAX & Compiler"),
    ]
    ax2.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.8)

    fig.tight_layout(w_pad=2.2)
    save_fig(fig, "fig3_eidothea_literature_landscape")


# ---------------------------------------------------------------------------
# Figure 4: Before vs After Simulation & Adjoint Architecture
# ---------------------------------------------------------------------------
def make_fig4():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    def draw_box(x, y, w, h, text, fc, ec, tc=C_DARK, fs=8.3, bold=False):
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, linewidth=1.3
        )
        ax.add_patch(box)
        ax.text(
            x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal"
        )

    def draw_arrow(x1, y1, x2, y2, color=C_DARK):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.5", color=color, lw=1.3)
        )

    # Row labels
    ax.text(1.0, 42.5, "BEFORE: Baseline Dense Pipeline (17.68 s warm | 44.20 s first-run | 606 dense LU solves)", fontsize=9.5, fontweight="bold", color=C_VERM)
    ax.text(1.0, 20.5, "AFTER: Final Combined Pipeline (0.411 s warm | 3.97 s first-run | 1 block-diag SuperLU)", fontsize=9.5, fontweight="bold", color=C_GREEN)

    # BEFORE row (y = 28..39)
    draw_box(1.5, 29, 17, 10, "19-way\nlax.switch\n(all functions)", "#FEE2E2", C_VERM)
    draw_arrow(19.2, 34, 21.8, 34, C_VERM)
    draw_box(22.5, 29, 18, 10, "Dense (101, 751, 751)\nSignal & Noise\n(99.5% zeros)", "#FEE2E2", C_VERM)
    draw_arrow(41.2, 34, 43.8, 34, C_VERM)
    draw_box(44.5, 29, 19, 10, "3x Dense LU\nSolve 1: A x = b\nSolve 2: A^H w = c\nSolve 3: A v = N w", "#FEE2E2", C_VERM)
    draw_arrow(64.2, 34, 66.8, 34, C_VERM)
    draw_box(67.5, 29, 30.5, 10, "Backward Pass (VJP)\n3x More Dense LU Solves\n+ Dense Outer Products (57M ops)", "#FEE2E2", C_VERM)

    # Divider
    ax.plot([1, 99], [24.2, 24.2], linestyle="--", color="#CBD5E1", lw=1.0)

    # AFTER row (y = 7..17)
    draw_box(1.5, 6, 18.5, 11, "Single custom_vjp\nNumPy jaxpr + JVP\ntranspose (1.52 s XLA)", "#E0F2FE", C_BLUE)
    draw_arrow(20.7, 11.5, 23.0, 11.5, C_BLUE)
    draw_box(23.7, 6, 18.0, 11, "Grouped Sparse COO\nNo lax.switch; sweep\nhoist (2,537 nnz)", "#E0F2FE", C_BLUE)
    draw_arrow(42.4, 11.5, 44.7, 11.5, C_BLUE)
    draw_box(45.4, 6, 20.5, 11, "Block-Diag _superlu\n101 freqs -> 1 call\n+ 1-step refinement\n(75,851x75,851)", "#DCFCE7", C_GREEN, bold=True)
    draw_arrow(66.6, 11.5, 68.9, 11.5, C_GREEN)
    draw_box(69.6, 6, 28.4, 11, "Adjoint + Sparse VJP\nRe(w^H N w) at 272 nnz\nReuse LU (trans='T'/'N')\nJAX fallback if needed", "#DCFCE7", C_GREEN)

    fig.tight_layout()
    save_fig(fig, "fig4_optimization_pipeline")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print("All 4 figures generated successfully.")
