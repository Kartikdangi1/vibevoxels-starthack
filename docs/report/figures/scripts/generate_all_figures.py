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
from matplotlib.patches import FancyBboxPatch, Patch

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

OUT_DIR = os.environ.get("FIG_OUT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

    # Panel A: Matrix density (measured nonzeros on uifo_800_3000)
    systems = ["Carrier\n(286 x 286)", "Signal\n(751 x 751)", "Noise N\n(751 x 751)"]
    nnz_counts = [680, 2537, 272]
    total_entries = [286**2, 751**2, 751**2]
    densities = [100.0 * nz / tot for nz, tot in zip(nnz_counts, total_entries)]
    zeros_pct = [100.0 - d for d in densities]

    x = np.arange(len(systems))
    w = 0.55
    ax1.bar(x, densities, width=w, color=C_BLUE, label="Nonzero entries (%)")
    ax1.bar(x, zeros_pct, bottom=densities, width=w, color="#E5E7EB", edgecolor="#CBD5E1", label="Zero entries (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(systems)
    ax1.set_ylabel("Matrix Entries (%)")
    ax1.set_ylim(0, 114)
    ax1.set_title("(a) System Matrix Sparsity (uifo_800_3000)", pad=8, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    for idx, (nz, d) in enumerate(zip(nnz_counts, densities)):
        ax1.text(
            idx, d + 4.5,
            f"{nz:,} nnz\n({d:.2f}%)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C_BLUE
        )

    # Panel B: Log-scale work reduction across 3 metrics
    metrics = [
        "751x751 LU\nFactorizations",
        "VJP Matrix\nMultiplies",
        "Quantum Noise\nMatvec Ops",
    ]
    before_vals = [606, 56_964_101, 56_964_101]
    after_vals = [101, 256_237, 27_472]
    speedups = ["6.0x", "222x", "2,074x"]

    y = np.arange(len(metrics))
    h = 0.34
    ax2.barh(y + h/2, before_vals, height=h, color=C_VERM, label="Baseline (Dense / Uncached)")
    ax2.barh(y - h/2, after_vals, height=h, color=C_GREEN, label="Combined v2 (Sparse + Adjoint)")

    ax2.set_xscale("log")
    ax2.set_xlim(1e1, 4e9)
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
# Figure 2: Benchmark Timing Comparison -- one session, AMD EPYC 9334, 1 CPU
# ---------------------------------------------------------------------------
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
FINAL_DIR = os.path.join(REPO, "results", "amd-epyc-final")
# (file prefix in results/amd-epyc-final, tick label); all measured in one session on one pinned core
FIG2_VARIANTS = [
    ("baseline", "Baseline\n(upstream)"),
    ("cpp", "C++ LU\n(dfsparse)"),
    ("stage3", "Native\nkernels"),
    ("fastreadout", "Host\nreadout"),
    ("v2_head", "combined-v2\n(committed)"),
    ("final", "Final\n(+E11/E12)"),
]
FIG2_SETUP = ("imports", "initialize", "load", "build")


def _session_medians(prefix, runs=3):
    """Median over runs of every first-run phase and of each run's warm phase medians."""
    import json
    import statistics as st
    rs = [json.load(open(os.path.join(FINAL_DIR, f"{prefix}_{i}.json"))) for i in range(1, runs + 1)]
    phase = lambda r, cat=None, keys=None: sum(p["seconds"] for p in r["phases"]
                                               if (cat is None or p["category"] == cat) and (keys is None or p["key"] in keys))
    warm = lambda k: st.median(r["timing_seconds"]["phase_runtime_medians"][k] for r in rs)
    return dict(
        first=st.median(r["timing_seconds"]["end_to_end"] for r in rs),
        setup=st.median(phase(r, keys=FIG2_SETUP) for r in rs),
        compile=st.median(phase(r, cat="compile") for r in rs),
        execute=st.median(phase(r, cat="execute") + phase(r, cat="output") for r in rs),
        warm=st.median(r["timing_seconds"]["runtime_median"] for r in rs),
        w_forward=warm("forward"), w_gradient=warm("gradient"), w_sens=warm("sensitivity"),
    )


def make_fig2():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.8, 4.4), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
    m = [_session_medians(k) for k, _ in FIG2_VARIANTS]
    configs = [lbl for _, lbl in FIG2_VARIANTS]
    x = np.arange(len(configs))
    w = 0.58
    gap = dict(edgecolor="white", linewidth=1.0)  # surface gap between stacked segments

    def stack(ax, xs, parts, colors, labels):
        bottom = np.zeros(len(xs))
        for vals, col, lab in zip(parts, colors, labels):
            ax.bar(xs, vals, bottom=bottom, width=w, color=col, label=lab, **gap)
            bottom += vals

    # (a) warm execution, all variants
    stack(ax1, x, [np.array([v["w_forward"] for v in m]), np.array([v["w_gradient"] for v in m]),
                   np.array([v["w_sens"] for v in m])],
          [C_BLUE, C_ORANGE, C_PURPLE], ["Forward simulation", "Reverse-mode gradient", "Sensitivity & loss"])
    for i, v in enumerate(m):
        lbl = f"{v['warm']:.2f} s" if i == 0 else f"{v['warm']:.3f} s\n{m[0]['warm'] / v['warm']:.0f}x"
        ax1.text(i, v["warm"] + m[0]["warm"] * 0.02, lbl, ha="center", va="bottom", fontsize=7.4, color=C_DARK,
                 fontweight="bold" if i == len(m) - 1 else "normal")
    ax1.set_xticks(x); ax1.set_xticklabels(configs, fontsize=7.4)
    ax1.set_ylabel("Median warm evaluation (s)")
    ax1.set_ylim(0, m[0]["warm"] * 1.22)
    ax1.set_title("(a) Warm evaluation", pad=8, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=7.6)

    # (b) first-run end-to-end, all variants (the scored number)
    parts = [np.array([v[k] for v in m]) for k in ("setup", "compile", "execute")]
    stack(ax2, x, parts, [C_GREY, C_SKY, C_GREEN], ["Imports & setup", "XLA trace / lower / compile", "Execute + transfer"])
    for i, v in enumerate(m):
        lbl = f"{v['first']:.1f} s" if i == 0 else f"{v['first']:.3f} s\n{m[0]['first'] / v['first']:.0f}x"
        ax2.text(i, v["first"] + m[0]["first"] * 0.02, lbl, ha="center", va="bottom", fontsize=7.4, color=C_DARK,
                 fontweight="bold" if i == len(m) - 1 else "normal")
    ax2.set_xticks(x); ax2.set_xticklabels(configs, fontsize=7.4)
    ax2.set_ylabel("First-run end-to-end (s)")
    ax2.set_ylim(0, m[0]["first"] * 1.22)
    ax2.set_title("(b) First run, complete (scored)", pad=8, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=7.6)

    # (c) zoom: first-run breakdown of the optimised variants only
    xz = np.arange(len(m) - 1)
    stack(ax3, xz, [p[1:] for p in parts], [C_GREY, C_SKY, C_GREEN],
          ["Imports & setup", "XLA trace / lower / compile", "Execute + transfer"])
    top = max(v["first"] for v in m[1:])
    for j, v in enumerate(m[1:]):
        ax3.text(j, v["first"] + top * 0.02, f"{v['first']:.3f} s\ncompile\n{v['compile']:.3f} s", ha="center",
                 va="bottom", fontsize=7.2, color=C_DARK, fontweight="bold" if j == len(m) - 2 else "normal")
    ax3.set_xticks(xz); ax3.set_xticklabels(configs[1:], fontsize=7.4)
    ax3.set_ylabel("First-run end-to-end (s)")
    ax3.set_ylim(0, top * 1.3)
    ax3.set_title("(c) Zoom: optimised variants", pad=8, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=7.4)

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig2_benchmark_comparison")



# ---------------------------------------------------------------------------
# Figure 3: 15-Year Eidothea Literature Landscape (2011-2026)
# ---------------------------------------------------------------------------
def make_fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1.1, 1.2]})

    # Panel A: 15-year publication trajectory from eidothea_literature_verdicts_15y.md
    years = np.arange(2011, 2027)
    counts = [33, 39, 54, 63, 73, 87, 72, 102, 117, 149, 160, 158, 163, 208, 330, 534]

    bars = ax1.bar(years, counts, color=C_SKY, edgecolor=C_BLUE, linewidth=0.7)
    bars[-1].set_color(C_BLUE)
    bars[-2].set_color(C_BLUE)
    ax1.plot(years, counts, color=C_DARK, marker="o", markersize=3.5, linewidth=1.2)

    ax1.set_xticks(years)
    ax1.set_xticklabels([str(y) for y in years], rotation=45, ha="right", fontsize=7.8)
    ax1.set_ylabel("Unique Papers in Catalog")
    ax1.set_ylim(0, 620)
    ax1.set_title("(a) 15-Year Eidothea Scan (2,342 Papers)", pad=8, fontweight="bold")

    # Verified milestone annotations (after Section 2 audit)
    milestones = [
        (2016, 87, "Basker\nSparse LU"),
        (2018, 102, "JAX"),
        (2021, 160, "JAXOpt"),
        (2022, 158, "APF\nOptics"),
        (2023, 163, "Urania /\nUIFO"),
        (2024, 208, "Linear\nSolvers AD"),
        (2025, 330, "Differometor"),
        (2026, 534, "SABLE &\nADSPICE"),
    ]
    for yr, val, txt in milestones:
        offsets = {2016: 18, 2018: 18, 2021: 18, 2022: 72, 2023: 18, 2024: 48, 2025: 18, 2026: 16}
        offset = offsets.get(yr, 18)
        ax1.annotate(
            txt, xy=(yr, val), xytext=(yr, val + offset),
            ha="center", va="bottom", fontsize=7.0, color=C_DARK,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6)
        )

    # Panel B: Top subfields across the 6 taxonomy groups (A-F)
    subfields = [
        ("jax_xla (D)", 187, C_PURPLE),
        ("quantum_noise (A)", 174, C_BLUE),
        ("autodiff_opt (D)", 165, C_PURPLE),
        ("sideband_mod (A)", 155, C_BLUE),
        ("gw_detector (A)", 144, C_BLUE),
        ("surrogate_ml (F)", 132, C_GREY),
        ("xd_fdfd_geo (F)", 112, C_GREY),
        ("xd_quantum_trans. (F)", 110, C_GREY),
        ("xd_ordering_graph (F)", 107, C_GREY),
        ("xd_power_grid (F)", 106, C_GREY),
        ("iter_refine (E)", 102, C_VERM),
        ("low_rank_update (B)", 99, C_GREEN),
        ("freq_sweep (B)", 83, C_GREEN),
        ("implicit_diff (C)", 78, C_ORANGE),
        ("host_exec (E)", 50, C_VERM),
        ("jvp_transpose (E)", 39, C_VERM),
    ]
    labels = [s[0] for s in subfields]
    vals = [s[1] for s in subfields]
    cols = [s[2] for s in subfields]

    y = np.arange(len(labels))
    ax2.barh(y, vals, color=cols, height=0.68)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=7.6)
    ax2.invert_yaxis()
    ax2.set_xlabel("Papers per Subfield (Top 16 of 27)")
    ax2.set_xlim(0, 225)
    ax2.set_title("(b) Subfields by Research Group (A-F)", pad=8, fontweight="bold")

    for idx, v in enumerate(vals):
        ax2.text(v + 3, idx, str(v), va="center", ha="left", fontsize=7.5, color=C_DARK)

    legend_handles = [
        Patch(facecolor=C_BLUE, label="A: Interferometry"),
        Patch(facecolor=C_GREEN, label="B: Sparse Solvers"),
        Patch(facecolor=C_ORANGE, label="C: Autodiff & Adjoint"),
        Patch(facecolor=C_PURPLE, label="D: JAX & Compiler"),
        Patch(facecolor=C_VERM, label="E: Host & Numerics"),
        Patch(facecolor=C_GREY, label="F: Cross-Domain"),
    ]
    ax2.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.3)

    fig.tight_layout(w_pad=2.2)
    save_fig(fig, "fig3_eidothea_literature_landscape")


# ---------------------------------------------------------------------------
# Figure 4: Before vs After Simulation & Adjoint Architecture (final/combined-v2)
# ---------------------------------------------------------------------------
def make_fig4():
    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    def draw_box(x, y, w, h, text, fc, ec, tc=C_DARK, fs=8.1, bold=False):
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

    # Row labels (AMD EPYC 9334 1-CPU apples-to-apples)
    ax.text(1.0, 42.8, "BEFORE: Baseline Dense Pipeline (15.48 s warm | 35.28 s first-run on EPYC 9334, 1 CPU | 606 dense LU solves)", fontsize=9.1, fontweight="bold", color=C_VERM)
    ax.text(1.0, 20.8, "AFTER (final): Host + Native Kernels + Host Readout + C++ LU + lean XLA (0.081 s warm [190x] | 0.585 s first-run [60x])", fontsize=9.1, fontweight="bold", color=C_GREEN)

    # BEFORE row (y = 29..39)
    draw_box(1.5, 29, 17, 10, "19-way\nlax.switch\n(all functions)", "#FEE2E2", C_VERM)
    draw_arrow(19.2, 34, 21.8, 34, C_VERM)
    draw_box(22.5, 29, 18, 10, "Dense (101, 751, 751)\nSignal & Noise\n(99.5% zeros)", "#FEE2E2", C_VERM)
    draw_arrow(41.2, 34, 43.8, 34, C_VERM)
    draw_box(44.5, 29, 19, 10, "3x Dense LU\nSolve 1: A x = b\nSolve 2: A^H w = c\nSolve 3: A v = N w", "#FEE2E2", C_VERM)
    draw_arrow(64.2, 34, 66.8, 34, C_VERM)
    draw_box(67.5, 29, 30.5, 10, "Backward Pass + Readout\n3x More Dense LU Solves\n+ Dense Outer Products (57M ops)", "#FEE2E2", C_VERM)

    # Divider
    ax.plot([1, 99], [24.2, 24.2], linestyle="--", color="#CBD5E1", lw=1.0)

    # AFTER row (y = 6..17)
    draw_box(1.5, 5.5, 20.0, 12, "Host custom_vjp +\n19 Wirtinger Kernels\n(native_kernels.py)\n+ JVP-T / Primal Cache", "#E0F2FE", C_BLUE)
    draw_arrow(22.2, 11.5, 24.3, 11.5, C_BLUE)
    draw_box(25.0, 5.5, 17.5, 12, "Grouped Sparse COO\nNo lax.switch; sweep\nhoist (2,537 nnz;\n272 noise nnz)", "#E0F2FE", C_BLUE)
    draw_arrow(43.2, 11.5, 45.3, 11.5, C_BLUE)
    draw_box(46.0, 5.5, 22.0, 12, "C++ Batched LU\n(dfsparse.cpp)\nA+A^T order, KLU pivot\nreuse + 1-step refine", "#DCFCE7", C_GREEN, bold=True)
    draw_arrow(68.7, 11.5, 70.8, 11.5, C_GREEN)
    draw_box(71.5, 5.5, 27.0, 12, "Adjoint + Host Readout\nRe(w^H N w), Sparse VJP\nHost signal/power_detector\nSuperLU / JAX fallbacks", "#DCFCE7", C_GREEN)

    fig.tight_layout()
    save_fig(fig, "fig4_optimization_pipeline")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print("All 4 figures generated successfully.")
