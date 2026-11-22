"""Before/after dumbbell charts: original Differometor vs final (c8066b8), same EPYC core."""
import json
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, "/root/final_c8066b8")
from plot_timings import GROUPS  # the official benchmark's five bars

O = Path("/root/full_results")
OUT = O / "charts"
OUT.mkdir(exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3"
BEFORE, AFTER = "#86b6ef", "#1c5cab"  # one hue, two shades (sequential blue 250 / 550)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})


def fmt(s):
    if s >= 1:
        return f"{s:.2f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.1f} ms"
    return f"{s * 1e6:.0f} µs"


def tick(s):
    if s >= 1:
        return f"{s:g} s"
    if s >= 1e-3:
        return f"{s * 1e3:g} ms"
    return f"{s * 1e6:g} µs"


def log_axis(ax, lo, hi):
    ax.set_xscale("log")
    lo, hi = 10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi))
    ax.set_xlim(lo / 8, hi * 4)
    ticks = [10 ** e for e in range(int(math.log10(lo)), int(math.log10(hi)) + 1)]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([tick(t) for t in ticks])
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


def dumbbell(ax, rows):
    """rows: [(label, before, after)] top to bottom."""
    lo = min(min(b, a) for _, b, a in rows)
    hi = max(max(b, a) for _, b, a in rows)
    log_axis(ax, lo, hi)
    y = list(range(len(rows)))[::-1]
    for yi, (label, b, a) in zip(y, rows):
        ax.plot([a, b], [yi, yi], color=BEFORE, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([b], [yi], s=90, color=BEFORE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([a], [yi], s=90, color=AFTER, edgecolor=SURFACE, linewidth=2, zorder=4)
        if max(a, b) / min(a, b) < 1.6:  # points overlap: one label to the right
            ax.annotate(f"{fmt(a)}  (original {fmt(b)})", (max(a, b), yi), xytext=(10, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=10, color=INK)
        else:
            ax.annotate(fmt(a), (a, yi), xytext=(-9, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
            ax.annotate(fmt(b), (b, yi), xytext=(9, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=10, color=INK2)
        ax.annotate(f"{b / a:,.0f}×" if b / a >= 10 else f"{b / a:.1f}×", (1, yi),
                    xycoords=("axes fraction", "data"), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.annotate("speed-up", (1, len(rows) - 0.3), xycoords=("axes fraction", "data"),
                xytext=(8, 4), textcoords="offset points", ha="left", va="bottom",
                fontsize=9.5, color=MUTED)


def legend(fig, x, y):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markersize=9, color=c, label=t)
               for c, t in ((BEFORE, "Original Differometor"), (AFTER, "Final (c8066b8)"))]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(x - 0.012, y), ncol=2,
               frameon=False, fontsize=10, labelcolor=INK2, handletextpad=0.3, columnspacing=1.6)


def grouped(report):
    values = {p["key"]: p["seconds"] for p in report["phases"]}
    return [(label, math.fsum(values.get(k, 0.0) for k in keys)) for label, keys in GROUPS]


def phase_chart(kind, title, workload):
    finals = [json.load(open(p)) for p in sorted((O / kind).glob("final_[0-9].json"))]
    orig = json.load(open(O / kind / "original.json"))
    # median run of the three final runs, by first-run total
    finals.sort(key=lambda r: r["timing_seconds"]["end_to_end"])
    fin = finals[len(finals) // 2]
    fo, ff = grouped(orig), grouped(fin)
    rows = [(lab, b, a) for (lab, b), (_, a) in zip(fo, ff)]
    t_o, t_f = orig["timing_seconds"]["end_to_end"], fin["timing_seconds"]["end_to_end"]
    w_o, w_f = orig["timing_seconds"]["runtime_median"], fin["timing_seconds"]["runtime_median"]
    rows += [("Complete first run", t_o, t_f), ("Warm evaluation (median)", w_o, w_f)]
    firsts = [r["timing_seconds"]["end_to_end"] for r in finals]

    fig = plt.figure(figsize=(11, 6.6), facecolor=SURFACE)
    ax = fig.add_axes([0.25, 0.14, 0.62, 0.56], facecolor=SURFACE)
    dumbbell(ax, rows)
    ax.axhline(1.5, color=GRID, linewidth=1)
    fig.text(0.04, 0.925, title, fontsize=20, fontweight="bold")
    fig.text(0.04, 0.865, f"{fmt(t_o)} → {fmt(t_f)} complete first run ({t_o / t_f:.0f}× faster), "
             f"warm {fmt(w_o)} → {fmt(w_f)} ({w_o / w_f:.0f}×)", fontsize=12.5, color=INK)
    fig.text(0.04, 0.82, f"{workload} · AMD EPYC 9334 · 1 pinned CPU (--cpus 1) · both PASS",
             fontsize=10.5, color=INK2)
    legend(fig, 0.045, 0.765)
    fig.text(0.04, 0.045, f"Log scale. Final: median of 3 runs (first run {min(firsts):.3f}–{max(firsts):.3f} s), "
             "10 warm repeats each; original: 1 run. Same machine and core.\n"
             "Rows follow the official benchmark's five phases; compilation = all tracing/lowering + XLA compile. "
             "Validation excluded, as in the rules.", fontsize=9, color=MUTED, linespacing=1.5)
    path = OUT / f"{kind}_original_vs_final.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path, (t_o, t_f, w_o, w_f, firsts)


def suite_chart():
    rows_first, rows_warm = [], []
    for nj in sorted((O / "suite" / "new").glob("*.json")):
        bj = O / "suite" / "base" / nj.name
        n, b = json.load(open(nj)), json.load(open(bj))
        label = nj.stem.replace("_", " ")
        rows_first.append((label, b["timing_seconds"]["end_to_end"], n["timing_seconds"]["end_to_end"]))
        rows_warm.append((label, b["timing_seconds"]["runtime_median"], n["timing_seconds"]["runtime_median"]))
    fig = plt.figure(figsize=(14, 7.2), facecolor=SURFACE)
    for i, (rows, name) in enumerate(((rows_first, "Complete first run"), (rows_warm, "Warm evaluation (median)"))):
        ax = fig.add_axes([0.13 + i * 0.47, 0.10, 0.31, 0.58], facecolor=SURFACE)
        dumbbell(ax, rows)
        if i:
            ax.set_yticklabels([])
        ax.set_title(name, loc="left", fontsize=12.5, color=INK, pad=22, x=-0.0)
    g1 = math.exp(st.mean(math.log(b / a) for _, b, a in rows_first))
    g2 = math.exp(st.mean(math.log(b / a) for _, b, a in rows_warm))
    fig.text(0.03, 0.93, "Benchmark suite: 10 optical setups", fontsize=20, fontweight="bold")
    fig.text(0.03, 0.88, f"Geometric-mean speed-up {g1:.0f}× on the complete first run, {g2:.0f}× warm; "
             "every case finite and matching the original (see comparison.txt)", fontsize=12.5)
    fig.text(0.03, 0.845, "aLIGO, Voyager and uIFO setups, 5–301 frequencies · AMD EPYC 9334 · 1 pinned CPU",
             fontsize=10.5, color=INK2)
    legend(fig, 0.035, 0.80)
    fig.text(0.03, 0.03, "Log scale. One run per case and code version, 5 warm repeats, same machine and core. "
             "No public reference exists for these setups, so correctness is checked against the original code's saved arrays.",
             fontsize=9, color=MUTED)
    path = OUT / "suite_original_vs_final.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path, g1, g2


if __name__ == "__main__":
    print(phase_chart("main", "Main benchmark: 3×3 optical grid", "101 frequencies, 193 design gradients"))
    print(phase_chart("aligo", "aLIGO benchmark", "101 frequencies (20–5000 Hz), 29 design gradients"))
    print(suite_chart())
