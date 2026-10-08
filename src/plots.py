"""
plots.py - Figure generation (spec Section 7, Figures 1-8, plus Figure 9 for
the sensitivity analysis and Figure 10 for the adoption threshold).

Project : After-Office Surge at the LRT-1 EDSA Interchange
Course  : CSS142P Modeling and Simulation, Mapua University
Authors : Aldea, De Leon, Jerusalem

The report is printed in black and white, so every chart separates its
series by marker shape, filled versus hollow markers and line style, never
by colour alone. Figures 2-9 are drawn from the CSV files in results/, so
they can be redrawn without re-running the experiments:

    python -m src.plots
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")       # write files only; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from src.model import Params, batch_arrivals, simulate

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

# =============================================================================
# STYLE
# =============================================================================

INK = "#1a1a1a"
GREY = "#6e6e6e"
LIGHT = "#b3b3b3"
GRID = "#d9d9d9"

BANK_STYLE = {
    "A": {"color": INK, "linestyle": "-", "label": "Bank A (beep + SJT)"},
    "B": {"color": GREY, "linestyle": "--", "label": "Bank B (QR only)"},
    "All": {"color": INK, "linestyle": "-", "label": "Single pool (all media)"},
}

# One style per QR share, assigned in the fixed order of params.qr_shares.
# Marker shape and fill carry identity; line style backs it up.
SERIES_STYLES = [
    {"color": INK, "marker": "o", "markerfacecolor": INK, "linestyle": "-"},
    {"color": INK, "marker": "s", "markerfacecolor": "white", "linestyle": "--"},
    {"color": GREY, "marker": "^", "markerfacecolor": GREY, "linestyle": "-."},
    {"color": GREY, "marker": "D", "markerfacecolor": "white", "linestyle": ":"},
    {"color": INK, "marker": "v", "markerfacecolor": "white", "linestyle": (0, (6, 2))},
]

# Bar fills for the three fare media (Figure 4).
MEDIUM_BARS = {
    "beep": {"color": "#d0d0d0", "hatch": "", "label": "beep card"},
    "sjt": {"color": "#8a8a8a", "hatch": "///", "label": "SJT"},
    "qr": {"color": INK, "hatch": "", "label": "beep QR"},
}


def apply_style():
    """Shared look for every figure: small type, hairline grid, light axes."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.edgecolor": GREY,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "hatch.color": "white",
        "hatch.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, name):
    """Write a figure to figures/ and close it."""
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / name)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def share_label(p_qr):
    """0.03 -> 'QR 3%'."""
    return f"QR {p_qr * 100:g}%"


def clock_label(seconds, _position=None):
    """Seconds after 16:00 as a clock time, e.g. 3600 -> '17:00'."""
    minutes = int(round(seconds / 60.0))
    return f"{16 + minutes // 60}:{minutes % 60:02d}"


def rows_for(table, p_qr):
    """Rows of a results table at one QR share (float-safe comparison)."""
    return table[np.isclose(table["p_qr"], p_qr)]


# =============================================================================
# CONFIGURATION AXIS - shared by Figures 2, 3 and 5
# =============================================================================

# Undivided sits apart on the left: it is the benchmark, not an allocation.
CONFIG_X = {"Undivided": 0.0, "6/1": 1.4, "5/2": 2.4, "4/3": 3.4, "3/4": 4.4, "2/5": 5.4}


def config_axis(ax, params):
    """Configuration categories on the x axis, with the present split marked."""
    labels = [f"{c}\n(present)" if c == params.present_configuration else c
              for c in CONFIG_X]
    ax.set_xticks(list(CONFIG_X.values()), labels)
    ax.axvline(0.7, color=LIGHT, linewidth=0.8)
    ax.set_xlim(-0.5, max(CONFIG_X.values()) + 0.5)
    ax.set_xlabel("Gate allocation, Bank A / Bank B")
    ax.grid(axis="x", visible=False)


def plot_by_allocation(ax, summary, metric, params, show_ci=True):
    """
    One series per QR share of `metric` against configuration. The divided
    allocations are joined by a line; Undivided is an unjoined marker.
    """
    for k, p_qr in enumerate(params.qr_shares):
        style = SERIES_STYLES[k]
        level = rows_for(summary, p_qr).set_index("config")
        configs = [c for c in CONFIG_X if c in level.index and not np.isnan(level.loc[c, f"{metric}_mean"])]
        x = np.array([CONFIG_X[c] for c in configs])
        y = level.loc[configs, f"{metric}_mean"].to_numpy(dtype=float)
        if show_ci:
            low = level.loc[configs, f"{metric}_ci_low"].to_numpy(dtype=float)
            high = level.loc[configs, f"{metric}_ci_high"].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=[y - low, high - y], fmt="none",
                        ecolor=style["color"], elinewidth=0.6, capsize=2)
        divided = np.array([c != "Undivided" for c in configs])
        ax.plot(x[divided], y[divided], label=share_label(p_qr), **style)
        ax.plot(x[~divided], y[~divided], linestyle="none",
                **{key: v for key, v in style.items() if key != "linestyle"})


# =============================================================================
# FIGURE 1 - queue length over time
# =============================================================================

def draw_queue_trace(ax, result, start_s, end_s):
    """Step plot of passengers waiting at each bank between two times."""
    for name, (times, waiting, _busy) in result.logs.items():
        ax.step(times, waiting, where="post", linewidth=0.9, **BANK_STYLE[name])
    ax.set_xlim(start_s, end_s)
    ax.xaxis.set_major_formatter(FuncFormatter(clock_label))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Passengers waiting")


def figure_1(params):
    """
    Figure 1: queue length at each bank over time under the present split,
    replication 0. (a) The whole surge window at 3% QR, where the burst
    envelope follows the demand profile. (b) and (c) The same 20 minutes
    around the 17:00 peak at 3% and 30% QR, on one y scale, with train
    arrivals marked: at 3% each burst is a cluster of brief spikes; at 30%
    Bank B is loaded close to capacity and each burst rises and drains.
    """
    present = params.present_configuration
    low_share, high_share = min(params.qr_shares), max(params.qr_shares)
    runs = {p: simulate(batch_arrivals(p, params, 0), present, params)
            for p in (low_share, high_share)}
    zoom_from = 60 * 60.0
    zoom_to = zoom_from + 20 * 60.0

    fig = plt.figure(figsize=(7.5, 7.8))
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.55)
    ax_full = fig.add_subplot(grid[0])
    ax_low = fig.add_subplot(grid[1])
    ax_high = fig.add_subplot(grid[2], sharey=ax_low)

    draw_queue_trace(ax_full, runs[low_share], 0, params.window_s)
    ax_full.set_xticks([h * 3600.0 for h in range(5)])
    ax_full.axvspan(0, params.warmup_s, color=GRID, alpha=0.6, linewidth=0)
    ax_full.axvspan(zoom_from, zoom_to, color=GRID, alpha=0.35, linewidth=0)
    ax_full.set_ylim(bottom=0)
    ax_full.text(params.warmup_s / 2, ax_full.get_ylim()[1] * 0.95, "warm-up",
                 ha="center", va="top", color=GREY, fontsize=8)
    ax_full.set_title(f"(a) Whole surge, {present}, {share_label(low_share)} "
                      "(light band: zoom window below)")
    ax_full.legend(loc="upper right")

    for ax, p_qr, tag in ((ax_low, low_share, "b"), (ax_high, high_share, "c")):
        draw_queue_trace(ax, runs[p_qr], zoom_from, zoom_to)
        trains = runs[p_qr].arrivals.train_times_s
        for t_train in trains[(trains >= zoom_from) & (trains < zoom_to)]:
            ax.axvline(t_train, color=GREY, linewidth=0.6, linestyle=":")
        ax.set_xticks([zoom_from + k * params.headway_s for k in range(6)])
        ax.set_title(f"({tag}) {clock_label(zoom_from)}-{clock_label(zoom_to)}, {present}, "
                     f"{share_label(p_qr)}; dotted lines = MRT-3 train arrivals")
    ax_low.set_ylim(bottom=0)
    ax_high.set_xlabel("Time of day")
    save(fig, "fig1_queue_over_time.png")


# =============================================================================
# FIGURES 2-6 - the experiment sweep
# =============================================================================

def figure_2(summary, params):
    """Figure 2: mean waiting time against gate allocation, per QR share."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    plot_by_allocation(ax, summary, "Wq_s", params)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    config_axis(ax, params)
    ax.set_ylabel("Mean waiting time Wq (s, log scale)")
    ax.set_title("Mean waiting time by gate allocation (95% CI bars)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "fig2_wait_by_allocation.png")


def figure_3(summary, params):
    """Figure 3: maximum queue length at each bank against gate allocation."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), sharey=True)
    for ax, bank, title in ((axes[0], "A", "(a) Bank A: beep + SJT queue"),
                            (axes[1], "B", "(b) Bank B: QR queue")):
        # Undivided has a single shared queue; it is shown in both panels.
        merged = summary.copy()
        undivided = merged["config"] == "Undivided"
        for suffix in ("_mean", "_ci_low", "_ci_high"):
            merged.loc[undivided, f"Lq_max_{bank}{suffix}"] = merged.loc[undivided, f"Lq_max_All{suffix}"]
        plot_by_allocation(ax, merged, f"Lq_max_{bank}", params)
        config_axis(ax, params)
        ax.set_title(title)
    axes[0].set_ylim(bottom=0)      # shared axis: set once both panels are drawn
    axes[0].set_ylabel("Peak passengers waiting, Lq_max")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.text(0.01, -0.02, "Undivided: the single shared queue, shown in both panels.",
             color=GREY, fontsize=8)
    fig.tight_layout()
    save(fig, "fig3_peak_queue_by_allocation.png")


def figure_4(summary, params):
    """Figure 4: mean waiting time per fare medium, grouped by configuration."""
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.4))
    width = 0.24
    x = np.arange(len(params.configurations))
    for ax, p_qr in zip(axes.flat, params.qr_shares):
        level = rows_for(summary, p_qr).set_index("config").loc[list(params.configurations)]
        for k, medium in enumerate(MEDIUM_BARS):
            style = MEDIUM_BARS[medium]
            ax.bar(x + (k - 1) * width, level[f"Wq_{medium}_s_mean"], width * 0.9,
                   color=style["color"], hatch=style["hatch"], label=style["label"],
                   edgecolor="white", linewidth=0)
        ax.set_xticks(x, [c.replace("Undivided", "Undiv.") for c in params.configurations],
                      fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_title(share_label(p_qr))
        ax.set_ylim(bottom=0)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean wait Wq (s)")
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", title="Fare medium")
    fig.suptitle("Waiting time per fare medium by configuration (each panel has its own scale)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_wait_per_class.png")


def figure_5(summary, params):
    """Figure 5: blocked-capacity time against configuration, per QR share."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    plot_by_allocation(ax, summary, "T_blk_gate_min", params)
    config_axis(ax, params)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Blocked-capacity time T_blk (gate-minutes)")
    ax.set_title("Gate time idle while the other bank had a queue (16:30-20:00)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "fig5_blocked_capacity.png")


def figure_6(optimal, params):
    """
    Figure 6, the headline: (a) the wait-optimal number of QR-only gates at
    each adoption level, beside the allocations that are best by peak queue
    and by clearance time; (b) what the present split and the optimal split
    cost per passenger against an undivided array.
    """
    fig, (ax_alloc, ax_cost) = plt.subplots(1, 2, figsize=(7.5, 3.8))
    shares = optimal["p_qr"].to_numpy() * 100

    def gates_on_b(column):
        """Bank B gate count of each allocation in `column`, e.g. '6/1' -> 1."""
        return optimal[column].str.split("/").str[1].astype(int)

    ax_alloc.axhline(int(params.present_configuration.split("/")[1]), color=LIGHT,
                     linewidth=1.0, linestyle="--")
    ax_alloc.text(shares[-1], int(params.present_configuration.split("/")[1]) + 0.12,
                  f"present {params.present_configuration}", ha="right", color=GREY, fontsize=8)
    ax_alloc.plot(shares, gates_on_b("best_by_wait"), color=INK, marker="o", markersize=8,
                  markerfacecolor=LIGHT, linestyle="-", label="lowest mean wait")
    ax_alloc.plot(shares, gates_on_b("best_by_queue"), color=GREY, marker="s", markersize=9,
                  markerfacecolor="none", linestyle="none", label="lowest peak queue")
    ax_alloc.plot(shares, gates_on_b("best_by_clear"), color=INK, marker="x", markersize=7,
                  linestyle="none", label="fastest burst clearance")
    for share, config in zip(shares, optimal["best_by_wait"]):
        ax_alloc.annotate(config, (share, int(config.split("/")[1])), xytext=(0, -14),
                          textcoords="offset points", ha="center", fontsize=8)
    top = max(4, *(gates_on_b(c).max() for c in ("best_by_wait", "best_by_queue", "best_by_clear")))
    ax_alloc.set_yticks(range(1, top + 1))
    ax_alloc.set_ylim(0.4, top + 0.6)
    ax_alloc.set_xticks(shares)
    ax_alloc.set_xlabel("beep QR adoption (% of passengers)")
    ax_alloc.set_ylabel("QR-only gates (Bank B) in best split")
    ax_alloc.set_title("(a) Optimal allocation")
    ax_alloc.legend(loc="upper left")

    for column, label, style in (
            ("dW_div_present_s", f"present {params.present_configuration}",
             {"color": INK, "marker": "o", "linestyle": "-"}),
            ("dW_div_best_s", "best division",
             {"color": GREY, "marker": "s", "markerfacecolor": "white", "linestyle": "--"})):
        y = optimal[f"{column}_mean"].to_numpy()
        low, high = optimal[f"{column}_ci_low"].to_numpy(), optimal[f"{column}_ci_high"].to_numpy()
        ax_cost.errorbar(shares, y, yerr=[y - low, high - y], fmt="none",
                         ecolor=style["color"], elinewidth=0.6, capsize=2)
        ax_cost.plot(shares, y, label=label, **style)
    ax_cost.set_xticks(shares)
    ax_cost.set_ylim(bottom=0)
    ax_cost.set_xlabel("beep QR adoption (% of passengers)")
    ax_cost.set_ylabel("Extra wait vs undivided (s per passenger)")
    ax_cost.set_title("(b) Cost of division, dW_div (95% CI)")
    ax_cost.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig6_optimal_allocation.png")


# =============================================================================
# FIGURES 7-9 - verification, arrival process, sensitivity
# =============================================================================

def draw_verification(ax, table, label, analytical_marker="o", simulated_marker="s",
                      linestyle="--"):
    """Analytical Wq as a line of hollow markers; simulated Wq as filled
    markers with 95% CI bars, nudged right so the two do not overlap."""
    x = table["load_rho"].to_numpy()
    sim = table["Wq_simulated_s"].to_numpy()
    ax.plot(x, table["Wq_analytical_s"], color=GREY, linestyle=linestyle,
            marker=analytical_marker, markerfacecolor="white", label=f"{label} analytical")
    ax.errorbar(x + 0.008, sim, yerr=[sim - table["ci95_low_s"], table["ci95_high_s"] - sim],
                fmt=simulated_marker, color=INK, markersize=4.5, elinewidth=0.8, capsize=2,
                label=f"{label} simulated")


def figure_7(verification, banks):
    """Figure 7: analytical against simulated Wq at the three test loads."""
    fig, (ax_pool, ax_banks) = plt.subplots(1, 2, figsize=(7.5, 3.6))
    c = int(verification["c"].iloc[0])
    draw_verification(ax_pool, verification, f"M/M/{c}")
    ax_pool.set_title(f"(a) Stage A: one pool of {c} gates")
    for bank, markers, linestyle in (("A", ("o", "s"), "--"), ("B", ("D", "^"), ":")):
        rows = banks[banks["bank"] == bank]
        draw_verification(ax_banks, rows, f"Bank {bank} M/M/{int(rows['c'].iloc[0])}",
                          *markers, linestyle=linestyle)
    ax_banks.set_title("(b) Stage C: each bank under 5/2")
    for ax in (ax_pool, ax_banks):
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.set_xticks([0.5, 0.7, 0.9])
        ax.set_xlabel("Utilisation rho")
        ax.legend(loc="upper left", fontsize=7.5)
    ax_pool.set_ylabel("Mean waiting time Wq (s, log scale)")
    fig.tight_layout()
    save(fig, "fig7_verification.png")


def figure_8(comparison, params):
    """Figure 8: batch against smooth arrivals under the present split."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2))
    panels = (("Wq_s", "(a) Mean wait Wq (s, log)", True),
              ("Lq_max_A", "(b) Peak queue, Bank A", False),
              ("Lq_max_B", "(c) Peak queue, Bank B", False))
    styles = {"batch": {"color": INK, "marker": "o", "linestyle": "-", "label": "train bursts"},
              "smooth": {"color": GREY, "marker": "s", "markerfacecolor": "white",
                         "linestyle": "--", "label": "smooth Poisson"}}
    for ax, (metric, title, log_scale) in zip(axes, panels):
        for arrivals, style in styles.items():
            rows = comparison[comparison["arrivals"] == arrivals]
            ax.plot(rows["p_qr"] * 100, rows[f"{metric}_mean"], **style)
        ax.set_title(title)
        ax.set_xticks(np.array(params.qr_shares) * 100)
        ax.set_xlabel("QR adoption (%)")
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        else:
            ax.set_ylim(bottom=0)
    axes[0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle(f"Same demand profile, configuration {params.present_configuration}: "
                 "what a smooth-arrival model would hide", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig8_batch_vs_smooth.png")


def figure_9(sensitivity, params):
    """
    Figure 9: sensitivity of the present split's cost of division. Each row
    is one input moved to the low (hollow) and high (filled) end of its
    range; the vertical line is the base case.
    """
    present = params.present_configuration
    shares = (min(params.sens_qr_shares), max(params.sens_qr_shares))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.2), sharey=True)
    rows = sensitivity[sensitivity["config"] == present]
    factors = [f for f in dict.fromkeys(rows["factor"]) if f != "base"]

    for ax, p_qr in zip(axes, shares):
        level = rows_for(rows, p_qr)
        base = level[level["factor"] == "base"]["dW_div_s_mean"].iloc[0]
        ax.axvline(base, color=LIGHT, linewidth=1.0)
        for y, factor in enumerate(factors):
            values = level[level["factor"] == factor]["dW_div_s_mean"].to_numpy()
            if len(values) == 2:
                ax.plot(values, [y, y], color=GREY, linewidth=1.0)
                ax.plot(values[0], y, marker="o", color=INK, markerfacecolor="white", linestyle="none")
                ax.plot(values[1], y, marker="o", color=INK, linestyle="none")
            else:
                ax.plot(values, [y] * len(values), marker="*", markersize=9, color=INK, linestyle="none")
        ax.set_title(f"{share_label(p_qr)} (base {base:.3f} s)")
        ax.set_xlabel(f"Cost of {present} vs undivided (s per passenger)")
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
    labels = []
    for factor in factors:
        levels = list(dict.fromkeys(rows[rows["factor"] == factor]["level"]))
        labels.append(f"{factor}\n{' / '.join(levels)}" if len(levels) == 2 else f"{factor}\n{levels[0]}")
    axes[0].set_yticks(range(len(factors)), labels, fontsize=7.5)
    axes[0].invert_yaxis()
    keys = [plt.Line2D([], [], marker="o", color=INK, markerfacecolor="white", linestyle="none",
                       label="low end of range"),
            plt.Line2D([], [], marker="o", color=INK, linestyle="none", label="high end of range"),
            plt.Line2D([], [], marker="*", markersize=9, color=INK, linestyle="none",
                       label="combined case"),
            plt.Line2D([], [], color=LIGHT, linewidth=1.0, label="base case")]
    fig.legend(handles=keys, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    save(fig, "fig9_sensitivity.png")


def figure_10(threshold, params):
    """
    Figure 10: mean wait of the competing splits at QR adoption from 3% to
    30% in 1-point steps, with the band where each split is best named at
    the top and the threshold (where one more QR gate first helps
    significantly) marked.
    """
    present = params.present_configuration
    n_a, n_b = (int(n) for n in present.split("/"))
    one_more = f"{n_a - 1}/{n_b + 1}"
    shares = threshold["p_qr"].to_numpy() * 100
    styles = [{"color": GREY, "marker": "o", "markerfacecolor": "white", "linestyle": "--"},
              {"color": INK, "marker": "o", "markerfacecolor": INK, "linestyle": "-"},
              {"color": INK, "marker": "s", "markerfacecolor": "white", "linestyle": "-."}]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for config, style in zip(params.threshold_configurations, styles):
        column = f"Wq_{config.replace('/', '_')}_s"
        label = f"{config} (present)" if config == present else config
        ax.plot(shares, threshold[f"{column}_mean"], markersize=4, label=label, **style)
        ax.fill_between(shares, threshold[f"{column}_ci_low"], threshold[f"{column}_ci_high"],
                        color=style["color"], alpha=0.12, linewidth=0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))

    # Name the band where each split has the lowest mean wait.
    best = threshold["best"].to_numpy()
    start = 0
    for k in range(1, len(best) + 1):
        if k == len(best) or best[k] != best[start]:
            middle = (shares[start] + shares[k - 1]) / 2
            ax.text(middle, 1.02, f"{best[start]} best", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=GREY)
            if k < len(best):
                ax.axvline((shares[k - 1] + shares[k]) / 2, color=LIGHT, linewidth=0.8)
            start = k

    significant = threshold[threshold["present_minus_one_more_qr_gate_s_ci_low"] > 0]
    if len(significant):
        at = significant["p_qr"].min() * 100
        ax.axvline(at, color=INK, linewidth=1.0, linestyle=":")
        ax.annotate(f"{one_more} significantly better\nthan {present} from {at:g}%",
                    xy=(at, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, va="center")
    ax.set_xlabel("beep QR adoption (% of passengers)")
    ax.set_ylabel("Mean waiting time Wq (s, log scale)")
    ax.set_title("When do two QR gates stop being enough? (shading: 95% CI)", pad=16)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "fig10_threshold.png")


def figure_model_flow():
    """
    Figure 0: process flow diagram of the conceptual model (proposal
    Methodology step 3) - how a passenger moves through the simulation.
    """
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.5, 8.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def node(x, y, text, width=3.6, height=0.95, dashed=False, fill="white"):
        """A rounded box centred on (x, y); returns its top and bottom centres."""
        ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=fill, edgecolor=INK, linewidth=0.9,
                                    linestyle="--" if dashed else "-"))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.2, color=INK)
        return (x, y + height / 2), (x, y - height / 2)

    def arrow(start, end, label=None):
        """An arrow between two box edges, with an optional label at its middle."""
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 0.9})
        if label:
            ax.text((start[0] + end[0]) / 2 + 0.1, (start[1] + end[1]) / 2, label,
                    fontsize=7.5, color=GREY, ha="left", va="center")

    _top, train = node(2.6, 11.1, "MRT-3 train every 4 min\nbatch = 70 x 15-min block multiplier",
                       width=4.3)
    _top, street = node(7.4, 11.1, "Street-level arrivals\nPoisson, 0.5/min x block multiplier",
                        width=4.3)
    reach_top, reach = node(5.0, 9.45, "Passenger reaches the gate array\n"
                                       "(train passengers spread Uniform 0-90 s)", width=4.6)
    assign_top, assign = node(5.0, 7.85, "Fare medium: multinomial draw (beep, SJT, QR at adoption p)\n"
                                         "Gate time: triangular, by medium", width=6.2)
    check_top, check = node(5.0, 6.25, "Eligibility matrix:\nwhich gates may this passenger use?",
                            width=4.0, fill="#eeeeee")
    a_top, a_bottom = node(1.85, 4.3, "Bank A queue (FIFO)\nbeep + SJT, n_A gates", width=3.2)
    b_top, b_bottom = node(5.0, 4.3, "Bank B queue (FIFO)\nQR only, n_B gates", width=2.8)
    u_top, u_bottom = node(8.2, 4.3, "Undivided: one queue (FIFO)\nevery medium, 7 gates",
                           width=3.2, dashed=True)
    gate_top, gate = node(5.0, 2.55, "Gate service: hold one gate\nfor the passenger's gate time",
                          width=4.0)
    leave_top, _bottom = node(5.0, 0.95, "Depart to the platform\n"
                                         "record wait, queue length and gate state", width=4.6)

    arrow(train, (reach_top[0] - 1.2, reach_top[1]))
    arrow(street, (reach_top[0] + 1.2, reach_top[1]))
    arrow(reach, assign_top)
    arrow(assign, check_top)
    arrow((check[0] - 1.0, check[1]), a_top, "beep, SJT")
    arrow(check, b_top, "QR")
    arrow((check[0] + 1.0, check[1]), u_top, "any medium\n(Undivided)")
    for bottom in (a_bottom, b_bottom, u_bottom):
        arrow(bottom, gate_top)
    arrow(gate, leave_top)
    ax.text(0.1, 0.05, "Dashed: the Undivided configuration replaces both banks with one pool.",
            fontsize=7.5, color=GREY)
    ax.set_title("Process flow of the simulation model")
    save(fig, "fig0_model_flow.png")


def make_all_figures(params):
    """Draw Figures 0-10. Figure 1 re-simulates one run; the rest read results/."""
    apply_style()

    def read(name):
        """Load one results CSV."""
        return pd.read_csv(RESULTS_DIR / name)

    summary = read("summary.csv")
    figure_model_flow()
    figure_1(params)
    figure_2(summary, params)
    figure_3(summary, params)
    figure_4(summary, params)
    figure_5(summary, params)
    figure_6(read("optimal_allocation.csv"), params)
    figure_7(read("verification.csv"), read("verification_banks.csv"))
    figure_8(read("batch_vs_smooth.csv"), params)
    if (RESULTS_DIR / "sensitivity.csv").exists():
        figure_9(read("sensitivity.csv"), params)
    if (RESULTS_DIR / "threshold.csv").exists():
        figure_10(read("threshold.csv"), params)


if __name__ == "__main__":
    make_all_figures(Params())
