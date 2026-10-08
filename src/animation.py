"""
animation.py - Animated replay of the Side A concourse for the oral defense
(spec Section 7).

Project : After-Office Surge at the LRT-1 EDSA Interchange
Course  : CSS142P Modeling and Simulation, Mapua University
Authors : Aldea, De Leon, Jerusalem

Every frame is drawn from the trace that simulate(trace=True) records once
per simulated second; nothing is animated from inside the SimPy loop. Two
GIFs are written to figures/:

    animation_5_2.gif          the present 5/2 split
    animation_comparison.gif   5/2 (left) against Undivided (right), driven by
                               the same passengers and the same random seeds

Run on its own with:   python -m src.animation
"""

import matplotlib

matplotlib.use("Agg")       # write files only; no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from src.model import (MEDIA, Params, batch_arrivals, blocked_gates_steps, simulate,
                       step_integral)
from src.plots import FIGURES_DIR

# =============================================================================
# LOOK - colour is allowed here (not a printed figure), but shape still
# identifies the fare medium in case the projector washes colour out
# =============================================================================

MEDIUM_LOOK = {             # Okabe-Ito colours, safe for colour-blind viewers
    "beep": {"marker": "o", "color": "#0072B2", "label": "beep card"},
    "sjt": {"marker": "s", "color": "#E69F00", "label": "SJT"},
    "qr": {"marker": "^", "color": "#CC79A7", "label": "beep QR"},
}
GATE_IDLE = "white"
GATE_BUSY = "#e6e6e6"
GATE_BLOCKED = "#f6c6c0"    # idle, while the other bank has a queue
INK = "#1a1a1a"

# =============================================================================
# SCENE GEOMETRY (axis units; the drawing is a schematic, not to scale)
# =============================================================================

# The seven physical turnstiles, left to right: five on the left, then two on
# the right past the ticket vending machine, as observed.
PHYSICAL_GATE_X = (1.2, 2.1, 3.0, 3.9, 4.8, 8.6, 9.5)
GATE_Y = 7.4
TVM_X = 6.7
PLATFORM = (8.5, 9.7)       # y extent of the platform band
QUEUE_TOP = 6.55            # y of the first queue row
SPACING = 0.42              # gap between queued passengers
ENTRANCE = (6.7, 0.45)      # where the footbridge lands in the concourse

# Display-only timings. The walk across the concourse is not modelled; a
# passenger is drawn walking in during the few seconds BEFORE their modelled
# arrival, so the queue counts on screen always match the model.
WALK_S = 4.0
EXIT_S = 2.0


def gate_positions(gates):
    """
    Centre of every gate, per bank. Bank A takes the leftmost physical
    gates and Bank B the rightmost; Undivided uses all seven as one pool.
    """
    if "All" in gates:
        return {"All": [(x, GATE_Y) for x in PHYSICAL_GATE_X]}
    n_a = gates["A"]
    return {"A": [(x, GATE_Y) for x in PHYSICAL_GATE_X[:n_a]],
            "B": [(x, GATE_Y) for x in PHYSICAL_GATE_X[n_a:]]}


def queue_layout(gates, gate_xy):
    """Per bank, the x centre of its queue and how many people fit in a row."""
    layout = {}
    for name, positions in gate_xy.items():
        xs = [x for x, _y in positions]
        centre = TVM_X if name == "All" else (min(xs) + max(xs)) / 2
        layout[name] = (centre, 3 if name == "B" else 5)
    return layout


def clock(seconds):
    """Seconds after 16:00 as a clock time with seconds, e.g. 17:04:30."""
    s = int(round(seconds))
    return f"{16 + s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# =============================================================================
# ONE PANEL - a top-down view driven by one traced run
# =============================================================================

class ConcoursePanel:
    """
    One top-down view of the gate array, redrawn from a traced RunResult.

    Each frame places every visible passenger: walking in across the
    concourse, standing in their bank's queue, at a gate, or stepping onto
    the platform. Gates that stand idle while the other bank has a queue
    are tinted, and the readout counts the blocked gate-minutes.
    """

    def __init__(self, ax, result, params, clip_start_s, title):
        """Draw the static scene and create the artists updated each frame."""
        self.result = result
        self.clip_start = clip_start_s
        self.gate_xy = gate_positions(result.gates)
        self.queue = queue_layout(result.gates, self.gate_xy)
        self.times = result.arrivals.times_s
        self.exit_order = np.argsort(result.finished)
        self.finished_sorted = result.finished[self.exit_order]
        self.blocked_times, self.blocked_gates = blocked_gates_steps(result)
        start_snapshot = result.trace[int(clip_start_s)]["banks"]
        self.served_before = {name: start_snapshot[name]["served"] for name in result.gates}

        ax.set_xlim(0, 10.6)
        ax.set_ylim(0, 12.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10, loc="left", color=INK)

        # Unpaid concourse below the gate line, platform above it.
        ax.add_patch(Rectangle((0.2, 0.1), 10.2, GATE_Y - 0.5, color="#f2f2f2", zorder=0))
        ax.text(0.4, 0.35, "Unpaid concourse", fontsize=7.5, color="#777777")
        ax.add_patch(Rectangle((0.2, PLATFORM[0]), 10.2, PLATFORM[1] - PLATFORM[0],
                               color="#dde6ef", zorder=0))
        ax.text(0.4, PLATFORM[1] - 0.35, "Platform (northbound)", fontsize=7.5, color="#55606b")
        ax.annotate("from MRT-3 footbridge", xy=ENTRANCE, xytext=(ENTRANCE[0], -0.35),
                    ha="center", fontsize=7.5, color="#777777",
                    arrowprops={"arrowstyle": "->", "color": "#999999"})
        ax.add_patch(Rectangle((TVM_X - 0.45, GATE_Y - 0.35), 0.9, 0.7, facecolor="white",
                               edgecolor="#999999", linewidth=0.8, zorder=1))
        ax.text(TVM_X, GATE_Y, "TVM", ha="center", va="center", fontsize=7, color="#777777")

        self.gate_boxes = {}
        for name, positions in self.gate_xy.items():
            self.gate_boxes[name] = []
            for x, y in positions:
                box = Rectangle((x - 0.3, y - 0.38), 0.6, 0.76, facecolor=GATE_IDLE,
                                edgecolor="#555555", linewidth=0.9, zorder=2)
                ax.add_patch(box)
                self.gate_boxes[name].append(box)
            xs = [x for x, _y in positions]
            label = {"All": "Every gate: every fare medium",
                     "A": "Bank A: beep + SJT", "B": "Bank B: QR only"}[name]
            ax.text((min(xs) + max(xs)) / 2, GATE_Y + 0.62, label, ha="center",
                    fontsize=7.5, color=INK, zorder=3)

        self.dots = {m: ax.scatter([], [], s=52, marker=MEDIUM_LOOK[m]["marker"],
                                   color=MEDIUM_LOOK[m]["color"], edgecolor="white",
                                   linewidth=0.5, zorder=4) for m in MEDIA}
        self.readout = ax.text(0.3, 11.95, "", fontsize=8, va="top", family="monospace",
                               color=INK)

    def queue_place(self, name, k):
        """Where the k-th person (0 = head) in a bank's queue stands."""
        centre, per_row = self.queue[name]
        row, col = divmod(k, per_row)
        return centre + (col - (per_row - 1) / 2) * SPACING, QUEUE_TOP - row * SPACING

    def update(self, t):
        """Redraw passengers, gate tints and readout for simulated time t (s)."""
        snapshot = self.result.trace[int(round(t))]["banks"]
        media = self.result.arrivals.media
        points = {m: [] for m in MEDIA}

        # In the queue and at the gates, straight from the trace.
        queue_length = {}
        for name, bank in snapshot.items():
            for k, i in enumerate(bank["queue_ids"]):
                points[media[i]].append(self.queue_place(name, k))
            for slot, i in enumerate(bank["gate_ids"]):
                if i is not None:
                    points[media[i]].append(self.gate_xy[name][slot])
            queue_length[name] = len(bank["queue_ids"])

        # Walking in: passengers who reach the gate array in the next WALK_S s.
        first, last = np.searchsorted(self.times, [t, t + WALK_S], side="right")
        for i in range(first, last):
            name = self.result.bank[i]
            target = self.queue_place(name, queue_length[name])
            queue_length[name] += 1
            progress = 1.0 - (self.times[i] - t) / WALK_S
            points[media[i]].append((ENTRANCE[0] + progress * (target[0] - ENTRANCE[0]),
                                     ENTRANCE[1] + progress * (target[1] - ENTRANCE[1])))

        # Leaving: passengers who finished at a gate in the last EXIT_S s.
        first, last = np.searchsorted(self.finished_sorted, [t - EXIT_S, t], side="right")
        for i in self.exit_order[first:last]:
            x, y = self.gate_xy[self.result.bank[i]][self.result.gate_index[i]]
            progress = (t - self.result.finished[i]) / EXIT_S
            points[media[i]].append((x, y + progress * (PLATFORM[1] - 0.3 - y)))

        for medium, dots in self.dots.items():
            dots.set_offsets(np.array(points[medium]).reshape(-1, 2))

        # Gate tints: busy, idle, or idle while the other bank has a queue.
        for name, boxes in self.gate_boxes.items():
            other_queue = any(snapshot[o]["waiting"] > 0 for o in snapshot if o != name)
            for slot, box in enumerate(boxes):
                if snapshot[name]["gate_ids"][slot] is not None:
                    box.set_facecolor(GATE_BUSY)
                else:
                    box.set_facecolor(GATE_BLOCKED if other_queue else GATE_IDLE)

        lines = []
        for name, bank in snapshot.items():
            label = "All gates" if name == "All" else f"Bank {name}"
            lines.append(f"{label:9s} waiting {bank['waiting']:3d}   "
                         f"served {bank['served'] - self.served_before[name]:4d}")
        blocked = step_integral(self.blocked_times, self.blocked_gates, self.clip_start, t) / 60.0
        lines.append(f"Blocked capacity {blocked:6.2f} gate-min")
        self.readout.set_text("\n".join(lines))


# =============================================================================
# WRITING THE GIFS
# =============================================================================

def legend_handles():
    """Legend entries: one marker per fare medium, plus the blocked-gate tint."""
    handles = [Line2D([], [], marker=look["marker"], color=look["color"], linestyle="none",
                      markersize=7, label=look["label"]) for look in MEDIUM_LOOK.values()]
    handles.append(Patch(facecolor=GATE_BLOCKED, edgecolor="#555555",
                         label="idle gate the queue cannot use"))
    return handles


def write_gif(fig, panels, params, path, dpi):
    """Animate `panels` over the clip, one frame per simulated second."""
    start = params.animation_start_min * 60.0
    frames = int(params.animation_minutes * 60)
    clock_text = fig.text(0.5, 0.985, "", ha="center", va="top", fontsize=13,
                          fontweight="bold", color=INK)

    def draw_frame(k):
        """Update every panel and the clock to frame k."""
        t = start + k
        clock_text.set_text(clock(t))
        for panel in panels:
            panel.update(t)

    animation = FuncAnimation(fig, draw_frame, frames=frames, interval=1000 / params.animation_fps)
    animation.save(path, writer=PillowWriter(fps=params.animation_fps), dpi=dpi)
    plt.close(fig)
    print(f"  wrote figures/{path.name} ({path.stat().st_size / 1e6:.1f} MB, {frames} frames)")


def make_animations(params, p_qr=None, rep=0, dpi=100):
    """
    Write both GIFs for one replication at QR share p_qr (default: base).
    Both panels of the comparison replay the same arrival table, so the
    only difference between them is the gate configuration.
    """
    p_qr = params.share_qr if p_qr is None else p_qr
    start = params.animation_start_min * 60.0
    arrivals = batch_arrivals(p_qr, params, rep)
    present = simulate(arrivals, params.present_configuration, params, trace=True)
    undivided = simulate(arrivals, "Undivided", params, trace=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    share = f"QR {p_qr * 100:g}%"

    fig, ax = plt.subplots(figsize=(5.6, 6.9))
    panel = ConcoursePanel(ax, present, params, start,
                           f"LRT-1 EDSA Side A entry, present {params.present_configuration} split, {share}")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.08)
    write_gif(fig, [panel], params, FIGURES_DIR / "animation_5_2.gif", dpi)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10.4, 6.6))
    panels = [ConcoursePanel(ax_left, present, params, start,
                             f"Present {params.present_configuration} split, {share}"),
              ConcoursePanel(ax_right, undivided, params, start,
                             f"Undivided: every gate takes every medium, {share}")]
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.07, wspace=0.04)
    write_gif(fig, panels, params, FIGURES_DIR / "animation_comparison.gif", dpi)


if __name__ == "__main__":
    make_animations(Params())
