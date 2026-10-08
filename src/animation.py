"""
animation.py - Animated replay of the Side A concourse (spec Section 7).

Project : After-Office Surge at the LRT-1 EDSA Interchange
Course  : CSS142P Modeling and Simulation, Mapua University
Authors : Aldea, De Leon, Jerusalem

Every frame is drawn from the trace that simulate(trace=True) records once
per simulated second; nothing is animated from inside the SimPy loop.

ConcourseScene works out, for any second of a traced run, where every
passenger stands and what every gate is doing. Two things draw from it, so
they always show the same picture:

  - the GIFs written to figures/ (the spec's presentation artifacts):
        animation_5_2.gif          the present 5/2 split
        animation_comparison.gif   5/2 (left) against Undivided (right), on the
                                   same passengers and the same random seeds
  - the live replay in the app (replay_data(), drawn in the browser).

Run on its own with:   python -m src.animation
"""

import math

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
GATE_FILL = {"idle": "white", "busy": "#e6e6e6",
             "blocked": "#f6c6c0"}  # blocked = idle while the other bank has a queue
GATE_CODE = {"idle": "i", "busy": "b", "blocked": "x"}     # one letter per gate in replay data
POINTS_PER_UNIT = 31        # text size in the GIFs: scene units -> points
INK = "#1a1a1a"
BANK_TITLES = {"All": "Every gate: every fare medium",
               "A": "Bank A: beep + SJT", "B": "Bank B: QR only"}

# =============================================================================
# SCENE GEOMETRY - a schematic of the Side A concourse as observed (scene
# units, not to scale)
#
# Passengers walk up the unpaid corridor from the MRT-3 footbridge at the
# bottom. The two QR gates come first on the corridor's left wall; a short
# walk further up are the five beep/SJT gates, with the ticket vending
# machines beside them and the staffed ticket booths beyond the corridor on
# the right. Through any gate lies the paid side and the northbound platform,
# to the left.
# =============================================================================

EXTENT = (10.6, 12.2)                    # width and height of one panel
CORRIDOR = (4.45, 0.1, 4.15, 10.2)       # x, y, width, height: unpaid corridor
PLATFORM_AREA = (0.2, 4.3, 3.4, 6.0)     # x, y, width, height: paid side, platform
TVM_BOX = (6.55, 6.85, 1.25, 0.55)       # ticket vending machines
BOOTH_BOX = (8.8, 7.0, 1.6, 3.3)         # staffed ticket booths
GATE_X = 4.0
NORMAL_GATE_Y = (9.6, 9.0, 8.4, 7.8, 7.2)    # the five beep/SJT gates, top down
QR_GATE_Y = (5.6, 5.0)                       # the two QR gates, nearer the footbridge
GATE_SIZE = (0.7, 0.46)                      # width, height of one gate box
QUEUE_X = 4.8               # first queue column, just in front of the gates
UNDIVIDED_QUEUE_Y = (6.62, 6.2)              # one shared queue, between the groups
SPACING = 0.42              # gap between queued passengers
ENTRANCE = (6.5, 0.8)       # where the footbridge lands
EXIT_X = 1.2                # passengers fade out here, on the platform

# Display-only walking. The walk is not modelled (the spec treats it as a
# fixed offset); a passenger is drawn walking up the corridor at a constant
# speed during the seconds BEFORE their modelled arrival, so the queue
# counts on screen always match the model.
WALK_SPEED = 1.8                                  # scene units per second
MAX_WALK_S = math.hypot(*EXTENT) / WALK_SPEED     # longest possible walk
EXIT_S = 2.0


def gate_positions(gates):
    """
    Centre of every gate, per bank. Bank A takes the physical gates from the
    top (the five beep/SJT positions first) and Bank B the rest, nearest the
    footbridge; Undivided uses all seven as one pool.
    """
    physical = [(GATE_X, y) for y in NORMAL_GATE_Y + QR_GATE_Y]
    if "All" in gates:
        return {"All": physical}
    n_a = gates["A"]
    return {"A": physical[:n_a], "B": physical[n_a:]}


def queue_rows(gate_xy):
    """
    Per bank, the y of each queue row. A bank's queue lines up in front of
    its own gates and grows out into the corridor; the Undivided pool has one
    shared queue between the two groups of gates.
    """
    return {name: (UNDIVIDED_QUEUE_Y if name == "All" else tuple(y for _x, y in positions))
            for name, positions in gate_xy.items()}


def scene_shapes(gate_xy):
    """
    The static drawing of one panel as plain dicts (rectangles and text),
    so the GIF writer and the browser player draw exactly the same scene.
    Text size is in scene units.
    """
    def box(area, fill, stroke=None):
        """A filled rectangle given as (x, y, width, height)."""
        x, y, w, h = area
        return {"kind": "rect", "x": x, "y": y, "w": w, "h": h, "fill": fill, "stroke": stroke}

    def words(x, y, text, color="#777777", align="left", size=0.24):
        """A one-line text label."""
        return {"kind": "text", "x": x, "y": y, "text": text, "color": color,
                "align": align, "size": size}

    shapes = [
        box(PLATFORM_AREA, "#dde6ef"),
        words(0.4, 9.95, "Platform", "#55606b"),
        words(0.4, 9.6, "(northbound train)", "#55606b"),
        words(1.9, 7.2, "← to the trains", "#55606b", "center"),
        box(CORRIDOR, "#f2f2f2"),
        words(CORRIDOR[0] + 0.15, 3.2, "Unpaid concourse"),
        words(ENTRANCE[0], 0.25, "↑ from MRT-3 footbridge", align="center"),
        box(TVM_BOX, "#ffffff", "#999999"),
        words(TVM_BOX[0] + TVM_BOX[2] / 2, TVM_BOX[1] + 0.2, "TVMs", align="center", size=0.22),
        box(BOOTH_BOX, "#f6dcc6", "#c9a27e"),
        words(BOOTH_BOX[0] + BOOTH_BOX[2] / 2, 8.85, "Ticket", "#8a5a2b", "center", 0.22),
        words(BOOTH_BOX[0] + BOOTH_BOX[2] / 2, 8.5, "booths", "#8a5a2b", "center", 0.22),
    ]
    for name, positions in gate_xy.items():
        top = max(y for _x, y in positions)
        shapes.append(words(QUEUE_X - 0.25, top + GATE_SIZE[1] / 2 + 0.12, BANK_TITLES[name],
                            INK, size=0.23))
    return shapes


def clock(seconds):
    """Seconds after 16:00 as a clock time with seconds, e.g. 17:04:30."""
    s = int(round(seconds))
    return f"{16 + s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# =============================================================================
# THE SCENE - what is where, at any second of a traced run
# =============================================================================

class ConcourseScene:
    """
    The gate array in one traced RunResult, from `clip_start_s` onwards.

    passengers(t) places every visible passenger: walking in across the
    concourse, standing in their bank's queue, at a gate, or stepping onto
    the platform. gate_states(t) says whether each gate is busy, idle, or
    blocked (idle while the other bank has a queue). readout(t) gives the
    on-screen counters, counted from the start of the clip.
    """

    def __init__(self, result, clip_start_s):
        """Index the run so each second can be looked up quickly."""
        self.result = result
        self.clip_start = clip_start_s
        self.gate_xy = gate_positions(result.gates)
        self.queue_rows = queue_rows(self.gate_xy)
        self.times = result.arrivals.times_s
        self.exit_order = np.argsort(result.finished)
        self.finished_sorted = result.finished[self.exit_order]
        self.blocked_times, self.blocked_gates = blocked_gates_steps(result)
        start_snapshot = result.trace[int(clip_start_s)]["banks"]
        self.served_before = {name: start_snapshot[name]["served"] for name in result.gates}

    def snapshot(self, t):
        """The traced state of every bank at second t."""
        return self.result.trace[int(round(t))]["banks"]

    def queue_place(self, name, k):
        """
        Where the k-th person (0 = head) in a bank's queue stands: the first
        column fills in front of the gates, then the queue grows out into the
        corridor one column at a time.
        """
        rows = self.queue_rows[name]
        column, row = divmod(k, len(rows))
        return QUEUE_X + column * SPACING, rows[row]

    def passengers(self, t):
        """Positions of every visible passenger at second t, grouped by medium."""
        snapshot = self.snapshot(t)
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

        # Walking up from the footbridge at one speed, so beep and SJT holders,
        # who must pass the QR gates to reach Bank A, take longer to arrive.
        first, last = np.searchsorted(self.times, [t, t + MAX_WALK_S], side="right")
        for i in range(first, last):
            name = self.result.bank[i]
            target = self.queue_place(name, queue_length[name])
            walk_s = math.dist(ENTRANCE, target) / WALK_SPEED
            remaining = self.times[i] - t
            if remaining > walk_s:
                continue        # still on the footbridge, not yet in view
            queue_length[name] += 1
            progress = 1.0 - remaining / walk_s
            points[media[i]].append((ENTRANCE[0] + progress * (target[0] - ENTRANCE[0]),
                                     ENTRANCE[1] + progress * (target[1] - ENTRANCE[1])))

        # Leaving: through the gate and onto the platform, for EXIT_S seconds.
        first, last = np.searchsorted(self.finished_sorted, [t - EXIT_S, t], side="right")
        for i in self.exit_order[first:last]:
            x, y = self.gate_xy[self.result.bank[i]][self.result.gate_index[i]]
            progress = (t - self.result.finished[i]) / EXIT_S
            points[media[i]].append((x + progress * (EXIT_X - x), y))
        return points

    def gate_states(self, t):
        """Per bank, 'busy', 'idle' or 'blocked' for each gate at second t."""
        snapshot = self.snapshot(t)
        states = {}
        for name, bank in snapshot.items():
            other_queue = any(snapshot[o]["waiting"] > 0 for o in snapshot if o != name)
            states[name] = ["busy" if who is not None else ("blocked" if other_queue else "idle")
                            for who in bank["gate_ids"]]
        return states

    def readout(self, t):
        """Counter lines: waiting and served (since the clip began) per bank,
        and blocked capacity since the clip began."""
        lines = []
        for name, bank in self.snapshot(t).items():
            label = "All gates" if name == "All" else f"Bank {name}"
            lines.append(f"{label:9s} waiting {bank['waiting']:3d}   "
                         f"served {bank['served'] - self.served_before[name]:4d}")
        blocked = step_integral(self.blocked_times, self.blocked_gates, self.clip_start, t) / 60.0
        lines.append(f"Blocked capacity {blocked:6.2f} gate-min")
        return lines


# =============================================================================
# GIF PANELS - the scene drawn with matplotlib
# =============================================================================

class ConcoursePanel:
    """One top-down matplotlib view of a ConcourseScene, updated per frame."""

    def __init__(self, ax, scene, title):
        """Draw the static scene and create the artists updated each frame."""
        self.scene = scene
        ax.set_xlim(0, EXTENT[0])
        ax.set_ylim(0, EXTENT[1])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10, loc="left", color=INK)

        # Static scene: the same shapes the browser player draws.
        for shape in scene_shapes(scene.gate_xy):
            if shape["kind"] == "rect":
                ax.add_patch(Rectangle((shape["x"], shape["y"]), shape["w"], shape["h"],
                                       facecolor=shape["fill"], linewidth=0.8, zorder=0,
                                       edgecolor=shape["stroke"] or shape["fill"]))
            else:
                ax.text(shape["x"], shape["y"], shape["text"], color=shape["color"],
                        ha=shape["align"], fontsize=shape["size"] * POINTS_PER_UNIT, zorder=3)

        self.gate_boxes = {}
        for name, positions in scene.gate_xy.items():
            self.gate_boxes[name] = []
            for x, y in positions:
                box = Rectangle((x - GATE_SIZE[0] / 2, y - GATE_SIZE[1] / 2), *GATE_SIZE,
                                facecolor=GATE_FILL["idle"], edgecolor="#555555",
                                linewidth=0.9, zorder=2)
                ax.add_patch(box)
                self.gate_boxes[name].append(box)

        self.dots = {m: ax.scatter([], [], s=52, marker=MEDIUM_LOOK[m]["marker"],
                                   color=MEDIUM_LOOK[m]["color"], edgecolor="white",
                                   linewidth=0.5, zorder=4) for m in MEDIA}
        self.readout = ax.text(0.3, 11.95, "", fontsize=8, va="top", family="monospace",
                               color=INK)

    def update(self, t):
        """Redraw passengers, gate tints and readout for simulated time t (s)."""
        for medium, points in self.scene.passengers(t).items():
            self.dots[medium].set_offsets(np.array(points).reshape(-1, 2))
        for name, states in self.scene.gate_states(t).items():
            for box, state in zip(self.gate_boxes[name], states):
                box.set_facecolor(GATE_FILL[state])
        self.readout.set_text("\n".join(self.scene.readout(t)))


def legend_handles():
    """Legend entries: one marker per fare medium, plus the blocked-gate tint."""
    handles = [Line2D([], [], marker=look["marker"], color=look["color"], linestyle="none",
                      markersize=7, label=look["label"]) for look in MEDIUM_LOOK.values()]
    handles.append(Patch(facecolor=GATE_FILL["blocked"], edgecolor="#555555",
                         label="idle gate the queue cannot use"))
    return handles


def write_gif(fig, panels, params, path, dpi):
    """Animate `panels` over the clip, one frame per simulated second."""
    start = params.animation_start_min * 60.0
    frames = int(params.gif_minutes * 60)
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


def make_animations(params, rep=0, dpi=100):
    """
    Write the spec's two GIFs at the base QR share: the present split on
    its own, and beside Undivided. Both comparison panels replay the same
    arrival table, so the only difference between them is the gates.
    """
    start = params.animation_start_min * 60.0
    present = params.present_configuration
    share = f"QR {params.share_qr * 100:g}%"
    arrivals = batch_arrivals(params.share_qr, params, rep)
    divided = ConcourseScene(simulate(arrivals, present, params, trace=True), start)
    undivided = ConcourseScene(simulate(arrivals, "Undivided", params, trace=True), start)
    FIGURES_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 6.9))
    panel = ConcoursePanel(ax, divided, f"LRT-1 EDSA Side A entry, present {present} split, {share}")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.08)
    write_gif(fig, [panel], params, FIGURES_DIR / "animation_5_2.gif", dpi)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10.4, 6.6))
    panels = [ConcoursePanel(ax_left, divided, f"Present {present} split, {share}"),
              ConcoursePanel(ax_right, undivided, f"Undivided: every gate takes every medium, {share}")]
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.07, wspace=0.04)
    write_gif(fig, panels, params, FIGURES_DIR / "animation_comparison.gif", dpi)


# =============================================================================
# LIVE REPLAY DATA - the same scene, sent to the app's browser player
# =============================================================================

def replay_data(params, config, p_qr, start_s, seconds, rep=0):
    """
    Everything the app's browser player needs to draw a replay: `config`
    beside Undivided (or Undivided alone) on the same passengers, from
    `start_s` for `seconds` simulated seconds, one frame per second.

    Positions come from ConcourseScene, exactly as in the GIFs; the player
    only draws them. Returned as plain lists and dicts, ready for JSON.
    """
    arrivals = batch_arrivals(p_qr, params, rep)
    configs = [config] if config == "Undivided" else [config, "Undivided"]
    medium_index = {m: k for k, m in enumerate(MEDIA)}
    setting = f"QR {p_qr * 100:g}%, {params.mean_arrival_rate_per_min:g}/min"
    frame_times = [start_s + k for k in range(int(seconds))]

    panels = []
    for name in configs:
        scene = ConcourseScene(simulate(arrivals, name, params, trace=True), start_s)
        title = ("Undivided: every gate takes every medium" if name == "Undivided"
                 else f"{name} split" + (" (present)" if name == params.present_configuration else ""))
        frames = []
        for t in frame_times:
            points = [[round(x, 2), round(y, 2), medium_index[m]]
                      for m, xy in scene.passengers(t).items() for x, y in xy]
            gates = {bank: "".join(GATE_CODE[state] for state in states)
                     for bank, states in scene.gate_states(t).items()}
            frames.append({"p": points, "g": gates, "r": scene.readout(t)})
        panels.append({"title": f"{title}, {setting}",
                       "gates": {bank: [[x, y] for x, y in xy] for bank, xy in scene.gate_xy.items()},
                       "shapes": scene_shapes(scene.gate_xy),
                       "frames": frames})

    return {"clock": [clock(t) for t in frame_times],
            "panels": panels,
            "media": [{"label": MEDIUM_LOOK[m]["label"], "color": MEDIUM_LOOK[m]["color"],
                       "shape": {"o": "circle", "s": "square", "^": "triangle"}[MEDIUM_LOOK[m]["marker"]]}
                      for m in MEDIA],
            "gateFill": {GATE_CODE[state]: fill for state, fill in GATE_FILL.items()},
            "geometry": {"extent": EXTENT, "gateSize": GATE_SIZE}}


if __name__ == "__main__":
    make_animations(Params())
