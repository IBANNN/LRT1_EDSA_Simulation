"""
app.py - Interactive front-end for the oral defense (Stage E).

Project : After-Office Surge at the LRT-1 EDSA Interchange
Course  : CSS142P Modeling and Simulation, Mapua University
Authors : Aldea, De Leon, Jerusalem

Launch from the project root with:   streamlit run app.py

This app does not simulate anything itself. Every number comes from the same
Stage D model the report uses (src/model.py): train-burst arrivals,
16:00-20:00, first 30 minutes discarded, replication k seeded exactly as in
the experiments. Only four inputs are adjustable; everything else stays at
the base case of the assumption register.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.model import (MEDIA, Params, batch_arrivals, measure, simulate,
                       with_mean_arrival_rate)

BASE = Params()

BANK_LABELS = {"A": "Bank A (beep + SJT)", "B": "Bank B (QR only)",
               "All": "All 7 gates (every medium)"}
BANK_COLOURS = {"A": "#0072B2", "B": "#E69F00", "All": "#0072B2"}   # Okabe-Ito
BANK_DASHES = {"A": [1, 0], "B": [5, 3], "All": [1, 0]}


# =============================================================================
# RUNNING THE MODEL
# =============================================================================

@st.cache_data(show_spinner=False)
def run_scenario(config, p_qr, rate_per_min, replications):
    """
    Simulate `replications` runs of one scenario, plus the Undivided
    benchmark on the same passengers (for the cost of division).

    Returns (metrics of each run, Undivided Wq of each run, queue logs of
    the first replication as {bank: (times_s, waiting)}). Cached, so
    re-running an identical scenario is instant.
    """
    params = with_mean_arrival_rate(BASE, rate_per_min)
    rows, undivided_wq, first_logs = [], [], None
    for rep in range(replications):
        arrivals = batch_arrivals(p_qr, params, rep)
        result = simulate(arrivals, config, params)
        rows.append(measure(result, params))
        if config == "Undivided":
            undivided_wq.append(rows[-1]["Wq_s"])
        else:
            undivided_wq.append(measure(simulate(arrivals, "Undivided", params), params)["Wq_s"])
        if rep == 0:
            first_logs = {name: (times, waiting) for name, (times, waiting, _busy)
                          in result.logs.items()}
    return pd.DataFrame(rows), np.array(undivided_wq), first_logs


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def seconds(value):
    """A waiting time for display: '0.083 s', '2.16 s', or a dash if absent."""
    if value is None or np.isnan(value):
        return "-"
    return f"{value:.3f} s" if value < 1 else f"{value:.2f} s"


def passengers(value, replications):
    """A queue length: whole number for one run, one decimal for a mean."""
    return f"{value:.0f}" if replications == 1 else f"{value:.1f}"


def queue_chart(logs, params):
    """
    Step chart of passengers waiting at each bank over the surge
    (replication 1). Scroll to zoom and drag to pan along the time axis.
    """
    frames = [pd.DataFrame({"minute": times / 60.0, "waiting": waiting,
                            "bank": BANK_LABELS[name]})
              for name, (times, waiting) in logs.items()]
    data = pd.concat(frames, ignore_index=True)
    names = list(logs)

    # Minutes after 16:00 shown as clock time, e.g. 75 -> 17:15.
    clock = ("floor(16 + datum.value / 60) + ':' + "
             "(floor(datum.value % 60) < 10 ? '0' : '') + floor(datum.value % 60)")
    lines = alt.Chart(data).mark_line(interpolate="step-after", strokeWidth=1.3).encode(
        x=alt.X("minute:Q", title="Time of day",
                scale=alt.Scale(domain=[0, params.window_min]),
                axis=alt.Axis(labelExpr=clock, tickMinStep=1)),
        y=alt.Y("waiting:Q", title="Passengers waiting"),
        color=alt.Color("bank:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=[BANK_LABELS[n] for n in names],
                                        range=[BANK_COLOURS[n] for n in names])),
        strokeDash=alt.StrokeDash("bank:N", legend=None,
                                  scale=alt.Scale(domain=[BANK_LABELS[n] for n in names],
                                                  range=[BANK_DASHES[n] for n in names])),
    ).interactive(bind_y=False)
    warm_up = alt.Chart(pd.DataFrame({"start": [0.0], "end": [params.warmup_min]})).mark_rect(
        color="#999999", opacity=0.15).encode(x="start:Q", x2="end:Q")
    return alt.layer(warm_up, lines).properties(height=340)


# =============================================================================
# PAGE
# =============================================================================

st.set_page_config(page_title="LRT-1 EDSA fare gates", layout="wide")
st.title("LRT-1 EDSA Side A entry gates: live model")
st.caption("Runs the project's SimPy model (src/model.py) with MRT-3 train-burst arrivals, "
           "16:00-20:00, first 30 minutes discarded. All other inputs are at the base case.")

with st.sidebar.form("controls"):
    st.header("Scenario")
    undivided = st.toggle("Undivided: every gate accepts every medium", value=False)
    n_a = st.slider("Gates on Bank A (beep + SJT)", 2, BASE.total_gates, 5,
                    help=f"Bank B (QR only) gets the other {BASE.total_gates} - n_A gates. "
                         "Ignored when Undivided is on.")
    qr_percent = st.slider("beep QR adoption (% of passengers)", 0, 40, 3)
    rate = st.slider("Mean arrival rate (passengers per minute)", 10, 30, 18,
                     help="Batch size per train = (rate - 0.5 background) x 4-minute headway; "
                          "18 per minute gives the base 70.")
    replications = st.slider("Replications", 1, 10, 5)
    submitted = st.form_submit_button("Run", type="primary", use_container_width=True)

if submitted:
    n_b = BASE.total_gates - n_a
    note = None
    if undivided:
        config = "Undivided"
    elif n_b == 0 and qr_percent > 0:
        st.error(f"With all {BASE.total_gates} gates on Bank A, QR passengers would have no gate "
                 "they are allowed to use. Lower Bank A to 6 or fewer, or set QR adoption to 0%.")
        st.stop()
    elif n_b == 0:
        config = "Undivided"
        note = (f"{BASE.total_gates}/0 with no QR passengers is the same as one pool of "
                f"{BASE.total_gates} gates, so it is run as Undivided.")
    else:
        config = f"{n_a}/{n_b}"
    st.session_state["scenario"] = (config, qr_percent / 100, rate, replications, note)

if "scenario" not in st.session_state:
    st.info("Set a scenario in the sidebar and press **Run**.")
    st.stop()

config, p_qr, rate, replications, note = st.session_state["scenario"]
with st.spinner(f"Simulating {config} at {p_qr:.0%} QR, {rate} per minute, "
                f"{replications} replication(s)..."):
    runs, undivided_wq, logs = run_scenario(config, p_qr, rate, replications)
params = with_mean_arrival_rate(BASE, rate)

mean = runs.mean(numeric_only=True)
cost_of_division = float(np.mean(runs["Wq_s"].to_numpy() - undivided_wq))

# --- One-line verdict --------------------------------------------------------
if config == "Undivided":
    verdict = (f"All {BASE.total_gates} gates: {passengers(mean['Lq_max_All'], replications)} "
               "waiting at peak · no blocked capacity (one shared queue)")
else:
    verdict = (f"Bank A: {passengers(mean['Lq_max_A'], replications)} waiting at peak · "
               f"Bank B: {passengers(mean['Lq_max_B'], replications)} waiting · "
               f"{mean['T_blk_gate_min']:.1f} gate-minutes blocked")
scope = "one run" if replications == 1 else f"mean of {replications} runs"
st.subheader(verdict)
st.caption(f"Configuration {config} · QR {p_qr:.0%} · {rate} passengers per minute "
           f"(batch {params.batch_size:g} per train) · {scope}")
if note:
    st.info(note)

# --- Waiting times -------------------------------------------------------------
st.markdown("**Mean waiting time in queue**")
columns = st.columns(4)
columns[0].metric("All passengers", seconds(mean["Wq_s"]))
for column, medium, label in zip(columns[1:], MEDIA, ("beep card", "SJT", "beep QR")):
    column.metric(label, seconds(mean[f"Wq_{medium}_s"]))

# --- Queues and blocked capacity ----------------------------------------------
st.markdown("**Queues and capacity**")
columns = st.columns(4)
if config == "Undivided":
    columns[0].metric("Peak queue, all gates", passengers(mean["Lq_max_All"], replications))
    columns[1].metric("Peak queue, Bank B", "-", help="No separate QR bank when Undivided.")
else:
    columns[0].metric("Peak queue, Bank A", passengers(mean["Lq_max_A"], replications))
    columns[1].metric("Peak queue, Bank B", passengers(mean["Lq_max_B"], replications))
columns[2].metric("Blocked capacity", f"{mean['T_blk_gate_min']:.1f} gate-min",
                  help="Gate time idle at one bank while the other bank had a queue, after warm-up.")
columns[3].metric("Cost of division", f"{cost_of_division:+.3f} s",
                  help="Extra mean wait per passenger versus Undivided, on the same passengers.")

# --- Queue length over time ------------------------------------------------------
st.markdown("**Queue length over time** (replication 1; grey band = warm-up; "
            "scroll to zoom, drag to pan, double-click to reset)")
st.altair_chart(queue_chart(logs, params), use_container_width=True)
