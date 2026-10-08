# LRT-1 EDSA Side A Fare Gate Simulation

Discrete-event simulation (Python + SimPy) of the seven entry turnstiles at the
LRT-1 EDSA Station Side A concourse during the 16:00–20:00 after-office surge.
CSS142P Modeling and Simulation — Aldea, De Leon, Jerusalem.

**Live app:** https://lrt1-edsa-gates.streamlit.app/ — no install needed. It
sleeps after 12 hours without visitors; the first visit after that takes about
a minute to wake it.

## How to run

```
pip install -r requirements.txt
python -m src.experiments
```

Run from this folder (the project root). The single command runs everything in
order — verification, stage checks, validation and degeneracy tests, the
900-run sweep, the adoption-threshold search, the sensitivity analysis, all
figures and both animations — in about 45 minutes.

To run one step only, name it: `python -m src.experiments SWEEP`. The steps are
`A B C D VALID SWEEP THRESH SENS FIGURES ANIM`. Figures and animations can also be redrawn on
their own (`python -m src.plots`, `python -m src.animation`); the figures read
the CSVs in `results/`.

### Interactive app (for the oral defense)

```
streamlit run app.py
```

If Windows says `streamlit` is not recognised, start it through Python instead:

```
python -m streamlit run app.py
```

Opens in the browser at http://localhost:8501 (paste the address in if it does
not open by itself); keep the terminal open while using it and press Ctrl+C
there to stop.

- **Pre-made scenario** (sidebar): one click runs a situation from the
  proposal — the present split, the Undivided benchmark, the best split today,
  QR use at 20% and 30%, the best split at 30%, heavier demand, and "lines
  forming" (30 per minute, beyond the register's range, to show what it takes
  for a queue to build with every train). Or set the
  sliders (gate split or Undivided, QR adoption, arrival rate, replications)
  and press **Run**.
- **Results** tab: waits overall and per fare medium, peak queues, blocked
  capacity, cost of division, and the queue-length chart.
- **Animation** tab: a live top-down replay of the current scenario beside
  Undivided on the same passengers, with play/pause, speed and a time slider.

The app calls the same model as the experiments (`src/model.py`); it does not
simulate anything itself. A typical run of 5 replications takes about 2 seconds.

## Files

| File | What it does |
|---|---|
| `MODEL.md` | Model specification: process flow, entity-attribute table, eligibility matrix, assumption register. |
| `src/model.py` | The simulation. All parameters are in the `Params` dataclass at the top; the eligibility matrix is under GATE LAYOUT. |
| `src/mmc.py` | Analytical M/M/c (Erlang C) and the M/G/c approximation used to check the simulation. |
| `src/experiments.py` | Verification, stage checks, validation, sweep, recommendations, threshold search, sensitivity analysis. |
| `src/plots.py` | Figures 0–10 (greyscale-safe: series differ by marker and line style). |
| `src/animation.py` | The concourse scene from the model's `trace=True` snapshots: draws the two GIFs and feeds the app's live replay. |
| `src/scenarios.py` | The app's pre-made scenarios and what each one demonstrates. |
| `src/replay_player.html` | Browser player for the app's live replay (drawing only). |
| `app.py` | Stage E Streamlit front-end: re-runs the model live with chosen inputs. |
| `results/verification.csv` | Stage A: one pool of 7 gates as M/M/7, simulated vs analytical Wq at ρ = 0.5, 0.7, 0.9. |
| `results/verification_banks.csv` | Stage C: Bank A as M/M/5 and Bank B as M/M/2 under 5/2. |
| `results/stage_b_check.csv` | Stage B: three fare media on shared gates — overall and per-class waits. |
| `results/stage_c_check.csv` | Stage C: every configuration at 3% and 30% QR — traffic and utilisation per bank, misrouted count (always 0). |
| `results/stage_d_check.csv` | Stage D: batch arrivals, every configuration at 3% QR, full metric set. |
| `results/validation.csv` | Volume vs published ridership, and degeneracy tests at zero, very light and saturating load. |
| `results/event_trace_sample.csv` | Every event for 60 passengers from 17:00, for event-trace inspection. |
| `results/runs.csv` | The sweep: one row per run (config, QR share, replication, every metric). |
| `results/summary.csv` | Mean and 95% CI of every metric per (QR share, configuration), plus the cost of division. |
| `results/optimal_allocation.csv` | Best allocation at each QR share (by wait, peak queue and clearance) and what 5/2 costs. |
| `results/batch_vs_smooth.csv` | 5/2 under train bursts vs smooth arrivals with the same demand profile. |
| `results/recommendations.csv` | Recommended split per QR share, with cost of division and expected change per passenger class. |
| `results/threshold.csv` | QR adoption from 3% to 30% in 1-point steps: where two QR gates stop being enough. |
| `results/sensitivity.csv`, `sensitivity_optimal.csv` | One-factor-at-a-time sensitivity analysis. |
| `figures/concourse_layout.png`, `concourse_layout_deck.png` | The Side A concourse as observed (report Figure 1), in greyscale for print and in the deck's colours. |
| `figures/fig0`–`fig10` | Figure 0 (process flow), Figures 1–8 of the spec, Figure 9 (sensitivity) and Figure 10 (adoption threshold). |
| `figures/animation_5_2.gif`, `animation_comparison.gif` | 8-minute replay from 17:00 (two train cycles) played in about 29 s for slides; 5/2 alone, and 5/2 beside Undivided on the same passengers. |

## Conventions and modelling decisions

- **Configurations.** Every feasible split, 6/1 to 2/5, plus Undivided
  (proposal Objective 6). 7/0 is infeasible: QR holders would have no gate.
- **Time unit.** The SimPy clock runs in seconds. Parameters keep the units of
  the assumption register and are converted only in `Params`.
- **Statistics.** 30 replications per cell; passengers arriving in the first
  30 minutes are discarded; means carry 95% t-based confidence intervals.
- **Common random numbers.** Three random streams per replication (arrivals,
  fare medium, service), seeded from the replication index. Each passenger's
  arrival time, medium and service time are drawn before the run, so every
  configuration replays exactly the same passengers. The cost of division is
  the paired difference against Undivided within each replication.
- **Arrivals.** One MRT-3 train every 4 minutes from 16:00; batch size is
  70 × the normalised block multiplier, rounded to a whole passenger, spread
  Uniform(0, 90 s). Street-level background arrivals are Poisson at
  0.5/min × the same multiplier. After normalisation the peak multiplier is 1.43.
- **Burst clearance time.** Time from a train's arrival until its bank's queue
  empties for the *last* time before the next train (0 if no queue formed);
  censored, and counted in residual carryover, if the queue is still non-empty
  when the next train arrives.
- **Smooth comparison (Figure 8).** Non-stationary Poisson arrivals with the same
  15-minute profile and expected volume as the batch process, so the only
  difference is batching.
- **Verification settings.** Stage A uses μ from the base-mix mean service
  time (2.92 s). The per-bank test sets the QR share to 17.1% so both banks run
  at the same utilisation; it is a test setting, not a scenario.
- **Sensitivity.** Headway is varied with the mean arrival rate held fixed
  (batch size scales with headway); service-time cases scale all three
  triangular points. A combined case pairs the largest batch with the shortest
  dispersal.
- **Animation.** Passengers enter from the MRT-3 footbridge on the QR side and
  walk past the QR gates to Bank A. Which physical gate a passenger uses, and
  the walk in and out, are display only; queue counts on screen match the model.
