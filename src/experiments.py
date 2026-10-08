"""
experiments.py - Verification tests, stage checks, the experiment sweep and
the sensitivity analysis.

Project : After-Office Surge at the LRT-1 EDSA Interchange
Course  : CSS142P Modeling and Simulation, Mapua University
Authors : Aldea, De Leon, Jerusalem

Run from the project root:
    python -m src.experiments          everything, in order (about 25 minutes)
    python -m src.experiments SWEEP    one step only; the steps are
                                       A B C D SWEEP SENS FIGURES ANIM
"""

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.mmc import mgc_wq_approx, mmc_summary
from src.model import (MEDIA, Params, batch_arrivals, blocked_capacity_gate_min,
                       exponential_arrivals, fare_mix, gate_layout,
                       idle_gates_while_queue_s, mean_triangular, measure,
                       mix_service_stats, mmc_arrivals, simulate, smooth_arrivals,
                       smooth_profile_arrivals, step_integral)
from src.animation import make_animations
from src.plots import make_all_figures

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def mean_ci(values, confidence):
    """
    Mean and confidence interval across replications, using the
    t-distribution with n - 1 degrees of freedom (spec Section 5).

    Returns (mean, low, high).
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = x.mean()
    half_width = stats.t.ppf((1 + confidence) / 2, df=n - 1) * x.std(ddof=1) / math.sqrt(n)
    return mean, mean - half_width, mean + half_width


def verification_rates(params):
    """
    Gate count c, service rate mu (per second) and the arrival rate lambda
    (per second) for each load in params.verification_loads.

    mu comes from the base-case mean service time, so lambda = rho * c * mu
    gives the same utilisation in Stage A (exponential service) and Stage B
    (triangular service with the same mean).
    """
    c = params.total_gates
    mean_service_s, _scv = mix_service_stats(params, params.share_qr)
    mu_per_s = 1.0 / mean_service_s
    return c, mu_per_s, {rho: rho * c * mu_per_s for rho in params.verification_loads}


# =============================================================================
# STAGE A - verification against the analytical M/M/c queue
# =============================================================================

def run_verification(params):
    """
    Compare simulated and analytical Wq for an M/M/c queue at each load in
    params.verification_loads.

    c is the full array of 7 gates and mu comes from the base-case mean
    service time, so the only thing that changes between rows is lambda.
    The test passes at a load when the analytical Wq lies inside the
    simulated 95% confidence interval.
    """
    c, mu_per_s, lambdas = verification_rates(params)

    rows = []
    for rho, lam_per_s in lambdas.items():
        analytical = mmc_summary(lam_per_s, mu_per_s, c)

        rep_means = []
        for rep in range(params.replications):
            arrivals = mmc_arrivals(lam_per_s, mu_per_s, params, rep)
            rep_means.append(measure(simulate(arrivals, "Undivided", params), params)["Wq_s"])
        sim_mean, ci_low, ci_high = mean_ci(rep_means, params.confidence)

        passed = ci_low <= analytical["Wq"] <= ci_high
        print(f"  verification rho={rho:.1f}: analytical Wq={analytical['Wq']:.3f}s  "
              f"simulated {sim_mean:.3f}s [{ci_low:.3f}, {ci_high:.3f}]  "
              f"{'PASS' if passed else 'FAIL'}")

        rows.append({
            "load_rho": rho,
            "lambda_per_min": lam_per_s * 60,
            "mu_per_min": mu_per_s * 60,
            "c": c,
            "P_wait": analytical["P_wait"],
            "Wq_analytical_s": analytical["Wq"],
            "Wq_simulated_s": sim_mean,
            "ci95_low_s": ci_low,
            "ci95_high_s": ci_high,
            "result": "PASS" if passed else "FAIL",
        })
    return pd.DataFrame(rows)


# =============================================================================
# STAGE B - three fare media sharing all seven gates
# =============================================================================

def run_stage_b_check(params):
    """
    Three fare media with triangular service, all 7 gates shared
    (Undivided), smooth Poisson arrivals, base-case QR share.

    Runs at the Stage A loads and at base demand (18 per minute). For each
    it reports:
      - overall Wq beside the Stage A M/M/c value at the same lambda and the
        M/G/c approximation, which predicts how much the less variable
        triangular service should shorten the wait;
      - Wq and W (wait + own service) per fare medium;
      - the paired difference Wq(QR) - Wq(beep) within each replication,
        with its 95% CI. In a shared first-come-first-served queue a
        passenger's wait does not depend on its own medium, so this CI
        should contain zero.
    """
    c, mu_per_s, lambdas = verification_rates(params)
    mean_service_s, scv = mix_service_stats(params, params.share_qr)

    cases = [(f"rho={rho:.1f}", lam) for rho, lam in lambdas.items()]
    cases.append(("base demand", params.mean_arrival_rate_per_s))

    rows = []
    for label, lam_per_s in cases:
        runs = [measure(simulate(smooth_arrivals(lam_per_s, params.share_qr, params, rep),
                                 "Undivided", params), params)
                for rep in range(params.replications)]
        runs = pd.DataFrame(runs)

        row = {
            "case": label,
            "rho": lam_per_s * mean_service_s / c,
            "lambda_per_min": lam_per_s * 60,
            "Wq_MMc_s": mmc_summary(lam_per_s, mu_per_s, c)["Wq"],
            "Wq_MGc_approx_s": mgc_wq_approx(lam_per_s, mean_service_s, scv, c),
        }
        row["Wq_sim_s"], row["Wq_ci_low_s"], row["Wq_ci_high_s"] = \
            mean_ci(runs["Wq_s"], params.confidence)
        for medium in MEDIA:
            row[f"Wq_{medium}_s"] = runs[f"Wq_{medium}_s"].mean()
        for medium in MEDIA:
            row[f"W_{medium}_s"] = runs[f"W_{medium}_s"].mean()

        diff = runs["Wq_qr_s"] - runs["Wq_beep_s"]
        _, row["dWq_qr_beep_ci_low_s"], row["dWq_qr_beep_ci_high_s"] = \
            mean_ci(diff, params.confidence)
        rows.append(row)

        print(f"  stage B {label:11s}: Wq={row['Wq_sim_s']:.3f}s  "
              f"(M/M/c {row['Wq_MMc_s']:.3f}s, M/G/c approx {row['Wq_MGc_approx_s']:.3f}s)")

    print(f"  service time of a random passenger: mean {mean_service_s:.3f}s, "
          f"scv {scv:.4f}, so M/G/c predicts Wq ~ {(1 + scv) / 2:.3f} x M/M/c")
    return pd.DataFrame(rows)


# =============================================================================
# STAGE C - the strict partition
# =============================================================================

def count_misrouted(result):
    """
    Passengers served at a bank their fare medium may not use, checked
    against the rule as worded in the spec rather than the model's own
    lookup table: QR only at Bank B, beep and SJT only at Bank A.
    Always 0 for Undivided, which has a single pool.
    """
    if result.config == "Undivided":
        return 0
    is_qr = result.arrivals.media == "qr"
    return int(np.sum(is_qr & (result.bank == "A")) + np.sum(~is_qr & (result.bank == "B")))


def run_stage_c_check(params):
    """
    Every configuration at the lowest and highest QR shares in the grid,
    base demand, smooth Poisson arrivals, triangular service.

    Reports how traffic and utilisation split between the banks and counts
    passengers served at a bank their medium may not use (must be 0).
    Under 5/2 at 3% QR, Bank B should be nearly idle while Bank A carries
    nearly all the traffic.
    """
    rate = params.mean_arrival_rate_per_s
    rows = []
    for p_qr in (min(params.qr_shares), max(params.qr_shares)):
        for config in params.configurations:
            results = [simulate(smooth_arrivals(rate, p_qr, params, rep), config, params)
                       for rep in range(params.replications)]
            runs = pd.DataFrame([measure(r, params) for r in results])

            row = {"p_qr": p_qr, "config": config, "Wq_s": runs["Wq_s"].mean()}
            for medium in MEDIA:
                row[f"Wq_{medium}_s"] = runs[f"Wq_{medium}_s"].mean()
            for name in gate_layout(config, params):
                row[f"share_{name}"] = (runs[f"n_{name}"] / runs["n_passengers"]).mean()
                row[f"rho_{name}"] = runs[f"rho_{name}"].mean()
            row["misrouted"] = sum(count_misrouted(r) for r in results)
            rows.append(row)
            print(f"  stage C p_qr={p_qr:.2f} {config:9s}: Wq={row['Wq_s']:.3f}s  "
                  f"misrouted={row['misrouted']}")

    columns = (["p_qr", "config", "Wq_s"] + [f"Wq_{m}_s" for m in MEDIA]
               + ["share_All", "share_A", "share_B", "rho_All", "rho_A", "rho_B", "misrouted"])
    return pd.DataFrame(rows)[columns]


def bank_mean_service_s(params):
    """
    Mean triangular service time (seconds) of the passengers each bank
    serves: Bank A takes beep and SJT in their fixed base ratio, Bank B
    takes QR only.
    """
    mix = fare_mix(params, 0.0)     # the beep:SJT ratio does not depend on p_qr
    s_beep = mean_triangular(*params.service_triangle_s("beep"))
    s_sjt = mean_triangular(*params.service_triangle_s("sjt"))
    s_a = (mix["beep"] * s_beep + mix["sjt"] * s_sjt) / (mix["beep"] + mix["sjt"])
    return {"A": s_a, "B": mean_triangular(*params.service_triangle_s("qr"))}


def run_bank_verification(params, config="5/2"):
    """
    Per-bank M/M/c verification through the partitioned model, under the
    present 5/2 split (proposal Objective 5).

    With Poisson arrivals and an independent random fare medium per
    passenger, each bank receives its own Poisson stream. Under strict
    division Bank A is therefore an M/M/5 queue and Bank B an M/M/2 queue,
    independent of each other. Service is exponential, with each bank's
    mean triangular service time.

    The QR share is set so that both banks run at the same utilisation,
    which lets one run test both banks at each load:
        (1 - p) * s_A / n_A  =  p * s_B / n_B
    It is a verification setting, not an adoption scenario.
    """
    gates = gate_layout(config, params)
    s = bank_mean_service_s(params)
    a_term, b_term = s["A"] / gates["A"], s["B"] / gates["B"]
    p_qr = a_term / (a_term + b_term)
    exponential_means = {"beep": s["A"], "sjt": s["A"], "qr": s["B"]}
    print(f"  bank verification uses p_qr = {p_qr:.4f} so both banks share one load")

    rows = []
    for rho in params.verification_loads:
        lam_per_s = rho * gates["A"] / ((1 - p_qr) * s["A"])
        runs = pd.DataFrame([
            measure(simulate(exponential_arrivals(lam_per_s, p_qr, exponential_means, params, rep),
                             config, params), params)
            for rep in range(params.replications)])

        for name, bank_share in (("A", 1 - p_qr), ("B", p_qr)):
            lam_bank = lam_per_s * bank_share
            analytical = mmc_summary(lam_bank, 1.0 / s[name], gates[name])
            sim_mean, ci_low, ci_high = mean_ci(runs[f"Wq_bank_{name}_s"], params.confidence)
            passed = ci_low <= analytical["Wq"] <= ci_high
            print(f"  bank {name} M/M/{gates[name]} rho={rho:.1f}: analytical Wq="
                  f"{analytical['Wq']:.3f}s  simulated {sim_mean:.3f}s "
                  f"[{ci_low:.3f}, {ci_high:.3f}]  {'PASS' if passed else 'FAIL'}")
            rows.append({
                "load_rho": rho,
                "bank": name,
                "c": gates[name],
                "lambda_per_min": lam_bank * 60,
                "mu_per_min": 60.0 / s[name],
                "Wq_analytical_s": analytical["Wq"],
                "Wq_simulated_s": sim_mean,
                "ci95_low_s": ci_low,
                "ci95_high_s": ci_high,
                "result": "PASS" if passed else "FAIL",
                "rho_simulated": runs[f"rho_{name}"].mean(),
            })
    return pd.DataFrame(rows)


# =============================================================================
# STAGE D - batch arrivals and the full metric set
# =============================================================================

def self_test(result, params):
    """
    Checks every Stage D run must pass. Returns a list of failures; an
    empty list means the run passed.

      - conservation: every passenger was served, and finished after arriving
      - blocked-capacity time is exactly zero under Undivided
      - work conservation: no gate ever stood idle while its own bank had a
        queue (if this failed, the model would be creating waits from nothing)
    """
    failures = []
    if np.isnan(result.waits).any() or (result.finished < result.arrivals.times_s).any():
        failures.append(f"{result.config}: not every passenger was served")
    if result.config == "Undivided" and blocked_capacity_gate_min(result, params) != 0.0:
        failures.append("Undivided: blocked-capacity time is not zero")
    run_end = result.finished.max()
    for name in result.gates:
        if idle_gates_while_queue_s(result, name, name, 0.0, run_end) > 0:
            failures.append(f"{result.config}: idle gate at bank {name} with a queue waiting")
    return failures


def queue_by_cycle_phase(results, bank, params, bin_s=15.0):
    """
    Time-average queue length at `bank` by position within the train cycle.

    Bin j covers the seconds [j*bin_s, (j+1)*bin_s) after a train arrives,
    averaged over every post-warm-up train in every replication. A sawtooth
    shows as high values in the first bins (the burst walking in) falling to
    about zero in the last ones (queue drained before the next train).
    """
    n_bins = int(params.headway_s // bin_s)
    totals = np.zeros(n_bins)
    n_cycles = 0
    for result in results:
        times, waiting, _busy = result.logs[bank]
        trains = result.arrivals.train_times_s
        for t_train in trains[trains >= params.warmup_s]:
            for j in range(n_bins):
                start = t_train + j * bin_s
                totals[j] += step_integral(times, waiting, start, start + bin_s) / bin_s
            n_cycles += 1
    return totals / n_cycles


def summarise_config(runs, baseline_wq, config, params):
    """
    One row of means across replications for a configuration, including
    the cost of division dW_div: the paired difference in overall Wq
    against Undivided in the same replication (common random numbers make
    the pairing valid), with its 95% CI.
    """
    row = {"config": config}
    row["Wq_s"], row["Wq_ci_low_s"], row["Wq_ci_high_s"] = mean_ci(runs["Wq_s"], params.confidence)
    for medium in MEDIA:
        row[f"Wq_{medium}_s"] = runs[f"Wq_{medium}_s"].mean()
    row["dW_s"] = runs["dW_s"].mean()
    row["dW_div_s"], row["dW_div_ci_low_s"], row["dW_div_ci_high_s"] = \
        mean_ci(runs["Wq_s"].values - baseline_wq.values, params.confidence)
    for name in gate_layout(config, params):
        row[f"Lq_max_{name}"] = runs[f"Lq_max_{name}"].mean()
        row[f"T_clr_{name}_s"] = runs[f"T_clr_{name}_s"].mean()
        row[f"T_clr_censored_{name}"] = runs[f"T_clr_censored_{name}"].sum()
        row[f"R_{name}"] = runs[f"R_{name}"].mean()
        row[f"rho_{name}"] = runs[f"rho_{name}"].mean()
    row["T_blk_gate_min"] = runs["T_blk_gate_min"].mean()
    row["X_per_hr"] = runs["X_per_hr"].mean()
    return row


def run_stage_d_check(params):
    """
    Batch arrivals with the full metric set: every configuration at the
    base QR share, 30 replications each.

    Also runs the self-tests on every replication, measures the queue by
    position in the train cycle (the sawtooth check), compares batch with
    smooth arrivals under 5/2, and draws Figure 1 from replication 0 of 5/2.
    Returns the per-configuration summary table.
    """
    p_qr = params.share_qr
    reps = range(params.replications)
    failures = []
    runs_by_config = {}
    results_5_2 = None

    for config in params.configurations:
        results = [simulate(batch_arrivals(p_qr, params, rep), config, params) for rep in reps]
        for result in results:
            failures += self_test(result, params)
        runs_by_config[config] = pd.DataFrame([measure(r, params) for r in results])
        if config == "5/2":
            results_5_2 = results
        print(f"  stage D {config:9s}: Wq={runs_by_config[config]['Wq_s'].mean():.3f}s  "
              f"T_blk={runs_by_config[config]['T_blk_gate_min'].mean():.2f} gate-min")

    print(f"  self-tests on {len(params.configurations) * params.replications} runs: "
          + ("all passed" if not failures else f"{len(failures)} FAILED"))
    for failure in failures[:10]:
        print("    " + failure)

    rows = [summarise_config(runs, runs_by_config["Undivided"]["Wq_s"], config, params)
            for config, runs in runs_by_config.items()]
    table = pd.DataFrame(rows)

    # Sawtooth check: queue by position in the 4-minute train cycle.
    for bank in ("A", "B"):
        profile = queue_by_cycle_phase(results_5_2, bank, params)
        print(f"  5/2 Bank {bank} mean queue by 15-s phase after a train: "
              + " ".join(f"{q:.2f}" for q in profile))

    # Batch versus smooth arrivals, same configuration and same demand profile.
    smooth = pd.DataFrame([measure(simulate(smooth_profile_arrivals(p_qr, params, rep),
                                            "5/2", params), params) for rep in reps])
    batch = runs_by_config["5/2"]
    for label, column in (("Wq overall (s)", "Wq_s"), ("Lq_max Bank A", "Lq_max_A"),
                          ("T_blk (gate-min)", "T_blk_gate_min")):
        print(f"  5/2 batch vs smooth, {label:16s}: batch {batch[column].mean():.3f}  "
              f"smooth {smooth[column].mean():.3f}")

    return table


# =============================================================================
# EXPERIMENT SWEEP - every configuration x every QR share x 30 replications
# =============================================================================

def add_mean_ci(row, name, values, params):
    """
    Add <name>_mean, <name>_ci_low and <name>_ci_high to `row`, ignoring
    NaN. With fewer than two values there is no interval, so the CI is NaN
    (a metric that does not apply, such as Bank B under Undivided, is NaN
    throughout).
    """
    values = pd.Series(values).dropna()
    if len(values) >= 2:
        mean, low, high = mean_ci(values, params.confidence)
    else:
        mean, low, high = (values.mean() if len(values) else np.nan), np.nan, np.nan
    row[f"{name}_mean"], row[f"{name}_ci_low"], row[f"{name}_ci_high"] = mean, low, high


def run_sweep(params):
    """
    The full experiment grid of Section 3.5 under batch arrivals: one row
    per run with config, p_qr, rep and every metric (results/runs.csv).

    For each (QR share, replication) the arrival table is generated once and
    replayed through all five configurations, so the configurations are
    compared on literally the same passengers (common random numbers).
    """
    reps = range(params.replications)
    rows = []
    for p_qr in params.qr_shares:
        arrivals = [batch_arrivals(p_qr, params, rep) for rep in reps]
        for config in params.configurations:
            cell = [{"config": config, "p_qr": p_qr, "rep": rep,
                     **measure(simulate(arrivals[rep], config, params), params)}
                    for rep in reps]
            rows += cell
            cell = pd.DataFrame(cell)
            print(f"  sweep p_qr={p_qr:.2f} {config:9s}: Wq={cell['Wq_s'].mean():.3f}s  "
                  f"T_blk={cell['T_blk_gate_min'].mean():6.2f} gate-min  ({len(cell)} runs)")
    return pd.DataFrame(rows)


def summarise_runs(runs, params):
    """
    results/summary.csv: for every (QR share, configuration) cell, the mean
    and 95% CI of every metric across replications, plus the cost of
    division dW_div - the paired difference in overall Wq against Undivided
    in the same replication, which common random numbers make valid.
    """
    metric_columns = [c for c in runs.columns if c not in ("config", "p_qr", "rep")]
    rows = []
    for (p_qr, config), cell in runs.groupby(["p_qr", "config"], sort=False):
        undivided = runs[(runs.p_qr == p_qr) & (runs.config == "Undivided")].set_index("rep")
        cell = cell.set_index("rep")
        row = {"p_qr": p_qr, "config": config, "replications": len(cell)}
        for column in metric_columns:
            add_mean_ci(row, column, cell[column], params)
        add_mean_ci(row, "dW_div_s", cell["Wq_s"] - undivided["Wq_s"], params)
        rows.append(row)
    return pd.DataFrame(rows)


def optimal_allocation(runs, params):
    """
    results/optimal_allocation.csv: the recommended allocation at each QR
    share, judged three ways among the divided configurations:

      best_by_wait  : lowest mean Wq - the research question's criterion,
                      and the same as the lowest cost of division;
      best_by_queue : lowest mean peak queue at the worse of the two banks;
      best_by_clear : lowest mean burst clearance time at the worse bank.

    For the wait-optimal allocation and for the present split it reports
    the cost of division, and the saving of the optimum over the present
    split, each as a paired 95% CI.
    """
    present = params.present_configuration
    divided = [c for c in params.configurations if c != "Undivided"]
    rows = []
    for p_qr in params.qr_shares:
        level = runs[runs.p_qr == p_qr]
        cell = {c: level[level.config == c].set_index("rep") for c in params.configurations}
        mean_wq = {c: cell[c]["Wq_s"].mean() for c in divided}
        worst_queue = {c: cell[c][["Lq_max_A", "Lq_max_B"]].max(axis=1).mean() for c in divided}
        worst_clear = {c: cell[c][["T_clr_A_s", "T_clr_B_s"]].max(axis=1).mean() for c in divided}
        best = min(mean_wq, key=mean_wq.get)

        row = {"p_qr": p_qr,
               "best_by_wait": best,
               "best_by_queue": min(worst_queue, key=worst_queue.get),
               "best_by_clear": min(worst_clear, key=worst_clear.get),
               "Wq_best_s": mean_wq[best],
               "Wq_present_s": mean_wq[present],
               "Wq_undivided_s": cell["Undivided"]["Wq_s"].mean()}
        add_mean_ci(row, "dW_div_best_s", cell[best]["Wq_s"] - cell["Undivided"]["Wq_s"], params)
        add_mean_ci(row, "dW_div_present_s", cell[present]["Wq_s"] - cell["Undivided"]["Wq_s"], params)
        add_mean_ci(row, "saving_best_vs_present_s", cell[present]["Wq_s"] - cell[best]["Wq_s"], params)
        row["Lq_max_worst_best"] = worst_queue[best]
        row["Lq_max_worst_present"] = worst_queue[present]
        row["Lq_max_undivided"] = cell["Undivided"]["Lq_max_All"].mean()
        rows.append(row)
        print(f"  optimum at p_qr={p_qr:.2f}: {best} by wait, {row['best_by_queue']} by peak queue, "
              f"{row['best_by_clear']} by clearance; present {present} costs "
              f"{row['dW_div_present_s_mean']:.3f}s vs Undivided")
    return pd.DataFrame(rows)


def run_batch_vs_smooth(runs, params):
    """
    results/batch_vs_smooth.csv (Figure 8): the present configuration under
    batch arrivals (taken from the sweep) and under smooth arrivals with
    the same 15-minute demand profile, at every QR share. The gap is the
    queueing a smooth Poisson assumption would conceal.
    """
    present = params.present_configuration
    reps = range(params.replications)
    rows = []
    for p_qr in params.qr_shares:
        smooth = pd.DataFrame([measure(simulate(smooth_profile_arrivals(p_qr, params, rep),
                                                present, params), params) for rep in reps])
        batch = runs[(runs.p_qr == p_qr) & (runs.config == present)]
        for label, cell in (("batch", batch), ("smooth", smooth)):
            row = {"p_qr": p_qr, "config": present, "arrivals": label}
            for column in ("Wq_s", "Wq_qr_s", "Lq_max_A", "Lq_max_B", "T_blk_gate_min"):
                add_mean_ci(row, column, cell[column], params)
            rows.append(row)
        print(f"  {present} p_qr={p_qr:.2f}: Wq batch {batch['Wq_s'].mean():.3f}s "
              f"vs smooth {smooth['Wq_s'].mean():.3f}s")
    return pd.DataFrame(rows)


# =============================================================================
# SENSITIVITY ANALYSIS - one factor at a time about the base case
# =============================================================================

def sensitivity_cases(params):
    """
    The one-factor-at-a-time cases as (factor, level, Params). Each case
    moves one input to one end of its range and leaves the rest at base.

    - Headway is varied with the mean arrival rate held fixed, so batch size
      scales with it (70 x h / 4): more, smaller trains at 3 min; fewer,
      larger ones at 5 min. Varying demand itself is the batch-size case.
    - Service-time cases scale all three triangular points by the factor.
    - The last case combines the largest batch with the shortest dispersal:
      the densest bursts the ranges allow.
    """
    def scaled(triangle, factor):
        """A triangular (low, mode, high) with every point times `factor`."""
        return tuple(x * factor for x in triangle)

    cases = [("base", "base", params)]
    for size in params.sens_batch_size:
        cases.append(("batch size", f"{size:g} passengers", replace(params, batch_size=size)))
    for headway in params.sens_headway_min:
        cases.append(("headway", f"{headway:g} min",
                      replace(params, headway_min=headway,
                              batch_size=params.batch_size * headway / params.headway_min)))
    for dispersal in params.sens_dispersal_sec:
        cases.append(("dispersal", f"{dispersal:g} s", replace(params, dispersal_sec=dispersal)))
    for rate in params.sens_background_rate_per_min:
        cases.append(("background rate", f"{rate:g} per min",
                      replace(params, background_rate_per_min=rate)))
    for factor in params.sens_service_beep_scale:
        cases.append(("beep service", f"x{factor:g}",
                      replace(params, service_beep_sec=scaled(params.service_beep_sec, factor))))
    for factor in params.sens_service_sjt_scale:
        cases.append(("SJT service", f"x{factor:g}",
                      replace(params, service_sjt_sec=scaled(params.service_sjt_sec, factor))))
    for factor in params.sens_service_qr_scale:
        cases.append(("QR service", f"x{factor:g}",
                      replace(params, service_qr_sec=scaled(params.service_qr_sec, factor))))
    densest = replace(params, batch_size=max(params.sens_batch_size),
                      dispersal_sec=min(params.sens_dispersal_sec))
    cases.append(("combined", f"batch {max(params.sens_batch_size):g} + "
                              f"dispersal {min(params.sens_dispersal_sec):g} s", densest))
    return cases


def run_sensitivity(params):
    """
    results/sensitivity.csv: every sensitivity case at the QR shares in
    params.sens_qr_shares, all five configurations, 30 replications each.
    One row per (case, QR share, configuration) with mean Wq, the cost of
    division (paired 95% CI), the peak queue and burst clearance time at the
    worse bank, censored bursts, and blocked-capacity time.
    """
    reps = range(params.replications)
    rows = []
    for factor, level, case in sensitivity_cases(params):
        for p_qr in params.sens_qr_shares:
            arrivals = [batch_arrivals(p_qr, case, rep) for rep in reps]
            cell = {config: pd.DataFrame([measure(simulate(a, config, case), case) for a in arrivals])
                    for config in case.configurations}
            undivided_wq = cell["Undivided"]["Wq_s"]
            for config, runs in cell.items():
                row = {"factor": factor, "level": level, "p_qr": p_qr, "config": config,
                       "Wq_s_mean": runs["Wq_s"].mean()}
                add_mean_ci(row, "dW_div_s", runs["Wq_s"] - undivided_wq, params)
                row["Lq_max_worst_mean"] = runs[["Lq_max_All", "Lq_max_A", "Lq_max_B"]].max(axis=1).mean()
                row["T_clr_worst_s_mean"] = runs[["T_clr_All_s", "T_clr_A_s", "T_clr_B_s"]].max(axis=1).mean()
                row["T_clr_censored_total"] = runs[["T_clr_censored_All", "T_clr_censored_A",
                                                    "T_clr_censored_B"]].sum(axis=1).sum()
                row["T_blk_gate_min_mean"] = runs["T_blk_gate_min"].mean()
                rows.append(row)
            divided = {c: cell[c]["Wq_s"].mean() for c in case.configurations if c != "Undivided"}
            best = min(divided, key=divided.get)
            print(f"  sensitivity {factor:15s} {level:28s} p_qr={p_qr:.2f}: best {best:4s} "
                  f"present costs {divided[params.present_configuration] - undivided_wq.mean():.3f}s")
    return pd.DataFrame(rows)


def sensitivity_optimum(table, params):
    """
    results/sensitivity_optimal.csv: for each sensitivity case and QR share,
    the wait-optimal divided configuration and the cost of the present
    split, to show whether any recommendation flips within the ranges.
    """
    rows = []
    for (factor, level, p_qr), cell in table.groupby(["factor", "level", "p_qr"], sort=False):
        divided = cell[cell.config != "Undivided"].set_index("config")
        present = divided.loc[params.present_configuration]
        rows.append({"factor": factor, "level": level, "p_qr": p_qr,
                     "best_by_wait": divided["Wq_s_mean"].idxmin(),
                     "dW_div_present_s": present["dW_div_s_mean"],
                     "dW_div_present_ci_low_s": present["dW_div_s_ci_low"],
                     "dW_div_present_ci_high_s": present["dW_div_s_ci_high"],
                     "Lq_max_worst_present": present["Lq_max_worst_mean"],
                     "Lq_max_undivided": cell.set_index("config").loc["Undivided", "Lq_max_worst_mean"],
                     "T_clr_censored_present": present["T_clr_censored_total"]})
    return pd.DataFrame(rows)


def print_table(df):
    """Print a results table to the console with 4 decimal places."""
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    """Run the requested stages (default: all) and write results to results/."""
    params = Params()
    RESULTS_DIR.mkdir(exist_ok=True)
    stages = [s.upper() for s in sys.argv[1:]] or ["A", "B", "C", "D", "SWEEP", "SENS",
                                                   "FIGURES", "ANIM"]

    if "A" in stages:
        print(f"Stage A: M/M/c verification ({params.replications} replications per load)")
        table = run_verification(params)
        table.to_csv(RESULTS_DIR / "verification.csv", index=False)
        print()
        print_table(table)
        print()

    if "B" in stages:
        print(f"Stage B: three media, shared gates ({params.replications} replications per case)")
        table = run_stage_b_check(params)
        table.to_csv(RESULTS_DIR / "stage_b_check.csv", index=False)
        print()
        print_table(table)
        print()

    if "C" in stages:
        print(f"Stage C: strict partition, all configurations ({params.replications} replications each)")
        table = run_stage_c_check(params)
        table.to_csv(RESULTS_DIR / "stage_c_check.csv", index=False)
        print()
        print_table(table)
        print()
        print("Stage C: per-bank M/M/c verification under 5/2")
        table = run_bank_verification(params)
        table.to_csv(RESULTS_DIR / "verification_banks.csv", index=False)
        print()
        print_table(table)
        print()

    if "D" in stages:
        print(f"Stage D: batch arrivals, full metric set ({params.replications} replications each)")
        table = run_stage_d_check(params)
        table.to_csv(RESULTS_DIR / "stage_d_check.csv", index=False)
        print()
        print_table(table)
        print()

    if "SWEEP" in stages:
        n_runs = len(params.configurations) * len(params.qr_shares) * params.replications
        print(f"Experiment sweep: {n_runs} runs, batch arrivals")
        runs = run_sweep(params)
        runs.to_csv(RESULTS_DIR / "runs.csv", index=False)
        summarise_runs(runs, params).to_csv(RESULTS_DIR / "summary.csv", index=False)
        print("Optimal allocation by QR share")
        optimal_allocation(runs, params).to_csv(RESULTS_DIR / "optimal_allocation.csv", index=False)
        print(f"Batch versus smooth arrivals under {params.present_configuration}")
        run_batch_vs_smooth(runs, params).to_csv(RESULTS_DIR / "batch_vs_smooth.csv", index=False)
        print()

    if "SENS" in stages:
        print("Sensitivity analysis, one factor at a time")
        table = run_sensitivity(params)
        table.to_csv(RESULTS_DIR / "sensitivity.csv", index=False)
        sensitivity_optimum(table, params).to_csv(RESULTS_DIR / "sensitivity_optimal.csv", index=False)
        print()

    if "FIGURES" in stages:
        print("Figures")
        make_all_figures(params)
        print()

    if "ANIM" in stages:
        print("Animations")
        make_animations(params)
        print()


if __name__ == "__main__":
    main()
