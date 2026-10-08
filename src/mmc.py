"""
mmc.py - Analytical M/M/c queue (Erlang C), used to verify the simulation.

Project : After-Office Surge at the LRT-1 EDSA Interchange
Course  : CSS142P Modeling and Simulation, Mapua University
Authors : Aldea, De Leon, Jerusalem

The formulas are unit-free: give lambda and mu in the same time unit
(both per second, or both per minute) and Wq comes back in that unit.
"""

import math


def erlang_c(lam, mu, c):
    """
    Probability that an arriving customer has to wait (Erlang C formula).

    lam : arrival rate
    mu  : service rate of ONE server
    c   : number of servers

    P_wait = (a^c / (c! (1 - rho))) / ( sum_{n=0}^{c-1} a^n / n!  +  a^c / (c! (1 - rho)) )
    where a = lam / mu is the offered load in Erlangs and rho = a / c.
    """
    a = lam / mu
    rho = a / c
    if rho >= 1:
        # No steady state exists: the queue grows without bound.
        raise ValueError(f"Unstable queue: rho = {rho:.3f} must be below 1")

    all_busy_term = a**c / (math.factorial(c) * (1 - rho))
    some_idle_terms = sum(a**n / math.factorial(n) for n in range(c))
    return all_busy_term / (some_idle_terms + all_busy_term)


def mmc_summary(lam, mu, c):
    """
    Steady-state M/M/c results as a dict: a, rho, P_wait, Wq and Lq.

    Wq = P_wait / (c*mu - lam)   mean time in queue
    Lq = lam * Wq                mean number in queue (Little's law)
    """
    a = lam / mu
    rho = a / c
    p_wait = erlang_c(lam, mu, c)
    wq = p_wait / (c * mu - lam)
    return {"a": a, "rho": rho, "P_wait": p_wait, "Wq": wq, "Lq": lam * wq}


def mgc_wq_approx(lam, mean_service, scv, c):
    """
    Approximate Wq for an M/G/c queue (Allen-Cunneen approximation with
    Poisson arrivals):

        Wq(M/G/c) ~= Wq(M/M/c) * (1 + scv) / 2

    where scv is the squared coefficient of variation of the service time.
    Exponential service has scv = 1 and recovers M/M/c exactly; less
    variable service (scv < 1) waits proportionally less. It is exact for
    c = 1 and a close approximation for c > 1, so it is used as a
    reasonableness check, not a pass/fail test.
    """
    mu = 1.0 / mean_service
    return mmc_summary(lam, mu, c)["Wq"] * (1 + scv) / 2
