"""
scenarios.py - The pre-made scenarios for the defense app. Each one is a
situation the proposal asks about, so it can be shown with one click
instead of by moving sliders.

Project : After-Office Surge at the LRT-1 EDSA Interchange
Course  : CSS142P Modeling and Simulation, Mapua University
Authors : Aldea, De Leon, Jerusalem
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    """
    One pre-made situation.

    key          : short id
    title        : name shown in the app
    config       : "Undivided" or "n_A/n_B"
    p_qr         : beep QR adoption share
    rate_per_min : mean arrival rate (18 is the base case)
    shows        : what the scenario demonstrates, and where the proposal asks it
    """
    key: str
    title: str
    config: str
    p_qr: float
    rate_per_min: int
    shows: str


SCENARIOS = (
    Scenario("present", "Present situation (5/2, 3% QR)", "5/2", 0.03, 18,
             "The split as observed in September 2026 at today's low QR use: Bank B "
             "sits almost idle while Bank A takes 97% of passengers (research question)."),
    Scenario("undivided", "Undivided benchmark (3% QR)", "Undivided", 0.03, 18,
             "Every gate accepts every medium: the benchmark that the cost of division "
             "is measured against (Objective 7)."),
    Scenario("best_today", "Best split today (6/1, 3% QR)", "6/1", 0.03, 18,
             "The allocation with the lowest mean wait at 3% QR (Objective 9)."),
    Scenario("qr_20", "QR use grows to 20% (5/2)", "5/2", 0.20, 18,
             "Past the adoption level where two QR gates stop being enough (Objective 8)."),
    Scenario("qr_30", "The opposite failure (5/2, 30% QR)", "5/2", 0.30, 18,
             "QR holders queue at Bank B while Bank A's gates stand idle (Problem Statement)."),
    Scenario("best_30", "Best split at 30% QR (4/3)", "4/3", 0.30, 18,
             "The recommended allocation for a digitalised future (Objective 9)."),
    Scenario("peak_demand", "Heavier demand (5/2, 3% QR, 22 per minute)", "5/2", 0.03, 22,
             "The top of the assumption register's arrival-rate range (sensitivity)."),
    Scenario("lines_forming", "Lines forming (5/2, 3% QR, 30 per minute)", "5/2", 0.03, 30,
             "Beyond the register's 14-22 per minute range: at the 17:00 peak each train now "
             "brings about as many Bank A passengers as its five gates can handle, so a line "
             "builds with every train and drains before the next. Shows what it takes for "
             "lines to form."),
)

