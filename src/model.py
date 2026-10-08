"""
model.py - Discrete-event simulation of the LRT-1 EDSA Side A entry fare gates.

Project : After-Office Surge at the LRT-1 EDSA Interchange
Course  : CSS142P Modeling and Simulation, Mapua University
Authors : Aldea, De Leon, Jerusalem

TIME UNIT: the SimPy clock runs in SECONDS. Parameters are written below in
the units of the assumption register (minutes for the window, headway and
rates; seconds for dispersal and service) so they can be checked line by line
against it. They are converted to seconds in one place only: the *_s
properties of Params.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import simpy


# =============================================================================
# PARAMETERS - every number in the model lives here (spec Section 3)
# =============================================================================

@dataclass(frozen=True)
class Params:
    """All model inputs. Values are the base case of the assumption register."""

    # --- 3.1 Fixed structure ---------------------------------------------
    total_gates: int = 7
    window_min: float = 240.0           # surge window, 16:00-20:00
    warmup_min: float = 30.0            # arrivals before this are discarded

    # --- 3.2 Arrivals ------------------------------------------------------
    mean_arrival_rate_per_min: float = 18.0
    peak_factor: float = 1.4            # realised through block_multipliers
    headway_min: float = 4.0            # MRT-3 train interval
    batch_size: float = 70.0            # transfer passengers per train at mean demand
    dispersal_sec: float = 90.0         # footbridge spread of one batch
    background_rate_per_min: float = 0.5
    block_min: float = 15.0             # length of one demand block
    block_multipliers: tuple = (0.70, 0.85, 1.00, 1.20,
                                1.40, 1.40, 1.30, 1.25,
                                1.15, 1.05, 0.95, 0.85,
                                0.80, 0.70, 0.60, 0.50)

    # --- 3.3 Fare medium mix (base case) -----------------------------------
    share_beep: float = 0.80
    share_sjt: float = 0.17
    share_qr: float = 0.03

    # --- 3.4 Service times: triangular (low, mode, high) in seconds ---------
    service_beep_sec: tuple = (2.0, 2.5, 3.5)
    service_sjt_sec: tuple = (2.5, 3.5, 5.0)
    service_qr_sec: tuple = (3.5, 5.0, 8.0)

    # --- 3.5 Experiment grid -----------------------------------------------
    configurations: tuple = ("Undivided", "6/1", "5/2", "4/3", "3/4")
    present_configuration: str = "5/2"  # as observed, September 2026
    qr_shares: tuple = (0.03, 0.05, 0.10, 0.20, 0.30)
    replications: int = 30

    # --- Section 4 Stage A / Section 5: verification and statistics --------
    verification_loads: tuple = (0.5, 0.7, 0.9)
    confidence: float = 0.95

    # --- Sensitivity ranges, varied one factor at a time ---------------------
    # Section 3.2 "Range" column; service-time ranges from the proposal's
    # assumption register (beep and SJT +/-20%, QR +/-25%).
    sens_batch_size: tuple = (50.0, 100.0)
    sens_headway_min: tuple = (3.0, 5.0)
    sens_dispersal_sec: tuple = (60.0, 120.0)
    sens_background_rate_per_min: tuple = (0.0, 2.0)
    sens_service_beep_scale: tuple = (0.8, 1.2)
    sens_service_sjt_scale: tuple = (0.8, 1.2)
    sens_service_qr_scale: tuple = (0.75, 1.25)
    sens_qr_shares: tuple = (0.03, 0.10, 0.30)

    # --- Animation (spec Section 7) -------------------------------------------
    animation_start_min: float = 60.0   # 17:00, the start of the peak block
    animation_minutes: float = 12.0     # three train cycles
    animation_fps: int = 30

    # --- Unit conversions (the only place minutes become seconds) ----------
    @property
    def window_s(self):
        """Surge window length in seconds."""
        return self.window_min * 60.0

    @property
    def warmup_s(self):
        """Warm-up period in seconds."""
        return self.warmup_min * 60.0

    @property
    def mean_arrival_rate_per_s(self):
        """Mean total arrival rate in passengers per second."""
        return self.mean_arrival_rate_per_min / 60.0

    @property
    def headway_s(self):
        """MRT-3 headway (interval between train batches) in seconds."""
        return self.headway_min * 60.0

    @property
    def block_s(self):
        """Length of one demand block in seconds."""
        return self.block_min * 60.0

    @property
    def background_rate_per_s(self):
        """Background (street-level) arrival rate in passengers per second."""
        return self.background_rate_per_min / 60.0

    def demand_multipliers(self):
        """
        Block multipliers rescaled so their mean is exactly 1.0 (Section 3.2),
        which keeps the window's total demand equal to the mean-demand value.
        After rescaling the peak is 1.40 / 0.98125 = 1.43.
        """
        multipliers = np.array(self.block_multipliers)
        if len(multipliers) * self.block_min != self.window_min:
            raise ValueError("block_multipliers must cover the surge window exactly")
        return multipliers / multipliers.mean()

    def service_triangle_s(self, medium):
        """(low, mode, high) service time in seconds for one fare medium."""
        return {"beep": self.service_beep_sec,
                "sjt": self.service_sjt_sec,
                "qr": self.service_qr_sec}[medium]


# =============================================================================
# FARE MEDIA - mix and service-time statistics
# =============================================================================

# The three passenger classes. The order matters only for reporting.
MEDIA = ("beep", "sjt", "qr")


def fare_mix(params, p_qr):
    """
    Share of each fare medium when beep QR adoption is `p_qr` (Section 3.3).

    The beep-to-SJT ratio is held at its base value (80:17) and the two are
    rescaled to fill whatever QR does not take:
        p_beep = (80/97) * (1 - p_qr),   p_sjt = (17/97) * (1 - p_qr)
    """
    non_qr_base = params.share_beep + params.share_sjt
    return {"beep": params.share_beep / non_qr_base * (1 - p_qr),
            "sjt": params.share_sjt / non_qr_base * (1 - p_qr),
            "qr": p_qr}


def mean_triangular(low, mode, high):
    """Mean of a triangular distribution: (low + mode + high) / 3."""
    return (low + mode + high) / 3.0


def variance_triangular(low, mode, high):
    """Variance of a triangular distribution."""
    return (low**2 + mode**2 + high**2 - low * mode - low * high - mode * high) / 18.0


def with_mean_arrival_rate(params, rate_per_min):
    """
    A copy of `params` with a different mean total arrival rate (used by the
    interactive app's rate slider).

    The batch process is driven by batch size, so the batch size is
    re-derived from the rate as in the assumption register: transfer
    passengers per minute (total rate minus background) times the headway.
    At 18 per minute this gives (18 - 0.5) x 4 = 70, the base value.
    """
    batch_size = (rate_per_min - params.background_rate_per_min) * params.headway_min
    return replace(params, mean_arrival_rate_per_min=rate_per_min, batch_size=batch_size)


def mix_service_stats(params, p_qr):
    """
    Mean (seconds) and squared coefficient of variation of the service time
    of a randomly chosen passenger, at QR adoption `p_qr`.

    The mixture's second moment is the share-weighted sum of each medium's
    E[S^2] = Var + mean^2. These figures feed the analytical cross-checks in
    Stages A and B; the simulation itself never uses them.
    """
    mix = fare_mix(params, p_qr)
    mean = 0.0
    second_moment = 0.0
    for medium, share in mix.items():
        tri = params.service_triangle_s(medium)
        m = mean_triangular(*tri)
        mean += share * m
        second_moment += share * (variance_triangular(*tri) + m**2)
    scv = (second_moment - mean**2) / mean**2
    return mean, scv


# =============================================================================
# RANDOM NUMBER STREAMS - common random numbers (spec Section 5)
# =============================================================================

def make_streams(rep):
    """
    Return three independent generators (arrivals, fare medium, service)
    seeded from the replication index alone.

    Because the seed depends only on `rep`, replication 7 of every
    configuration sees exactly the same passengers. SeedSequence.spawn is
    NumPy's recommended way to split one seed into independent streams.
    """
    arrival_seed, medium_seed, service_seed = np.random.SeedSequence(rep).spawn(3)
    return (np.random.default_rng(arrival_seed),
            np.random.default_rng(medium_seed),
            np.random.default_rng(service_seed))


# =============================================================================
# ARRIVAL TABLES - every passenger is drawn before the simulation starts
# =============================================================================

@dataclass
class Arrivals:
    """
    The passengers of one replication, in arrival order.

    times_s       : arrival time at the gate array (seconds from 16:00)
    media         : fare medium of each passenger
    service_s     : the gate service time each passenger will need
    train_times_s : when each MRT-3 train arrived (batch arrivals only;
                    empty for smooth arrivals)

    Drawing the service time at arrival (rather than when service starts)
    keeps common random numbers intact: passenger i needs the same time at
    the gate whatever configuration it is replayed under.
    """
    times_s: np.ndarray
    media: np.ndarray
    service_s: np.ndarray
    train_times_s: np.ndarray = field(default_factory=lambda: np.array([]))


def poisson_times(rate_per_s, window_s, rng):
    """Homogeneous Poisson arrival times on [0, window_s), built from exponential gaps."""
    times = []
    t = rng.exponential(1.0 / rate_per_s)
    while t < window_s:
        times.append(t)
        t += rng.exponential(1.0 / rate_per_s)
    return np.array(times)


def piecewise_poisson_times(block_rates_per_s, params, rng):
    """
    Non-stationary Poisson arrival times whose rate is constant within each
    demand block and equal to block_rates_per_s[k] in block k.

    Each block is generated as its own homogeneous Poisson stream; this is
    exact because exponential gaps are memoryless, so restarting the clock
    at a block boundary changes nothing.
    """
    pieces = []
    for k, rate in enumerate(block_rates_per_s):
        if rate > 0:
            pieces.append(k * params.block_s + poisson_times(rate, params.block_s, rng))
    return np.concatenate(pieces) if pieces else np.array([])


def draw_media_and_service(n, p_qr, params, medium_rng, service_rng):
    """
    Fare medium and service time for n passengers, in arrival order.

    The medium is a multinomial draw over fare_mix(p_qr); the service time is
    then drawn from that medium's triangular distribution.
    """
    mix = fare_mix(params, p_qr)
    media = medium_rng.choice(MEDIA, size=n, p=[mix[m] for m in MEDIA])
    service = np.array([service_rng.triangular(*params.service_triangle_s(m)) for m in media])
    return media, service


def mmc_arrivals(lam_per_s, mu_per_s, params, rep):
    """
    Stage A arrival table: one passenger class, Poisson arrivals at a constant
    rate, and EXPONENTIAL service times (M/M/c needs exponential service).
    """
    arrival_rng, _medium_rng, service_rng = make_streams(rep)
    times = poisson_times(lam_per_s, params.window_s, arrival_rng)
    service = service_rng.exponential(1.0 / mu_per_s, size=len(times))
    media = np.full(len(times), "any")      # single class: medium is irrelevant
    return Arrivals(times, media, service)


def smooth_arrivals(rate_per_s, p_qr, params, rep):
    """
    Smooth (constant-rate Poisson) arrival table with three fare media.

    Each passenger's medium is a multinomial draw over fare_mix(p_qr), and
    its service time is drawn from that medium's triangular distribution.
    """
    arrival_rng, medium_rng, service_rng = make_streams(rep)
    times = poisson_times(rate_per_s, params.window_s, arrival_rng)
    media, service = draw_media_and_service(len(times), p_qr, params, medium_rng, service_rng)
    return Arrivals(times, media, service)


def batch_arrivals(p_qr, params, rep):
    """
    Stage D arrival table: MRT-3 train batches plus a street-level background.

    Trains: one arrives every headway, starting at 16:00. Its batch is the
    base batch size times the normalised multiplier of the block the train
    arrives in, rounded to a whole passenger. Each passenger of the batch
    reaches the gates Uniform(0, dispersal) seconds after the train, which
    spreads the burst over the footbridge walk.

    Background: a non-stationary Poisson stream at the background rate times
    the same block multiplier.
    """
    arrival_rng, medium_rng, service_rng = make_streams(rep)
    multipliers = params.demand_multipliers()

    train_times = np.arange(0.0, params.window_s, params.headway_s)
    batches = []
    for t_train in train_times:
        block = int(t_train // params.block_s)
        size = round(params.batch_size * multipliers[block])
        batches.append(t_train + arrival_rng.uniform(0.0, params.dispersal_sec, size))
    background = piecewise_poisson_times(params.background_rate_per_s * multipliers,
                                         params, arrival_rng)

    times = np.sort(np.concatenate(batches + [background]))
    media, service = draw_media_and_service(len(times), p_qr, params, medium_rng, service_rng)
    return Arrivals(times, media, service, train_times)


def smooth_profile_arrivals(p_qr, params, rep):
    """
    The smooth counterpart of batch_arrivals(), for the batch-versus-smooth
    comparison (Figure 8).

    A non-stationary Poisson stream with the same 15-minute demand profile and
    the same expected volume as the batch process (batch passengers per
    headway plus background, times the block multiplier), but no bursts. Any
    difference between the two is therefore caused by batching alone.
    """
    arrival_rng, medium_rng, service_rng = make_streams(rep)
    mean_rate_per_s = params.batch_size / params.headway_s + params.background_rate_per_s
    times = piecewise_poisson_times(mean_rate_per_s * params.demand_multipliers(),
                                    params, arrival_rng)
    media, service = draw_media_and_service(len(times), p_qr, params, medium_rng, service_rng)
    return Arrivals(times, media, service)


def exponential_arrivals(lam_per_s, p_qr, mean_service_by_medium, params, rep):
    """
    Verification-only arrival table for the per-bank M/M/c test (Stage C).

    Poisson arrivals with multinomial fare media as in smooth_arrivals(), but
    with EXPONENTIAL service whose mean is given per medium. Giving beep and
    SJT the same mean makes Bank A a single-class M/M/n_A queue, and QR alone
    makes Bank B an M/M/n_B queue.
    """
    arrival_rng, medium_rng, service_rng = make_streams(rep)
    times = poisson_times(lam_per_s, params.window_s, arrival_rng)

    mix = fare_mix(params, p_qr)
    media = medium_rng.choice(MEDIA, size=len(times), p=[mix[m] for m in MEDIA])
    service = np.array([service_rng.exponential(mean_service_by_medium[m]) for m in media])
    return Arrivals(times, media, service)


# =============================================================================
# GATE LAYOUT - configurations and the eligibility matrix
# =============================================================================

# Eligibility matrix under strict division (as observed). It is block-diagonal:
# the fare medium alone decides the bank, and no passenger can switch.
#
#              Bank A   Bank B
#     beep       yes      no
#     sjt        yes      no
#     qr         no       yes
#
DIVIDED_BANK_FOR = {"beep": "A", "sjt": "A", "qr": "B"}

# Under the Undivided configuration the matrix is full: there is one pool of
# all seven gates and every medium may use it.
UNDIVIDED_BANK = "All"

# Every bank name that can appear, in reporting order.
BANK_NAMES = (UNDIVIDED_BANK, "A", "B")


def gate_layout(config, params):
    """
    Gate count per bank for a configuration label.

    "Undivided" gives one pool of all gates. "n_A/n_B" (for example "5/2")
    gives n_A gates on Bank A (beep and SJT) and n_B on Bank B (QR only).
    """
    if config == "Undivided":
        return {UNDIVIDED_BANK: params.total_gates}

    n_a, n_b = (int(n) for n in config.split("/"))
    if n_a + n_b != params.total_gates:
        raise ValueError(f"{config}: banks must add up to {params.total_gates} gates")
    if n_a < 1 or n_b < 1:
        raise ValueError(f"{config}: each bank needs at least one gate")
    return {"A": n_a, "B": n_b}


def bank_for(medium, config):
    """The one bank a passenger holding `medium` may use under `config`."""
    if config == "Undivided":
        return UNDIVIDED_BANK
    return DIVIDED_BANK_FOR[medium]


# =============================================================================
# SIMULATION
# =============================================================================

@dataclass
class RunResult:
    """
    Everything recorded in one replication.

    config   : configuration label, e.g. "5/2"
    arrivals : the passengers that were replayed
    gates    : gate count per bank
    bank     : the bank each passenger used
    waits    : each passenger's time in queue, seconds
    finished : the time each passenger left the gate, seconds
    logs     : per bank, the state after every change as three arrays
               (time_s, number waiting, number of gates busy); see settle()
    gate_index : which gate of its bank each passenger used (display only)
    trace    : one snapshot per simulated second if simulate(trace=True),
               otherwise None; see simulate()
    """
    config: str
    arrivals: Arrivals
    gates: dict
    bank: np.ndarray
    waits: np.ndarray
    finished: np.ndarray
    logs: dict
    gate_index: np.ndarray
    trace: list = None


def simulate(arrivals, config, params, trace=False):
    """
    Run one replication of `arrivals` through the gate layout `config`.

    A `source` process releases each passenger at its arrival time. Each
    passenger goes to the one bank its fare medium allows, joins that bank's
    FIFO queue, takes the first free gate, holds it for its service time
    and leaves. Arrivals stop at the end of the window, but the run continues
    until the last passenger is served, so no waiting time is cut off.

    Every time a bank's state changes (someone joins, starts or finishes),
    the bank's queue length and busy-gate count are logged. All time-based
    metrics are computed from these logs, not by polling on a time grid.

    With trace=True a separate `tracer` process also takes a snapshot every
    simulated second, for the animation. Per bank it stores: number waiting,
    number of gates busy, the fare medium of each waiting passenger, the
    passengers in the queue (in order) and at each gate, and the number
    served so far. The tracer only reads the state, so it cannot change
    the results.
    """
    gates = gate_layout(config, params)
    env = simpy.Environment()
    banks = {name: simpy.Resource(env, capacity=n) for name, n in gates.items()}

    n = len(arrivals.times_s)
    waits = np.full(n, np.nan)
    finished = np.full(n, np.nan)
    bank_used = np.empty(n, dtype=object)

    # Our own counters, so the logged state is easy to check by eye.
    waiting = dict.fromkeys(gates, 0)
    busy = dict.fromkeys(gates, 0)
    records = {name: [(0.0, 0, 0)] for name in gates}     # empty at 16:00

    # Display-only bookkeeping for the animation: who is in each queue, in
    # order, and which physical gate each passenger stands at. SimPy does not
    # need this; a Resource only counts its free gates.
    in_queue = {name: [] for name in gates}
    at_gate = {name: [None] * count for name, count in gates.items()}
    last_freed = {name: [0.0] * count for name, count in gates.items()}
    gate_index = np.full(n, -1)
    served = dict.fromkeys(gates, 0)
    snapshots = [] if trace else None

    def record(name):
        """Log the current state of one bank."""
        records[name].append((env.now, waiting[name], busy[name]))

    def passenger(i):
        """One passenger: go to the eligible bank, queue, be served, depart."""
        name = bank_for(arrivals.media[i], config)
        bank_used[i] = name
        joined = env.now
        waiting[name] += 1
        in_queue[name].append(i)
        record(name)
        with banks[name].request() as request:
            yield request
            waiting[name] -= 1
            busy[name] += 1
            in_queue[name].remove(i)
            # Take the free gate that has been idle longest, so the display
            # spreads passengers over all of a bank's gates.
            free = [g for g, who in enumerate(at_gate[name]) if who is None]
            slot = min(free, key=lambda g: last_freed[name][g])
            at_gate[name][slot] = i
            gate_index[i] = slot
            record(name)
            waits[i] = env.now - joined
            yield env.timeout(arrivals.service_s[i])
        busy[name] -= 1
        at_gate[name][slot] = None
        last_freed[name][slot] = env.now
        served[name] += 1
        record(name)
        finished[i] = env.now

    def source():
        """Release passengers into the model at their pre-drawn arrival times."""
        for i, t in enumerate(arrivals.times_s):
            yield env.timeout(t - env.now)
            env.process(passenger(i))

    def tracer():
        """Snapshot the state of every bank once per simulated second."""
        while env.now <= params.window_s:
            snapshots.append({"t": env.now, "banks": {name: {
                "waiting": waiting[name],
                "busy": busy[name],
                "waiting_media": [arrivals.media[j] for j in in_queue[name]],
                "queue_ids": list(in_queue[name]),
                "gate_ids": list(at_gate[name]),
                "served": served[name],
            } for name in gates}})
            yield env.timeout(1.0)

    env.process(source())
    if trace:
        env.process(tracer())
    env.run()   # no `until`: run until every passenger has been served

    logs = {name: settle(recs) for name, recs in records.items()}
    return RunResult(config, arrivals, gates, bank_used, waits, finished, logs,
                     gate_index, snapshots)


def settle(records):
    """
    Turn a bank's raw log into three arrays (time, waiting, busy) with one
    entry per distinct time: the state once every event at that instant has
    been processed.

    Without this, a passenger who arrives at an idle gate would show up as
    "1 waiting" for zero seconds before being served, inflating the maximum
    queue length.
    """
    times, waiting, busy = (np.array(column) for column in zip(*records))
    last_at_this_time = np.append(times[1:] != times[:-1], True)
    return times[last_at_this_time], waiting[last_at_this_time], busy[last_at_this_time]


# =============================================================================
# MEASUREMENT - metrics from one RunResult (spec Section 6)
# =============================================================================

def step_integral(times, values, t0, t1):
    """
    Integral over [t0, t1] of a step function that holds values[k] from
    times[k] until times[k+1] (the last value holds indefinitely).
    """
    edges = np.clip(np.append(times, np.inf), t0, t1)
    return float(np.sum(values * np.diff(edges)))


def state_at(times, values, at):
    """Value of a step function at time(s) `at`: the last logged value at or before `at`."""
    return values[np.searchsorted(times, at, side="right") - 1]


def step_max(times, values, t0, t1):
    """Largest value a step function takes at any instant in [t0, t1)."""
    inside = (times > t0) & (times < t1)
    return int(max(state_at(times, values, t0), values[inside].max(initial=0)))


def idle_while_queue_steps(result, idle_bank, queue_bank):
    """
    Step function (times, gates): the number of idle gates at `idle_bank`
    whenever `queue_bank` has a non-empty queue, and 0 otherwise.
    """
    t_idle, _waiting, busy_idle = result.logs[idle_bank]
    t_queue, waiting_queue, _busy = result.logs[queue_bank]
    times = np.union1d(t_idle, t_queue)       # every instant either bank changed
    idle = result.gates[idle_bank] - state_at(t_idle, busy_idle, times)
    queue_nonempty = state_at(t_queue, waiting_queue, times) > 0
    return times, idle * queue_nonempty


def idle_gates_while_queue_s(result, idle_bank, queue_bank, t0, t1):
    """
    Gate-seconds in [t0, t1] during which gates at `idle_bank` stood idle
    while `queue_bank` had a non-empty queue.

    With the same bank twice this must be zero, because a gate never stands
    idle while its own queue is non-empty (work conservation); the Stage D
    checks use that as a self-test.
    """
    times, gates = idle_while_queue_steps(result, idle_bank, queue_bank)
    return step_integral(times, gates, t0, t1)


def blocked_gates_steps(result):
    """
    Step function (times, gates) of blocked capacity: idle gates at Bank A
    while Bank B has a queue, plus idle gates at Bank B while Bank A has a
    queue. Both T_blk and the animation's running counter come from here.

    Under Undivided there is only one bank, so there is no pair of banks to
    add up and the function is zero everywhere by construction.
    """
    all_times = np.unique(np.concatenate([result.logs[name][0] for name in result.gates]))
    blocked = np.zeros(len(all_times))
    for idle_bank in result.gates:
        for queue_bank in result.gates:
            if idle_bank != queue_bank:
                times, gates = idle_while_queue_steps(result, idle_bank, queue_bank)
                blocked += state_at(times, gates, all_times)
    return all_times, blocked


def blocked_capacity_gate_min(result, params, t0=None, t1=None):
    """
    T_blk in gate-minutes over [t0, t1] (default: the post-warm-up window),
    computed exactly from the state-change logs rather than by polling.
    """
    t0 = params.warmup_s if t0 is None else t0
    t1 = params.window_s if t1 is None else t1
    times, blocked = blocked_gates_steps(result)
    return step_integral(times, blocked, t0, t1) / 60.0


def burst_metrics(times, waiting, train_times, params):
    """
    Burst clearance time and residual carryover at one bank.

    For every train arriving at t_b after warm-up (the next one arrives at
    t_b + headway):
      R     : passengers still queueing at the instant the next train arrives.
      T_clr : time from t_b until the queue empties for the LAST time before
              the next train (0 if no queue formed). "Last", because the
              queue can touch zero several times while a burst is still
              walking in, and only the final emptying means it has cleared.
              If the queue is still non-empty when the next train arrives,
              the burst is censored: it gets no T_clr and is counted instead.

    Returns (mean T_clr over uncensored bursts in seconds,
             number of censored bursts, mean R).
    """
    # Every instant at which the queue went from non-empty to empty.
    emptied_at = times[1:][(waiting[1:] == 0) & (waiting[:-1] > 0)]

    clear_times, carryovers, censored = [], [], 0
    for t_b in train_times[train_times >= params.warmup_s]:
        t_next = t_b + params.headway_s
        carry = int(state_at(times, waiting, t_next))
        carryovers.append(carry)
        if carry > 0:
            censored += 1
            continue
        emptied = emptied_at[(emptied_at > t_b) & (emptied_at <= t_next)]
        clear_times.append(emptied[-1] - t_b if len(emptied) else 0.0)

    mean_clear = float(np.mean(clear_times)) if clear_times else np.nan
    return mean_clear, censored, float(np.mean(carryovers))


def measure(result, params):
    """
    Every Section 6 metric for one replication. Waits are taken over
    passengers arriving after warm-up; time-based metrics over the part of
    the window after warm-up.

    Whole array
      Wq_s             : mean wait in queue, all passengers
      Wq_<medium>_s    : mean wait in queue, per fare medium
      W_<medium>_s     : mean time at the array (wait + own service), per medium
      dW_s             : waiting time disparity - highest minus lowest per-class Wq
      T_blk_gate_min   : blocked-capacity time, gate-minutes
      X_per_hr         : throughput - passengers finishing service per hour
      n_passengers     : passengers counted
    Per bank (<bank> is A and B, or All under Undivided)
      n_<bank>, Wq_bank_<bank>_s
      rho_<bank>       : utilisation - share of the time each gate is busy
      Lq_max_<bank>    : most passengers waiting at any instant
      T_clr_<bank>_s, T_clr_censored_<bank>, R_<bank> : see burst_metrics();
                         NaN for smooth arrivals, which have no trains
    A medium or bank with no passengers in the run gets NaN.
    The cost of division (dW_div) compares two configurations, so it is
    computed in experiments.py, not here.
    """
    arrivals = result.arrivals
    t0, t1 = params.warmup_s, params.window_s
    counted = arrivals.times_s >= t0
    metrics = {"Wq_s": float(np.mean(result.waits[counted])),
               "n_passengers": int(counted.sum())}

    for medium in MEDIA:
        chosen = counted & (arrivals.media == medium)
        if chosen.any():
            metrics[f"Wq_{medium}_s"] = float(np.mean(result.waits[chosen]))
            metrics[f"W_{medium}_s"] = float(np.mean(result.waits[chosen]
                                                     + arrivals.service_s[chosen]))
        else:
            metrics[f"Wq_{medium}_s"] = np.nan
            metrics[f"W_{medium}_s"] = np.nan

    class_waits = [metrics[f"Wq_{m}_s"] for m in MEDIA if not np.isnan(metrics[f"Wq_{m}_s"])]
    metrics["dW_s"] = max(class_waits) - min(class_waits) if class_waits else np.nan

    metrics["T_blk_gate_min"] = blocked_capacity_gate_min(result, params)
    done_in_window = (result.finished >= t0) & (result.finished < t1)
    metrics["X_per_hr"] = done_in_window.sum() / ((t1 - t0) / 3600.0)

    # Every bank name gets a column, NaN where the configuration lacks that
    # bank, so every run produces the same columns in the same order.
    for name in BANK_NAMES:
        if name not in result.gates:
            for key in ("n_{}", "Wq_bank_{}_s", "rho_{}", "Lq_max_{}",
                        "T_clr_{}_s", "T_clr_censored_{}", "R_{}"):
                metrics[key.format(name)] = np.nan
            continue
        n_gates = result.gates[name]
        chosen = counted & (result.bank == name)
        times, waiting, busy = result.logs[name]
        metrics[f"n_{name}"] = int(chosen.sum())
        metrics[f"Wq_bank_{name}_s"] = (float(np.mean(result.waits[chosen]))
                                        if chosen.any() else np.nan)
        metrics[f"rho_{name}"] = step_integral(times, busy, t0, t1) / (n_gates * (t1 - t0))
        metrics[f"Lq_max_{name}"] = step_max(times, waiting, t0, t1)

        if len(arrivals.train_times_s):
            clear, censored, carry = burst_metrics(times, waiting, arrivals.train_times_s, params)
        else:
            clear, censored, carry = np.nan, np.nan, np.nan
        metrics[f"T_clr_{name}_s"] = clear
        metrics[f"T_clr_censored_{name}"] = censored
        metrics[f"R_{name}"] = carry
    return metrics
