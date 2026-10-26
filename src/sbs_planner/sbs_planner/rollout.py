"""Control-space rollout generation for a differential-drive PMV.

Pure numpy, no ROS, so it can be unit-tested without a ROS environment.

A rollout is a two-segment piecewise-constant command (v, w) integrated forward
through the vehicle's actuation lag. Two segments rather than one because a
single constant (v, w) is a circular arc and cannot express "turn, then
straighten", which is what a corner needs -- with one arc the cost of every
candidate at a corner is misleading even though replanning papers over it.

The lag is inside the integrator on purpose. Measured from /joystick against
/odom across four bags and two driver pairs:

    linear    T = 0.41 +/- 0.07 s
    angular   T = 0.49 +/- 0.06 s

Over a 3 s horizon that is ~7 time constants, so candidates genuinely diverge;
over the 1.0-1.5 s the ROS 1 planner used, the vehicle has barely reached the
commanded speed and candidates are hard to tell apart. A rollout that ignored
the lag would promise motion the vehicle cannot deliver, and the first executed
step is exactly where that error lands.

Dead time is deliberately not modelled: it came out grouped by recording
session (0.20 s vs 0.30 s) rather than by vehicle, which is the signature of a
timestamp offset, not physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VehicleModel:
    """Actuation lag and the envelope the recorded driving stayed inside."""

    tau_v: float = 0.41          # linear time constant [s]
    tau_w: float = 0.49          # angular time constant [s]
    v_max: float = 1.30          # recorded p99 was 1.26 m/s
    v_min: float = 0.0           # no reversing in this data
    w_max: float = 0.80          # recorded p99 of |w| was 0.77 rad/s
    #: Acceleration and deceleration are separate, because they are separate
    #: questions. Measured over the 94 recorded runs, smoothed over 0.3 s:
    #:
    #:                     median   p90   p95   p99    max
    #:     accelerating      0.20  0.69  0.84  0.97   1.75
    #:     decelerating      0.20  0.66  0.84  1.18   2.93
    #:
    #: In ordinary driving the drivers were symmetric -- identical medians and
    #: p95s both ways -- so `a_max` and `d_max` share a value at about their
    #: p85. It is the *tails* that differ: braking reaches 2.93 m/s^2 against
    #: 1.75 accelerating, so when these drivers had to stop they used three
    #: times what the planner assumed.
    #:
    #: Neither of these bounds the sampler. The candidate set is bounded by
    #: `v_span` -- 0.45 m/s per segment, which the 0.41 s lag turns into about
    #: 1.1 m/s^2 of achieved deceleration on the executed step -- together
    #: with `v_min`/`v_max` and `lat_acc_max`. So the vehicle can already brake
    #: harder than `d_max`; what these numbers do is answer questions about
    #: it.
    a_max: float = 0.60          # comfortable acceleration [m/s^2]
    d_max: float = 0.60          # comfortable deceleration, for planning
    #: What the vehicle may use when the alternative is a collision: above the
    #: drivers' p99 of 1.18 and half the 2.93 they were observed to reach, so
    #: it is inside what humans did in this building.
    #:
    #: Deliberately *not* used by `wall_speed_cap` or any other speed cap.
    #: Those return the fastest speed whose stopping distance fits the gap
    #: ahead, so a larger deceleration there permits a *higher* speed and
    #: drives the vehicle closer to things -- the opposite of the intent. The
    #: caps keep `d_max`, and this is for asking whether a stop is available
    #: at all.
    d_stop: float = 1.50         # emergency deceleration [m/s^2]
    lat_acc_max: float = 1.50    # |v * w|, comfort limit on cornering


@dataclass
class RolloutConfig:
    horizon: float = 3.0         # matches the 3 s analysis window
    dt: float = 0.1              # matches the 10 Hz resample
    segment_split: float = 1.5   # when the command switches
    #: Let the speed grid reach `d_stop` of braking, which needs a larger
    #: downward span than upward. False restores the symmetric grid, as the
    #: control arm for measuring it.
    asymmetric_speed: bool = True

    # Resolution near the current state matters more than reach. Only the first
    # step of the first segment is ever executed, and the planner replans at
    # 10 Hz, so it can ramp over successive cycles -- but it cannot command a
    # speed that is not in the grid. Sampling +/-0.9 m/s with 4 points put the
    # samples 0.6 m/s apart and left no candidate within 0.2 m/s of the
    # preferred speed, so the planner chose maximum speed for want of anything
    # closer.
    v_span: float = 0.45         # m/s either side of the current command
    w_span: float = 0.60         # rad/s
    # The pair terms cost the *product* of the two vehicles' grid sizes, so
    # candidate count is the dominant lever on tick time: 7x7x3x3 gave 357
    # candidates and a 3.8M-element pair block that ran at 2 Hz against a
    # 10 Hz deadline.
    #
    # Of the two axes, *speed* is the one to keep fine. Holding the pair
    # being side by side means nulling an along-track offset, and the only control that
    # does that is a small speed difference: coarsening this axis to 5 points
    # (0.225 m/s steps) held separation just as well, 1.42 m, while the
    # along-track error went from 0.44 to 0.78 rad and the along-track offset from
    # 0.57 to 0.97 m -- the pair kept its distance but sat diagonal, one ahead
    # of the other. Steering is coarser instead; lane keeping is a slower
    # business and the actuation lag smooths the 0.30 rad/s steps.
    n_v: int = 7                 # first segment: it is what gets executed
    n_w: int = 5
    # Second segment is lookahead only -- never executed -- so it is the cheap
    # place to trim. The pair terms cost the product of the two grid sizes:
    # 3x3 here gives 357 candidates and a 3.8M-element pair block that missed
    # the 10 Hz deadline; 2x2 gives 196 and comfortably makes it.
    n_v2: int = 2
    n_w2: int = 2
    model: VehicleModel = field(default_factory=VehicleModel)

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon / self.dt))

    # NOTE: do not coarsen one axis of the pair. The cooperative concepts work
    # only because both vehicles solve the *same* bimatrix -- each one's matrix
    # is the transpose of the other's -- so two agents choosing independently
    # land on the two halves of one joint optimum, with nothing shared between
    # them. Give the ego a fine grid and the partner a coarse one and the two
    # are solving different games, their halves stop matching, and the
    # formation falls apart even though every term is correct. Buy speed by
    # coarsening *both* (n_v2/n_w2 below), never one.

    @property
    def split_step(self) -> int:
        return int(round(self.segment_split / self.dt))


def command_grid(value: float, span: float, count: int,
                 low: float, high: float,
                 span_down: float | None = None) -> np.ndarray:
    """Samples around the current value, clipped to the vehicle envelope.

    Sampling around the state rather than over the whole box keeps every
    candidate reachable, so the set stays small without losing anything the
    vehicle could actually do next.

    `span_down` makes the grid asymmetric, which matters for speed: a
    symmetric span means the hardest available braking equals the hardest
    available acceleration, and the vehicle then cannot ask to stop. Measured,
    from 0.93 m/s the slowest command on offer was 0.48 m/s -- a 0.45 m/s
    change, which the 0.41 s lag turns into 1.10 m/s^2, below the 1.50 the
    recorded drivers used when they had to stop. The vehicle was structurally
    unable to brake as hard as the corpus says it can.

    The current value stays on the grid, so holding the present command is
    always among the options. `count == 2` keeps the two extremes it always
    had, since the second rollout segment samples only those.
    """
    if count <= 1:
        return np.array([np.clip(value, low, high)])
    down = span if span_down is None else span_down
    if count == 2:
        grid = np.array([value - down, value + span])
    else:
        half = count // 2
        grid = np.concatenate([
            np.linspace(value - down, value, half + 1),
            np.linspace(value, value + span, count - half)])
    return np.unique(np.clip(grid, low, high))


def sample_commands(v: float, w: float, cfg: RolloutConfig) -> np.ndarray:
    """Every (v_a, w_a, v_b, w_b) the vehicle could reach from (v, w).

    The second segment is sampled around the *first* segment's command, not the
    current state, since that is where the vehicle will be by then.
    """
    m = cfg.model
    # How much speed the vehicle may shed in one command, if it has to: the
    # emergency deceleration through the actuation lag. Never less than the
    # symmetric span, so this only ever adds candidates -- it cannot empty a
    # mask, which is what has defeated other attempts at braking authority
    # here.
    down = (max(cfg.v_span, m.d_stop * m.tau_v)
            if cfg.asymmetric_speed else cfg.v_span)
    first_v = command_grid(v, cfg.v_span, cfg.n_v, m.v_min, m.v_max,
                           span_down=down)
    first_w = command_grid(w, cfg.w_span, cfg.n_w, -m.w_max, m.w_max)

    rows = []
    for va in first_v:
        for wa in first_w:
            if abs(va * wa) > m.lat_acc_max:
                continue
            for vb in command_grid(va, cfg.v_span, cfg.n_v2,
                                   m.v_min, m.v_max, span_down=down):
                for wb in command_grid(wa, cfg.w_span, cfg.n_w2,
                                       -m.w_max, m.w_max):
                    if abs(vb * wb) > m.lat_acc_max:
                        continue
                    rows.append((va, wa, vb, wb))
    return np.asarray(rows, dtype=float)


def integrate(state: tuple[float, float, float, float, float],
              commands: np.ndarray, cfg: RolloutConfig) -> dict[str, np.ndarray]:
    """Integrate every command row forward, vectorised over rollouts.

    Returns arrays of shape (n_rollouts, n_steps). The unicycle is advanced with
    the *achieved* velocities, and the heading is updated before the position so
    a rollout matches how odo_simulator integrates -- otherwise the planner and
    the vehicle disagree by half a step of yaw at every timestep.
    """
    x0, y0, yaw0, v0, w0 = state
    n, steps = len(commands), cfg.n_steps
    split = cfg.split_step
    m = cfg.model

    av = np.exp(-cfg.dt / m.tau_v)
    aw = np.exp(-cfg.dt / m.tau_w)

    v_cmd = np.where(np.arange(steps)[None, :] < split,
                     commands[:, 0:1], commands[:, 2:3])
    w_cmd = np.where(np.arange(steps)[None, :] < split,
                     commands[:, 1:2], commands[:, 3:4])

    x = np.empty((n, steps)); y = np.empty((n, steps))
    yaw = np.empty((n, steps)); v = np.empty((n, steps)); w = np.empty((n, steps))

    cur_x = np.full(n, x0); cur_y = np.full(n, y0); cur_yaw = np.full(n, yaw0)
    cur_v = np.full(n, v0); cur_w = np.full(n, w0)

    for k in range(steps):
        cur_v = av * cur_v + (1.0 - av) * v_cmd[:, k]
        cur_w = aw * cur_w + (1.0 - aw) * w_cmd[:, k]
        cur_yaw = cur_yaw + cur_w * cfg.dt
        cur_x = cur_x + np.cos(cur_yaw) * cur_v * cfg.dt
        cur_y = cur_y + np.sin(cur_yaw) * cur_v * cfg.dt
        x[:, k] = cur_x; y[:, k] = cur_y; yaw[:, k] = cur_yaw
        v[:, k] = cur_v; w[:, k] = cur_w

    return {"x": x, "y": y, "yaw": yaw, "v": v, "w": w}


def generate(state, cfg: RolloutConfig | None = None):
    """Sample commands from the current state and integrate them."""
    cfg = cfg or RolloutConfig()
    commands = sample_commands(state[3], state[4], cfg)
    return commands, integrate(state, commands, cfg)
