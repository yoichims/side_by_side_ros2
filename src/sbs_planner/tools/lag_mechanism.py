"""Controlled test: one vehicle on a straight route tracks a lateral target
that ramps inward.  Shipped rollout generator, cross-track + lane-yaw + speed
terms at the shipped weights and scales.  Nothing else in the cost.

Variants, one change each against `shipped`:
  shipped   tau 0.41/0.49, target = lane at the vehicle's current x,
            compared with the horizon-mean of the rollout's offset
  no_lag    tau ~ 0 in both the planner model and the "vehicle"
  per_point target evaluated at each rollout point: |mean_k(y_k - lane(x_k))|
"""
import sys
import numpy as np
sys.path.insert(0, "src/sbs_planner")
from sbs_planner.rollout import RolloutConfig, VehicleModel, generate

W_CT, K_CT = 6.00, 0.203
W_YAW, K_YAW = 3.00, 0.092
W_SPD, K_SPD = 3.00, 0.20
V_PREF = 1.0
DROP, X0 = 0.34, 5.0


def lane(x, g):
    return -np.clip((x - X0) * g, 0.0, DROP)


def run(g, variant):
    model = VehicleModel()
    if variant == "no_lag":
        model.tau_v = model.tau_w = 1e-3
    cfg = RolloutConfig(model=model)
    s = (0.0, 0.0, 0.0, V_PREF, 0.0)
    log = []
    for _ in range(250):
        _, r = generate(s, cfg)
        if variant == "per_point":
            ct = np.abs((r["y"] - lane(r["x"], g)).mean(axis=1))
        else:
            ct = np.abs(r["y"].mean(axis=1) - lane(s[0], g))
        cost = (W_CT / K_CT * ct
                + W_YAW / K_YAW * np.abs(r["yaw"]).mean(axis=1)
                + W_SPD / K_SPD * np.abs(r["v"].mean(axis=1) - V_PREF))
        i = int(np.argmin(cost))
        s = (r["x"][i, 0], r["y"][i, 0], r["yaw"][i, 0], r["v"][i, 0], r["w"][i, 0])
        log.append(s)
        if s[0] > X0 + DROP / g + 8:
            break
    a = np.array(log)
    x, y, v = a[:, 0], a[:, 1], a[:, 3]
    e = y - lane(x, g)
    # distance lag: shift L minimising |y(x) - lane(x - L)|
    Ls = np.arange(0, 4.01, 0.05)
    lag = Ls[np.argmin([np.abs(y - lane(x - L, g)).mean() for L in Ls])]
    return np.abs(e).max(), lag, lag / np.median(v), np.abs(e[-20:]).mean()


print(f"{'sweep m/m':>9} {'variant':>9} {'peak |e| m':>10} {'lag m':>6} {'lag s':>6} {'settled':>8}")
for g in (0.05, 0.10, 0.20):
    for variant in ("shipped", "no_lag", "per_point"):
        pk, L, T, end = run(g, variant)
        print(f"{g:9.2f} {variant:>9} {pk:10.3f} {L:6.2f} {T:6.2f} {end:8.3f}")
