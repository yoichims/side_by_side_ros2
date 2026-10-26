#!/usr/bin/env python3
"""Replay recorded states through the one-tick-then-brake test.

    ./in_container.sh python3 src/sbs_planner/tools/brake_replay.py RUN.csv VEHICLE [LATENCY]

For each recorded state in the 2 s before the body first entered a wall, how
many shipped candidates pass "execute one tick, then brake, stays clear" against
the planner's inflated walls, and does braking *now* stay clear?
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src' / 'sbs_planner'))
sys.path.insert(0, str(ROOT / 'src' / 'sbs_planner' / 'tools'))
import stall_harness as h                                       # noqa: E402
from sbs_planner.cost import footprint_clearance                 # noqa: E402
from sbs_planner.rollout import integrate                        # noqa: E402


def main():
    run, v = sys.argv[1], sys.argv[2]
    df = pd.read_csv(run, comment='#')
    wb = df[f'{v}_wall_body'].values
    # T_HIT_ARG: optional 4th argument, seconds into the run of the contact
    # to look before; default the first tick inside a wall.
    t_hit = (df.t.iloc[0] + float(sys.argv[4]) if len(sys.argv) > 4
             else df.t[int(np.flatnonzero(wb < 0)[0])])
    walls = h._walls()
    cfg = h.RolloutConfig(horizon=3.0, dt=0.1, segment_split=1.5,
                          v_span=0.45, w_span=0.60, n_v=7, n_w=5, n_v2=2,
                          n_w2=2, model=h.VehicleModel())
    latency = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    bcfg = replace(cfg, horizon=1.6 + latency, segment_split=cfg.dt + latency)
    now_cfg = replace(cfg, horizon=1.6 + latency, segment_split=latency or cfg.dt)
    print(' t-hit   v     w    wall(raw)  inflated-now  brake-now  pass/total  fastest passing v')
    for _, r in df[(df.t >= t_hit - 2.0) & (df.t <= t_hit)].iloc[::1].iterrows():
        state = (r[f'{v}_x'], r[f'{v}_y'], r[f'{v}_yaw'], r[f'{v}_v'], r[f'{v}_w'])
        commands, _ = h.generate(np.array(state), cfg)
        cmds = np.column_stack([commands[:, :2], np.zeros((len(commands), 2))])
        stop = integrate(state, cmds, bcfg)
        ok = footprint_clearance(stop, walls, None, h.FOOTPRINT_BOX) >= 0.0
        # "brake now" still runs the current command through the latency
        now = integrate(state, np.array([[state[3], state[4], 0.0, 0.0]]) if latency
                        else np.zeros((1, 4)), now_cfg)
        brake_now = float(footprint_clearance(now, walls, None, h.FOOTPRINT_BOX)[0])
        here = float(footprint_clearance(
            {k: np.array([[state[j]]]) for j, k in enumerate(('x', 'y', 'yaw'))},
            walls, None, h.FOOTPRINT_BOX)[0])
        fast = commands[ok, 0].max() if ok.any() else float('nan')
        print(f'{r.t - t_hit:+5.1f}  {state[3]:+.2f} {state[4]:+.2f}   {r[f"{v}_wall_body"]:+.3f}     '
              f'{here:+.3f}      {brake_now:+.3f}    {int(ok.sum()):3d}/{len(ok)}      {fast:+.2f}')


if __name__ == '__main__':
    main()
