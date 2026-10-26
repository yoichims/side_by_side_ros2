#!/usr/bin/env python3
"""An A/B over repeats, reported as mean and spread.

Wall and cone contact are both shown, in millimetres of penetration. They are
separate columns because they are separate failures and because reporting only
one of them hid eight cone collisions of up to 146 mm behind a wall-only
"no contact" result for a week.

    python3 src/sbs_planner/tools/compare_repeats.py [repeat_runs]

Reports each arm as mean and spread across repeats, because the spread is the
point: a difference smaller than it is not a result. Per-layout rows are shown
with their own repeat range so a claim about one layout can be checked against
how much that layout moves on its own.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarise_layouts import Contaminated, load, measure


def stalled_seconds(path):
    """Seconds either vehicle spent standing still.

    Reported alongside the stall *count*, because the two disagree and the
    count is the misleading one. A vehicle parked for 100 s scores one stall;
    a vehicle that drives, sticks briefly, recovers and drives on scores three
    while covering twice the route. Counting episodes punishes the better
    behaviour, and recovery working is exactly what turns one long episode
    into several short ones.
    """
    g = load(path)
    dt = float(np.median(np.diff(g('t'))))
    total = 0.0
    for v in ('v1', 'v2'):
        still = np.where(np.abs(g(f'{v}_v')) < 0.05)[0]
        if not len(still):
            continue
        for q in np.split(still, np.where(np.diff(still) > 10)[0] + 1):
            if len(q) > 25:
                total += len(q) * dt
    return total / 2.0

ROOT = Path(__file__).resolve().parents[3]
FLOOR = 0.025


def main():
    root = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'repeat_runs')
    print(f'\narms differ in one parameter only; see {root.name}/*.log for '
          f'which')
    runs = defaultdict(list)
    refused = 0
    for path in sorted(root.glob('*_r[0-9].csv')):
        layout, arm, direction, _ = path.stem.rsplit('_', 3)
        try:
            m = measure(path)
            m['stalled_s'] = stalled_seconds(path)
            runs[(layout, arm, direction)].append(m)
        except Contaminated:
            refused += 1
    if not runs:
        print('no runs yet')
        return
    n = sum(len(v) for v in runs.values())
    print(f'\n{n} runs, {len(runs)} cells'
          + (f', {refused} refused' if refused else ''))

    def cell(arm, direction, key, fn=np.mean):
        vals = [fn([m[key] for m in ms])
                for (l, a, d), ms in runs.items() if a == arm and d == direction]
        return float(np.mean(vals)) if vals else float('nan')

    def worst(arm, direction, key):
        """The deepest single contact in the arm, not an average of maxima.

        `cell(..., max)` takes the worst within each layout and then averages
        across layouts, which dilutes one deep collision into nothing: a
        146 mm cone penetration showed as 37 mm that way. A safety number is
        an extremum -- this project has made the same mistake before,
        reporting cone clearance as a median of per-run minima and hiding a
        29 mm penetration in a paper draft.
        """
        vals = [m[key] for (l, a, d), ms in runs.items()
                if a == arm and d == direction for m in ms]
        return float(max(vals)) if vals else float('nan')

    print(f"\n{'':22s} {'stalls/run':>11s} {'stalled s':>10s} "
          f"{'progress':>10s} {'wall mm':>9s} {'cone mm':>9s} {'conv %':>8s}")
    print(f"  {'':20s} {'':11s} {'':10s} {'':10s} "
          f"{'deepest single contact':>19s}")
    for direction in ('ccw', 'cw'):
        for arm in ('nofix', 'fix'):
            # Generic on purpose. These labels were once hard-coded to the
            # parameter of the day, and a table that names the wrong one is
            # worse than a table that names none: the runner's ARM_FLAG says
            # what the arms differ in, and the run directory records it.
            label = 'control (off)' if arm == 'nofix' else 'treatment (on)'
            print(f"  {direction:4s} {label:17s} "
                  f"{cell(arm, direction, 'stalls'):9.2f} "
                  f"{cell(arm, direction, 'stalled_s'):10.0f} "
                  f"{cell(arm, direction, 'prog'):10.2f} "
                  f"{1000*worst(arm, direction, 'contact'):9.0f} "
                  f"{1000*worst(arm, direction, 'cone_contact'):9.0f} "
                  f"{cell(arm, direction, 'conv'):8.0f}")

    print(f"\nper layout: mean over repeats, with the repeat range in brackets")
    print(f"{'layout':10s} {'dir':4s} {'stalls nofix -> fix':>24s} "
          f"{'progress nofix -> fix':>30s}")
    for (layout, arm, direction), ms in sorted(runs.items()):
        if arm != 'nofix':
            continue
        other = runs.get((layout, 'fix', direction))
        if not other:
            continue
        def fmt(group, key):
            v = [m[key] for m in group]
            return (f'{np.mean(v):.2f}' if key != 'stalls' else f'{np.mean(v):.1f}') + \
                   (f'[{min(v):.2f}-{max(v):.2f}]' if key != 'stalls'
                    else f'[{min(v):.0f}-{max(v):.0f}]')
        print(f"  {layout:10s} {direction:4s} "
              f"{fmt(ms,'stalls'):>11s} ->{fmt(other,'stalls'):>11s} "
              f"{fmt(ms,'prog'):>14s} ->{fmt(other,'prog'):>14s}")


if __name__ == '__main__':
    main()
