#!/usr/bin/env python3
"""Runtime corridor measurement against its matched control.

    python3 src/sbs_planner/tools/compare_room.py

`room_runs/` holds the treatment arm -- `runtime_room` and `w_centre` on --
and `layout_runs/*_stale_*` is its control: the same layouts, the same
shipped profile, the code as it was before. Pairing them run for run is what
makes the comparison worth anything; the arms differ in exactly one thing.

Reports per direction and per layout, because cw and ccw are not equivalent
and one direction improving while the other regresses is the outcome that
would matter most.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarise_layouts import Contaminated, measure

ROOT = Path(__file__).resolve().parents[3]
FLOOR = 0.025


def arms(treat_dir, control_dir, treat_tag='room', control_tag='stale'):
    pairs = []
    for treat in sorted((ROOT / treat_dir).glob(f'*_{treat_tag}_*.csv')):
        layout, _, direction = treat.stem.rsplit('_', 2)
        control = ROOT / control_dir / f'{layout}_{control_tag}_{direction}.csv'
        if not control.exists():
            continue
        try:
            pairs.append((layout, direction, measure(control), measure(treat)))
        except Contaminated as bad:
            print(f'  refused: {bad}')
    return pairs


def main():
    # compare_room.py <treat_dir> <control_dir> [<treat_tag> <control_tag>]
    argv = sys.argv[1:]
    pairs = arms(argv[0] if argv else 'room_runs',
                 argv[1] if len(argv) > 1 else 'layout_runs',
                 *(argv[2:4] if len(argv) > 3 else ()))
    if not pairs:
        print('no matched pairs')
        return
    print(f'\n{len(pairs)} matched pairs, control -> runtime room\n')
    head = (f"{'':10s} {'stalls':>13s} {'contact >25mm':>14s} "
            f"{'worst mm':>13s} {'cone m':>15s} {'conv %':>13s} {'prog':>13s}")
    print(head)
    for direction in ('ccw', 'cw'):
        rows = [p for p in pairs if p[1] == direction]
        if not rows:
            continue
        def both(key, fn):
            return fn([r[2][key] for r in rows]), fn([r[3][key] for r in rows])
        s0, s1 = both('stalls', sum)
        c0 = sum(1 for r in rows if r[2]['contact'] > FLOOR)
        c1 = sum(1 for r in rows if r[3]['contact'] > FLOOR)
        w0, w1 = both('contact', max)
        o0, o1 = both('cone', min)
        v0, v1 = both('conv', np.mean)
        p0, p1 = both('prog', np.mean)
        print(f'{direction:10s} {s0:6d} ->{s1:5d} {c0:6d} ->{c1:6d} '
              f'{1000*w0:6.0f} ->{1000*w1:5.0f} {o0:+7.3f} ->{o1:+6.3f} '
              f'{v0:6.0f} ->{v1:5.0f} {p0:6.2f} ->{p1:5.2f}')

    print('\nper run: contact mm, then conversational %')
    print(f"{'layout':10s} {'dir':4s} {'contact':>16s} {'conv':>14s} "
          f"{'prog':>14s}  stalls")
    for layout, direction, a, b in sorted(pairs):
        flag = ''
        if b['contact'] > FLOOR and b['contact'] > a['contact']:
            flag = '  <-- worse'
        if a['contact'] > FLOOR >= b['contact']:
            flag = '  <-- fixed'
        print(f'{layout:10s} {direction:4s} {1000*a["contact"]:7.0f} ->'
              f'{1000*b["contact"]:6.0f} {a["conv"]:6.0f} ->{b["conv"]:6.0f} '
              f'{a["prog"]:6.2f} ->{b["prog"]:6.2f}  {a["stalls"]} -> '
              f'{b["stalls"]}{flag}')


if __name__ == '__main__':
    main()
