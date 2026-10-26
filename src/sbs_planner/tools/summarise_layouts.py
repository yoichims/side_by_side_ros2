#!/usr/bin/env python3
"""What the offline corridor knowledge is worth, across moved obstacles.

    python3 src/sbs_planner/tools/summarise_layouts.py layout_runs/

The lane offsets and the side-by-side flag are derived offline from the map *and*
one particular set of objects. Each layout is therefore run twice: once with
that profile regenerated for the layout it will actually meet, and once with
the profile left as shipped -- which is what happens the moment a cone is
moved and nobody regenerates.

The gap between the two arms is the measurement. If it is small, the offline
profile is a convenience and the planner generalises to obstacle
configurations it was not built around. If it is large, that knowledge is
load-bearing, and the free-space reasoning has to move to runtime before
obstacles can move at all.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

#: Half a map cell: below this a contact is not distinguishable from passing
#: close, because clearance is measured to occupied cell centres.
FLOOR = 0.025
#: Largest believable movement per tick. At the 1.05 m/s cap a vehicle covers
#: 0.105 m in the 0.1 s between rows, so anything approaching this is not the
#: planner driving badly -- it is two publishers on the same odometry topic,
#: which happens whenever a second instance is launched. Such a run looks
#: catastrophic and means nothing, so it is refused rather than averaged in.
MAX_STEP = 0.20


def load(path):
    rows, head = [], None
    for line in open(path):
        if line.startswith('#'):
            continue
        parts = next(csv.reader([line]))
        if head is None:
            head = parts
            continue
        if len(parts) != len(head):
            # A torn row: the metrics file is flushed per line, and a run cut
            # short by `timeout` can leave a partial one mid-file -- measured,
            # three of twelve runs had exactly one, starting mid-number. One
            # row in 1679 is a write artefact, not behaviour, so it is dropped
            # rather than allowed to raise IndexError several callers deep.
            continue
        rows.append(parts)
    idx = {n: i for i, n in enumerate(head)}
    def col(name):
        return np.array([float(r[idx[name]]) if r[idx[name]] not in ('', 'nan')
                         else np.nan for r in rows])
    return col


class Contaminated(Exception):
    """The run cannot be measured: two publishers, or a missing input."""


def measure(path):
    g = load(path)
    t = g('t') - g('t')[0]
    step = max(float(np.nanmax(np.hypot(np.diff(g(f'{v}_x')), np.diff(g(f'{v}_y')))))
               for v in ('v1', 'v2'))
    if step > MAX_STEP:
        raise Contaminated(f'{path.name}: {step:.2f} m in one tick')
    # A whole clearance column of NaN means the metrics node never received
    # the map, which happens when a second stack owns the map server. The
    # per-tick test does not see it -- the poses are perfectly sane, the run
    # simply has no wall data -- so it would otherwise be averaged in as a run
    # with no wall contact, which is indistinguishable from a good one.
    for column in ('v1_wall_body', 'v2_wall_body'):
        if np.all(np.isnan(g(column))):
            raise Contaminated(f'{path.name}: {column} is entirely NaN, so '
                               f'the metrics node never had the map')
    out = {}
    stalls = 0
    for v in ('v1', 'v2'):
        still = np.where(np.abs(g(f'{v}_v')) < 0.05)[0]
        stalls += sum(1 for q in np.split(still, np.where(np.diff(still) > 10)[0] + 1)
                      if len(q) > 25)
    out['stalls'] = stalls
    out['contact'] = max(-min(0.0, float(np.nanmin(g(f'{v}_wall_body'))))
                         for v in ('v1', 'v2'))
    out['cone'] = min(float(np.nanmin(g('v1_obj_body'))),
                      float(np.nanmin(g('v2_obj_body'))))
    # Penetration *into a cone*, the same quantity `contact` reports for walls.
    # Its absence is why a week of campaigns was read as collision-free while
    # a vehicle was driving 146 mm through a cone: `contact` is wall-only, and
    # `cone` above is a clearance, so a negative value there is easy to skim
    # past next to a column of positive ones. Report the two symmetrically.
    out['cone_contact'] = max(
        -min(0.0, float(np.nanmin(g(f'{v}_obj_body'))))
        for v in ('v1', 'v2'))
    out['worst_contact'] = max(out['contact'], out['cone_contact'])
    moving = (np.abs(g('v1_v')) > 0.05) & (np.abs(g('v2_v')) > 0.05)
    turn = np.degrees(np.arctan2(np.abs(g('v1_along')), np.abs(g('v1_lateral'))))
    out['conv'] = 100 * float(np.nanmean((turn < 25) & (g('sep') < 2.0)))
    # The *pair's* lateral separation -- `v1_lateral` is the partner's offset
    # in v1's own frame, not v1's cross-track error against the route. The
    # drivers held 1.31 m. A reading of 5-19 m is therefore the formation
    # coming apart after one vehicle stalls while the other drives on; it is
    # not a vehicle leaving the corridor, and it has been misread that way.
    out['lat'] = float(np.nanmedian(np.abs(g('v1_lateral'))[moving]))
    out['filed'] = 100 * float(np.nanmean(np.abs(g('v1_along'))[moving] > 1.0))
    s1, s2 = g('v1_s'), g('v2_s')
    loop = max(np.nanmax(s1), np.nanmax(s2))
    def travelled(s):
        d = np.diff(s)
        d[d < -0.5 * loop] += loop
        d[d > 0.5 * loop] -= loop
        return float(np.nansum(d[np.abs(d) < 2.0]))
    out['prog'] = 0.5 * (travelled(s1) + travelled(s2)) / (t[-1] - t[0])
    return out


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'layout_runs')
    runs = defaultdict(list)
    void = []
    for path in sorted(root.glob('layout_*.csv')):
        parts = path.stem.split('_')
        arm, direction = parts[2], parts[3]
        try:
            runs[(arm, direction)].append((parts[1], measure(path)))
        except Contaminated as bad:
            void.append(str(bad))
    for bad in void:
        print(f'  REFUSED {bad}')
    if not runs:
        raise SystemExit(f'no runs found in {root}')

    print(f'\n{"arm":10}{"dir":>5}{"runs":>6}{"stalls":>8}{"contact>25mm":>14}'
          f'{"worst":>9}{"cone min":>10}{"conv":>7}{"filed":>8}{"prog":>7}')
    for (arm, direction), items in sorted(runs.items()):
        m = [x for _, x in items]
        print(f'{arm:10}{direction:>5}{len(m):>6}{sum(x["stalls"] for x in m):>8}'
              f'{sum(1 for x in m if x["contact"] > FLOOR):>14}'
              f'{1000 * max(x["contact"] for x in m):>7.0f}mm'
              f'{min(x["cone"] for x in m):>+10.3f}'
              f'{np.median([x["conv"] for x in m]):>6.0f}%'
              f'{np.median([x["filed"] for x in m]):>7.0f}%'
              f'{np.median([x["prog"] for x in m]):>7.2f}')

    # the comparison the campaign exists for, layout by layout
    for direction in sorted({d for _, d in runs}):
        pairs = []
        derived = dict(runs.get(('derived', direction), []))
        stale = dict(runs.get(('stale', direction), []))
        common = sorted(set(derived) & set(stale))
        if not common:
            continue
        print(f'\n{direction}: per layout, profile regenerated vs left stale')
        print(f'{"layout":9}{"contact mm":>22}{"cone clearance":>20}{"conv %":>16}')
        print(f'{"":9}{"derived":>11}{"stale":>11}{"derived":>10}{"stale":>10}'
              f'{"derived":>8}{"stale":>8}')
        for k in common:
            a, b = derived[k], stale[k]
            pairs.append((a, b))
            print(f'{k:9}{1000*a["contact"]:>11.0f}{1000*b["contact"]:>11.0f}'
                  f'{a["cone"]:>+10.3f}{b["cone"]:>+10.3f}'
                  f'{a["conv"]:>8.0f}{b["conv"]:>8.0f}')
        if pairs:
            dc = [1000 * (b['contact'] - a['contact']) for a, b in pairs]
            dv = [b['conv'] - a['conv'] for a, b in pairs]
            do = [b['cone'] - a['cone'] for a, b in pairs]
            print(f'\n  stale minus derived, median over {len(pairs)} layouts: '
                  f'contact {np.median(dc):+.0f} mm, cone clearance '
                  f'{np.median(do):+.3f} m, conversational {np.median(dv):+.0f} points')


if __name__ == '__main__':
    raise SystemExit(main())
