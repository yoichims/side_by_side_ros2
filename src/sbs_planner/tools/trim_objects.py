#!/usr/bin/env python3
"""Shrink surveyed cone objects to the size a cone actually is.

    python3 src/sbs_planner/tools/trim_objects.py            # report only
    python3 src/sbs_planner/tools/trim_objects.py --write

The cells in `objects.yaml` are laser returns, and a return is not the object:
measured over the nine surveyed cones, four come out 0.25-0.30 m across on
21-26 cells, which is a traffic cone, while `cone_1` comes out 0.45 x 0.50 on
**63** cells -- three cones' worth of returns for one cone. The surplus is
noise, and it is not harmless: these cones stand in an aisle whose outer
passage is already the tightest part of the route, so an inflated cone is
taken out of the one lane that has least to give.

This is the opposite of carving the *map*, which must stay exactly as
surveyed -- walls are structure and the planner's strictness is the thing to
fix downstream. An object is a measurement of a thing whose true size is
known, so fitting it to that size is a correction, not a convenience.

Each object is kept as its own object -- cones stand in pairs here and merging
a pair's centroids puts a phantom obstacle in the gap the vehicles drive
through. The core is found by density rather than by the centroid, so a
one-sided halo cannot drag the centre off the cone.
"""
import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
ENV = ROOT / 'environments/experiment'
#: Half the width of the cones that were cleanly observed -- 0.25-0.30 m over
#: cones 3, 4, 8 and 9 -- rounded to the map's own 0.05 m grid.
RADIUS = 0.15


def core(cells, radius):
    """Densest neighbourhood of a cell cloud, and its centroid."""
    tree = cKDTree(cells)
    counts = np.array([len(tree.query_ball_point(c, radius)) for c in cells])
    seed = cells[int(np.argmax(counts))]
    return cells[tree.query_ball_point(seed, radius)].mean(axis=0)


def trim(cells, radius):
    centre = core(cells, radius)
    keep = np.linalg.norm(cells - centre, axis=1) <= radius + 1e-9
    return cells[keep], centre


def span(a):
    return float(a.max() - a.min()) + 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--objects', default=str(ENV / 'objects.yaml'))
    ap.add_argument('--radius', type=float, default=RADIUS)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    path = Path(args.objects)
    spec = yaml.safe_load(path.read_text())
    print(f"{'name':9s} {'cells':>12s} {'across m':>14s} {'along m':>14s}")
    for o in spec['objects']:
        if 'cells' not in o:
            continue
        cells = np.array(o['cells'], float)
        kept, centre = trim(cells, args.radius)
        print(f"{o['name'][:9]:9s} {len(cells):5d} ->{len(kept):5d} "
              f"{span(cells[:, 0]):6.2f} ->{span(kept[:, 0]):6.2f} "
              f"{span(cells[:, 1]):6.2f} ->{span(kept[:, 1]):6.2f}")
        o['cells'] = [[round(float(x), 4), round(float(y), 4)] for x, y in kept]

    if not args.write:
        print('\nreport only; pass --write to apply')
        return
    # No file-based backup. It would be worse than none: this is close to
    # idempotent -- trimming an already-trimmed file keeps the same cells --
    # so a second run would happily save the *trimmed* objects under a name
    # claiming to be the survey. The surveyed original is in git history,
    # which cannot be overwritten by running this twice.
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f'wrote {path.name} at radius {args.radius:.2f} m')
    print('REGENERATE THE LANE PROFILES: the side-by-side flag reads the objects.')


if __name__ == '__main__':
    main()
