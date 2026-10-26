#!/usr/bin/env python3
"""Generate cone layouts along the route, for testing against moved obstacles.

    python3 src/sbs_planner/tools/make_layouts.py --count 12 --out layouts/

Everything the planner knows about the corridor that is *not* the map comes
from a layout like this, and the lane offsets and the side-by-side flag are derived
from one particular layout. Moving the cones is therefore the test of whether
that offline knowledge is doing real work or is merely fitted to the four cones
it was built around.

Placements are sampled along the centreline rather than at random in the plane,
so every cone sits somewhere the pair will actually meet it, and each is
checked to leave a band beside it that the pair can actually track through --
`MIN_BAND`, taken from the tightest site in the surveyed environment. A layout
tighter than anything real tests nothing but the stall recovery.

Cones are written as `circle` primitives at a measured radius, not as cell
lists. The surveyed cones are LiDAR surface plus smear -- three mapped blobs
for the two cones in `docs/cones_center.png` -- so authoring them gives a known
geometry for the first time, and makes the object extent something the
experiment controls rather than inherits.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbs_planner.cost import FOOTPRINT_BOX, free_bands

ROOT = Path(__file__).resolve().parents[3]
ENV = ROOT / 'environments' / 'experiment'
#: A traffic cone's base. The surveyed cells measured 0.25-0.45 m across, but
#: the four cleanly observed ones -- 3, 4, 8 and 9 -- come out 0.25-0.30 m on
#: 21-26 cells, and the wide ones are noise: `cone_1` carried 63 cells for one
#: cone. `tools/trim_objects.py` fits the surveyed objects to this radius.
CONE_RADIUS = 0.15
#: An obstacle here is a *pair* of cones, not one. Measured on the surveyed
#: environment, three of the four obstacles are two cones 0.47, 0.53 and
#: 0.57 m apart across the aisle, and the fourth is three cones packed 0.07-
#: 0.09 m apart. So a single circle is not the thing the drivers met: a pair
#: spans about 0.83 m including the cone bodies, and the 0.23 m between the
#: two is far too narrow to drive through, which makes the obstacle wide
#: rather than a gate. Generating single cones understated every layout.
PAIR_GAP = 0.53
#: Body half-width plus the clearance the planner asks of an object: below
#: this a side is not passable and the layout would only test recovery.
PASSABLE = 0.275 + 0.10
#: The clearance the planner asks of an object, for the band test below.
PAIR_CLEARANCE = 0.10
#: How wide the free band beside an obstacle must be, measured as the span of
#: body-*centre* positions that clear everything.
#:
#: Not zero, which is what "a body fits" means and what this checked before.
#: The pair tracks its lane to about 0.08 m, so a band of 0.05-0.20 m is at or
#: below the error the vehicles actually carry: geometrically passable, and in
#: practice not. Measured over 32 runs on layouts generated that way, the four
#: whose tightest band was 0.05-0.20 m took 14 of the 20 stalls, while the four
#: at 0.45-0.70 m took 6.
#:
#: 0.30 m is not a tuned number -- it is the tightest band anywhere in the
#: surveyed environment, over 14 obstacle sites in both directions, where the
#: rest run 0.35-1.80 m. A layout tighter than that is not a moved obstacle,
#: it is an obstacle course the drivers never faced.
MIN_BAND = 0.30
#: And the band has to be *reachable*. A passable side a vehicle cannot get to
#: is not a passable side: the lateral target is capped at 1.00 m of shift,
#: which is the range the recorded drivers used (0.55-1.00 m at a cone) and
#: not a tunable, so a band further than that from a lane can only be watched
#: from the wrong side of the obstacle.
#:
#: Measured, the first version of this gate produced exactly that: cw layout 07
#: left a 1.05 m band at +0.70 to +1.75 while the outer lane sat at -0.44, a
#: 1.19 m shift. The target refused to move, the vehicle drove into the closed
#: side and pinned at (29.43, -1.16) for 138 s of a 168 s run -- in all six
#: runs, both arms. The surveyed environment needs at most 0.89 m (ccw) and
#: 0.92 m (cw), so this bound excludes nothing the drivers ever faced.
#: 0.90, not the planner's 1.00. A layout that needs 0.99 m sits on the cap
#: with nothing left for tracking error, and the pair tracks its target to
#: about 0.08 m. Measured, layout 08 needed 1.01 m and produced four of the
#: nine real wall contacts in a 48-run campaign, worst 122 mm.
MAX_REACH = 0.90
#: How far either side of an obstacle the reachability test looks, in
#: centreline samples of 0.1 m. An obstacle narrows the corridor for a metre
#: or more around itself, and the tightest point is not its own sample: the
#: first version of this test asked only there, passed a layout needing
#: 1.01 m, and the worst point turned out to be 0.7 m further on where the
#: lane offset had moved. The band is what the obstacle does to the corridor,
#: not what it does at one point of it.
REACH_WINDOW = 20
#: How far off the centreline a cone may be placed. The pair rides at about
#: +/-0.6 m and the body reaches 0.275 m beyond that, so a cone past this is
#: simply not in the way and the layout tests nothing. The free band can be
#: metres wide in the open parts of the floor, which is why this is a separate
#: bound rather than a consequence of the corridor.
REACH = 1.10


def read_map():
    info = yaml.safe_load((ENV / 'map.yaml').read_text())
    res, (ox, oy) = info['resolution'], info['origin'][:2]
    with open(ENV / info['image'], 'rb') as handle:
        assert handle.readline().strip() == b'P5'
        line = handle.readline()
        while line.startswith(b'#'):
            line = handle.readline()
        width, height = map(int, line.split())
        int(handle.readline())
        image = np.frombuffer(handle.read(width * height),
                              dtype=np.uint8).reshape(height, width)
    # Occupied *and* never-observed cells. Free space cannot be inferred from
    # the absence of an occupied cell: unexplored map has none, so a layout
    # whose passable side runs through it would be passable only on paper.
    occupancy = (255.0 - image.astype(float)) / 255.0
    occupied = occupancy > info['occupied_thresh']
    blind = ~occupied & ~(occupancy < info.get('free_thresh', 0.196))
    rows, cols = np.nonzero(occupied | blind)
    return cKDTree(np.column_stack([ox + (cols + 0.5) * res,
                                    oy + ((height - 1 - rows) + 0.5) * res]))


def centreline(direction='ccw'):
    rows = list(csv.DictReader(
        open(ENV / 'centerlines' / f'{direction}_centerline.csv', newline='')))
    return (np.array([float(r['x']) for r in rows]),
            np.array([float(r['y']) for r in rows]),
            np.array([float(r['yaw']) for r in rows]))


def corridor(walls, x, y, yaw, span=3.0, step=0.05):
    """Free lateral interval about the centreline here, as (left, right)."""
    offsets = np.arange(-span, span + step / 2, step)
    px = x - offsets * math.sin(yaw)
    py = y + offsets * math.cos(yaw)
    clear = walls.query(np.column_stack([px, py]))[0] > CONE_RADIUS
    best, run = (0, -1), []
    for i, ok in enumerate(clear):
        if ok:
            run.append(i)
            if run[0] <= len(offsets) // 2 <= run[-1] and len(run) > best[0]:
                best = (len(run), run[0])
        else:
            run = []
    if best[1] < 0:
        return None
    lo = offsets[best[1]]
    hi = offsets[best[1] + best[0] - 1]
    return float(lo), float(hi)


def layout(rng, walls, cx, cy, yaw, lane_offsets, n_cones, routes):
    """One set of obstacles: positions along the route, offsets across it.

    `n_cones` counts *obstacles*; each is emitted as a pair of cones.
    """
    cones, used = [], []
    tries = 0
    while len(used) < n_cones and tries < 4000:
        tries += 1
        k = int(rng.integers(40, len(cx) - 40))      # not on top of the start
        if any(abs(k - j) < 60 for j in used):       # spread them out
            continue
        band = corridor(walls, cx[k], cy[k], yaw[k])
        if band is None:
            continue
        lo, hi = band
        # A pair is the obstacle, so the corridor has to hold the pair's full
        # span plus a passable side, not one cone's.
        if hi - lo < 2 * PASSABLE + PAIR_GAP + 2 * CONE_RADIUS:
            continue                                  # no room to pass at all
        # Placed by *kind*, not at a uniform random offset. A cone on a lane,
        # a cone in the middle of the aisle and a cone near a wall pose
        # different problems -- the first makes one vehicle shift or fall
        # behind, the second is the case the drivers straddled, the third is
        # nearly free -- and sampling the offset uniformly mostly produces the
        # third. The first attempt did exactly that: every layout left passing
        # room on both sides of every cone, and the side-by-side flag came back
        # 100% of the route, testing nothing.
        kind = rng.choice(['outer_lane', 'inner_lane', 'centre', 'edge'],
                          p=[0.3, 0.3, 0.25, 0.15])
        lane = lane_offsets[k]
        if kind == 'outer_lane':
            off = float(lane[0])
        elif kind == 'inner_lane':
            off = float(lane[1])
        elif kind == 'centre':
            off = float(rng.uniform(-0.12, 0.12))
        else:
            edge = lo if rng.random() < 0.5 else hi
            off = float(edge - math.copysign(CONE_RADIUS + 0.05, edge))
        # `off` is where the pair's centre goes; the two cones straddle it.
        half = 0.5 * PAIR_GAP + CONE_RADIUS
        off = float(np.clip(off, max(lo + half, -REACH),
                            min(hi - half, REACH)))
        # Does a passable band actually survive this pair? Asked with the
        # planner's own `free_bands`, against walls and every cone already
        # placed, rather than inferred from the corridor arithmetic -- the
        # arithmetic says a body fits, which is not the same question.
        pair = [(cx[k] - h * math.sin(yaw[k]), cy[k] + h * math.cos(yaw[k]))
                for h in (off - 0.5 * PAIR_GAP, off + 0.5 * PAIR_GAP)]
        cones_tree = cKDTree(np.array(pair))
        # Asked of *both* routes. Placement samples the ccw centreline, but a
        # layout is run in both directions and the two centrelines are
        # different curves: a pair leaving 0.30 m measured from one can leave
        # 0.25 m measured from the other, which is exactly what the first
        # version of this gate let through. Which curve the pair rides is not
        # the obstacle's business.
        ok = True
        for rx, ry, ryaw, tree in routes:
            j = int(tree.query([pair[0]])[1][0])
            for step in range(-REACH_WINDOW, REACH_WINDOW + 1):
                m = (j + step) % len(rx)
                bands = free_bands(rx[m], ry[m], ryaw[m],
                                   [walls, cones_tree], PAIR_CLEARANCE,
                                   FOOTPRINT_BOX, span=1.75, step=0.05)
                wide = [(a, b) for a, b in bands if b - a >= MIN_BAND]
                if not wide:
                    ok = False
                    break
                # Reachable from both lanes, not merely present.
                # `lane_offsets` is indexed on the ccw profile, so the local
                # pair of offsets is the closest thing to "where the vehicles
                # will be" available here.
                here = lane_offsets[min(m, len(lane_offsets) - 1)]
                if max(min(abs(float(np.clip(side, a, b)) - side)
                           for a, b in wide) for side in here) > MAX_REACH:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        used.append(k)
        for side in (-1.0, +1.0):
            here = off + side * 0.5 * PAIR_GAP
            cones.append(dict(
                name=f'cone_{len(cones) + 1}',
                circle=dict(x=round(float(cx[k] - here * math.sin(yaw[k])), 3),
                            y=round(float(cy[k] + here * math.cos(yaw[k])), 3),
                            radius=CONE_RADIUS),
                _s=round(k * 0.1, 1), _offset=round(here, 2),
                _kind=str(kind)))
    return cones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=12)
    ap.add_argument('--cones', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=str(ROOT / 'layouts'))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    walls = read_map()
    cx, cy, yaw = centreline()
    routes = []
    for direction in ('ccw', 'cw'):
        rx, ry, ryaw = centreline(direction)
        routes.append((rx, ry, ryaw, cKDTree(np.column_stack([rx, ry]))))
    prof = list(csv.DictReader(
        open(ENV / 'centerlines' / 'ccw_offsets.csv', newline='')))
    lane_offsets = [(float(r['outer']), float(r['inner'])) for r in prof]
    rng = np.random.default_rng(args.seed)
    print(f'{"layout":10}{"cones":>7}   positions (arc length m @ lateral offset m)')
    for i in range(1, args.count + 1):
        cones = layout(rng, walls, cx, cy, yaw, lane_offsets, args.cones,
                       routes)
        where = '  '.join(f'{c.pop("_kind")[:6]}@{c.pop("_s")}m'
                          f'({c.pop("_offset"):+.2f})' for c in cones)
        path = out / f'layout_{i:02d}.yaml'
        body = yaml.safe_dump({'resolution': 0.05, 'structure': [],
                               'objects': cones},
                              sort_keys=False, default_flow_style=False)
        path.write_text(
            '# Generated by tools/make_layouts.py -- cones placed along the\n'
            '# route to test the planner against obstacles it was not tuned\n'
            '# around. Circles at a measured cone radius, not surveyed cells.\n'
            + body)
        print(f'{path.name:10}{len(cones):>7}   {where}')


if __name__ == '__main__':
    raise SystemExit(main())
