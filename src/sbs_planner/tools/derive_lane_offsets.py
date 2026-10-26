#!/usr/bin/env python3
"""Derive per-position lane offsets from the corridor the map actually has.

    python3 tools/derive_lane_offsets.py <centerline.csv> <map.yaml> [--write]

The lane offsets in use are constants -- the median of each recorded lane,
-0.50 and +0.53, so 1.03 m apart -- and both the corpus and the geometry say
the spacing should vary with the corridor.

From the corpus (23,477 obstacle-free samples, grouped by the corridor
half-width at the pair's midpoint):

    half-width      |lateral|   |along|
      0.0-0.8 m        0.88       0.78
      0.8-1.0 m        0.83       0.48
      1.0-1.3 m        1.17       0.40
      1.3-1.8 m        1.01       0.39
      1.8+   m         1.35       0.42

So the drivers' 1.31 m median is what they held *where there was room*; in
tight corridors they closed to 0.83-0.88 m and their stagger doubled. And
measured against this map with a 0.05 m margin, 1.31 m is unavailable over
31% of the ccw route and 15% of cw. A constant is the wrong shape of thing:
0.97 m holds the tight-corridor spacing everywhere, including the two thirds
of the route where 1.47 m is available.

Walls only, deliberately. Cones are runtime data -- injected by message and
handled by the object logic -- so baking them into the nominal lanes would
make the route obstacle-specific and wrong the moment the cones move.

The output is a profile, not a path: arc length against the two offsets, for
`Reference` to read. Regenerating it for a different map is one command,
which is what makes the planner portable -- see the note on computing a
centreline from the distance transform.
"""

import argparse
import csv
import math
import os

import numpy as np
import yaml
from scipy.spatial import cKDTree

#: Same rectangle as the planner's clearance check.
BACK, FRONT, HALF = -0.35, 0.55, 0.275
#: Body clearance a nominal lane must hold. This is not the smallest
#: clearance that is *possible* -- the drivers' own minimum was 0.001 m -- it
#: is how much the lane must leave spare for the pair to miss the wall while
#: tracking the lane imperfectly. The pair holds its target to about 0.08 m
#: at the median and worse in corners, and a profile derived at 0.05 m
#: stalled 4 of 6 ccw runs against 1 of 10 on the constants: geometrically
#: feasible, and unusable once tracking error appears. 0.15 m was not enough
#: either: the ccw corner near (20.2, 8.5) stalled three separate runs, with
#: the vehicle arriving carrying about 0.10 m of error against the 0.114 m
#: the profile actually delivered there. Raising it was tried against the
#: recurring wedge near (13.0, -7.4) and did not fix it: 0.25 m still wedged
#: (1 of 6 ccw runs, same site, same +0.02 m wall clearance), and the cost
#: was monotone -- lateral spacing 1.14 / 1.11 / 1.04 m and head turn 14.0 /
#: ~16 / 19.2 deg at 0.20 / 0.25 / 0.30. The wedge is not a clearance
#: problem: the vehicle arrives 0.23 m off a lane that is already clear,
#: because the pair holds formation through a pinch where `side by side` is 0.
#: The margin only has to cover tracking error, which is what it does.
MARGIN = 0.20
#: Rate-limiting the lane's lateral sweep was tried on 2026-10-06 and
#: **rejected**. It is recorded here because the offline case for it is
#: strong and someone will think of it again.
#:
#: The motivation was measured, not guessed. At the recurring cw stall near
#: (20.7, 14.5) the lane target sweeps inward 0.34 m over about 4 m of
#: travel, the vehicle tracks it with 0.22 m of lag, and the lag alone brings
#: its body to 0.088 m -- under `object_clearance` -- at which point nothing
#: is feasible and it stops for the rest of the run. No cone is involved and
#: the pair is correctly formed at 1.15-1.20 m.
#:
#: The implementation capped |d offset / ds| at 0.015 m/m, enforced backwards
#: along arc so a change starts early rather than arriving late, every move
#: clamped into its own free band; and took the pair's half-width as the
#: running minimum of what the band allows over the next 3.2 m, so the pair
#: narrows *before* a pinch instead of while entering it. Offline it did
#: everything intended: sweep p95 0.191 -> 0.093 (cw) and 0.200 -> 0.077
#: (ccw), the inner lane's travel through the stall window 0.303 -> 0.094 m,
#: its median clearance as the planner sees it 0.230 -> 0.441 m, and the
#: margin guarantee intact at +0.200 m.
#:
#: Closed loop it was worse, over 24 runs per arm against the baseline:
#: stalls 4 -> 8 and wall contact 0 -> 3 runs, worst 114 mm. A third arm
#: isolating the object-aware `side_by_side` flag alone gave 6 stalls and
#: 2 contact runs, so both changes hurt and about equally.
#:
#: What the three arms agree on is the thing to chase: filing rose 15-21%
#: -> 26-32% -> 32-37% and convergence fell 53/57 -> 38/47 -> 37/43,
#: monotonically with the failures. The failures live in the file path, not
#: in the abreast case, and a profile change that files more will lose
#: whatever it wins geometrically. Fix filing before touching the lanes.
#:
#: Two variants failed earlier and are worth not repeating: leading the whole
#: profile by the 3.2 m lag puts the lane outside the corridor (worst
#: clearance -0.037 m, 7.5% of samples below 0.10 m; only about 1.0 m of
#: lead fits), and anticipating only the pair's translation achieves nothing
#: (p95 stays at 0.20) because the sweep is mostly the pair widening and
#: closing with the corridor, not translating -- which also means that taking
#: all the width the band offers at every sample *is* the sweep.
#: Spacing the drivers held with room, and with none.
SPACING_OPEN, SPACING_TIGHT = 1.35, 0.83
#: Smoothing window for the offset profile, in metres of arc. A target that
#: steps is worse than one that is slightly wrong: a discontinuous lateral
#: target measurably destabilised the pair when it was tried at runtime.
SMOOTH_M = 3.0
#: Body clearance the planner requires at runtime, and so the clearance the
#: side-by-side flag is judged at -- `object_clearance` in the planner.
PAIR_CLEARANCE = 0.10


def read_map(path):
    info = yaml.safe_load(open(path))
    res = info['resolution']
    ox, oy = info['origin'][0], info['origin'][1]
    image = os.path.join(os.path.dirname(path), info['image'])
    with open(image, 'rb') as handle:
        assert handle.readline().strip() == b'P5'
        line = handle.readline()
        while line.startswith(b'#'):
            line = handle.readline()
        width, height = map(int, line.split())
        int(handle.readline())
        grid = np.frombuffer(handle.read(width * height),
                             dtype=np.uint8).reshape(height, width)
    occupancy = (255.0 - grid.astype(float)) / 255.0
    if info.get('negate', 0):
        occupancy = 1.0 - occupancy
    rows, cols = np.nonzero(occupancy > info.get('occupied_thresh', 0.65))
    return np.column_stack([ox + (cols + 0.5) * res,
                            oy + ((height - 1 - rows) + 0.5) * res])


def read_unknown(path):
    """Cell centres the map has never observed.

    The map has three states and reading only the occupied one treats the
    third as free: unexplored space has no occupied cells, so a lane or a band
    measured through it looks perfectly clear. Measured on this map, 95.9% of
    the grid is unknown and it supplied the widest apparent corridor on 49% of
    the ccw route.

    These are not obstacles -- the planner's clearance constraints never see
    them. They bound where a body may be *placed*, which is the question this
    tool asks.
    """
    info = yaml.safe_load(open(path))
    res = info['resolution']
    ox, oy = info['origin'][0], info['origin'][1]
    image = os.path.join(os.path.dirname(path), info['image'])
    with open(image, 'rb') as handle:
        assert handle.readline().strip() == b'P5'
        line = handle.readline()
        while line.startswith(b'#'):
            line = handle.readline()
        width, height = map(int, line.split())
        int(handle.readline())
        grid = np.frombuffer(handle.read(width * height),
                             dtype=np.uint8).reshape(height, width)
    occupancy = (255.0 - grid.astype(float)) / 255.0
    if info.get('negate', 0):
        occupancy = 1.0 - occupancy
    blind = (~(occupancy > info.get('occupied_thresh', 0.65))
             & ~(occupancy < info.get('free_thresh', 0.196)))
    rows, cols = np.nonzero(blind)
    return np.column_stack([ox + (cols + 0.5) * res,
                            oy + ((height - 1 - rows) + 0.5) * res])


def clearance(points, tree, x, y, yaw):
    """Signed distance from the body rectangle to the nearest point.

    Exact over everything within 3 m, not the four nearest to base_link:
    `footprint_clearance` takes the latter shortcut and overestimates by up to
    0.28 m on a long wall, which is tolerable as a ranking but not when the
    answer becomes reference data.
    """
    near = points[tree.query_ball_point([x, y], 3.0)]
    if not len(near):
        return 9.0
    cos, sin = math.cos(yaw), math.sin(yaw)
    dx = near[:, 0] - x
    dy = near[:, 1] - y
    along = dx * cos + dy * sin
    across = -dx * sin + dy * cos
    gap_along = np.maximum(BACK - along, along - FRONT)
    gap_across = np.abs(across) - HALF
    return float((np.hypot(np.maximum(gap_along, 0.0),
                           np.maximum(gap_across, 0.0))
                  + np.minimum(np.maximum(gap_along, gap_across), 0.0)).min())


def free_bands(points, tree, x, y, yaw, span=2.0, step=0.02, margin=None):
    """Every run of body-centre offsets that clears, not just the best one.

    A single band answers "can the pair ride here"; the list answers "can they
    ride here *around* something", which is a different question and the one
    a central obstacle poses. See `straddles`.
    """
    if margin is None:
        margin = MARGIN
    offsets = np.arange(-span, span + 1e-9, step)
    ok = np.array([clearance(points, tree, x - o * math.sin(yaw),
                             y + o * math.cos(yaw), yaw) >= margin
                   for o in offsets])
    bands, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            bands.append((float(offsets[i]), float(offsets[j])))
            i = j + 1
        else:
            i += 1
    return bands


def straddles(bands, need_pair, max_pair):
    """Can one body sit in one band and the other in a different band?

    This is what the drivers did at a cone standing in the middle of the
    aisle: they passed one either side of it and stayed side_by_side, rather than
    falling into file. Photographed at the third cw obstacle, where the
    corridor is 2.40 m wide, the cones occupy 0.47 m of it near the centre,
    and 0.82-0.97 m is left on one side with 1.08-1.26 m on the other -- each
    enough for a body, neither enough for two.

    The single-band test cannot express that, and answers the question it does
    ask correctly, so the pair was told to file at exactly the obstacle the
    humans rode through side by side.

    Two bodies are placed at the points of their bands nearest to each other's
    side, so the pair rides as close as the geometry allows, and the result
    counts only if the resulting separation is a formation the drivers would
    recognise: at least `need_pair` so the bodies do not overlap, at most
    `max_pair` so "side_by_side" still means side by side.
    """
    best = None
    for a in range(len(bands)):
        for b in range(a + 1, len(bands)):
            lo_a, hi_a = bands[a]
            lo_b, hi_b = bands[b]
            # bands are ordered, so b lies outboard of a
            gap = max(need_pair, lo_b - hi_a)
            left = hi_a if lo_b - hi_a >= need_pair else min(hi_a, lo_b - need_pair)
            if left < lo_a:
                continue
            right = left + gap
            if right > hi_b or right < lo_b:
                right = min(hi_b, max(lo_b, left + need_pair))
            sep = right - left
            if need_pair <= sep <= max_pair:
                if best is None or sep < best:
                    best = sep
    return best


def free_band(points, tree, x, y, yaw, span=2.0, step=0.02, margin=None):
    """The widest run of body-centre offsets that clears, around the centre.

    `margin` defaults to MARGIN, which is the room a nominal *lane* leaves
    spare for tracking error. The side-by-side flag asks a different question --
    whether two bodies physically fit here -- and must use the clearance the
    planner actually requires at runtime, not the lane's design margin. At
    0.20 m the flag called the ccw start site unfittable when its band is
    0.78 m wide and two bodies need 0.65 m.
    """
    best = None
    for lo, hi in free_bands(points, tree, x, y, yaw, span, step, margin):
        # Prefer the band containing the centreline; else the widest.
        score = (lo <= 0.0 <= hi, hi - lo)
        if best is None or score > best[0]:
            best = (score, lo, hi)
    return (None, None) if best is None else (best[1], best[2])


def main():
    global MARGIN
    parser = argparse.ArgumentParser()
    parser.add_argument('centerline')
    parser.add_argument('map_yaml')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--margin', type=float, default=None,
                        help='body clearance the lane must leave spare')
    parser.add_argument('--objects', default=None,
                        help='objects.yaml, included in the side-by-side flag')
    args = parser.parse_args()
    if args.margin is not None:
        MARGIN = args.margin

    # Walls *and* unexplored space. A lane offset justified by cells nothing
    # has ever observed is not a lane offset; see `read_unknown`.
    surveyed = read_map(args.map_yaml)
    blind = read_unknown(args.map_yaml)
    points = np.vstack([surveyed, blind]) if len(blind) else surveyed
    tree = cKDTree(points)
    print(f'map: {len(surveyed)} occupied cells, {len(blind)} never observed')

    # The offsets come from walls alone -- cones are runtime data and a lane
    # baked around them is wrong the moment they move. The *side-by-side flag* is
    # the opposite case: whether the pair has to travel in file is exactly a
    # question about what is in the way, so it takes the objects too, and it
    # must be regenerated when they change.
    blocked = points
    if args.objects:
        cells = [np.array(o['cells']) for o in
                 yaml.safe_load(open(args.objects))['objects'] if 'cells' in o]
        if cells:
            blocked = np.vstack([points] + cells)
    blocked_tree = cKDTree(blocked)

    rows = list(csv.DictReader(open(args.centerline, newline='')))
    cx = np.array([float(r['x']) for r in rows])
    cy = np.array([float(r['y']) for r in rows])
    tx, ty = np.gradient(cx), np.gradient(cy)
    norm = np.hypot(tx, ty)
    tx, ty = tx / norm, ty / norm
    yaw = np.arctan2(ty, tx)
    step = np.hypot(np.diff(cx, append=cx[0]), np.diff(cy, append=cy[0]))
    arc = np.cumsum(step)

    # Which side each lane sits on, taken from the recorded lanes so the
    # output drops into the existing convention rather than inventing one.
    lane_dir = os.path.dirname(args.centerline)
    prefix = os.path.basename(args.centerline).replace('_centerline.csv', '')
    outer_sign = None
    path = os.path.join(lane_dir, f'{prefix}_lane_outer.csv')
    if os.path.exists(path):
        lane = list(csv.DictReader(open(path, newline='')))
        lx = np.array([float(r['x']) for r in lane])
        ly = np.array([float(r['y']) for r in lane])
        k = cKDTree(np.column_stack([cx, cy])).query(
            np.column_stack([lx, ly]))[1]
        off = -(lx - cx[k]) * ty[k] + (ly - cy[k]) * tx[k]
        outer_sign = float(np.sign(np.median(off)))
    if outer_sign is None:
        raise SystemExit(f'missing {path}, cannot fix the side convention')

    # Can two bodies stand side by side here at all? The free band is of
    # body *centres*, so two of them need 2 * half + clearance between them.
    # This replaces a runtime proxy: `side_by_side` compared the two lateral
    # targets, and when both vehicles shifted for the same cone they
    # converged, room collapsed from 1.31 m to 0.33 m, and the file rule
    # fired on 84-96% of the first five seconds of every ccw run -- at a site
    # whose band is 0.78 m wide, where they fit comfortably.
    need_pair = 2.0 * HALF + PAIR_CLEARANCE
    side_by_side = np.zeros(len(cx), dtype=int)
    straddle = np.zeros(len(cx), dtype=int)
    for i in range(len(cx)):
        bands = free_bands(blocked, blocked_tree, cx[i], cy[i], yaw[i],
                           margin=PAIR_CLEARANCE)
        widest = max((hi - lo for lo, hi in bands), default=-1.0)
        side_by_side[i] = int(widest >= need_pair)
        if not side_by_side[i]:
            straddle[i] = int(straddles(bands, need_pair,
                                        SPACING_OPEN) is not None)

    centre = np.zeros(len(cx))
    spacing = np.zeros(len(cx))
    for i in range(len(cx)):
        lo, hi = free_band(points, tree, cx[i], cy[i], yaw[i])
        if lo is None:
            centre[i], spacing[i] = 0.0, SPACING_TIGHT
            continue
        width = min(SPACING_OPEN, max(0.0, hi - lo))
        # Centred on the centreline wherever the band allows it, and moved
        # only as far as it must to fit. Centring on the *corridor* instead
        # would shift the pair up to 0.38 m off the recorded route, since the
        # centreline is the drivers' mean path and not the medial axis -- that
        # buys clearance by changing the experiment.
        spacing[i] = width
        centre[i] = float(np.clip(0.0, lo + width / 2.0, hi - width / 2.0))

    def smooth(values):
        width = max(3, int(SMOOTH_M / max(np.median(step), 1e-3)) | 1)
        kernel = np.ones(width) / width
        return np.convolve(np.r_[values[-width:], values, values[:width]],
                           kernel, mode='same')[width:-width]

    centre, spacing = smooth(centre), smooth(spacing)
    outer = centre + outer_sign * spacing / 2.0
    inner = centre - outer_sign * spacing / 2.0

    # Smoothing can push an offset back into a wall and clamping it back puts
    # a step in the profile, so the two are alternated until they settle.
    # Clamping once at the end left 71 mm of lateral change per 10 cm of
    # travel -- as steep as the runtime nudge that measurably destabilised
    # the pair, and the reason this is a smooth profile rather than a
    # per-tick decision at all.
    bands = [free_band(points, tree, cx[i], cy[i], yaw[i])
             for i in range(len(cx))]
    pulled = 0
    # 12, not more. Each smooth-and-clamp cycle pulls wide sections toward
    # narrow ones, so the alternation is not a projection that preserves
    # spacing: at 200 sweeps the median spacing eroded from 1.26 m to 1.07 m,
    # giving back the whole gain over the constants to buy a smoothness the
    # pair does not need.
    for sweep in range(12):
        moved = 0
        for i, (lo, hi) in enumerate(bands):
            if lo is None:
                continue
            for offsets in (outer, inner):
                fixed = float(np.clip(offsets[i], lo, hi))
                if abs(fixed - offsets[i]) > 1e-6:
                    offsets[i] = fixed
                    moved += 1
        pulled = moved
        if not moved:
            break
        outer, inner = smooth(outer), smooth(inner)

    # A final clamp with no smoothing after it, so the margin is *guaranteed*
    # rather than approached. Without this the last smoothing pass pushed
    # points back inside: a profile requested at 0.15 m delivered 0.114 m at
    # the ccw corner near (20.2, 8.5), and a vehicle arriving there with
    # 0.10 m of tracking error stalled against the wall -- three times.
    # The steps this introduces are small because the alternation has already
    # converged; the printed max step is the check.
    for i, (lo, hi) in enumerate(bands):
        if lo is None:
            continue
        for offsets in (outer, inner):
            offsets[i] = float(np.clip(offsets[i], lo, hi))

    worst = min(
        min(clearance(points, tree, cx[i] - o * math.sin(yaw[i]),
                      cy[i] + o * math.cos(yaw[i]), yaw[i])
            for o in (outer[i], inner[i]))
        for i in range(len(cx)))
    gap = np.abs(outer - inner)
    print(f'{prefix}: {len(cx)} samples, outer side {outer_sign:+.0f}')
    print(f'  spacing   median {np.median(gap):.2f}  p10 {np.percentile(gap,10):.2f}'
          f'  p90 {np.percentile(gap,90):.2f}  min {gap.min():.2f}  max {gap.max():.2f}')
    print(f'  outer     median {np.median(outer):+.2f}  '
          f'range {outer.min():+.2f}..{outer.max():+.2f}')
    print(f'  inner     median {np.median(inner):+.2f}  '
          f'range {inner.min():+.2f}..{inner.max():+.2f}')
    print(f'  worst body clearance on the derived lanes: {worst:+.3f} m'
          f'   ({pulled} offsets clamped after smoothing)')
    both = np.clip(side_by_side + straddle, 0, 1)
    print(f'  straddle rescues {int(straddle.sum())} samples '
          f'({100 * straddle.mean():.1f}% of the route); side by side one way '
          f'or the other on {100 * both.mean():.0f}%')
    print(f'  side by side possible on {100 * side_by_side.mean():.0f}% of the route'
          f'{" (walls + objects)" if args.objects else " (walls only)"}')
    print(f'  for comparison, the constants in use: -0.50 / +0.53, '
          f'spacing 1.03, and the drivers held 1.31 with room')

    if args.write:
        out = os.path.join(lane_dir, f'{prefix}_offsets.csv')
        with open(out, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['s', 'outer', 'inner', 'side_by_side', 'straddle'])
            for s, o, n, a, t in zip(arc, outer, inner, side_by_side, straddle):
                writer.writerow([f'{s:.4f}', f'{o:.4f}', f'{n:.4f}',
                                 int(a), int(t)])
        print(f'  wrote {out}')


if __name__ == '__main__':
    main()
