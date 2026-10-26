#!/usr/bin/env python3
"""The site where the vehicle pins beside a cone pair, drawn.

    python3 src/sbs_planner/tools/plot_pinned_site.py --out docs/pinned_site.pdf

Top-down view of one decision: where the free space actually is over the next
few metres, where the lane goes, where `shifted_lateral_target` points, and
what the vehicle did. The claim to check by eye is that the target lands in
the gap *between* two free bands -- somewhere no body can be -- because the
rule picks the option nearest the lane and never asks the corridor.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src' / 'sbs_planner'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sbs_planner.cost import (FOOTPRINT_BOX, free_bands,        # noqa: E402
                              shifted_lateral_target)
from sbs_planner.reference import Reference                      # noqa: E402
from derive_lane_offsets import read_map                         # noqa: E402
import os                                                        # noqa: E402


def unexplored(map_yaml):
    """Cell centres the map has never observed.

    Free space is otherwise inferred from the absence of an occupied cell,
    which makes unexplored map read as perfectly clear.
    """
    info = yaml.safe_load(open(map_yaml))
    res, (ox, oy) = info['resolution'], info['origin'][:2]
    with open(os.path.join(os.path.dirname(map_yaml), info['image']), 'rb') as h:
        assert h.readline().strip() == b'P5'
        line = h.readline()
        while line.startswith(b'#'):
            line = h.readline()
        width, height = map(int, line.split())
        int(h.readline())
        grid = np.frombuffer(h.read(width * height),
                             dtype=np.uint8).reshape(height, width)
    occ = (255.0 - grid.astype(float)) / 255.0
    blind = ~(occ > info.get('occupied_thresh', 0.65)) \
        & ~(occ < info.get('free_thresh', 0.196))
    rows, cols = np.nonzero(blind)
    return np.column_stack([ox + (cols + 0.5) * res,
                            oy + ((height - 1 - rows) + 0.5) * res])
from summarise_layouts import load                               # noqa: E402

ENV = ROOT / 'environments' / 'experiment'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layout', default='layout_05')
    ap.add_argument('--direction', default='ccw')
    ap.add_argument('--run', default='repeat_runs/layout_05_fix_ccw_r1.csv')
    ap.add_argument('--at', nargs=2, type=float, default=[20.99, 15.72])
    ap.add_argument('--out', default='docs/pinned_site.pdf')
    args = ap.parse_args()

    walls = read_map(str(ENV / 'map.yaml'))
    cones = np.array([[o['circle']['x'], o['circle']['y']] for o in
                      yaml.safe_load((ROOT / 'layouts' / f'{args.layout}.yaml')
                                     .read_text())['objects'] if 'circle' in o])
    radius = [o['circle']['radius'] for o in
              yaml.safe_load((ROOT / 'layouts' / f'{args.layout}.yaml')
                             .read_text())['objects'] if 'circle' in o][0]
    ref = Reference.load(str(ENV / 'centerlines'
                             / f'{args.direction}_centerline.csv'))
    X, Y = args.at
    i = int(cKDTree(ref.centre).query([[X, Y]])[1][0])
    blind = unexplored(str(ENV / 'map.yaml'))
    near_blind = blind[np.hypot(blind[:, 0] - X, blind[:, 1] - Y) < 12.0]
    trees = [cKDTree(walls), cKDTree(cones), cKDTree(near_blind)]

    yaw_here = ref.yaw[i]
    lane = ref.lane_offset_at([X, Y], 'outer')

    fig, ax = plt.subplots(figsize=(9, 7))
    near = walls[np.hypot(walls[:, 0] - X, walls[:, 1] - Y) < 6.0]
    shown = blind[np.hypot(blind[:, 0] - X, blind[:, 1] - Y) < 6.0]
    ax.plot(shown[:, 0], shown[:, 1], 's', ms=1.6, color='#e0d7c0',
            label='never observed')
    ax.plot(near[:, 0], near[:, 1], 's', ms=1.6, color='0.45', label='wall')

    # Free bands over the route ahead, drawn as the segments a body centre
    # may occupy. This is the thing the target is supposed to land inside.
    # Stations where the vehicle's own side is shut are drawn in red, because
    # that stretch is the whole argument.
    xs, ys, yaws, reach = ref.centre_ahead([X, Y], 4.0, 0.2)
    seen = {'open': False, 'shut': False}
    for x, y, yaw, r in zip(xs, ys, yaws, reach):
        bands = free_bands(float(x), float(y), float(yaw), trees, 0.10,
                           FOOTPRINT_BOX, span=1.75, step=0.05)
        shut = not any(lo <= lane <= hi for lo, hi in bands)
        key = 'shut' if shut else 'open'
        colour = '#c62828' if shut else '#2e7d32'
        for lo, hi in bands:
            ax.plot([x - lo * np.sin(yaw), x - hi * np.sin(yaw)],
                    [y + lo * np.cos(yaw), y + hi * np.cos(yaw)],
                    color=colour, lw=4, alpha=0.5, solid_capstyle='butt',
                    label=('free band, lane side shut' if shut
                           else 'free band, lane side open')
                    if not seen[key] else None)
            seen[key] = True

    for c in cones:
        ax.add_patch(plt.Circle(c, radius, color='#c62828', zorder=5))
    ax.plot([], [], 'o', color='#c62828', label='cones')

    # lane target and the shift the planner asks for
    yaw = yaw_here
    cos, sin = np.cos(yaw), np.sin(yaw)
    d = cones - np.array([X, Y])
    along = d[:, 0] * cos + d[:, 1] * sin
    cross = -d[:, 0] * sin + d[:, 1] * cos
    sel = (along > -0.35) & (along < 4.0)
    target = shifted_lateral_target(lane, cross[sel], 0.10, FOOTPRINT_BOX[2],
                                    current=lane)
    for off, colour, name in ((lane, '#1565c0', f'outer lane {lane:+.2f}'),
                              (target, '#ef6c00',
                               f'target the planner asks for {target:+.2f}')):
        px = ref.centre[i, 0] - off * np.sin(yaw)
        py = ref.centre[i, 1] + off * np.cos(yaw)
        ax.plot(px, py, 'o', ms=13, mfc='none', mew=2.5, color=colour,
                label=name, zorder=7)

    g = load(ROOT / args.run)
    t = g('t') - g('t')[0]
    w = (t > 60) & (t < 170)
    ax.plot(g('v1_x')[w], g('v1_y')[w], '-', color='k', lw=1.2, alpha=0.8,
            label='v1, t = 60-170 s')
    ax.plot(X, Y, '*', ms=18, color='k', zorder=8, label='where it pins')

    ax.set_aspect('equal')
    ax.set_xlim(X - 4, X + 4)
    ax.set_ylim(Y - 3.5, Y + 3.5)
    ax.set_title(f'{args.layout} {args.direction}: the asked-for target lies '
                 )
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f'wrote {out}')
    print(f'  outer lane {lane:+.2f}   target {target:+.2f}')
    bands = free_bands(ref.centre[i, 0], ref.centre[i, 1], yaw, trees, 0.10,
                       FOOTPRINT_BOX, span=1.75, step=0.05)
    print(f'  bands here: {[(round(a,2), round(b,2)) for a, b in bands]}')
    print(f'  target inside a band: '
          f'{any(a <= target <= b for a, b in bands)}')


if __name__ == '__main__':
    main()
