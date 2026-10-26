#!/usr/bin/env python3
"""What the pair does laterally while it passes a cone.

    python3 src/sbs_planner/tools/plot_lateral_at_cone.py \
        --runs 'brake_ab/layout_02_*_cw_r[0-9].csv' --layout layout_02 \
        --direction cw --at 26.11 5.21 --out docs/lateral_at_cone.pdf

Cross-track of both vehicles against distance along the route, through the
site, with the cones' own cross-track and the free bands behind them. The
question it answers is the one the contact tables cannot: *where the pair put
itself*, rather than how deep it went.
"""
import argparse
import glob
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
from sbs_planner.cost import FOOTPRINT_BOX, free_bands            # noqa: E402
from sbs_planner.reference import Reference                        # noqa: E402
from derive_lane_offsets import read_map                           # noqa: E402
from plot_pinned_site import unexplored                            # noqa: E402
from summarise_layouts import load, measure, Contaminated          # noqa: E402

ENV = ROOT / 'environments' / 'experiment'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--layout', required=True)
    ap.add_argument('--direction', default='cw')
    ap.add_argument('--at', nargs=2, type=float, required=True)
    ap.add_argument('--span', type=float, default=6.0)
    ap.add_argument('--out', default='docs/lateral_at_cone.pdf')
    args = ap.parse_args()

    ref = Reference.load(str(ENV / 'centerlines'
                             / f'{args.direction}_centerline.csv'))
    centre = cKDTree(ref.centre)
    cones = np.array([[o['circle']['x'], o['circle']['y']] for o in
                      yaml.safe_load((ROOT / 'layouts'
                                      / f'{args.layout}.yaml').read_text())
                      ['objects'] if 'circle' in o])
    walls = cKDTree(read_map(str(ENV / 'map.yaml')))
    blind = cKDTree(unexplored(str(ENV / 'map.yaml')))
    objs = cKDTree(cones)

    # Arc length along the centreline, so several runs share an x axis.
    step = np.hypot(*np.diff(np.vstack([ref.centre, ref.centre[:1]]),
                             axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(step)])[:len(ref.centre)]
    here = int(centre.query([args.at])[1][0])
    s0 = arc[here]

    def along_of(x, y):
        idx = centre.query(np.column_stack([x, y]))[1]
        s = arc[idx] - s0
        total = arc[-1]
        return (s + total / 2) % total - total / 2

    fig, ax = plt.subplots(figsize=(10, 6))

    # free bands through the site, as a grey corridor
    window = np.arange(here - 70, here + 70) % len(ref.centre)
    for i in window:
        bands = free_bands(ref.centre[i, 0], ref.centre[i, 1], ref.yaw[i],
                           [walls, objs, blind], 0.10, FOOTPRINT_BOX,
                           span=1.75, step=0.05)
        s = arc[i] - s0
        if abs(s) > args.span:
            continue
        for lo, hi in bands:
            ax.plot([s, s], [lo, hi], color='0.82', lw=3,
                    solid_capstyle='butt', zorder=0)
    ax.plot([], [], color='0.82', lw=3, label='where a body may be')

    # the cones themselves, in cross-track
    for c in cones:
        i = int(centre.query([c])[1][0])
        s = arc[i] - s0
        if abs(s) > args.span:
            continue
        cos, sin = np.cos(ref.yaw[i]), np.sin(ref.yaw[i])
        d = c - ref.centre[i]
        ax.plot(s, -d[0] * sin + d[1] * cos, 'o', ms=11, color='#c62828',
                zorder=6)
    ax.plot([], [], 'o', color='#c62828', label='cone centres')

    for name, colour in (('v1', '#1565c0'), ('v2', '#ef6c00')):
        first = True
        for path in sorted(glob.glob(str(ROOT / args.runs))):
            try:
                measure(Path(path))
            except Contaminated:
                continue
            g = load(Path(path))
            x, y = g(f'{name}_x'), g(f'{name}_y')
            s = along_of(x, y)
            cross = ref.signed_cross_track(x[:, None], y[:, None])
            keep = np.abs(s) <= args.span
            order = np.argsort(s[keep])
            ax.plot(s[keep][order], cross[keep][order], '-', color=colour,
                    lw=1.0, alpha=0.6,
                    label=f'{name} cross-track' if first else None)
            first = False
            # mark contact
            d = g(f'{name}_obj_body')
            hit = np.where((d < -0.025) & keep)[0]
            if len(hit):
                ax.plot(s[hit], cross[hit], 'x', ms=9, mew=2.5,
                        color='#b71c1c', zorder=7,
                        label='body inside a cone' if first else None)

    ax.axvline(0, color='0.6', ls=':', lw=1)
    ax.set_xlabel('distance along the route through the site  [m]')
    ax.set_ylabel('cross-track offset, + left  [m]')
    ax.set_title(f'{args.layout} {args.direction}: the pair through the cone '
                 f'at ({args.at[0]:.1f}, {args.at[1]:.1f})')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
