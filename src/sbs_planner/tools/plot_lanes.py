#!/usr/bin/env python3
"""Draw the derived lanes on the map, with the obstacles and the file segments.

    ./in_container.sh python3 src/sbs_planner/tools/plot_lanes.py
    ./in_container.sh python3 src/sbs_planner/tools/plot_lanes.py --out docs/lanes

The spacing plot answers "how wide is the formation here"; this answers "where
exactly does it go", which is the question that matters when a lane turns out
to pass through something. Each obstacle gets an inset at its own scale,
because the interesting distances there are centimetres on a route tens of
metres long.

Everything is read from the shipped files, so a picture that disagrees with a
run is evidence about the run, not about the picture.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
ENV = ROOT / 'environments' / 'experiment'
FOOTPRINT_BOX = (-0.35, 0.55, 0.275)
LOOKAHEAD = 2.6

plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})


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
    rows, cols = np.nonzero(((255.0 - image.astype(float)) / 255.0)
                            > info['occupied_thresh'])
    return np.column_stack([ox + (cols + 0.5) * res,
                            oy + ((height - 1 - rows) + 0.5) * res])


def lanes(direction):
    centre = list(csv.DictReader(
        open(ENV / 'centerlines' / f'{direction}_centerline.csv', newline='')))
    cx = np.array([float(r['x']) for r in centre])
    cy = np.array([float(r['y']) for r in centre])
    yaw = np.array([float(r['yaw']) for r in centre])
    prof = list(csv.DictReader(
        open(ENV / 'centerlines' / f'{direction}_offsets.csv', newline='')))
    s = np.array([float(r['s']) for r in prof])
    flag = np.array([int(r['side_by_side']) for r in prof])
    strad = np.array([int(r.get('straddle', 0)) for r in prof])
    out = {}
    for side in ('outer', 'inner'):
        off = np.array([float(r[side]) for r in prof])
        out[side] = (cx - off * np.sin(yaw), cy + off * np.cos(yaw))
    step = float(np.median(np.diff(s)))
    span = max(1, int(round(LOOKAHEAD / step)))
    index = (np.arange(len(flag))[:, None] + np.arange(span + 1)[None, :]) % len(flag)
    either = np.clip(flag + strad, 0, 1)
    return (cx, cy, out, flag, flag[index].all(axis=1),
            either[index].all(axis=1), s)


def objects():
    doc = yaml.safe_load((ENV / 'objects.yaml').read_text())
    return [(o['name'], np.asarray(o['cells'], float))
            for o in (doc.get('objects') or [])]


def draw(ax, walls, cx, cy, lane, effective, either, cones, lw=1.4, dots=1.2):
    ax.scatter(walls[:, 0], walls[:, 1], s=dots, c='0.72', marker='s',
               linewidths=0, zorder=1)
    for name, cells in cones:
        ax.scatter(cells[:, 0], cells[:, 1], s=dots * 5, c='#d1651b',
                   marker='s', linewidths=0, zorder=3)
    ax.plot(cx, cy, ls=':', lw=0.7, color='0.55', zorder=2)
    for side, colour in (('outer', '#1f4e9c'), ('inner', '#128a5a')):
        x, y = lane[side]
        ax.plot(x, y, lw=lw, color=colour, zorder=4,
                label=f'{side} lane' if lw > 1 else None)
        # amber: the single-band test says file, but the pair can straddle
        # the obstacle -- one body each side -- which is what the drivers did.
        # red: no side-by-side arrangement fits at all.
        for mask, colour in ((~effective & either, '#e0a21b'),
                             (~either, '#c0392b')):
            idx = np.nonzero(mask)[0]
            for group in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                if len(group):
                    ax.plot(x[group], y[group], lw=lw + 1.4, color=colour,
                            zorder=5, solid_capstyle='butt')
    ax.set_aspect('equal')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'docs'))
    args = ap.parse_args()
    walls = read_map()
    cones = objects()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    for direction in ('ccw', 'cw'):
        cx, cy, lane, raw, effective, either, s = lanes(direction)
        # obstacle groups, for the insets
        groups, used = [], set()
        for name, cells in cones:
            if name in used:
                continue
            centre = cells.mean(0); members = [cells]; used.add(name)
            for other, cell2 in cones:
                if other not in used and np.hypot(*(cell2.mean(0) - centre)) < 1.0:
                    members.append(cell2); used.add(other)
            groups.append(np.vstack(members).mean(0))
        groups.sort(key=lambda p: np.hypot(cx - p[0], cy - p[1]).argmin())

        # The route is taller than it is wide, so the overview gets a tall
        # panel on the left and the obstacle insets stack to the right of it;
        # a wide panel with equal aspect would be mostly white paper.
        pad = 1.5
        w = (cx.max() - cx.min()) + 2 * pad
        h = (cy.max() - cy.min()) + 2 * pad
        fig = plt.figure(figsize=(7.2, 4.4))
        left_w = 0.40
        main_ax = fig.add_axes([0.015, 0.02, left_w, 0.88])
        draw(main_ax, walls, cx, cy, lane, effective, either, cones)
        main_ax.set_xlim(cx.min() - pad, cx.max() + pad)
        main_ax.set_ylim(cy.min() - pad, cy.max() + pad)
        main_ax.set_xticks([]); main_ax.set_yticks([])
        for spine in main_ax.spines.values():
            spine.set_visible(False)
        step = float(np.median(np.diff(s)))
        filed_m = float((~effective).sum()) * step
        left_m = float((~either).sum()) * step
        fig.text(0.015, 0.958,
                 f'{direction} — derived lanes over {s.max():.0f} m',
                 fontsize=9, weight='bold')
        fig.text(0.015, 0.925,
                 f'amber: straddle, one body each side ({filed_m - left_m:.1f} m)   '
                 f'·   red: must file ({left_m:.1f} m)', fontsize=8)
        main_ax.legend(fontsize=7, frameon=False, loc='lower left')
        for i, centre in enumerate(groups):
            row, col = divmod(i, 2)
            ax = fig.add_axes([0.46 + col * 0.27, 0.48 - row * 0.45,
                               0.25, 0.40])
            draw(ax, walls, cx, cy, lane, effective, either, cones, lw=2.0, dots=4.0)
            ax.set_xlim(centre[0] - 1.8, centre[0] + 1.8)
            ax.set_ylim(centre[1] - 1.8, centre[1] + 1.8)
            ax.set_xticks([]); ax.set_yticks([])
            k = int(np.hypot(cx - centre[0], cy - centre[1]).argmin())
            filed_here = ('side_by_side' if effective[k]
                          else 'straddle' if either[k] else 'file')
            ax.set_title(f'obstacle {i + 1} — {filed_here}', fontsize=7.5)
            for spine in ax.spines.values():
                spine.set_color('0.8')
        path = outdir / f'lanes_{direction}.pdf'
        fig.savefig(path, dpi=200); plt.close(fig)
        print(f'wrote {path}  (file segments: {filed_m:.1f} m)')


if __name__ == '__main__':
    raise SystemExit(main())
