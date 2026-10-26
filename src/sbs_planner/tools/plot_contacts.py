#!/usr/bin/env python3
"""Where the bodies touch a wall, over a set of metrics runs.

    python3 src/sbs_planner/tools/plot_contacts.py 'runs/*.csv' --out docs/contacts.pdf

A table of depths says how bad the contact is but not whether it is one place
or many, and those call for different fixes: one site is a geometry problem to
be solved, contact scattered along the route is a tracking problem. This draws
every episode on the map, sized by depth, so the distinction is immediate.

Depth is signed body-to-wall distance from the metrics CSV, which measures
against the *surveyed* map -- never an inflated one -- so a millimetre here is
comparable with the corpus and with every other run.
"""
import argparse
import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
ENV = ROOT / 'environments' / 'experiment'
#: Half the diagonal of a map cell: the true wall surface lies within this of
#: the cell centre clearance is measured against, so contact shallower than
#: this is not evidence of anything.
RESOLUTION_FLOOR = 0.025

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


def route():
    """Both centrelines, to crop the view and to give the contacts context."""
    out = []
    for direction in ('ccw', 'cw'):
        path = ENV / 'centerlines' / f'{direction}_centerline.csv'
        if not path.exists():
            continue
        rows = list(csv.DictReader(open(path, newline='')))
        out.append(np.array([[float(r['x']), float(r['y'])] for r in rows]))
    return out


def load(path):
    rows, head = [], None
    for line in open(path):
        if line.startswith('#'):
            continue
        parts = next(csv.reader([line]))
        if head is None:
            head = parts
            continue
        rows.append(parts)
    index = {n: i for i, n in enumerate(head)}
    def column(name):
        return np.array([float(r[index[name]])
                         if r[index[name]] not in ('', 'nan') else np.nan
                         for r in rows])
    return column


def episodes(paths):
    """One entry per contact: an unbroken run of negative clearance.

    Walls *and* cones. This read `wall_body` alone, and a campaign of 48 runs
    was therefore reported as having one contact of 26 mm while vehicles were
    driving up to 146 mm through cones. The cone data was in every file; the
    plot simply never asked for it.
    """
    out = []
    for path in paths:
        g = load(path)
        for vehicle in ('v1', 'v2'):
            for column, kind in (('wall_body', 'wall'), ('obj_body', 'cone')):
                room = g(f'{vehicle}_{column}')
                if np.all(np.isnan(room)):
                    continue
                index = np.where(room < 0)[0]
                for group in np.split(index,
                                      np.where(np.diff(index) > 1)[0] + 1):
                    if not len(group):
                        continue
                    k = group[int(np.argmin(room[group]))]
                    out.append(dict(x=g(f'{vehicle}_x')[k],
                                    y=g(f'{vehicle}_y')[k],
                                    depth=-float(room[k]), vehicle=vehicle,
                                    kind=kind, run=Path(path).stem))
    return out


def cluster(hits, radius=2.0):
    if not hits:
        return []
    points = np.array([[h['x'], h['y']] for h in hits])
    used = np.zeros(len(points), bool)
    groups = []
    for i in range(len(points)):
        if used[i]:
            continue
        near = (np.hypot(*(points - points[i]).T) < radius) & ~used
        used |= near
        groups.append([hits[j] for j in np.nonzero(near)[0]])
    return sorted(groups, key=lambda g: -max(h['depth'] for h in g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('patterns', nargs='+')
    ap.add_argument('--out', default=str(ROOT / 'docs' / 'contacts.pdf'))
    args = ap.parse_args()
    paths = sorted({p for pattern in args.patterns for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit('no metrics files matched')
    walls = read_map()
    hits = episodes(paths)
    groups = cluster(hits)
    deep = [h for h in hits if h['depth'] > RESOLUTION_FLOOR]

    lines = route()
    span = np.vstack(lines) if lines else np.array([[h['x'], h['y']] for h in hits])
    pad = 4.0
    lo, hi = span.min(axis=0) - pad, span.max(axis=0) + pad
    near = walls[(walls[:, 0] > lo[0]) & (walls[:, 0] < hi[0])
                 & (walls[:, 1] > lo[1]) & (walls[:, 1] < hi[1])]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.scatter(near[:, 0], near[:, 1], s=1.4, c='0.78', marker='s',
               linewidths=0, zorder=1)
    for line in lines:
        ax.plot(line[:, 0], line[:, 1], lw=0.6, ls=':', color='0.62', zorder=2)
    for h in hits:
        real = h['depth'] > RESOLUTION_FLOOR
        ax.scatter(h['x'], h['y'], s=24 + 2600 * h['depth'],
                   facecolor=('#c0392b' if real else 'none'),
                   edgecolor=('#c0392b' if real else '#8d9aa8'),
                   alpha=0.55 if real else 0.9, linewidths=0.9, zorder=3)
    for group in groups:
        worst = max(h['depth'] for h in group)
        if worst <= RESOLUTION_FLOOR:
            continue
        cx = float(np.mean([h['x'] for h in group]))
        cy = float(np.mean([h['y'] for h in group]))
        n_deep = sum(1 for h in group if h['depth'] > RESOLUTION_FLOOR)
        # Label toward the middle of the route, which is the open part of
        # any loop, rather than off the nearest edge of the page.
        mid = span.mean(axis=0)
        step = np.array([mid[0] - cx, mid[1] - cy], float)
        step /= max(np.hypot(*step), 1e-6)
        ax.annotate(f'({cx:.1f}, {cy:.1f})  —  {n_deep} of {len(group)} episodes '
                    f'over {1000 * RESOLUTION_FLOOR:.0f} mm, worst '
                    f'{1000 * worst:.0f} mm',
                    (cx, cy), xytext=(cx + 4.5 * step[0], cy + 4.5 * step[1]),
                    fontsize=7.5, color='#c0392b',
                    ha='left' if step[0] >= 0 else 'right',
                    va='bottom' if step[1] >= 0 else 'top',
                    arrowprops=dict(arrowstyle='-', lw=0.7, color='#c0392b'))
    ax.set_aspect('equal')
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f'Wall contact over {len(paths)} runs: {len(hits)} episodes, '
                 f'{len(deep)} deeper than the map resolution', fontsize=9,
                 loc='left')
    ax.scatter([], [], s=40, facecolor='#c0392b', edgecolor='#c0392b',
               alpha=0.55, label=f'deeper than {1000 * RESOLUTION_FLOOR:.0f} mm '
                                 '(real contact)')
    ax.scatter([], [], s=40, facecolor='none', edgecolor='#8d9aa8',
               label='within map resolution')
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    fig.tight_layout(pad=0.4)
    fig.savefig(args.out, dpi=200)
    print(f'wrote {args.out}')
    print(f'\n{"site":22}{"episodes":>10}{"real":>7}{"worst":>9}{"vehicles":>11}')
    for group in groups:
        cx = float(np.mean([h['x'] for h in group]))
        cy = float(np.mean([h['y'] for h in group]))
        n_deep = sum(1 for h in group if h['depth'] > RESOLUTION_FLOOR)
        print(f'({cx:6.2f},{cy:6.2f}){"":7}{len(group):>7}{n_deep:>7}'
              f'{1000 * max(h["depth"] for h in group):8.0f}mm'
              f'{"/".join(sorted({h["vehicle"] for h in group})):>11}')


if __name__ == '__main__':
    raise SystemExit(main())
