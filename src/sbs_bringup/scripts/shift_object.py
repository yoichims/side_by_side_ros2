#!/usr/bin/env python3
"""Move a named object in objects.yaml across the route.

    shift_object.py <objects.yaml> <centerline.csv> <shift_m> <name> [name...]

`shift_m` is signed cross-track metres, positive to the left of the direction
the centreline runs. Cells are moved perpendicular to the local centreline
tangent, so a group keeps its shape on a curve.

Written for the middle obstacle, which the survey placed 0.247 m off centre:
it left 0.01 m beside the outer lane against 0.48 m beside the inner one, so
the outer vehicle was wholly blocked while its partner had room, and the pair
filed instead of splitting around it. Editing the surveyed map to fix that
would destroy the distinction between what was measured and what was
invented; objects.yaml exists precisely for things that are not in the
survey, so the object moves and the map stays as surveyed.

Comments carrying each object's centroid are rewritten to match, and the file
is otherwise reproduced line for line -- yaml.dump would discard the comments
that say where these cells came from.
"""
import csv
import re
import sys

import numpy as np


def frame(path):
    rows = list(csv.DictReader(open(path, newline='')))
    x = np.array([float(r['x']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    tx, ty = np.gradient(x), np.gradient(y)
    n = np.hypot(tx, ty)
    return x, y, tx / n, ty / n


def main():
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    target, reference, shift = sys.argv[1], sys.argv[2], float(sys.argv[3])
    names = set(sys.argv[4:])
    cx, cy, tx, ty = frame(reference)

    def moved(x, y):
        k = int(np.argmin((cx - x) ** 2 + (cy - y) ** 2))
        # +shift is to the left of travel: the left normal is (-ty, tx).
        return x - shift * ty[k], y + shift * tx[k]

    lines = open(target).read().splitlines()
    out, current, touched = [], None, {}
    cell = re.compile(r'^(\s*- )\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]\s*$')
    for line in lines:
        name = re.match(r'^\s*- name:\s*(\S+)\s*$', line)
        if name:
            current = name.group(1)
            out.append(line)
            continue
        m = cell.match(line)
        if m and current in names:
            x, y = moved(float(m.group(2)), float(m.group(3)))
            out.append(f'{m.group(1)}[{x:.3f}, {y:.3f}]')
            touched.setdefault(current, []).append((x, y))
            continue
        out.append(line)

    missing = names - set(touched)
    if missing:
        raise SystemExit(f'no cells found for: {", ".join(sorted(missing))}')

    # Refresh the centroid comments so they do not describe the old position.
    final, current = [], None
    for line in out:
        name = re.match(r'^\s*- name:\s*(\S+)\s*$', line)
        if name:
            current = name.group(1)
        note = re.match(r'^(\s*)# (\d+) cells, centroid \([^)]*\), '
                        r'radius ([\d.]+) m\s*$', line)
        if note and current in touched:
            pts = np.array(touched[current])
            line = ('%s# %s cells, centroid (%.2f, %.2f), radius %s m'
                    % (note.group(1), note.group(2), pts[:, 0].mean(),
                       pts[:, 1].mean(), note.group(3)))
        final.append(line)

    open(target, 'w').write('\n'.join(final) + '\n')
    for name, pts in sorted(touched.items()):
        pts = np.array(pts)
        print(f'{name}: {len(pts)} cells -> centroid '
              f'({pts[:, 0].mean():.2f}, {pts[:, 1].mean():.2f})')
    print(f'shifted {shift:+.3f} m across {reference}')


if __name__ == '__main__':
    main()
