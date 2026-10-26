#!/usr/bin/env python3
"""Wall and cone contact from logged runs, measured exactly and independently.

    python3 src/sbs_planner/tools/exact_contact.py ARM_DIR [ARM_DIR ...]

For every logged pose, the body-to-wall clearance against *every* surveyed
map cell within 0.8 m of base_link (the body reaches at most ~0.62 m from
it), point to rectangle. Deliberately does not call cost.footprint_clearance:
that function is what the planner checks itself with, and until 2026-10-22 it
missed corners, so a contact
count that shares it cannot catch its errors. Host-runnable, no ROS.

Cones are scored the same way against the *placed* geometry: surveyed cells
for the drivers' course, sampled discs for a generated layout, chosen from the
run name as the runners write it. The `vN_obj_body` column cannot stand in for
this: it is measured against what the detectors published, so a cone the
planner was never sent is a cone the log never sees approach.
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src' / 'sbs_planner'))
from sbs_planner.cost import FOOTPRINT_BOX                      # noqa: E402

CONTACT = 0.025          # half a map cell: the reporting floor used throughout


def _pgm(path):
    b = Path(path).read_bytes()
    parts, i = [], 0
    while len(parts) < 4:
        while b[i:i + 1].isspace():
            i += 1
        if b[i:i + 1] == b'#':
            while b[i:i + 1] != b'\n':
                i += 1
            continue
        j = i
        while not b[j:j + 1].isspace():
            j += 1
        parts.append(b[i:j])
        i = j
    i += 1
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(b[i:i + w * h], np.uint8).reshape(h, w)


def wall_points(env=ROOT / 'environments' / 'experiment'):
    meta = dict(line.split(':', 1) for line in
                (env / 'map.yaml').read_text().splitlines() if ':' in line)
    res = float(meta['resolution'])
    ox, oy = [float(v) for v in meta['origin'].strip(' []').split(',')[:2]]
    img = _pgm(env / 'map.pgm')
    r, c = np.nonzero(img < 50)
    return np.column_stack([ox + (c + 0.5) * res,
                            oy + (img.shape[0] - r - 0.5) * res])


def cone_points(objects_file, spacing=0.025):
    """Every placed object as points: surveyed cells, or a filled disc."""
    pts = []
    for o in yaml.safe_load(Path(objects_file).read_text()).get('objects') or []:
        if 'cells' in o:
            pts += [tuple(c) for c in o['cells']]
            continue
        c = o['circle']
        g = np.arange(-c['radius'], c['radius'] + 1e-9, spacing)
        gx, gy = np.meshgrid(g, g)
        inside = gx ** 2 + gy ** 2 <= c['radius'] ** 2
        pts += list(zip(c['x'] + gx[inside], c['y'] + gy[inside]))
    return np.array(pts, float).reshape(-1, 2)


def objects_for(run):
    """The objects file a run was launched with, from its name; None if clear."""
    name = Path(run).name
    m = re.match(r'(layout_\d+)_', name)
    if m:
        return ROOT / 'layouts' / f'{m.group(1)}.yaml'
    if name.startswith('obstacle_'):
        return ROOT / 'environments' / 'experiment' / 'objects.yaml'
    return None


def cone_clearance(run, df, v):
    """Exact body-to-placed-cone clearance per row, inf where there are none."""
    path = objects_for(run)
    if path is None:
        return np.full(len(df), np.inf)
    points = cone_points(path)
    return exact(points, cKDTree(points), df[f'{v}_x'].values,
                 df[f'{v}_y'].values, df[f'{v}_yaw'].values)


def exact(points, tree, x, y, yaw, reach=0.8):
    back, front, half = FOOTPRINT_BOX
    out = np.full(len(x), np.inf)
    for i, idx in enumerate(tree.query_ball_point(np.column_stack([x, y]), reach)):
        if not idx:
            continue
        d = points[idx] - (x[i], y[i])
        cs, sn = np.cos(yaw[i]), np.sin(yaw[i])
        u = d[:, 0] * cs + d[:, 1] * sn
        v = -d[:, 0] * sn + d[:, 1] * cs
        ga = np.maximum(back - u, u - front)
        gc = np.abs(v) - half
        out[i] = np.min(np.hypot(np.maximum(ga, 0), np.maximum(gc, 0))
                        + np.minimum(np.maximum(ga, gc), 0))
    return out


def main():
    points = wall_points()
    tree = cKDTree(points)
    print(f"{'arm':28} {'':5} {'runs':>4} {'contact runs':>12} {'worst':>7} "
          f"{'ticks <-25mm':>12}  worst run")
    for arm in sys.argv[1:]:
        files = sorted(glob.glob(f'{arm.rstrip("/")}/*.csv'))
        score = {k: dict(runs=0, ticks=0, worst=0.0, where='')
                 for k in ('walls', 'cones')}
        for f in files:
            df = pd.read_csv(f, comment='#')
            hit = dict(walls=False, cones=False)
            for v in ('v1', 'v2'):
                for kind, c in (
                        ('walls', exact(points, tree, df[f'{v}_x'].values,
                                        df[f'{v}_y'].values,
                                        df[f'{v}_yaw'].values)),
                        ('cones', cone_clearance(f, df, v))):
                    s = score[kind]
                    hit[kind] |= c.min() < -CONTACT
                    s['ticks'] += int((c < -CONTACT).sum())
                    if c.min() < s['worst']:
                        s['worst'], s['where'] = c.min(), f'{Path(f).stem} {v}'
            for kind in hit:
                score[kind]['runs'] += hit[kind]
        for kind, s in score.items():
            print(f'{arm if kind == "walls" else "":28} {kind:5} '
                  f'{len(files):4d} {s["runs"]:12d} {s["worst"] * 1000:6.0f}mm '
                  f'{s["ticks"]:12d}  {s["where"]}')


if __name__ == '__main__':
    main()
