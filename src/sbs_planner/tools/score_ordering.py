"""Score the pass ordering against the offline ground truth.

    python3 tools/score_ordering.py "runs/before_DIR_*.csv" "runs/after_DIR_*.csv"

DIR is replaced by ccw and cw. Each cone group is checked at the pair's
closest approach: the vehicle whose lane is NOT blocked should be ahead.

Which lane a cone group blocks is a fact about the map and objects.yaml. It
must be read from them, not inferred from how far each lateral target moved
-- that proxy reports "both lanes blocked" at every site, where the geometry
says exactly one is, and scoring with it made a working rule look broken
(3 of 6 "correct") while a change that genuinely broke ccw looked like an
improvement. Every ordering verdict in this project before 2026-09-11 came
from the proxy and should be disregarded.
"""
import csv, glob, os, sys
from pathlib import Path

import numpy as np
import yaml

# Located from this file, not from an absolute path: the workspace directory
# gets renamed and moved, and a hardcoded one silently breaks the tool without
# breaking anything that would fail a test.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src' / 'sbs_planner'))
from sbs_planner.reference import Reference

BASE = str(ROOT / 'environments' / 'experiment') + '/'
HALF, CLEAR = 0.275, 0.10
GROUPS = (('A', ['cone_1', 'cone_2']), ('B', ['cone_3', 'cone_4']),
          ('C', ['cone_5', 'cone_6', 'cone_7']), ('D', ['cone_8', 'cone_9']))
cones = {o['name']: np.array(o['cells'])
         for o in yaml.safe_load(open(BASE + 'objects.yaml'))['objects']}

def truth(direction):
    ref = Reference.load(BASE + f'centerlines/{direction}_centerline.csv')
    out = []
    for label, names in GROUPS:
        pts = np.vstack([cones[n] for n in names])
        mid = pts.mean(0)
        cross = np.asarray(ref.signed_cross_track(pts[:, 0:1], pts[:, 1:2])).ravel()
        o = ref.lane_offset_at(mid, 'outer')
        i = ref.lane_offset_at(mid, 'inner')
        bo = bool(np.any(np.abs(cross - o) < HALF + CLEAR))
        bi = bool(np.any(np.abs(cross - i) < HALF + CLEAR))
        out.append((label, mid, bo, bi))
    return out

def load(path):
    rows, head = [], None
    for line in open(path):
        if line.startswith('#'):
            continue
        p = next(csv.reader([line]))
        if head is None:
            head = p; continue
        rows.append(p)
    c = {n: i for i, n in enumerate(head)}
    return lambda n: np.array([float(r[c[n]]) if r[c[n]] not in ('', 'nan')
                               else np.nan for r in rows])

def verdicts(path, direction, sites):
    g = load(path)
    # v1 is outer, v2 inner; v1_along > 0 means v2 is ahead of v1
    ok = bad = 0
    detail = []
    for label, mid, bo, bi in sites:
        d = np.hypot((g('v1_x') + g('v2_x')) / 2 - mid[0],
                     (g('v1_y') + g('v2_y')) / 2 - mid[1])
        near = d < 2.5
        if near.sum() < 5:
            continue
        k = int(np.argmin(np.where(near, d, np.inf)))
        along = g('v1_along')[k]          # + => inner (v2) ahead
        if bo and not bi:                 # outer blocked -> inner should lead
            good = along > -0.2
        elif bi and not bo:               # inner blocked -> outer should lead
            good = along < 0.2
        else:
            continue                      # both or neither: no preference
        ok += good; bad += not good
        detail.append(f'{label}:{"ok" if good else "WRONG"}({along:+.2f})')
    return ok, bad, detail

for direction in ('ccw', 'cw'):
    sites = truth(direction)
    print(f'\n{direction}: ' + '  '.join(
        f'{l}={"outer" if bo else "inner"}-blocked' for l, _, bo, bi in sites))
    for tag, pattern in (('before', sys.argv[1]), ('after', sys.argv[2])):
        files = sorted(glob.glob(pattern.replace('DIR', direction)))
        T = B = 0
        for f in files:
            if not os.path.exists(f):
                continue
            a, b, det = verdicts(f, direction, sites)
            T += a; B += b
        if T + B:
            print(f'   {tag:7} {T} correct, {B} wrong  ({100*T/(T+B):.0f}%)')
