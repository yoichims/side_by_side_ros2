#!/usr/bin/env python3
"""Does the planner have a forward option at the poses where it froze?

    python3 src/sbs_planner/tools/stall_harness.py

A stall is rare in a full run -- about one cw run in four, 170 s each -- so
judging a fix by running the simulator means reading a difference of one or
two episodes out of six, which at that rate is indistinguishable from noise.
Several changes were accepted or rejected that way and had to be revisited.

This puts the vehicle straight at a recorded stall pose instead, with the
velocity, heading and obstacle layout it actually had, and asks the one
question the stall is made of: **is any forward candidate admissible?** The
answer is deterministic and takes a second.

It calls the shipped `admissible` and `generate`, not a copy of their logic,
so a harness pass means the planner itself has an option -- not that a
reimplementation does. What it does not do is prove a change is safe: the
poses here are four points, and a rule that frees them can still wedge a
vehicle somewhere else. Full runs remain the check on that; this is the check
on the mechanism.
"""

import math
import sys
from pathlib import Path

import numpy as np
from dataclasses import replace
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src' / 'sbs_planner'))

from sbs_planner.cost import ContextModel, admissible            # noqa: E402
from sbs_planner.rollout import (RolloutConfig, VehicleModel,    # noqa: E402
                                 generate)
from sbs_planner.world import inflate                            # noqa: E402
from sbs_planner.cost import best_recovery, footprint_clearance  # noqa: E402

ENV = ROOT / 'environments' / 'experiment'
FOOTPRINT_BOX = (-0.35, 0.55, 0.275)
#: Must match `wall_inflate` in the launch file; see `_walls`.
WALL_INFLATE = 0.10
#: Must match `escape_speed` in the launch file.
ESCAPE_SPEED = 0.25
#: Must match `escape_tolerance`. It ships at 0.0, and the default in
#: `best_recovery` is not it -- at site C the 0.05 default moves the floor
#: from -0.021 to -0.071 m and admits candidates the planner refuses.
ESCAPE_TOLERANCE = 0.0

#: Poses where a vehicle froze in a closed-loop run, with the run it came
#: from and how long it stayed there.
#:
#: Sites A and B are cw and in open space -- the wall is 0.56-1.67 m away --
#: so neither is the wall wedge. Both now pass, which is why site C was
#: added: it is the wedge, and it is what stalls today.
#:
#: Site C is the recurring cw site near (20.7, 14.5), measured over 24 runs
#: of the baseline planner. Always v2, always the inner lane, body 0.072-0.113 m from a
#: wall, and the pair correctly formed at 1.15-1.20 m separation with no cone
#: involved. The planner's own diagnostic says the same thing at every stuck
#: tick: `0 clear walls, 60 clear objects`. Three of the four held for the
#: whole remainder of the run.
STALLS = [
    ('site A, rev_2',   26.379, -3.766, -2.216,  0.000, 92),
    ('site A, fabcw_1', 26.396, -3.810, -2.295,  0.000, 87),
    ('site B, rev_3',   17.981, 16.878, -0.496,  0.000, 40),
    ('site B, dg_2',    18.038, 17.008, -0.377,  0.037, 138),
    # site C: the wall wedge. Wall clearance in the comment.
    ('site C, l02_r1',  20.792, 14.362, -0.750,  0.027, 132),   # 0.088 m
    ('site C, l02_r3',  20.775, 14.380, -0.829,  0.007, 132),   # 0.113 m
    ('site C, l07_r1',  20.276, 14.827, -0.809,  0.018, 131),   # 0.076 m
    ('site C, l07_r3',  20.783, 14.364, -0.811,  0.047,  27),   # 0.072 m
]

#: Where the partner was at the same instant, for the site C poses.
#:
#: Recovery clears the partner's body as well as the map, so a recovery
#: check without it is not the decision the planner made. It is not a
#: detail: at site C the harness picked a forward move of +0.48 m/s with no
#: partner present, while the run -- with the partner 1.5 m away -- issued
#: 0.00 to 0.03 m/s rotations for 133 s.
PARTNERS = {
    'site C, l02_r1': (22.250, 14.043, -1.351),   # sep 1.49 m
    'site C, l02_r3': (22.272, 14.111, -1.358),   # sep 1.52 m
    'site C, l07_r1': (21.366, 13.843, -1.554),   # sep 1.47 m
    'site C, l07_r3': (21.075, 12.040, -1.881),   # sep 2.34 m
}


def _body(state):
    """A vehicle footprint as points, as `_partner_body` builds it."""
    back, front, half = FOOTPRINT_BOX
    uu, vv = np.meshgrid(np.linspace(back, front, 5),
                         np.linspace(-half, half, 3), indexing='ij')
    cos, sin = math.cos(state[2]), math.sin(state[2])
    return cKDTree(np.column_stack([
        (state[0] + uu * cos - vv * sin).ravel(),
        (state[1] + uu * sin + vv * cos).ravel()]))


def _walls():
    info = yaml.safe_load((ENV / 'map.yaml').read_text())
    res, (ox, oy) = info['resolution'], info['origin'][:2]
    with open(ENV / 'map.pgm', 'rb') as handle:
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
    points = np.column_stack([ox + (cols + 0.5) * res,
                              oy + ((height - 1 - rows) + 0.5) * res])
    # Thickened exactly as the planner thickens it. Without this the harness
    # measures a different map from the one the planner plans against, and
    # the whole claim of the file -- that a pass means the *planner* has an
    # option -- does not hold. It mattered: at site C the body is 0.072-0.113
    # m from a surveyed cell, which the planner sees as -0.028 to +0.013 m
    # once inflated, and the uninflated harness reported 20-30 free
    # candidates at poses the planner had logged `0 clear walls` for.
    return cKDTree(inflate(points, WALL_INFLATE, res))


def _objects(inflate=0.05, spacing=0.05):
    """The object cloud the planner actually sees, not the file.

    `publish_world_objects` dilates the `objects` section by `inflate` before
    publishing -- the survey holds only the LiDAR-visible surface of a cone,
    so an undilated cone is a thin hollow arc -- and leaves `structure`
    alone. Reading objects.yaml directly instead put the harness up to 0.073 m
    out against the clearances the runs recorded, which is most of the 0.10 m
    margin being tested, so the loading is shared with the publisher rather
    than reimplemented here.
    """
    sys.path.insert(0, str(ROOT / 'src' / 'sbs_bringup' / 'scripts'))
    from publish_world_objects import inflate_cells, load_objects
    groups, _, spacing = load_objects(ENV / 'objects.yaml', spacing)
    points = []
    for _, cells in groups:
        points.extend(inflate_cells(cells, inflate, spacing))
    structure, _, _ = load_objects(ENV / 'objects.yaml', spacing,
                                   section='structure')
    for _, cells in structure:
        points.extend(cells)
    return cKDTree(np.asarray(points, float)) if points else None


def check(name, x, y, yaw, v, held, walls, objects, cfg, model,
          v_min_fraction=0.30, object_clearance=0.10, clearance_steps=5):
    state = np.array([x, y, yaw, v, 0.0])
    commands, rollouts = generate(state, cfg)
    report = {}
    ok = admissible(rollouts, walls, v_min_fraction
                    * float(model.preferred_speed(0.0)),
                    0.0, clearance_steps, objects, object_clearance,
                    object_steps=cfg.n_steps, report=report)
    moves = commands[:, 0] >= 0.05
    free = bool((ok & moves).any())
    print(f'  {name:18} v={v:4.2f}  held {held:3d}s   '
          f'rule={report.get("rule"):22} '
          f'forward admissible: {int((ok & moves).sum()):3d}/{int(moves.sum()):3d}'
          f'   {"FREE" if free else "STUCK"}')
    return free


def _room(rollouts, walls, objects, wall_steps, object_steps):
    """Body clearance to the nearest of wall or object, as the node computes it."""
    room = footprint_clearance(rollouts, walls, wall_steps, FOOTPRINT_BOX)
    if objects is not None:
        room = np.minimum(room, footprint_clearance(
            rollouts, objects, object_steps, FOOTPRINT_BOX))
    return room


def main():
    walls, objects = _walls(), _objects()
    model = VehicleModel()
    cfg = RolloutConfig(horizon=3.0, dt=0.1, segment_split=1.5, v_span=0.45,
                        w_span=0.60, n_v=7, n_w=5, n_v2=2, n_w2=2, model=model)
    ctx = ContextModel()
    print(f'\nrecorded stall poses (preferred speed {ctx.preferred_speed(0.0):.3f} '
          f'm/s, floor {0.30 * ctx.preferred_speed(0.0):.3f} m/s)\n')
    free = [check(n, x, y, yaw, v, held, walls, objects, cfg, ctx)
            for n, x, y, yaw, v, held in STALLS]
    print(f'\n  {sum(free)}/{len(free)} poses have a forward option\n')
    # Speed sweep at one site: the trap is a function of speed, not place.
    name, x, y, yaw, _, held = STALLS[0]
    print('same pose, varying the speed the vehicle arrives with:\n')
    for v in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        check(f'{name} @{v:.2f}', x, y, yaw, v, held, walls, objects, cfg, ctx)
    print()

    # Can it get out? A pose with no forward option is only a stall if
    # recovery also fails, so the harness asks both questions. This calls the
    # shipped `best_recovery`, the same function the node calls, with the same
    # trees -- `min_move` is the only thing varied, so the two columns are the
    # old rule and the new one and nothing else.
    print('recovery at each pose  (escape_move 0.00 = the old rule)\n')
    # Built the same way the node builds it: reverse is a property of the
    # model, not of the grid.
    escape_cfg = replace(cfg, model=replace(cfg.model, v_min=-ESCAPE_SPEED))
    for name, x, y, yaw, v, held in STALLS:
        state = (x, y, yaw, v, 0.0)
        pose = {'x': np.array([[x]]), 'y': np.array([[y]]),
                'yaw': np.array([[yaw]])}
        mate = PARTNERS.get(name)
        body = _body(mate) if mate else None

        def path(r, body=body):
            room = _room(r, walls, objects, 5, 30)
            if body is not None:
                room = np.minimum(room, footprint_clearance(
                    r, body, 5, FOOTPRINT_BOX))
            return room

        def endc(t, body=body):
            end = _room(t, walls, objects, None, None)
            if body is not None:
                end = np.minimum(end, footprint_clearance(
                    t, body, None, FOOTPRINT_BOX))
            return end

        here = float(endc(pose)[0])
        out = []
        for mv in (0.00, 0.05):
            got = best_recovery(state, (cfg, escape_cfg), generate, path,
                                endc, here, min_speed=mv,
                                tolerance=ESCAPE_TOLERANCE)
            out.append('none' if got is None
                       else f'v={got[0]:+.2f} w={got[1]:+.2f}')
        print(f'  {name:18s} standing {here:+.3f} m  '
              f'{"partner" if mate else "alone  "}  '
              f'old: {out[0]:18s} new: {out[1]}')
    print()
    return 0 if all(free) else 1


if __name__ == '__main__':
    raise SystemExit(main())
