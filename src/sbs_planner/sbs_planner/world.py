"""What counts as fixed geometry, shared by every node that needs to know.

Three nodes answer "is this a wall?" and they must answer it identically: the
planner vetoes lateral targets and caps its speed on walls, the detector calls
anything that is *not* a wall an object, and the metrics node logs clearances
that are supposed to be the planner's own. When those three disagree the
symptoms are silent and confusing -- a detector that reports the aisle as an
obstacle to swerve around, or a log whose clearance is not the one the
constraint used. A drifted copy of exactly this logic has already cost a day
of measurements, so it lives here once.

Fixed geometry is the surveyed map plus a `structure` section from
objects.yaml. The distinction that matters downstream is not "in the survey or
not" but "negotiate it or drive between it": an object *causes* a lateral
target to move, so aisle walls entered as objects would have the pair swerving
around the aisle it is meant to be driving down. Structure is therefore
published separately, latched, and merged in here.
"""

import math

import numpy as np
from scipy.spatial import cKDTree


def grid_points(msg, threshold: int = 50):
    """Occupied cell centres of an OccupancyGrid, in the map frame."""
    data = np.asarray(msg.data, dtype=np.int16).reshape(
        msg.info.height, msg.info.width)
    rows, cols = np.nonzero(data > threshold)
    if not len(rows):
        return np.zeros((0, 2))
    res, origin = msg.info.resolution, msg.info.origin.position
    return np.column_stack([origin.x + (cols + 0.5) * res,
                            origin.y + (rows + 0.5) * res])


def unknown_points(msg, bounds=None):
    """Cell centres an OccupancyGrid has never observed, in the map frame.

    Not obstacles, and never treated as such by the clearance constraints --
    but not free space either, and the difference matters wherever the
    question is "where can a body be". Free space is otherwise inferred from
    the *absence* of an occupied cell, which makes unexplored map read as
    perfectly clear: 95.9% of this 4000x4000 map is unknown, and 30.2% of the
    3 m around one site where a vehicle pinned. A band measured through it is
    an invitation to drive somewhere nothing has ever looked.

    `bounds` is (x_min, y_min, x_max, y_max) and is not an optimisation -- it
    is the difference between this being usable and not. Unknown cells are the
    overwhelming majority of a mostly-empty grid: 15.3 million of the 16
    million here, against 19 thousand occupied. A k-d tree over all of them
    takes **3.5 s** to build where the occupied tree takes 3 ms, and built in
    a map callback that blocks the node for three and a half seconds -- during
    which the vehicles keep driving on the last command. Measured, every ccw
    run then left the route in its first four seconds, holding the opening
    corner's yaw rate straight into a wall, up to 274 mm deep.

    Only cells the route passes near can matter, because the bands are
    sampled within a couple of metres of the centreline, so the caller passes
    the route's own extent and the rest are never touched.
    """
    data = np.asarray(msg.data, dtype=np.int16).reshape(
        msg.info.height, msg.info.width)
    res, origin = msg.info.resolution, msg.info.origin.position
    if bounds is not None:
        x_min, y_min, x_max, y_max = bounds
        c0 = max(0, int((x_min - origin.x) / res))
        c1 = min(msg.info.width, int((x_max - origin.x) / res) + 2)
        r0 = max(0, int((y_min - origin.y) / res))
        r1 = min(msg.info.height, int((y_max - origin.y) / res) + 2)
        if c1 <= c0 or r1 <= r0:
            return np.zeros((0, 2))
        rows, cols = np.nonzero(data[r0:r1, c0:c1] < 0)
        rows, cols = rows + r0, cols + c0
    else:
        rows, cols = np.nonzero(data < 0)
    if not len(rows):
        return np.zeros((0, 2))
    return np.column_stack([origin.x + (cols + 0.5) * res,
                            origin.y + (rows + 0.5) * res])


def cloud_points(msg):
    """Points of a sensor_msgs/PointCloud as an (n, 2) array."""
    if not msg.points:
        return np.zeros((0, 2))
    return np.array([[p.x, p.y] for p in msg.points], float)


def inflate(points, radius: float, spacing: float = 0.05):
    """Thicken a point set by `radius`, on the same grid.

    The same operation `publish_world_objects` applies to the cones, and for
    a related reason. There it compensates for the survey seeing only the
    LiDAR-visible surface of an object; here it compensates for the map being
    a grid: clearance is measured to cell *centres*, so the true wall surface
    lies up to half a diagonal away and a body flush against the measured
    points is already into the wall.

    Applied by the planner alone. The metrics node and the object detector
    must keep the surveyed geometry: inflating the first would report contact
    before it happened and make every clearance figure incomparable with the
    corpus, and inflating the second would let the detector subtract more than
    the map contains and erase real objects standing near a wall.
    """
    points = np.asarray(points, float)
    if radius <= 0.0 or not len(points):
        return points
    steps = int(round(radius / spacing))
    if steps < 1:
        return points
    offsets = np.array([(i * spacing, j * spacing)
                        for i in range(-steps, steps + 1)
                        for j in range(-steps, steps + 1)
                        if math.hypot(i, j) * spacing <= radius + 1e-9])
    grown = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    return np.unique(np.round(grown / spacing).astype(np.int64),
                     axis=0) * spacing


def wall_tree(*groups):
    """One k-d tree over every group of fixed geometry given, or None."""
    stacked = [np.asarray(g) for g in groups if g is not None and len(g)]
    if not stacked:
        return None
    return cKDTree(np.vstack(stacked))
