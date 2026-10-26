"""The pair's reference path and the per-vehicle lane offset from it.

Loaded from the CSV stage 12 produces: the centreline is refined onto the
centre of every recorded pair midpoint, and `separation` is the pair width
actually held at each point (1.22-1.77 m), so the lanes follow the recorded
width rather than one hand-picked constant.

Each lane is refined onto the positions that vehicle actually occupied, not
produced by offsetting the centreline along its normal. The rigid offset put
27.5% of one lane inside a wall, because the pair does not stay perpendicular
to the path through a corner -- the inner vehicle cuts and the outer swings
wide. Data-derived lanes clear the map everywhere (minimum 0.28 m).

The lateral gap between the lanes is ~1.0 m while the recorded instantaneous
separation is ~1.43 m. Both are right: instantaneous separation includes the
longitudinal stagger the pair actually held (along-track error median 0.59 rad).

Which lane a vehicle takes is a *convention*, not shared information. An agent
assumes its partner is on the other lane of the same centreline -- like knowing
which side of a corridor people walk on. Nothing is transmitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

#: The centreline file carries the path attributes; each lane is its own file,
#: because the three curves are resampled independently and parallel columns
#: would imply an index correspondence that does not exist.
CENTRE_COLUMNS = ["x", "y", "yaw", "curvature", "separation"]

#: Vehicle 1 takes the outer lane: stage 1 orders the pair by path length, and
#: on a closed loop the longer path is the outside one.
SIDES = ("outer", "inner")


def _lane_curvature(points: np.ndarray, window: int = 15) -> np.ndarray:
    """Curvature along a closed lane, from heading change per unit arc length.

    Same construction as the centreline's own curvature (stage 12), so the two
    are comparable. Smoothed over a wrapping window because the lanes are built
    from recorded positions.
    """
    step = np.diff(np.vstack([points, points[:1]]), axis=0)
    yaw = np.unwrap(np.arctan2(step[:, 1], step[:, 0]))
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*step.T))])[:-1]
    ds = np.gradient(arc)
    curvature = np.gradient(yaw) / np.maximum(np.abs(ds), 1e-9)
    if window > 1:
        kernel = np.ones(window) / window
        padded = np.concatenate([curvature[-window:], curvature,
                                 curvature[:window]])
        curvature = np.convolve(padded, kernel, mode="same")[window:-window]
    return curvature


@dataclass
class Reference:
    centre: np.ndarray       # (n, 2)
    yaw: np.ndarray
    curvature: np.ndarray
    separation: np.ndarray
    lanes: dict              # {"left": (n,2), "right": (n,2)}
    #: Per-sample lane offsets derived from the corridor, aligned with
    #: `centre`, as {"outer": (n,), "inner": (n,)}. `None` falls back to each
    #: lane's median offset, which is what the planner used before the
    #: profile existed.
    offsets: dict | None = None

    def __post_init__(self):
        self._centre_tree = cKDTree(self.centre)
        self._lane_trees = {k: cKDTree(v) for k, v in self.lanes.items()}
        # Each lane has its own curvature, and on a corner they differ a lot:
        # the outer lane is the longer arc. Holding formation through a bend
        # means the two vehicles turn at the same rate but travel at different
        # speeds, so each one's speed target has to come from its own lane.
        # Scoring both against the midpoint curvature told the outer vehicle to
        # go too slow and the inner one too fast, and the formation broke.
        self._lane_curvature = {k: _lane_curvature(v)
                                for k, v in self.lanes.items()}
        # Which side of the centreline each lane lies on, +1 left. Measured
        # once over the whole route rather than looked up per point: on both
        # recorded directions every point of a lane falls on the same side
        # (1.000 of samples), so this is a property of the lane, and reading
        # it per point would only reintroduce noise into a constant.
        self._lane_side = {}
        self._lane_offset = {}
        for name, points in self.lanes.items():
            index = self._centre_tree.query(points)[1]
            cos, sin = np.cos(self.yaw[index]), np.sin(self.yaw[index])
            offset = (-(points[:, 0] - self.centre[index, 0]) * sin
                      + (points[:, 1] - self.centre[index, 1]) * cos)
            self._lane_side[name] = float(np.sign(np.median(offset)))
            # The lane's own signed distance from the centreline, which is the
            # formation the drivers actually held laterally. The recorded
            # *instantaneous* separation is wider (1.43 m against a 1.05 m lane
            # gap) because it includes the longitudinal stagger; scoring a
            # lateral target against that number is what pushed both vehicles
            # off their lanes to manufacture the difference.
            self._lane_offset[name] = float(np.median(offset))
        # Tangent direction of each lane, for scoring heading against the
        # route rather than only against the partner. Every formation term is
        # invariant to rigid rotation of the pair -- measured: identically
        # zero at spin rates from 0 to 0.8 rad/s -- so without an absolute
        # heading reference a co-rotating pair satisfies the whole social cost
        # while going nowhere.
        self._lane_tangent = {}
        for name, points in self.lanes.items():
            step = np.diff(np.vstack([points, points[:1]]), axis=0)
            self._lane_tangent[name] = np.arctan2(step[:, 1], step[:, 0])
        # Cumulative arc length per lane, for measuring progress *along* the
        # route. Distance to the lane is only the lateral half of path
        # following; without the longitudinal half a vehicle satisfies the cost
        # by driving in a circle that hugs the lane, which is precisely what
        # happened.
        self._lane_s = {}
        self._lane_total = {}
        for name, points in self.lanes.items():
            step = np.hypot(*np.diff(points, axis=0).T)
            arc = np.concatenate([[0.0], np.cumsum(step)])
            self._lane_s[name] = arc
            # Closed loop: the last point returns near the first, so the total
            # includes the closing step.
            self._lane_total[name] = float(
                arc[-1] + np.hypot(*(points[0] - points[-1])))
        # Cumulative arc length along the *centreline*, for asking a question
        # about the corridor rather than about either lane. The pair has to
        # agree on where it is looking, and the two vehicles sit on different
        # lanes of different lengths.
        centre_step = np.hypot(*np.diff(self.centre, axis=0).T)
        self._centre_s = np.concatenate([[0.0], np.cumsum(centre_step)])
        self._centre_total = float(
            self._centre_s[-1] + np.hypot(*(self.centre[0] - self.centre[-1])))

    @classmethod
    def load(cls, path: str | Path) -> "Reference":
        """Load a centreline and the two lane files beside it.

        `<dir>_centerline.csv` -> `<dir>_lane_outer.csv`, `<dir>_lane_inner.csv`
        """
        path = Path(path)
        raw = np.loadtxt(path, delimiter=",", skiprows=1)
        col = {name: raw[:, i] for i, name in enumerate(CENTRE_COLUMNS)}
        stem = path.name.replace("_centerline.csv", "")
        lanes = {}
        for side in SIDES:
            lane_path = path.with_name(f"{stem}_lane_{side}.csv")
            if not lane_path.exists():
                raise FileNotFoundError(f"missing lane file: {lane_path}")
            lanes[side] = np.loadtxt(lane_path, delimiter=",", skiprows=1)

        # Offsets derived from the corridor, if `derive_lane_offsets.py` has
        # been run for this route. One row per centreline sample, in the same
        # order, so the lookup is the nearest centreline index. Optional on
        # purpose: a route without a profile still loads and behaves as it
        # did.
        offsets = None
        offset_path = path.with_name(f"{stem}_offsets.csv")
        if offset_path.exists():
            table = np.loadtxt(offset_path, delimiter=",", skiprows=1)
            if len(table) != len(col["x"]):
                raise ValueError(
                    f"{offset_path.name} has {len(table)} rows against "
                    f"{len(col['x'])} centreline samples; regenerate it")
            offsets = {"outer": table[:, 1], "inner": table[:, 2]}
            if table.shape[1] > 3:
                offsets["side_by_side"] = table[:, 3]

        return cls(centre=np.column_stack([col["x"], col["y"]]),
                   yaw=col["yaw"], curvature=col["curvature"],
                   separation=col["separation"], lanes=lanes,
                   offsets=offsets)

    def curvature_at(self, point) -> float:
        """Curvature where the pair's midpoint sits on the centreline."""
        query = np.atleast_2d(np.asarray(point, float))
        index = int(np.atleast_1d(self._centre_tree.query(query)[1])[0])
        return float(abs(self.curvature[index]))

    def signed_curvature_at(self, point) -> float:
        """Curvature *with its sign* where the pair's midpoint sits.

        `curvature_at` returns the magnitude, which is what the speed model
        wants. Which way the route bends is what says which lane is on the
        outside, and that needs the sign.
        """
        query = np.atleast_2d(np.asarray(point, float))
        index = int(np.atleast_1d(self._centre_tree.query(query)[1])[0])
        return float(self.curvature[index])

    def lane_curvature_at(self, point, side: str) -> float:
        """Curvature of one lane, where that vehicle sits on it."""
        query = np.atleast_2d(np.asarray(point, float))
        index = int(np.atleast_1d(self._lane_trees[side].query(query)[1])[0])
        return float(abs(self._lane_curvature[side][index]))

    def outer_side_at(self, point) -> str:
        """Which lane lies on the outside of the curve here.

        The outside of a bend is the longer arc, so it is the lane with the
        smaller local curvature. Read from the lane geometry rather than from
        the sign of the centreline curvature, so it stays correct if the route
        ever bends both ways.
        """
        query = np.atleast_2d(np.asarray(point, float))
        return min(SIDES, key=lambda side: abs(
            self._lane_curvature[side][
                int(np.atleast_1d(self._lane_trees[side].query(query)[1])[0])]))

    def side_sign_at(self, point, side: str, previous: float = 0.0,
                     low: float = 0.10, high: float = 0.25) -> float:
        """+1 if `side` is outside the curve here, -1 if inside, 0 if neither.

        Derived from the signed centreline curvature and the lane's constant
        side, not by comparing the two lanes' smoothed curvatures. The
        comparison agreed with this 96% of the time it mattered, so the change
        is in stability, not in meaning.

        A straight corridor has no outside, and this used to answer as though
        it did. Measured on two logged runs, the old form flipped 185 times in
        159 s -- once a second -- because half the route sits below |kappa| =
        0.15 where the sign of the curvature is not meaningfully determined.
        Each flip swaps which vehicle is told to run faster, by up to 0.24 m/s
        of target, and the vehicles follow it: the achieved speed difference
        correlates +0.65 with the target split. That is a ~1 Hz dither in the
        speed command, and it is the single thing that most makes the motion
        look wrong.

        Hence a Schmitt trigger rather than one threshold. The split engages
        only above `high`, releases below `low`, and holds `previous` in
        between, so a corner is entered and left once instead of chattering
        across a boundary. On the same two runs this gives 19 transitions
        rather than 185 -- about one per real corner.

        `previous` is the caller's last answer. The default of 0.0 makes a
        stateless caller fall back to "no split in the band", which is the
        safe half of the hysteresis rather than the chattering half.
        """
        kappa = self.signed_curvature_at(point)
        if abs(kappa) >= high:
            return -float(np.sign(kappa)) * self._lane_side[side]
        if abs(kappa) < low:
            return 0.0
        return float(previous)

    def lane_distance(self, xs, ys, side: str) -> np.ndarray:
        """Mean distance of each rollout to a lane, over the horizon."""
        flat = np.column_stack([np.asarray(xs).ravel(), np.asarray(ys).ravel()])
        return self._lane_trees[side].query(flat)[0].reshape(
            np.asarray(xs).shape).mean(axis=1)

    def progress(self, xs, ys, side: str) -> np.ndarray:
        """Signed arc length advanced along a lane, per rollout.

        Wrapped to (-total/2, total/2] so a lap boundary reads as a small
        forward step rather than an 86 m jump backwards. A 3 s rollout covers
        at most ~3.9 m against a loop of ~86 m, so the wrap is unambiguous.
        """
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        tree = self._lane_trees[side]
        arc = self._lane_s[side]
        total = self._lane_total[side]

        start = arc[tree.query(np.column_stack([xs[:, 0], ys[:, 0]]))[1]]
        end = arc[tree.query(np.column_stack([xs[:, -1], ys[:, -1]]))[1]]
        raw = end - start
        return raw - total * np.round(raw / total)

    def lane_ahead(self, point, side: str, distance: float,
                   step: float = 0.10):
        """Poses along a lane, from the nearest point forward.

        Returns (xs, ys, yaws, reach) where `reach` is the arc distance of
        each pose from the query point. The speed cap needs to know how far
        ahead the corridor stops fitting, which is a question about the route
        rather than about any one rollout -- asking the rollouts instead
        cannot answer it, because a rollout only reaches as far as the speed
        being decided carries it.
        """
        tree = self._lane_trees[side]
        arc = self._lane_s[side]
        total = self._lane_total[side]
        lane = self.lanes[side]
        tangent = self._lane_tangent[side]

        query = np.atleast_2d(np.asarray(point, float))
        start = int(np.atleast_1d(tree.query(query)[1])[0])
        reach = np.arange(0.0, float(distance) + 1e-9, step)
        # Wrapped, so a lookahead that crosses the lap boundary keeps going
        # round the loop instead of running off the end of the array.
        index = np.searchsorted(arc, (arc[start] + reach) % total)
        index = np.clip(index, 0, len(arc) - 1)
        return (lane[index, 0], lane[index, 1], tangent[index], reach)

    def centre_ahead(self, point, distance: float, step: float = 0.10):
        """Poses along the centreline, from the nearest point forward.

        `lane_ahead` answers "where does my lane go"; this answers "where
        does the corridor go", which is a question the pair has to agree on.
        Asking it of either lane would give the two vehicles different sample
        points -- the lanes are different curves of different lengths -- and
        the joint matrices would stop being transposes of each other.

        Returns (xs, ys, yaws, reach), `reach` being the arc distance of each
        pose from the query point.
        """
        query = np.atleast_2d(np.asarray(point, float))
        start = int(np.atleast_1d(self._centre_tree.query(query)[1])[0])
        reach = np.arange(0.0, float(distance) + 1e-9, step)
        index = np.searchsorted(self._centre_s,
                                (self._centre_s[start] + reach)
                                % self._centre_total)
        index = np.clip(index, 0, len(self._centre_s) - 1)
        return (self.centre[index, 0], self.centre[index, 1],
                self.yaw[index], reach)

    def lane_offset(self, side: str) -> float:
        """Median signed cross-track offset of a lane, +left.

        A summary of the lane, not the lane. Every point of a lane falls on
        the same *side* of the centreline (1.000 of samples on both recorded
        directions), which is what makes the median meaningful -- but the
        magnitude varies, from -0.21 to -0.73 on the ccw outer lane, so
        driving at the median is not driving on the lane. Use
        `lane_offset_at` for a target; this is for the side test and for
        reporting.
        """
        return self._lane_offset[side]

    def fits_side_by_side(self, point, lookahead: float = 0.0) -> bool:
        """Can two bodies stand side by side here, and for `lookahead` metres?

        Answered offline from the map and the objects by
        `derive_lane_offsets.py`, not inferred at runtime from the two
        lateral targets. The runtime proxy compared those targets, and when
        both vehicles shifted for the same cone they converged: room fell
        from 1.31 m to 0.33 m and the pair was told to file at a site whose
        free band is 0.78 m wide, where two bodies need 0.65 m. A fact about
        the corridor should not be re-derived every tick from quantities that
        move.

        `lookahead` asks the same question of the route just ahead, because
        the answer is needed before the pinch, not inside it. Falling into
        file takes about two seconds to build a stagger, and the pinch that
        wedged a vehicle at (13.0, -7.35) is only 0.8 m long: read at the
        pair's own position, the flag clears just as the corridor closes and
        the vehicles are already squeezed. The window is measured along the
        centreline, which is a fact about the corridor and so belongs here
        rather than in a per-tick computation by the caller.

        True when no profile carries the flag, so a route without one behaves
        as it did.
        """
        if self.offsets is None or "side_by_side" not in self.offsets:
            return True
        query = np.atleast_2d(np.asarray(point, float))
        here = int(np.atleast_1d(self._centre_tree.query(query)[1])[0])
        flag = self.offsets["side_by_side"]
        if lookahead <= 0.0:
            return bool(flag[here])
        step = float(np.linalg.norm(np.diff(self.centre[:2], axis=0)))
        span = max(1, int(round(lookahead / max(step, 1e-6))))
        index = (here + np.arange(span + 1)) % len(flag)
        return bool(flag[index].all())

    def lane_offset_at(self, point, side: str) -> float:
        """Signed cross-track offset of the lane *here*, +left.

        The lane is a recorded curve, not a constant offset of the
        centreline: on ccw outer the offset runs -0.21 to -0.73 with a median
        of -0.50. Aiming at the median put the outer vehicle 0.05 m further
        out than the lane actually goes at the corner near (20.2, 8.5), which
        has about 0.02 m of body margin at -0.50 -- so the body reached the
        wall while the vehicle was, by its own measure, tracking its lane
        well. Reading the offset where the vehicle is removes that error
        wherever it occurs, on any map, because the lane data carries it.
        """
        query = np.atleast_2d(np.asarray(point, float))
        if self.offsets is not None:
            here = int(np.atleast_1d(self._centre_tree.query(query)[1])[0])
            return float(self.offsets[side][here])

        # No profile: read the recorded lane's own local offset, which is
        # still closer to the lane than its median is.
        index = int(np.atleast_1d(self._lane_trees[side].query(query)[1])[0])
        lane = self.lanes[side][index]
        # Measured in the same frame as `signed_cross_track`, so the target
        # and the achieved value are the same quantity.
        centre = int(np.atleast_1d(
            self._centre_tree.query(np.atleast_2d(lane))[1])[0])
        cos, sin = np.cos(self.yaw[centre]), np.sin(self.yaw[centre])
        return float(-(lane[0] - self.centre[centre, 0]) * sin
                     + (lane[1] - self.centre[centre, 1]) * cos)

    def band_excursion(self, xs, ys) -> np.ndarray | None:
        """How far each rollout goes outside the band between the two lanes.

        The band is [min(outer, inner), max(outer, inner)] of the derived
        offsets at each point's nearest centreline sample; the result is the
        largest distance beyond it, per rollout, shaped like the first axis
        of `xs`. None without a derived profile.
        """
        if self.offsets is None:
            return None
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        flat = np.column_stack([xs.ravel(), ys.ravel()])
        index = self._centre_tree.query(flat)[1]
        cos, sin = np.cos(self.yaw[index]), np.sin(self.yaw[index])
        cross = (-(flat[:, 0] - self.centre[index, 0]) * sin
                 + (flat[:, 1] - self.centre[index, 1]) * cos)
        a = np.asarray(self.offsets['outer'], float)[index]
        b = np.asarray(self.offsets['inner'], float)[index]
        beyond = np.maximum(np.maximum(cross - np.maximum(a, b),
                                       np.minimum(a, b) - cross), 0.0)
        return beyond.reshape(xs.shape).max(axis=-1)

    def signed_cross_track(self, xs, ys) -> np.ndarray:
        """Signed offset of each rollout from the centreline, +left.

        Mean over the horizon, shaped (n_rollouts,). Signed on purpose: a
        distance is minimised just as well from the wrong side of the route,
        which is why nothing stopped the pair swapping sides or stacking onto
        one lane.
        """
        xs, ys = np.asarray(xs), np.asarray(ys)
        flat = np.column_stack([xs.ravel(), ys.ravel()])
        index = self._centre_tree.query(flat)[1]
        cos = np.cos(self.yaw[index])
        sin = np.sin(self.yaw[index])
        offset = (-(flat[:, 0] - self.centre[index, 0]) * sin
                  + (flat[:, 1] - self.centre[index, 1]) * cos)
        return offset.reshape(xs.shape).mean(axis=1)

    def tangent_error(self, xs, ys, yaws, side: str) -> np.ndarray:
        """Mean |heading - local lane tangent| per rollout, wrapped.

        The absolute heading reference the formation terms lack. Unlike a
        penalty on |w|, this does not charge for cornering: the reference
        direction turns with the route, so following a bend costs nothing
        while spinning on the spot costs the full deviation.
        """
        xs, ys, yaws = np.asarray(xs), np.asarray(ys), np.asarray(yaws)
        flat = np.column_stack([xs.ravel(), ys.ravel()])
        index = self._lane_trees[side].query(flat)[1]
        want = self._lane_tangent[side][index].reshape(xs.shape)
        return np.abs(np.arctan2(np.sin(yaws - want),
                                 np.cos(yaws - want))).mean(axis=1)

    def arc_length_at(self, point, side: str) -> float:
        """Arc length of the nearest point on a lane, from its start.

        `progress` returns a *difference* over a rollout, which is what the
        cost wants; an absolute position along the route is what a log wants,
        so that a lap reads as arc length climbing and wrapping rather than as
        a sequence of small forward steps.
        """
        query = np.atleast_2d(np.asarray(point, float))
        index = int(np.atleast_1d(self._lane_trees[side].query(query)[1])[0])
        return float(self._lane_s[side][index])

    def nearer_lane(self, point) -> str:
        """Which lane a position is closest to, for auto-assigning sides."""
        query = np.atleast_2d(np.asarray(point, float))
        return min(self._lane_trees,
                   key=lambda side: float(np.atleast_1d(
                       self._lane_trees[side].query(query)[0])[0]))
