"""Utility of a pair of rollouts, one per vehicle.

Terms come from measurement (stage 13 of the analysis pipeline), on data with
localiser jumps excluded -- 5.23% of samples, and including them materially
changed the ranking, so the numbers below are the clean ones.

Two categories, deliberately distinguished:

DISCRIMINATIVE -- earn their place by ranking the rollout the human actually
drove above the alternatives. Single-feature rank, then greedy gain:

    range_rate      84.8%   +34.8 over chance   rate the gap changes
    rel_yaw_rate    75.8%    +0.8               turning together
    accel           67.1%    +1.7               comfort
    rel_dist_end    80.0%    +1.4               where the gap ends up

STRUCTURAL -- rank poorly over a 3 s horizon but the problem is ill-posed
without them. Kept on purpose:

    speed     59.4%  Without a speed target, standing still is optimal: zero
                     acceleration, zero range rate, on path, separation held.
                     Greedy ranks one-step prediction given the human was
                     already moving, so it cannot see this.
    path_dist 64.2%  Its job is over 80 m, not 3 s. Zeroed, the pair happily
                     co-rotates at 0.6 rad/s -- a rigid pair turning together
                     costs nothing in the formation terms.
    progress  --     Arc length advanced along the lane, rewarded rather than
                     penalised. Distance to the lane is only the lateral half
                     of path following: with speed demanding ~1.09 m/s and
                     nothing asking for progress, driving a circle that hugs
                     the lane is optimal, and that is what the vehicle did.

That distinction is the main lesson from calibration: discrimination ranking is
not the same as closed-loop necessity.

Dropped, with reasons:

    range_rate_end 77.2%  correlates 0.70 with range_rate
    rel_dist       75.8%  correlates 0.74-0.80 with range_rate/rel_dist_end
    yaw_rate       70.2%  correlates 0.70 with rel_yaw_rate
    jerk           66.4%  adds nothing once accel is in
    along_track_err    62.5%  characterises the formation but barely responds to
                          the choice available in 3 s; keep as a validation
                          metric
    rel_speed      58.6%  correlates 0.72 with speed
    rel_dist_min   49.1%  chance -- worst-case closeness does not discriminate
    obj_dist       36.2%  *below* chance: the drivers passed closer to cones
                          than their alternatives, so as a repulsion term it
                          would push against observed behaviour. Its real role
                          is modulating the formation target, which
                          separation_vs_object already does.

    wall_dist      61.2%  a hard feasibility constraint, not a soft term: too
                          weak to weight, but the only thing preventing wall
                          collisions.

Preferred values are functions of observable context, not a table indexed by a
driving state: curvature moves speed and leaves the formation alone
(rho = +0.04), object distance moves the formation and leaves speed alone.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ContextModel:
    """Fitted preferred values. Defaults are the stage 13 fits."""

    # v*(kappa) = v0 / (1 + (kappa/kappa0)^power)
    v0: float = 1.094
    kappa0: float = 1.074
    power: float = 1.15
    # d_R*(o) = far + (near - far) * exp(-o/scale)
    sep_far: float = 1.413
    sep_near: float = 1.898
    sep_scale: float = 2.18
    #: Formation-conditioned separation, the measured alternative to the
    #: pooled law above. Corpus medians: 1.47 m abreast, 1.75 m filing,
    #: and 1.00 m abreast within 0.4 m of an object.
    sep_side: float = 1.47
    sep_file: float = 1.75
    sep_squeeze: float = 1.00
    sep_squeeze_range: float = 0.40
    #: Object distance is clipped here, matching the analysis.
    object_max: float = 5.0
    #: v_outer - v_inner = split * |w| * d, fitted on 116,345 turning samples.
    #: Rigid formation would give 1.0; the drivers hold only about a third of
    #: it, letting the along-track offset slip through a bend. Robust (median)
    #: estimate -- per-sample scatter makes least squares useless here, while
    #: the band medians it reproduces to R2 = 0.991.
    split: float = 0.326

    def preferred_speed(self, kappa):
        return self.v0 / (1.0 + (np.abs(kappa) / self.kappa0) ** self.power)

    def preferred_speed_for_side(self, kappa, separation, side_sign):
        """Speed target for one vehicle of the pair through a bend.

        Both vehicles turn at the same rate, so the outer one has the longer
        arc and must run faster to stay side by side. Scoring both against
        v*(kappa) told the outer vehicle to go too slow and the inner one too
        fast, and the formation broke on every corner.

        `side_sign` is +1 for the vehicle on the outside of the curve, -1 for
        the inside, 0 to disable the split.
        """
        pair = self.preferred_speed(kappa)
        offset = (0.5 * self.split * np.abs(kappa)
                  * np.asarray(separation, float))
        return pair * (1.0 + np.asarray(side_sign, float) * offset)

    def lateral_offset(self, lane_offset, obj_distance):
        """Target signed cross-track offset for one vehicle, +left.

        Anchored on the lane the drivers actually used, then widened when an
        object is close. `preferred_separation` was fitted to *instantaneous*
        separation, which includes the pair's longitudinal stagger, so using
        it directly as a lateral target asks for 1.46 m across lanes that sit
        1.05 m apart -- and the only ways to comply are to leave the lanes or
        to stay staggered. Only the object-driven *change* in it transfers to
        a lateral target, split between the two vehicles.
        """
        widening = 0.5 * (np.asarray(self.preferred_separation(obj_distance),
                                     float) - self.sep_far)
        return lane_offset + np.sign(lane_offset) * widening

    def preferred_separation(self, obj_distance, side_by_side=None):
        """Preferred centre-to-centre separation.

        `side_by_side` None keeps the object-distance law every result before
        2026-09-29 was measured under. True or False selects the measured
        constant for that formation instead.

        The law is a mixture artefact. Measured over 235k corpus samples, the
        drivers' separation barely moves with object distance *within* a
        formation -- abreast holds 1.45-1.51 m over the whole range, filing
        1.67-1.91 m -- while the proportion filing falls from 80% next to an
        object to 35% far from one. Pooling the two produces a decline from
        1.81 to 1.53 m that neither population exhibits, and fitting an
        exponential to it asks a side-by-side pair to hold a filing distance:
        the shipped `sep_far`/`sep_near` sit on the filing fit and 0.3 m wide
        of the abreast one. Two of the three pooled parameters also railed
        against their bounds, which is the fit saying so.

        The one real object effect is a squeeze: abreast separation drops to
        about 1.00 m within 0.4 m of an object, and only there.
        """
        d = np.minimum(np.asarray(obj_distance, float), self.object_max)
        if side_by_side is None:
            return self.sep_far + (self.sep_near - self.sep_far) * np.exp(
                -d / self.sep_scale)
        if not side_by_side:
            return np.full_like(d, self.sep_file, dtype=float)
        squeeze = np.clip(d / self.sep_squeeze_range, 0.0, 1.0)
        return self.sep_squeeze + (self.sep_side - self.sep_squeeze) * squeeze

    @classmethod
    def from_json(cls, path: str | Path) -> "ContextModel":
        payload = json.loads(Path(path).read_text())
        speed = payload["speed_vs_curvature"]
        sep = payload["separation_vs_object"]
        kwargs = {}
        if "speed_split" in payload:
            kwargs["split"] = payload["speed_split"]["coefficient"]
        return cls(v0=speed["v0"], kappa0=speed["kappa0"], power=speed["power"],
                   sep_far=sep["far"], sep_near=sep["near"],
                   sep_scale=sep["scale"], **kwargs)


@dataclass
class Scales:
    """Typical magnitude of each term, so a weight of 1.0 means equal influence.

    Without this the weights are uninterpretable and hand-tuning is hopeless:
    progress runs 1-4.3 m over a 3 s horizon while speed error runs ~0.04 m/s
    and range rate ~0.01-0.1 m/s, so at equal weights progress swamps
    everything -- weight 1 and weight 3 gave an identical choice because the
    term was already saturated.

    Values are the medians measured on the recorded corpus, except progress
    which is the median advance across a rollout set.
    """

    rel_range_rate: float = 0.20     # recorded |v1-v2| median 0.156 m/s
    rel_dist_end: float = 0.30   # separation error we care about
    rel_yaw_rate: float = 0.20   # recorded |w1-w2| median 0.15 rad/s
    motion_accel: float = 0.20          # recorded |a| median 0.139 m/s^2
    motion_speed: float = 0.20          # error against v*(kappa)
    path_dist: float = 0.30      # lane error we care about
    path_progress: float = 2.00       # median advance over 3 s
    rel_along_track: float = 0.60         # recorded along-track offset median 0.597 m
    #: |along| / separation = |cos R_a|: 0 abreast, 1 single file. Target is
    #: zero, so the term reads "be abreast". The corpus median is 0.499, which
    #: is the scale. Deliberately NOT an arctan2: ||bearing| - pi/2| over
    #: (n_own, n_partner, n_steps) cost 108 ms against a 100 ms budget, while
    #: this is one divide on an array already computed for `along_track`.
    #: 0.133, the corpus median of the *signed* feature |sin R_a - side|
    #: over 235k samples. The previous 0.50 was measured for the magnitude
    #: form |cos R_a| and was never re-measured when the signed form was
    #: added, so the term delivered 3.7x less authority than its weight
    #: implied -- the whole purpose of a scale is that a weight of 1.0 means
    #: equal influence, and it did not.
    rel_angle: float = 0.133
    rel_yaw: float = 0.013       # recorded 1-cos(dyaw) median (0.164 rad)
    # Both measured on the 48 obstacle-free recordings, both directions, both
    # vehicles, localiser jumps excluded: 234,578 samples. They were initially
    # borrowed from path_dist and rel_yaw, which overstated both.
    path_cross_track: float = 0.203   # median |cross-track - lane offset|
    path_lane_yaw: float = 0.092      # median |heading - lane tangent| [rad], 5.3 deg

    def get(self, name: str) -> float:
        return float(getattr(self, name))


@dataclass
class Weights:
    """Equal starting weights; IRL over the rollout sets replaces these."""

    # discriminative
    rel_range_rate: float = 1.0
    rel_dist_end: float = 1.0
    rel_yaw_rate: float = 1.0
    motion_accel: float = 1.0
    # structural
    motion_speed: float = 1.0
    path_dist: float = 1.0
    # Being *beside* the partner rather than ahead of or behind it, and
    # pointing the same way. Both rank poorly over 3 s -- the reason they were
    # dropped once -- because in 3 s no candidate changes the formation much.
    # But nothing else in the cost asks for 90 degrees, so without them there
    # is no side-by-side formation at all: the pair holds its gap and its turn
    # rate while drifting into file. Structural, like speed and path_dist.
    rel_along_track: float = 1.0
    #: Pull toward abreast: |along| / separation, target zero.
    #: See Scales.ra. Zero keeps the previous cost exactly.
    rel_angle: float = 0.0
    rel_yaw: float = 1.0
    # 0.5, not 1.0. Arc-length progress measured by nearest-point projection
    # rewards cutting a corner -- a rollout crossing the chord projects further
    # along the arc than one following it. At weight 1.0 the planner took
    # w = -0.60 with 0.46 m of lane error where 0.5 takes w = -0.20 with
    # 0.14 m. Enough to stop it circling, not enough to make it cut.
    path_progress: float = 0.5

    # Both default to zero, so adding them changes nothing until they are
    # asked for. Signed lateral position against the route, and heading
    # against the route: the two things the pair-relative terms cannot see.
    #
    # `cross_track` is what forbids swapping sides. Every other spatial term
    # is a distance, and a distance is minimised just as well from the wrong
    # side of the route -- measured over three runs, the pair sat on the same
    # lane 27-35% of the time and crossed sides 5-8 times per run, which the
    # recorded drivers never did once.
    #
    # `lane_yaw` is what forbids co-rotation. It replaces a penalty on |w|,
    # which cannot tell a corner from a spin: tested in closed loop, |w|
    # removed the spinning but pulled separation down to 0.65 m and was not
    # monotonic in its own weight.
    path_cross_track: float = 0.0
    path_lane_yaw: float = 0.0

    def names(self):
        return ["rel_range_rate", "rel_dist_end", "rel_yaw_rate", "motion_accel",
                "motion_speed", "path_dist", "path_progress", "rel_along_track", "rel_yaw",
                "path_cross_track", "path_lane_yaw", "rel_angle"]

    def vector(self):
        return np.array([getattr(self, n) for n in self.names()], float)


@dataclass
class PairContext:
    """Everything the cost needs that is not a rollout."""

    dt: float
    kappa_own: float = 0.0
    kappa_partner: float = 0.0
    object_distance: float = 5.0
    path_dist_own: np.ndarray | None = None      # (n_own,) mean over horizon
    path_dist_partner: np.ndarray | None = None  # (n_partner,)
    path_progress_own: np.ndarray | None = None       # (n_own,) arc length advanced
    path_progress_partner: np.ndarray | None = None   # (n_partner,)
    #: +1 for the vehicle outside the curve, -1 inside, 0 to disable the
    #: outer/inner speed split.
    side_sign_own: float = 0.0
    side_sign_partner: float = 0.0
    #: Which side of the ego the partner is assigned to, +1 left, -1
    #: right, 0 to use the sign-blind magnitude form of `rel_angle`.
    #: Derived from the two lane offsets, so it is a discrete
    #: assignment rather than a continuous pin to the route.
    rel_angle_side_own: float = 0.0
    #: Whether the pair is holding side by side this tick. None keeps the
    #: pooled object-distance separation law; True/False select the
    #: formation-conditioned constants. See ContextModel.
    side_by_side: bool | None = None
    #: Signed cross-track offset from the centreline, +left, per rollout, and
    #: the offset each vehicle is aiming for. `None` disables the term.
    cross_own: np.ndarray | None = None       # (n_own,)
    cross_partner: np.ndarray | None = None   # (n_partner,)
    cross_target_own: float = 0.0
    cross_target_partner: float = 0.0
    #: Along-track offset the pair should hold, metres, SIGNED: positive
    #: means the partner should be ahead of the ego, negative behind. The sign
    #: matters and leaving it out was a bug -- with the offset taken as a
    #: magnitude the term had minima at both orderings, so which vehicle led
    #: was undetermined and the blocked one sometimes rushed the gap and cut
    #: across the vehicle that had a clear lane.
    #:
    #: Zero means exactly level, which is the formation everywhere the
    #: corridor allows it. When
    #: an obstacle blocks one lane the two cross-track targets come closer
    #: than a body width, the pair cannot be side by side at all, and the recorded
    #: drivers resolved that by going into file: measured at the instant of
    #: the pass, 1.20-1.55 m of stagger with 0.28-0.70 m of lateral
    #: separation. So the formation is not abandoned, its target moves --
    #: which is the same pattern as `preferred_speed(kappa)` and
    #: `preferred_separation(object)`, and it means `side by side` keeps its weight
    #: and is what pulls the pair back together once the obstacle is passed.
    along_track_target: float = 0.0
    #: Accept either ordering: the along-track term is scored against
    #: +/- `along_track_target` and takes the cheaper, so the term expresses no
    #: preference about which vehicle leads and the rest of the cost decides.
    #:
    #: With this off, the sign of `along_track_target` imposes the ordering. That
    #: is a convention, not a result: measured with the term switched off, the
    #: cost's own preference is for the blocked vehicle to *lead* whenever the
    #: inner lane is the blocked one -- scoring 2 to 4 units better than the
    #: ordering the recorded drivers chose. The drivers yielded when blocked;
    #: this cost does not derive that, so imposing it is a modelling choice.
    along_track_either: bool = False
    #: Multiplier on each vehicle's preferred speed. Below 1.0 for the
    #: vehicle that is yielding while the pair passes an obstacle in file.
    #:
    #: Yielding cannot be left to `side by side` alone. Measured on a real run,
    #: the blocked vehicle accelerated from 0.75 to 1.09 m/s to take the gap
    #: while the vehicle with a clear lane braked to 0.63 m/s below its own
    #: target to let it in -- the opposite of the intended manoeuvre, and
    #: it happens because reordering needs a sustained speed difference that
    #: `speed` charges about six times what `side by side` repays. Scaling the
    #: yielder's target makes `speed` drive the manoeuvre rather than forbid
    #: it, and costs no lateral freedom, where raising `along_track` enough to
    #: overpower it took the inner vehicle's obstacle clearance to 9 mm.
    speed_scale_own: float = 1.0
    speed_scale_partner: float = 1.0
    #: Upper bound on the speed target from the corridor ahead -- see
    #: `wall_speed_cap`. Infinity leaves the preferred speed untouched.
    speed_cap_own: float = float('inf')
    speed_cap_partner: float = float('inf')
    #: Upper bound on the *separation* target from the corridor measured
    #: ahead. Infinity leaves `preferred_separation` untouched.
    #:
    #: `preferred_separation` rises as an object is approached -- to 1.71 m,
    #: which is what the drivers did, because they had somewhere to go. In a
    #: corridor that offers 1.2 m the same target is a demand for width that
    #: does not exist, and the cost buys it with the only margin left: the
    #: outer vehicle's. Measured at three cw sites, that vehicle *opened* the
    #: gap from 0.62 to 0.79 m by moving 0.20 m further out, and finished
    #: 51-125 mm inside the wall, stopped. Capping the target at the room
    #: actually there does not ask the pair to squeeze -- the term simply
    #: stops paying for width the corridor cannot give.
    #:
    #: A cap, like `speed_cap_*`, and for the same reason: applied to the
    #: target rather than as a mask on the candidates, so every candidate
    #: stays available and the admissible set cannot empty.
    separation_cap: float = float('inf')
    #: Mean |heading - lane tangent| per rollout. `None` disables the term.
    tangent_own: np.ndarray | None = None     # (n_own,)
    tangent_partner: np.ndarray | None = None   # (n_partner,)


#: The pairwise block is (n_own, n_partner, n_steps) -- 3.8M elements at the
#: default grid -- so it is memory-bandwidth bound and runs in single
#: precision. Positions are metres over a 3 s horizon, where float32 carries
#: about 1e-7 relative error: far below the centimetre the terms are compared
#: at, and it takes the block from 108 ms to 64 ms against a 100 ms budget.
PAIR_DTYPE = np.float32


def separation(own, partner, dtype=None):
    """Pairwise distance between every own rollout and every partner rollout.

    Shape (n_own, n_partner, n_steps).
    """
    x_own = own["x"] if dtype is None else own["x"].astype(dtype, copy=False)
    y_own = own["y"] if dtype is None else own["y"].astype(dtype, copy=False)
    x_par = (partner["x"] if dtype is None
             else partner["x"].astype(dtype, copy=False))
    y_par = (partner["y"] if dtype is None
             else partner["y"].astype(dtype, copy=False))
    dx = x_own[:, None, :] - x_par[None, :, :]
    dy = y_own[:, None, :] - y_par[None, :, :]
    return np.hypot(dx, dy)


def evaluate(own, partner, ctx: PairContext,
             weights: Weights | None = None,
             model: ContextModel | None = None,
             scales: Scales | None = None):
    """Utility matrices for both agents over the rollout pairs.

    Returns ``(u_own, u_partner, terms)`` with the matrices shaped
    (n_own, n_partner). ``terms["u_solo"]`` is the ego-only utility, shaped
    (n_own,), which is what the `independent` concept maximises. Utility is negative cost, so larger is better and the
    solution concepts can all be written as maximisations.

    The formation terms are symmetric -- both agents see the same gap and the
    same rate of change -- so they enter both matrices identically. Only the
    speed and path terms differ, each agent being scored against its own
    reference and curvature.
    """
    weights = weights or Weights()
    model = model or ContextModel()
    scales = scales or Scales()
    w, k = weights, scales

    sep = separation(own, partner, PAIR_DTYPE)
    n_own, n_partner, n_steps = sep.shape

    # Rate the gap opens or closes: the strongest single feature measured.
    range_rate = (np.abs(np.diff(sep, axis=2) / ctx.dt).mean(axis=2)
                  if w.rel_range_rate else 0.0)

    # Where the gap ends up, against the context-dependent target. `target` is
    # set below, with the speed targets that also depend on it.
    def _rel_dist_end(t):
        return np.abs(sep[:, :, -1] - t)

    # Own-only terms, broadcast across the partner axis.
    #
    # Deviation is averaged per step, not taken on the mean speed. Penalising
    # |mean(v) - v*| lets a rollout accelerate hard and then brake: its mean
    # matches v* exactly while the profile is nothing like the target. The
    # per-step form charges for the excursion.
    target = min(float(model.preferred_separation(ctx.object_distance,
                                                  ctx.side_by_side)),
                 float(ctx.separation_cap))
    # The cap is applied to the *target*, not as a mask on the candidates.
    # A hard limit is what produced the failure it exists to prevent: the
    # wall mask admitted only v = 0 and the vehicle stopped dead. Lowering
    # the target lets `speed` pull the vehicle down smoothly from metres out,
    # with every candidate still available.
    v_star_own = min(
        ctx.speed_scale_own * model.preferred_speed_for_side(
            ctx.kappa_own, target, ctx.side_sign_own),
        ctx.speed_cap_own)
    v_star_partner = min(
        ctx.speed_scale_partner * model.preferred_speed_for_side(
            ctx.kappa_partner, target, ctx.side_sign_partner),
        ctx.speed_cap_partner)
    motion_speed_own = np.abs(own["v"] - v_star_own).mean(axis=1)
    motion_speed_partner = np.abs(partner["v"] - v_star_partner).mean(axis=1)
    rel_dist_end = _rel_dist_end(target)

    path_own = (ctx.path_dist_own if ctx.path_dist_own is not None
                else np.zeros(n_own))
    path_partner = (ctx.path_dist_partner if ctx.path_dist_partner is not None
                    else np.zeros(n_partner))
    path_progress_own = (ctx.path_progress_own if ctx.path_progress_own is not None
                    else np.zeros(n_own))
    path_progress_partner = (ctx.path_progress_partner if ctx.path_progress_partner is not None
                        else np.zeros(n_partner))

    # Turning together: ranked 75.8%, above matching speed (58.6%). In a
    # corridor, holding formation through a corner is mostly about turning at
    # the same rate; the inner/outer speed difference is a consequence.
    if w.rel_yaw_rate:
        rel_yaw_rate = np.abs(
            own["w"].astype(PAIR_DTYPE, copy=False)[:, None, :]
            - partner["w"].astype(PAIR_DTYPE, copy=False)[None, :, :]).mean(2)
    else:
        rel_yaw_rate = 0.0

    # Side by side: how far ahead or behind the partner sits, along the
    # vehicle's own heading. Zero means exactly level. This is what makes a
    # vehicle ease off when it is ahead and push when it is behind -- the only
    # way to null an along-track offset is to change speed -- and nothing else
    # in the cost asks for it.
    #
    # As a projection rather than an angle, on purpose. The equivalent
    # ||bearing| - pi/2| needs an arctan2 over (n_own, n_partner, n_steps),
    # which cost 108 ms per call against a 100 ms budget at 10 Hz; the dot
    # product is 10 ms and it reads in metres. Headings enter through their
    # sine and cosine, taken once on the small (n, n_steps) arrays.
    cos_own = np.cos(own["yaw"]).astype(PAIR_DTYPE, copy=False)
    sin_own = np.sin(own["yaw"]).astype(PAIR_DTYPE, copy=False)
    cos_partner = np.cos(partner["yaw"]).astype(PAIR_DTYPE, copy=False)
    sin_partner = np.sin(partner["yaw"]).astype(PAIR_DTYPE, copy=False)
    if w.rel_along_track or w.rel_angle:
        x_own = own["x"].astype(PAIR_DTYPE, copy=False)
        y_own = own["y"].astype(PAIR_DTYPE, copy=False)
        x_partner = partner["x"].astype(PAIR_DTYPE, copy=False)
        y_partner = partner["y"].astype(PAIR_DTYPE, copy=False)
        dx = x_partner[None, :, :] - x_own[:, None, :]
        dy = y_partner[None, :, :] - y_own[:, None, :]
        proj_own = dx * cos_own[:, None, :] + dy * sin_own[:, None, :]
        proj_partner = (dx * cos_partner[None, :, :]
                        + dy * sin_partner[None, :, :])
        rel_along_track_own = np.abs(proj_own).mean(2)
        rel_along_track_partner = np.abs(proj_partner).mean(2)
        if w.rel_angle:
            # R_a, the partner's bearing about own heading, scale free: unlike
            # `rel_along_track`, 0.6 m of offset at 1.2 m apart is a different
            # formation from 0.6 m at 3 m apart and only this term can tell
            # them apart. `sep` is floored at one body width so an overlapping
            # pair cannot divide by zero.
            safe = np.maximum(sep, PAIR_DTYPE(2.0 * FOOTPRINT_BOX[2]))
            if ctx.rel_angle_side_own:
                # SIGNED form: sin R_a, the partner's *lateral* offset in own
                # frame over separation. +1 abreast on one side, -1 abreast on
                # the other, 0 in single file. Target is the assigned side, so
                # one term forbids both filing *and* crossing over -- which is
                # what R_a was defined to do, and what the magnitude form
                # below cannot do, because |cos R_a| is zero at both +pi/2 and
                # -pi/2 and so cannot tell the pair's arrangement from its
                # mirror image.
                #
                # sin rather than atan2 on purpose: ||bearing| - pi/2| over
                # (n_own, n_partner, n_steps) cost 108 ms against a 100 ms
                # budget, while this is one divide on an array already built.
                perp_own = (-dx * sin_own[:, None, :]
                            + dy * cos_own[:, None, :])
                # The partner sees the ego, so its vector is negated and its
                # target is the negation of ours.
                perp_partner = (dx * sin_partner[None, :, :]
                                - dy * cos_partner[None, :, :])
                side = PAIR_DTYPE(ctx.rel_angle_side_own)
                rel_angle_own = np.abs(perp_own / safe - side).mean(2)
                rel_angle_partner = np.abs(perp_partner / safe + side).mean(2)
            else:
                # Magnitude form, |cos R_a|: 0 abreast, 1 single file, target
                # zero. Sign blind. Kept because it is what every result
                # before 2026-09-29 was measured under.
                rel_angle_own = (np.abs(proj_own) / safe).mean(2)
                rel_angle_partner = (np.abs(proj_partner) / safe).mean(2)
        else:
            rel_angle_own = rel_angle_partner = 0.0
        if ctx.along_track_target:
            # Signed, so the ordering is determined. `dx, dy` runs from ego to
            # partner, so projecting it on a vehicle's own heading gives a
            # positive number when the partner is ahead of it -- and the same
            # expression serves both terms, because the partner's own target
            # is the negation of the ego's and so is its measure.
            #
            # The magnitude form is kept for the side by side case: mean|proj|
            # charges for swinging through zero, where |mean proj| does not,
            # and the obstacle-free behaviour was calibrated on it.
            target = PAIR_DTYPE(ctx.along_track_target)
            signed_own = (dx * cos_own[:, None, :]
                          + dy * sin_own[:, None, :]).mean(2)
            signed_partner = (dx * cos_partner[None, :, :]
                              + dy * sin_partner[None, :, :]).mean(2)
            if ctx.along_track_either:
                rel_along_track_own = np.minimum(np.abs(signed_own - target),
                                         np.abs(signed_own + target))
                rel_along_track_partner = np.minimum(
                    np.abs(signed_partner - target),
                    np.abs(signed_partner + target))
            else:
                rel_along_track_own = np.abs(signed_own - target)
                rel_along_track_partner = np.abs(signed_partner - target)
    else:
        # Both weights zero: neither term is wanted. `rel_angle_own` must
        # still exist, because the utility sum references every term
        # unconditionally and multiplies by the weight afterwards -- a zero
        # weight does not stop the name being looked up. Leaving it unbound
        # crashed both planners on every run of the ablation arm that zeroed
        # `rel_along_track` while `rel_angle` was also off.
        rel_along_track_own = rel_along_track_partner = 0.0
        rel_angle_own = rel_angle_partner = 0.0

    # Parallel formation: pointing the same way, not merely turning at the
    # same rate -- matching yaw *rates* leaves any fixed heading offset
    # intact. 1 - cos(dyaw) rather than |dyaw|, so the difference of angles is
    # never formed on the big array; mean_t(cos a cos b + sin a sin b)
    # separates into two matrix products.
    #
    # Skipped outright at zero weight. This is not micro-optimisation: the
    # target platform's numpy is built without BLAS (1.4 GFLOPS on a 600x600
    # matmul), so these two small products cost 159 ms of a 100 ms budget --
    # spent on a term that was then multiplied by zero.
    if w.rel_yaw:
        rel_yaw = np.maximum(1.0 - (cos_own @ cos_partner.T
                                    + sin_own @ sin_partner.T)
                             / float(n_steps), 0.0)
    else:
        rel_yaw = 0.0

    motion_accel_own = np.abs(np.diff(own["v"], axis=1) / ctx.dt).mean(axis=1)
    motion_accel_partner = np.abs(np.diff(partner["v"], axis=1) / ctx.dt).mean(axis=1)

    # Signed lateral position against the route. |measured - target| on the
    # *signed* offset, so the wrong side of the centreline is as far from the
    # target as it is wide, rather than being an equally good answer.
    def _cross(measured, target, count):
        if measured is None or not w.path_cross_track:
            return np.zeros(count)
        return np.abs(np.asarray(measured, float) - float(target))

    cross_own = _cross(ctx.cross_own, ctx.cross_target_own, n_own)
    cross_partner = _cross(ctx.cross_partner, ctx.cross_target_partner,
                           n_partner)

    def _tangent(measured, count):
        if measured is None or not w.path_lane_yaw:
            return np.zeros(count)
        return np.asarray(measured, float)

    tangent_own = _tangent(ctx.tangent_own, n_own)
    tangent_partner = _tangent(ctx.tangent_partner, n_partner)

    # Every term is divided by its typical magnitude, so the weights compare.
    shared = (w.rel_range_rate / k.rel_range_rate * range_rate
              + w.rel_dist_end / k.rel_dist_end * rel_dist_end
              + w.rel_yaw_rate / k.rel_yaw_rate * rel_yaw_rate
              + w.rel_yaw / k.rel_yaw * rel_yaw)
    # Progress is the one term that is rewarded, not charged.
    u_own = -(shared
              + w.rel_angle / k.rel_angle * rel_angle_own
              + w.rel_along_track / k.rel_along_track * rel_along_track_own
              + w.motion_speed / k.motion_speed * motion_speed_own[:, None]
              + w.motion_accel / k.motion_accel * motion_accel_own[:, None]
              + w.path_dist / k.path_dist * path_own[:, None]
              + w.path_cross_track / k.path_cross_track * cross_own[:, None]
              + w.path_lane_yaw / k.path_lane_yaw * tangent_own[:, None]
              - w.path_progress / k.path_progress * path_progress_own[:, None])
    u_partner = -(shared
                  + w.rel_angle / k.rel_angle * rel_angle_partner
                  + w.rel_along_track / k.rel_along_track * rel_along_track_partner
                  + w.motion_speed / k.motion_speed * motion_speed_partner[None, :]
                  + w.motion_accel / k.motion_accel * motion_accel_partner[None, :]
                  + w.path_dist / k.path_dist * path_partner[None, :]
                  + w.path_cross_track / k.path_cross_track * cross_partner[None, :]
                  + w.path_lane_yaw / k.path_lane_yaw * tangent_partner[None, :]
                  - w.path_progress / k.path_progress * path_progress_partner[None, :])

    # Utility with the partner ignored entirely, for the `independent`
    # baseline. Taking a column of `u_own` instead would score the ego against
    # one arbitrary partner candidate -- and candidate 0 is a standing spin, so
    # the "independent" agent was solving for the best response to a parked
    # partner, whose answer is to park alongside it.
    u_solo = -(w.motion_speed / k.motion_speed * motion_speed_own
               + w.motion_accel / k.motion_accel * motion_accel_own
               + w.path_dist / k.path_dist * path_own
               + w.path_cross_track / k.path_cross_track * cross_own
               + w.path_lane_yaw / k.path_lane_yaw * tangent_own
               - w.path_progress / k.path_progress * path_progress_own)

    terms = {"rel_range_rate": range_rate, "rel_dist_end": rel_dist_end,
             "rel_yaw_rate": rel_yaw_rate, "rel_along_track_own": rel_along_track_own,
             "rel_yaw": rel_yaw, "motion_accel_own": motion_accel_own,
             "motion_speed_own": motion_speed_own, "path_dist_own": path_own,
             "path_progress_own": path_progress_own,
             "path_cross_track_own": cross_own, "path_lane_yaw_own": tangent_own,
             "target_separation": target,
             "preferred_speed_own": float(np.ravel(v_star_own)[0]),
             "u_solo": u_solo}
    return u_own, u_partner, terms


def moving(own, v_min: float, steps: int | None = None):
    """Reject candidates that crawl, so standing still is not an option.

    Every formation term in this cost is a deviation penalty, and all of them
    are identically zero when nothing moves: an pair already side by side that stops
    holds its gap, its heading and its along-track offset for free, forever. Add the
    bounded progress reward and standing still is a genuine optimum of the
    objective, not a numerical accident -- and being a competing optimum, no
    choice of weights removes it. Sweeping the speed weight over 0.43-3.0 moved
    the boundary around (the loop parked at 0.43 one way and at 2.0 the other)
    without ever clearing it.

    So it is a constraint, like wall clearance, not a term. Stopping stays
    available as a *safety* behaviour, because `feasible` takes precedence:
    see `admissible`.

    `steps` spans the whole plan by default, and that matters more than it
    looks. Measured over the shipped fan: from rest, **0 of 80** candidates
    average 0.328 m/s over the first 0.5 s, because tau_v = 0.41 s means the
    vehicle is still spooling up -- the fastest possible average is 0.220.
    Over the full plan, 20 of 80 pass. Judged on half a second, a stopped
    vehicle fails the "you must be going somewhere" test *because it is
    stopped*, the fallback below collapses the mask to a single candidate --
    the fastest, which is also the one most likely to clip an obstacle -- and
    `admissible` drops to a rung that re-admits standing still. The vehicle
    then cannot climb back above 0.2 m/s and stays put: measured at 92 s and
    138 s in two cw runs. Whether a plan is going somewhere is a property of
    the plan, not of its first half-second.
    """
    speed = own["v"][:, :steps] if steps else own["v"]
    mean_speed = speed.mean(axis=1)
    ok = mean_speed >= v_min
    if not ok.any():
        ok = np.zeros(len(mean_speed), bool)
        ok[int(np.argmax(mean_speed))] = True
    return ok


def admissible(own, wall_tree, v_min: float, min_clearance: float = 0.0,
               steps: int | None = 5, object_tree=None,
               object_clearance: float | None = None,
               footprint=None, object_steps: int | None = None,
               report: dict | None = None,
               tau_v: float | None = None, a_max: float | None = None):
    """Candidates that clear the walls and objects, and are going somewhere.

    Objects belong here, not only in the cost. Detected objects used to feed
    `preferred_separation` alone -- widening the formation as a cone came up --
    while nothing stopped a rollout from driving straight over it: obstacle
    distance ranked *below* chance as a cost term (the drivers passed closer to
    cones than their alternatives), so it was correctly left out of the
    objective and then wrongly left out of everything.

    Object clearance is measured from the body, via `footprint_clearance`, not
    from base_link. `object_clearance` is therefore a margin between the
    vehicle's outline and the object, and its useful value is much smaller
    than the centre-based number it replaced.

    Objects are checked over a longer prefix than walls (`object_steps`, and
    `steps` if it is None). The short wall prefix is deliberate -- a 3 s hard
    wall check empties the set at every corner -- but that reasoning does not
    transfer. A wall runs alongside the route, so a candidate that clears it
    now will still clear it shortly. A cone sits *on* the route, and avoiding
    one means moving sideways by more than the body half-width, which at
    tau_omega = 0.49 s and 0.8 rad/s takes far longer than 0.5 s. Checked over
    0.5 s the cone becomes visible to the constraint only once no candidate
    can avoid it, the mask empties, the fallback drops the constraint, and the
    vehicle drives into it -- which is what a closed-loop run showed, with the
    body reaching 0.000 m of clearance.

    Precedence, in order: walls, then objects, then the speed floor. The floor
    is dropped before the object constraint, because stopping in front of a
    cone beats driving into it -- the previous order had it the other way
    round, so a boxed-in vehicle preferred a collision to a halt. Walls stay
    top of the list and the returned set is never empty: if nothing clears,
    the roomiest candidate is returned, since creeping along the best
    available option beats stopping dead in front of a wall.
    """
    # Kept as an array, not just the mask: `feasible` never returns an empty
    # set -- it falls back to the roomiest single candidate -- so `safe` can
    # mean "inside a wall, but least-badly", and a caller that needs to know
    # whether a candidate actually clears cannot get that from the mask.
    wall_room = footprint_clearance(own, wall_tree, steps,
                                    FOOTPRINT_BOX if footprint is None
                                    else footprint)
    safe = feasible(own, wall_tree, min_clearance, steps, footprint,
                    tau_v, a_max)
    room = np.full(len(safe), np.inf)
    if object_tree is not None:
        margin = (min_clearance if object_clearance is None
                  else object_clearance)
        box = FOOTPRINT_BOX if footprint is None else footprint
        ahead = steps if object_steps is None else object_steps
        # The full horizon only ranks candidates; whether the vehicle may
        # move is decided on the executed prefix (`room_near`, exact) and by
        # the brake test. So the cheap selection here (`footprint_clearance`).
        room = footprint_clearance(own, object_tree, ahead, box, outline=False)
    clear = room >= (0.0 if object_tree is None else margin)
    # The same clearance over the *executed* prefix. `room` spans the whole
    # horizon on purpose -- a cone has to be seen early enough to steer round,
    # which takes far longer than 0.5 s -- and that is right for choosing
    # between candidates. It is wrong as a test of whether the vehicle may
    # move at all: close to a cone every 3 s trajectory eventually clips it,
    # so the test empties and the vehicle stops dead while 0.131 m clear of
    # anything, which is what stalled one vehicle for 87 s. Only the first
    # step is executed and the planner replans at 10 Hz, which is the same
    # reason `feasible` checks walls over a prefix.
    room_near = (np.full(len(safe), np.inf) if object_tree is None else
                 footprint_clearance(own, object_tree, steps, box))
    # Not `steps`: the wall prefix is half a second, and a floor measured
    # over half a second is unreachable from rest. See `moving`.
    go = moving(own, v_min)

    # A moving candidate whose *executed prefix* clears the margin outranks a
    # standstill that merely looks clear over a horizon it will never reach.
    # The full-horizon test is for choosing between candidates; close to a
    # cone every 3 s trajectory eventually clips it, so as a test of whether
    # the vehicle may move at all it empties and the vehicle stops dead with
    # room to spare. Measured at the pose that froze a vehicle for 92 s: of 60
    # forward candidates the best cleared 0.070 m over 3 s -- none reached the
    # 0.10 m margin -- while 36 cleared it over the executed 0.5 s.
    #
    # This rung keeps the **full margin**, only over the prefix, and still
    # requires the speed floor. It is safe for the reason `feasible` checks
    # walls over a prefix: only the first step is executed and the planner
    # replans at 10 Hz, so the vehicle can never be committed to a state whose
    # next half-second comes within the margin. As it closes on a cone the
    # prefix test fails in its turn, and stopping becomes the right answer.
    # The drivers passed cones at 0.00-0.34 m, so creeping past at 0.070 m is
    # inside their envelope.
    #
    # Two weaker versions were measured and must not be reinstated. Dropping
    # the speed floor from this rung does nothing, because before the floor
    # itself was fixed `moving` collapsed to the single fastest candidate --
    # the one that travels furthest and so is exactly the one the prefix test
    # rejects. And relaxing the prefix test from the margin to mere
    # non-penetration, ranked above `walls+objects`, made things worse: 3
    # stall episodes in 6 cw runs instead of 1, 17.4% wall overlap, and a
    # vehicle wedged 0.009 m inside a wall.
    near = room_near >= (0.0 if object_tree is None else margin)
    names = ('walls+objects+moving', 'walls+near-objects+moving',
             'walls+objects')
    for name, candidate in zip(names, (safe & clear & go, safe & near & go,
                                       safe & clear)):
        if candidate.any():
            if report is not None:
                report.update(rule=name, n_safe=int(safe.sum()),
                              n_clear=int(clear.sum()),
                              n_chosen=int(candidate.sum()), go=go,
                              margin=(0.0 if object_tree is None else margin),
                              safe=safe, clear=clear, room=room, room_near=room_near, wall_room=wall_room)
            return candidate

    # Nothing clears the objects, even at a standstill. The object filter is
    # NOT dropped here: an earlier version fell through to a wall-only mask,
    # and the log recorded "140/140 candidates cleared walls, 0 cleared
    # objects" while the vehicle drove through a cone at 1.02 m/s. Removing
    # the rollouts that touch an obstacle is the whole point of the check, so
    # the worst case is the single roomiest candidate by object clearance --
    # the same shape of fallback `feasible` uses for walls, and it creeps away
    # from the obstacle instead of ignoring it.
    best = np.zeros(len(safe), dtype=bool)
    if object_tree is not None and tau_v is not None and a_max is not None:
        # Same reasoning as the wall fallback: when no plan clears the
        # objects, the one that arrives slowest beats the one with most room.
        best[least_impact(own, object_tree, box, ahead)] = True
    else:
        best[int(np.argmax(np.where(safe, room, -np.inf)))] = True
    if report is not None:
        # `safe` and `clear` go out so the caller can tell a genuinely clear
        # candidate from this least-bad one. The mask returned here keeps the
        # planner running, but every member of it touches something, and a
        # caller that drives it is driving into an obstacle.
        report.update(rule='roomiest from the object', n_safe=int(safe.sum()),
                      n_clear=int(clear.sum()), n_chosen=1, go=go,
                      margin=(0.0 if object_tree is None else margin),
                      safe=safe, clear=clear, room=room, room_near=room_near, wall_room=wall_room)
    return best


def shifted_lateral_target(lane_offset: float, obstacle_cross,
                           clearance: float, half_width: float = 0.275,
                           max_shift: float = 1.00,
                           past_centre: float | None = 0.45,
                           current: float | None = None,
                           min_slack: float = 0.20,
                           veto_cross=None,
                           veto_clearance: float = 0.08) -> float:
    """Lane target moved aside far enough to clear the nearest obstacle.

    `obstacle_cross` are the signed cross-track offsets of the obstacle points
    that matter -- the ones ahead of the vehicle -- in the same frame as
    `lane_offset`. Returns the target the vehicle should hold: the lane itself
    when nothing is in the way, otherwise the closest position to it that
    leaves the body `clearance` from every blocking obstacle.

    This is a *target*, not a constraint, and that is the whole point. A hard
    clearance constraint on the rollouts could only answer yes or no: checked
    over half a second it saw the obstacle too late for the 0.49 s yaw lag to
    do anything, and checked over the full horizon every candidate clipped
    something, so the mask emptied and the constraint was dropped -- a
    measured run drove through a cone at 1.02 m/s with the log reporting
    "0 of 140 candidates cleared objects". A moved target is a cost gradient
    available from metres away, so the vehicle eases across early, while the
    constraint stays as the backstop it should have been.

    Direction comes from where the obstacle actually is, not from a rule about
    which way to go. The archived map cannot calibrate that -- its cone cells
    are inflated and offset by about the size of the shift being fitted -- but
    a live detection has no such error.

    Recorded behaviour this reproduces: the vehicle whose side the obstacle is
    on moves aside and the partner holds its lane, so the pair narrows rather
    than one of them yielding. Over 46 obstacle runs the blocked vehicle held
    a median 0.33 m of body clearance, tightest quartile 0.20 m.
    """
    cross = np.atleast_1d(np.asarray(obstacle_cross, float))
    cross = cross[np.isfinite(cross)]
    if not len(cross):
        return float(lane_offset)
    need = float(half_width) + float(clearance)

    # Only an obstacle may *cause* the target to move. Walls come in through
    # `veto_cross`, which can rule a candidate out but never sets one off.
    #
    # Putting walls in as ordinary obstacles was a mistake worth recording:
    # the outer lane runs within 0.28 m of a wall in places, and `need` is
    # 0.375, so the corridor itself kept displacing the target. Measured over
    # two runs, the target was displaced on 57-64% of ticks with no obstacle
    # anywhere in the window on about a third of those, and the pair went
    # into file for the corridor rather than for anything in its way. A cone
    # is something to go around; a wall is the edge of the road.
    if not (np.abs(cross - lane_offset) < need).any():
        return float(lane_offset)

    veto = np.atleast_1d(np.asarray(veto_cross, float)) \
        if veto_cross is not None else np.zeros(0)
    veto = veto[np.isfinite(veto)]
    veto_need = float(half_width) + float(veto_clearance)

    # Candidate targets: just clear of each obstacle, either side. Clearance
    # is then checked against *every* obstacle, not only the ones that blocked
    # the lane -- an obstacle that leaves the lane alone can still block the
    # place the vehicle is being moved to, and checking only the blockers put
    # the target 0.275 m from an obstacle it had ignored.
    options = np.concatenate([cross - need, cross + need])
    ok = np.all(np.abs(options[:, None] - cross[None, :]) >= need - 1e-9,
                axis=1)
    if len(veto):
        ok &= np.all(
            np.abs(options[:, None] - veto[None, :]) >= veto_need - 1e-9,
            axis=1)
    # A target has to be reachable. Without this bound the search chains
    # exclusions between obstacles and lands somewhere absurd: measured, it
    # asked for a 1.27 m shift, the vehicle tracked it with 0.65 m of error
    # and drove through the cone on the way. Beyond `max_shift` the honest
    # answer is to hold the lane and let the hard clearance constraint slow
    # the vehicle, rather than to promise a swerve it cannot make. The
    # recorded drivers moved 0.55-1.00 m at a cone, crossing to the
    # centreline or past it, so the bound has to admit that much: an earlier
    # 0.60 m would have refused the behaviour being reproduced.
    ok &= np.abs(options - lane_offset) <= max_shift

    # How far past the centreline a vehicle may go to get round an obstacle.
    # Measured at the instant of the pass, the blocked driver sat within
    # 0.12 m of the centreline in all six cases -- but with its body still
    # overlapping the cone's cross-track band, clearing the mapped cells by
    # 0.00-0.34 m. So the drivers passed with essentially no lateral margin,
    # which is not a target for a vehicle that has to do this repeatably: to
    # hold a real margin it must go further into the partner's half than they
    # did, and this bound is set to allow that rather than to reproduce their
    # centimetres.
    #
    # It is a bound and not a hard limit for a reason: the pair is in file
    # while this happens, staggered by 1.2-1.55 m, so lateral overlap is not
    # a collision. What keeps them apart is the along-track target below.
    if past_centre is not None and lane_offset != 0.0:
        side = np.sign(lane_offset)
        ok &= side * options >= -abs(past_centre)

    if not ok.any():
        return float(lane_offset)

    # A target is no use if the gap it sits in is narrower than the vehicle
    # can hold. Clearing every obstacle by `need` leaves a free interval
    # around each candidate, and that interval has to be wide enough to
    # absorb tracking error, not merely wide enough for the body.
    #
    # Without this the rule threads needles. Measured on a real run, it aimed
    # the inner vehicle at -1.47 -- the gap between a cone pair and the wall,
    # a viable window 0.12 m wide -- which it could not hold: it wedged with
    # 0.13 m of wall clearance, the target flipped to the far side too late,
    # and the vehicle sat spinning for seven seconds with zero object
    # clearance. Crossing into the partner's half was available the whole
    # time and is what the recorded drivers did.
    if min_slack > 0.0 and len(options):
        bounds = np.concatenate([cross, veto]) if len(veto) else cross
        slack = np.empty(len(options))
        for i, option in enumerate(options):
            below = bounds[bounds <= option]
            above = bounds[bounds > option]
            low = below.max() + need if len(below) else -np.inf
            high = above.min() - need if len(above) else np.inf
            slack[i] = high - low
        ok &= slack >= min_slack

    if not ok.any():
        return float(lane_offset)

    viable = options[ok]

    # Prefer a target the vehicle can actually reach without crossing the
    # obstacle to get there. A lateral target says where to end up, not how to
    # get there, and going round an obstacle means passing it longitudinally
    # first -- so "clear once you arrive" is not the same as reachable. In a
    # closed-loop run the rule chose a far-side target at +0.876 while the
    # vehicle sat at +0.489 with the obstacle starting at +0.41: the route
    # there went straight through the cone, and by the time the constraint
    # could see it, no rollout in the set avoided the obstacle at all. The
    # recorded drivers took the near side every time.
    if current is not None and len(viable) > 1:
        lo = np.minimum(current, viable)
        hi = np.maximum(current, viable)
        crosses = np.array([np.any((cross > l) & (cross < h))
                            for l, h in zip(lo, hi)])
        if (~crosses).any():
            viable = viable[~crosses]

    # Nearest to the lane, and on a tie the side closer to the centreline.
    # The tie is not hypothetical: an obstacle sitting exactly on the lane
    # makes both sides equidistant, and leaving that to floating point let
    # the target flip between them from one tick to the next. Inward is also
    # the better default -- outward is the wall, and the recorded drivers
    # moved inward in five of six encounters.
    cost = np.abs(viable - lane_offset) + 1e-6 * np.abs(viable)
    return float(viable[int(np.argmin(cost))])


#: The vehicle body in base_link as (x_min, x_max, half_width), metres. Same
#: rectangle the safety limiter and the footprint publisher use: 0.90 m long,
#: 0.55 m wide, with base_link 0.35 m behind the front face.
FOOTPRINT_BOX = (-0.35, 0.55, 0.275)


#: Two discs per body, covering the rectangle exactly along its length and
#: 1.25 cm wide of it: centres at these offsets along base_link, this radius.
BODY_DISCS = ((0.2625, -0.0625), 0.2875)


def pair_gap(own, partner, steps: int | None = None, discs=BODY_DISCS):
    """Smallest body-to-body gap over the horizon, per pair. Negative = overlap.

    Same disc model as `pair_feasible`, but returning the margin rather than a
    verdict, so a caller with no feasible pair can still move the two apart
    instead of giving up. Shape (n_own, n_partner).
    """
    offsets, radius = discs

    def centres(state):
        x = state["x"][:, :steps] if steps else state["x"]
        y = state["y"][:, :steps] if steps else state["y"]
        yaw = state["yaw"][:, :steps] if steps else state["yaw"]
        cos, sin = np.cos(yaw), np.sin(yaw)
        return [(x + d * cos, y + d * sin) for d in offsets]

    best = None
    for ox, oy in centres(own):
        for px, py in centres(partner):
            dx = ox[:, None, :] - px[None, :, :]
            dy = oy[:, None, :] - py[None, :, :]
            gap = np.sqrt(dx * dx + dy * dy).min(axis=2) - 2.0 * radius
            best = gap if best is None else np.minimum(best, gap)
    return best


def pair_feasible(own, partner, steps: int | None = None,
                  discs=BODY_DISCS, margin: float = 0.0):
    """Which candidate PAIRS keep the two bodies apart, shape (n_own, n_par).

    Nothing else in the planner stops the two vehicles occupying the same
    space. Walls and objects are filtered, the speed floor is enforced, and
    the partner is handled only by the formation terms -- which are soft, so
    the pair can be driven through each other by any term that outweighs
    them. That is not a formation problem, it is a collision.

    It belongs in the bimatrix rather than in `admissible`, and that is the
    point of putting it here. Whether a pair of trajectories collides is a
    property of the pair alone: both vehicles compute the same answer from the
    same observed state, so both remove the same entries and the two matrices
    stay transposes of each other. A filter applied to the ego's own rollouts
    could not do that -- it would need to guess which candidate the partner
    picks, and guessing differently is exactly how the two agents end up
    solving different games.

    A centre-distance threshold will not serve. Two of these bodies side by side
    need 0.55 m of lateral room and end to end need 0.90 m, so any single
    distance is either too slack in one configuration or forbids the other --
    and the recorded drivers passed obstacles with their bodies overlapping
    laterally while staggered 1.2-1.55 m, which a 0.90 m threshold would
    refuse. So the test is per-pose and geometric: two discs per body, and a
    pair fails if any disc pair ever touches.
    """
    offsets, radius = discs
    reach = 2.0 * radius + margin

    def centres(state):
        x = state["x"][:, :steps] if steps else state["x"]
        y = state["y"][:, :steps] if steps else state["y"]
        yaw = state["yaw"][:, :steps] if steps else state["yaw"]
        cos, sin = np.cos(yaw), np.sin(yaw)
        return [(x + d * cos, y + d * sin) for d in offsets]

    own_discs = centres(own)
    par_discs = centres(partner)

    hit = None
    for ox, oy in own_discs:
        for px, py in par_discs:
            dx = ox[:, None, :] - px[None, :, :]
            dy = oy[:, None, :] - py[None, :, :]
            touch = (dx * dx + dy * dy) < reach * reach
            close = touch.any(axis=2)
            hit = close if hit is None else (hit | close)
    return ~hit


def best_recovery(state, cfgs, generate, path_clearance, end_clearance,
                  here, *, tolerance=0.05, resolution=0.05, min_speed=0.05,
                  unstick=False):
    """The move that most improves clearance, forwards or backwards, or None.

    Extracted from the planner node so that an offline harness can ask the
    question against the *shipped* rule rather than a reimplementation of it.
    The caller supplies `generate` and the two clearance functions, which is
    what keeps the k-d trees and the ROS node out of here.

    Scored on where the move *ends*, not on the worst point along it: every
    rollout starts at the pose the vehicle is already in, so when that pose is
    the tight one -- the only time recovery runs -- the path minimum is
    essentially the standing clearance and "strictly better than standing
    still" is unsatisfiable. The path minimum is kept as a filter, so a
    candidate may not penetrate on the way and may not go deeper when the body
    is already inside something.

    `min_speed` is the fix for a recovery that recovers nothing. A candidate
    only had to beat `here + 1e-3`, and a near-stationary rotation reorients
    the footprint enough to clear that by a fraction of a millimetre:
    measured at the wall wedge, recovery issued 0.00-0.03 m/s rotations for
    133 s, never moved, never improved its standing clearance, and spent its
    whole 60-tick budget doing it -- after which recovery stops being
    attempted and the stall is permanent.

    The bar is the commanded *speed*, not the distance the rollout covers.
    Horizon displacement was tried first and does not work: at 0.03 m/s the
    3 s rollout travels 0.09 m and passes any sane distance test, while only
    the first 0.1 s is ever executed, so the vehicle actually moves 3 mm
    before replanning and choosing another 3 mm. `min_speed` defaults to the
    same 0.05 m/s the planner already calls moving, so recovery has to pick a
    genuine forward or a genuine reverse rather than a crawl.

    This deliberately does *not* touch the improvement threshold, which is
    what `escape_unstick` loosened and was rejected for.
    """
    def terminal(rollouts):
        return {k: rollouts[k][:, -1:] for k in ('x', 'y', 'yaw')}

    floor = min(0.0, here)
    if here < resolution:
        floor -= tolerance
    best, best_end = None, (floor if (unstick and here < resolution)
                            else here + 1e-3)
    for cfg in cfgs:
        commands, rollouts = generate(state, cfg)
        room = path_clearance(rollouts)
        end = end_clearance(terminal(rollouts))
        moves = np.abs(commands[:, 0]) >= min_speed
        viable = (room >= floor) & moves
        if not viable.any():
            continue
        pick = int(np.argmax(np.where(viable, end, -np.inf)))
        if end[pick] > best_end:
            best_end = float(end[pick])
            best = (float(commands[pick, 0]), float(commands[pick, 1]))
    return best


#: Candidate selection for `footprint_clearance`: outline sample spacing (m),
#: nearest points per sample, and kd-tree query threads. See its docstring.
OUTLINE_SPACING = 0.30
OUTLINE_K = 3
QUERY_WORKERS = 4
#: What `outline=None` means. The planner sets it from its `exact_clearance`
#: parameter; False restores the base_link-only selection everywhere.
OUTLINE_ENABLED = True


def _outline(box, spacing):
    """Body-frame points every <= `spacing` around the footprint rectangle."""
    back, front, half = box
    nx = int(np.ceil((front - back) / spacing)) + 1
    ny = int(np.ceil(2.0 * half / spacing)) + 1
    xs = np.linspace(back, front, nx)
    ys = np.linspace(-half, half, ny)
    return np.array([(x, -half) for x in xs] + [(x, half) for x in xs]
                    + [(back, y) for y in ys[1:-1]]
                    + [(front, y) for y in ys[1:-1]])


def footprint_clearance(own, tree, steps: int | None = None,
                        box=FOOTPRINT_BOX, neighbours: int = 4,
                        per_step: bool = False, outline: bool | None = None):
    """Distance from the vehicle *body* to the nearest object, per rollout.

    `wall_clearance` measures from base_link, which is a point. base_link sits
    0.35 m behind the front face, so the body reaches 0.55 m ahead of the
    measured point: an object 0.45 m in front of base_link -- passing a 0.45 m
    clearance test -- is already 0.10 m inside the vehicle. Measured on a
    logged run, the inner vehicle came within 0.15 m of a detected object
    against a 0.45 m threshold, which is not a threshold being approached but
    a collision being permitted.

    So this transforms the nearest object points into the vehicle frame at
    each pose and returns the exact point-to-rectangle distance, which is
    direction-dependent in the way the real constraint is: passing a cone
    abeam needs 0.275 m plus margin, meeting it head-on needs 0.55 m plus
    margin, and one number on the centre cannot express both.

    Which points are checked is the part that matters. This used to take
    only the `neighbours` points nearest *base_link*, on the argument that
    detected objects arrive as dense clusters -- and then walls were moved onto
    it. Along a wall the four cells nearest base_link are all on that wall, so
    a second wall the nose is pushing into at a corner was never examined:
    measured, it reported +178 mm for a body 46-64 mm inside the wall, missed
    13 contacts over 25 mm in one 24-run arm, and fed every safety check in
    the planner. A pair of cones
    has the same shape of failure.

    So candidates come from points every `OUTLINE_SPACING` around the body
    outline, `OUTLINE_K` nearest each, plus the `neighbours` nearest
    base_link as before -- a superset, so this can never report more
    clearance than the old rule. Any point d from the body is within
    d + spacing/2 of some outline sample, so the error is bounded by half the
    spacing; with three per sample it was never more than 23 mm optimistic
    and missed no contact against brute force over 24,643 logged poses
    within 0.3 m of a wall. The queries run on `QUERY_WORKERS` threads:
    single-threaded it cost 9x the old rule, which a 100 ms tick cannot pay.

    `outline=False` keeps the old base_link-only selection, for estimates that
    are not safety checks and are called on many poses: the corridor-room
    bands, the foresight and centring preferences, and the RViz fan. With the
    outline everywhere the tick went to 253 ms median, 400 ms p90, and a
    vehicle planning on 0.4 s-old states hit a wall by 273 mm.

    Walls use this too, as of the run that measured why the vehicles were
    hitting them. The old argument for keeping walls on the point measure was
    that the recorded centre came within 0.13 m of a mapped cell, so the body
    must already have been overlapping and a body-based threshold would make
    the recorded driving infeasible. Measuring the recorded paths directly
    against this map refutes it: body clearance reached 0.003 m and 0.001 m
    for the two vehicles and ran below 0.05 m for 7-8% of the lap, but was
    never negative. The drivers held zero overlap with millimetres to spare,
    so zero overlap is the constraint -- and a 0.15 m *centre* threshold,
    against a 0.275 m half-width, had been permitting 0.125 m of it.
    """
    if tree is None:
        return np.full(len(own["x"]), np.inf)
    xs = own["x"][:, :steps] if steps else own["x"]
    ys = own["y"][:, :steps] if steps else own["y"]
    yaws = own["yaw"][:, :steps] if steps else own["yaw"]
    flat = np.column_stack([xs.ravel(), ys.ravel()])

    cos = np.cos(yaws).ravel()[:, None]
    sin = np.sin(yaws).ravel()[:, None]

    count = min(neighbours, tree.n)
    index = tree.query(flat, k=count, workers=QUERY_WORKERS)[1]
    index = np.asarray(index).reshape(len(flat), -1)
    if outline is None:
        outline = OUTLINE_ENABLED
    if outline:
        body = _outline(box, OUTLINE_SPACING)             # (m, 2), body frame
        ox = flat[:, 0:1] + body[None, :, 0] * cos - body[None, :, 1] * sin
        oy = flat[:, 1:2] + body[None, :, 0] * sin + body[None, :, 1] * cos
        k = min(OUTLINE_K, tree.n)
        around = tree.query(np.column_stack([ox.ravel(), oy.ravel()]), k=k,
                            workers=QUERY_WORKERS)[1]
        index = np.concatenate(
            [index, np.asarray(around).reshape(len(flat), -1)], axis=1)
    near = np.asarray(tree.data)[index]                  # (poses, k, 2)
    dx = near[:, :, 0] - flat[:, 0:1]
    dy = near[:, :, 1] - flat[:, 1:2]
    along = dx * cos + dy * sin
    across = -dx * sin + dy * cos

    # Signed: negative when the point is *inside* the footprint, and equal to
    # how far inside. Clamping both gaps at zero -- as this did -- collapses
    # every depth of penetration onto exactly 0.0, so a constraint of
    # "clearance >= 0" was satisfied by a body sitting deep inside a wall.
    # That silently disabled the wall check when the margin was set to zero.
    # Outside the rectangle the two agree exactly, so nothing else changes.
    x_min, x_max, half = box
    gap_along = np.maximum(x_min - along, along - x_max)
    gap_across = np.abs(across) - half
    outside = np.hypot(np.maximum(gap_along, 0.0), np.maximum(gap_across, 0.0))
    inside = np.minimum(np.maximum(gap_along, gap_across), 0.0)
    distance = (outside + inside).min(axis=1)
    # `per_step` returns the (poses, steps) array instead of the per-rollout
    # minimum. The drawing needs to know *where* a rollout first breaches, not
    # merely that it does.
    if per_step:
        return distance.reshape(xs.shape)
    return distance.reshape(xs.shape).min(axis=1)


def wall_clearance(own, wall_tree, steps: int | None = None):
    """Minimum distance to a wall over the first `steps` of each rollout."""
    if wall_tree is None:
        return np.full(len(own["x"]), np.inf)
    horizon = own["x"][:, :steps] if steps else own["x"]
    vertical = own["y"][:, :steps] if steps else own["y"]
    flat = np.column_stack([horizon.ravel(), vertical.ravel()])
    return wall_tree.query(flat)[0].reshape(horizon.shape).min(axis=1)


def stopping_distance(v: float, tau_v: float, a_max: float) -> float:
    """How far the vehicle travels before it can be stopped from `v`.

    The lag term is not optional: the command grid is a first-order response
    with tau_v = 0.41 s, so the vehicle keeps going for roughly v * tau_v
    before deceleration takes hold, on top of the v^2 / 2a it then needs.

    Which deceleration to pass depends on what is being asked. For a speed
    *cap* it is the comfortable one, `d_max`: the cap returns the fastest
    speed whose stopping distance fits the gap, so a larger value there lets
    the vehicle run closer to things. For "could this be stopped if it had
    to", it is `d_stop`.

    At 1.05 m/s the two differ substantially -- 1.35 m at d_max = 0.60 and
    0.80 m at d_stop = 1.50 -- and two thirds of the first figure is the
    v^2/2a term rather than the lag, which is why the deceleration matters
    more here than the response time does.
    """
    return v * tau_v + v * v / (2.0 * a_max)


def wall_speed_cap(clearance, reach, need: float, tau_v: float, a_max: float,
                   v_max: float) -> float:
    """Fastest speed that can still slow to fit the corridor ahead.

    `clearance` is the body-to-obstacle clearance at poses along the route,
    `reach` their arc distance from the vehicle, and `need` the clearance the
    vehicle actually requires -- which is the cross-track error it is
    carrying, not a constant. Returns the largest v whose `stopping_distance`
    fits inside the distance to the first pose too tight to hold it.

    This exists because the hard wall mask cannot prevent a stop. It checks
    `clearance_steps` = 5 steps, or 0.5 s, which is 0.53 m at 1.05 m/s, while
    stopping from that speed takes 1.35 m. Preview scales as 0.5 v and the
    requirement as 0.41 v + 0.83 v^2, so they only meet below 0.11 m/s: at
    every speed the vehicle actually uses, it can enter a situation the mask
    will later have no admissible way out of. Measured, that is exactly what
    happened -- v* = 1.05 one tick and v = 0 the next, body at 0.001 m from
    the wall, with the nearest cone 7.55 m away and the speed floor dropped.

    Keying `need` on the carried error rather than on a fixed margin was
    settled offline against all four lane files: with a zero margin the cap
    never fires anywhere (on its lane the body always fits), while a constant
    0.19 m throttles 18% of the ccw outer lap. Keyed on the error, a
    well-tracking vehicle (median 0.07-0.08 m) is capped over 2-6% of the lap
    and the median cap stays at v_max on every lane.
    """
    tight = np.nonzero(np.asarray(clearance) < need)[0]
    if not len(tight):
        return v_max
    distance = max(float(np.asarray(reach)[tight[0]]), 0.0)
    # invert stopping_distance: a(-tau + sqrt(tau^2 + 2d/a))
    v = a_max * (-tau_v + np.sqrt(tau_v * tau_v + 2.0 * distance / a_max))
    return float(min(v, v_max))


def braking_tail(own, tau_v: float, a_max: float, samples: int = 8):
    """The poses a plan would sweep while coming to a stop from its end.

    Straight ahead on the final heading, because that is the only place
    stopping distance means anything.

    The first version of this test compared `footprint_clearance` against
    `stopping_distance` and was hopelessly strict: that clearance is to the
    nearest obstacle in *any* direction, so in a corridor it reported that the
    vehicle could not stop before a wall it was merely driving past. Measured,
    it held wall contact to 0.6 mm and cost everything else -- 42 stall
    episodes over six runs, progress 0.88 -> 0.11. A wall beside the vehicle
    needs no stopping distance; only what lies in the path does.

    Appending the tail to the plan and requiring the whole sweep to clear is
    the standard argument for keeping a receding-horizon planner out of
    inevitable-collision states: if the plan's end state is stoppable and the
    planner replans at 10 Hz, every state it enters remains stoppable.
    """
    x = np.asarray(own["x"])[:, -1]
    y = np.asarray(own["y"])[:, -1]
    yaw = np.asarray(own["yaw"])[:, -1]
    v = np.asarray(own["v"])[:, -1]
    reach = (v * tau_v + v * v / (2.0 * a_max))[:, None]
    step = np.linspace(1.0 / samples, 1.0, samples)[None, :]
    return {"x": x[:, None] + np.cos(yaw)[:, None] * reach * step,
            "y": y[:, None] + np.sin(yaw)[:, None] * reach * step,
            "yaw": np.repeat(yaw[:, None], samples, axis=1)}


def can_stop(own, tree, tau_v: float, a_max: float, box=FOOTPRINT_BOX):
    """Whether each plan could still be brought to a stop without hitting.

    The test the fixed-window wall mask was missing. A 0.5 s window inspects
    0.53 m at 1.05 m/s while stopping takes 1.35 m, so the vehicle could be
    committed long before anything was rejected. Measured at the ccw window
    site over the approach to a 78 mm contact: the 0.5 s mask called all 120
    candidates safe until 0.7 s before impact and none after, while a check
    that reaches a stopping distance ahead separates them 2.2 s out with 74 of
    100 still clear. A constraint that never tells good from bad until nothing
    is good cannot steer anything.
    """
    if tree is None:
        return np.ones(len(np.asarray(own["x"])), dtype=bool)
    return footprint_clearance(braking_tail(own, tau_v, a_max),
                               tree, None, box) >= 0.0


def least_impact(own, tree, box=FOOTPRINT_BOX, steps: int | None = None):
    """Index of the plan that hits hardest last, for when none can stop.

    The fallback matters as much as the constraint. `feasible` and
    `admissible` never return an empty set: when nothing clears they hand back
    the **roomiest** candidate, and that is a trap -- a stricter constraint
    then empties the mask sooner, the fallback fires more often, and the
    candidate it returns is deeper into the obstacle than anything the looser
    planner would have chosen. Measured: inflating the map by 0.05 m to buy
    margin took the worst wall contact from 38 mm to 196 mm, by exactly that
    route.

    So when there is no way to stop in time, rank by the speed the body would
    carry into the obstacle rather than by room. A plan that touches at
    0.1 m/s is not the same event as one that touches at 1.1 m/s, and the
    roomiest plan is frequently the faster of the two.
    """
    room = footprint_clearance(own, tree, steps, box, per_step=True)
    speed = np.asarray(own["v"])[:, :room.shape[1]]
    hit = room < 0.0
    # speed at the first penetrating step; plans that never penetrate score 0
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), 0)
    impact = np.where(hit.any(axis=1), speed[np.arange(len(speed)), first], 0.0)
    # ties on impact speed go to the plan with more room
    worst = room.min(axis=1)
    return int(np.lexsort((-worst, impact))[0])


def feasible(own, wall_tree, min_clearance: float = 0.0,
             steps: int | None = 5, footprint=None,
             tau_v: float | None = None, a_max: float | None = None):
    """Reject rollouts that come too close to a wall *soon*.

    Measured from the vehicle *body*, like objects, not from base_link. The
    point measure was the bug behind "the robots crash into the walls": with
    a half-width of 0.275 m, requiring 0.15 m from base_link permits the body
    to sit 0.125 m inside a wall, so the constraint was satisfied at the
    moment of the collision. A clean run recorded 0.02 m of centre clearance,
    which is the body a quarter of a metre into the wall. `footprint_clearance`
    already made this argument for objects; walls were left on the old measure.

    The default margin is zero: the body may not be inside a mapped cell.
    That is the drivers' own limit, not a guess. Measured on the recorded
    pair against this map, body clearance reached 0.003 m and 0.001 m and ran
    below 0.05 m for 7-8% of the lap, below 0.15 m for 17-23%, and never went
    negative. Any positive margin therefore rejects a slice of the humans'
    own driving -- 0.05 m rejects a twelfth of it -- and in closed loop a
    0.05 m margin deadlocked the ccw run outright: the vehicle stopped at
    0.056 m because no candidate held 0.05, and `v_min = 0` leaves no action
    that recovers from a standstill.

    Two further choices, both learned from watching this fail:

    Checked over a prefix, not the whole horizon. Applying a hard constraint
    3 s out is the mistake the ROS 1 safety limiter made: over ~3 m of travel
    at a corner every candidate clips something, so the set empties and the
    vehicle stops in clear space. Only the first step is ever executed and the
    planner replans at 10 Hz, so the near future is what must be safe.

    Never returns an all-false mask: if nothing clears, the caller gets the
    single roomiest candidate. Creeping along the best available option beats
    stopping dead in front of a wall -- and it is what keeps `admissible`'s
    last resort meaningful, since that ranks by object clearance among the
    wall-safe set.
    """
    box = FOOTPRINT_BOX if footprint is None else footprint
    clearance = footprint_clearance(own, wall_tree, steps, box)
    ok = clearance >= min_clearance
    braking = tau_v is not None and a_max is not None and wall_tree is not None
    if braking:
        # Not a wider margin: the plan must end somewhere it can still stop.
        ok = ok & can_stop(own, wall_tree, tau_v, a_max, box)
    if not ok.any():
        ok = np.zeros(len(clearance), dtype=bool)
        # Nothing can stop in time. Rank by impact speed, not by room: the
        # roomiest plan is often the faster one into the obstacle.
        ok[least_impact(own, wall_tree, box, steps) if braking
           else int(np.argmax(clearance))] = True
    return ok


def free_bands(x, y, yaw, trees, clearance, box=FOOTPRINT_BOX,
               span: float = 2.0, step: float = 0.05):
    """Lateral positions a body can occupy here, as a list of (lo, hi).

    Measured from what the vehicle can see, not from a file. The offsets are
    of the body *centre*, signed left-positive about the given heading, and a
    band is a contiguous run of them that clears everything in `trees` by
    `clearance`.

    One vectorised clearance call over every sampled offset, because this runs
    every tick: 81 offsets at 0.05 m over +/-2 m is a fraction of the rollout
    fan the planner already evaluates.
    """
    offsets = np.arange(-span, span + step / 2, step)
    xs = x - offsets * math.sin(yaw)
    ys = y + offsets * math.cos(yaw)
    pose = {"x": xs[:, None], "y": ys[:, None],
            "yaw": np.full_like(xs, yaw)[:, None]}
    room = None
    for tree in trees:
        if tree is None:
            continue
        # An estimate of the corridor, not a safety check: see `outline`.
        gap = footprint_clearance(pose, tree, None, box, outline=False)
        room = gap if room is None else np.minimum(room, gap)
    if room is None:
        return [(float(offsets[0]), float(offsets[-1]))]
    ok = room >= clearance
    bands, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            bands.append((float(offsets[i]), float(offsets[j])))
            i = j + 1
        else:
            i += 1
    return bands


def formation_room(bands, need_pair: float, max_pair: float):
    """How the pair can hold formation here: side_by_side, straddling, or neither.

    `side_by_side` -- one band takes both bodies side by side.

    `straddle` -- no single band does, but two of them each take one body, so
    the pair passes with the obstacle *between* them and stays level. This is
    what the recorded drivers did at a cone standing in the middle of the
    aisle, and asking only the first question told the pair to fall into file
    there, which is the one thing that interrupts the conversation the
    formation exists for.

    Returned as the option and the separation it implies, so the caller can
    place the two vehicles; `None` separation means side by side at the usual
    target. The pair is placed as close as the bands allow, subject to the
    bodies not overlapping and the result still being a formation rather than
    two vehicles in separate corridors.
    """
    if any(hi - lo >= need_pair for lo, hi in bands):
        return "side_by_side", None
    best = None
    for a in range(len(bands)):
        for b in range(a + 1, len(bands)):
            lo_a, hi_a = bands[a]
            lo_b, hi_b = bands[b]
            left = min(hi_a, max(lo_a, lo_b - need_pair))
            right = max(lo_b, min(hi_b, left + need_pair))
            sep = right - left
            if need_pair <= sep <= max_pair and (best is None or sep < best[0]):
                best = (sep, left, right)
    if best is None:
        return "file", None
    return "straddle", best
