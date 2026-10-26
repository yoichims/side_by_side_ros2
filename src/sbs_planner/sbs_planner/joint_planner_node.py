#!/usr/bin/env python3
"""Joint MPC for one vehicle of a side-by-side pair.

Each cycle: sample this vehicle's rollouts, sample a *prediction* of the
partner's rollouts from the partner's observed state, score every pair, pick one
by the configured solution concept, and command the first segment of the choice.
Receding horizon -- only the first command is ever executed, then it replans.

Nothing about the partner's intent is received. The partner's pose and velocity
are what a driver could see; `/vN/odom` over wifi stands in for that perception
and can be replaced by a tracker without touching this node. There is no topic
carrying a plan, a chosen trajectory or a utility between vehicles, and if one
ever appears the design has leaked.

Solution concepts, all reductions of the same two matrices:

    independent   own terms only; the formation weights are zeroed
    anticipated   predict the partner by its own best row, then respond
    social        argmax over the pair of (u_own + u_partner)
    gamma         argmax over the pair of ((1-g) u_own + g u_partner)
    optimistic    argmax_i max_j u_own -- assumes the partner complements you
    maximin       argmax_i min_j u_own -- assumes the partner does its worst

Note that `gamma` at g=0 *is* `optimistic`: dropping the partner's utility
from the blend still leaves the reduction maximising over the partner's
column. Whose utility counts and how the partner's column is resolved are
independent choices, and only `anticipated` and `maximin` resolve that column
by anything other than a maximum.

Evaluator and selector live in one node deliberately: the swappable parts are
the cost function and the concept, both of which are modules and parameters
here. Splitting them across nodes would put ~1 MB of matrix on the wire at
10 Hz for no gain. CostMatrix is still published when `publish_matrix` is set,
because recording it is what an offline inverse-RL fit needs.
"""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Twist
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial import cKDTree
from sbs_msgs.msg import CostMatrix
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker

from sbs_planner.cost import (FOOTPRINT_BOX, ContextModel, PairContext,
                              Weights, admissible, evaluate,
                              footprint_clearance, free_bands, pair_gap,
                              shifted_lateral_target, stopping_distance,
                              wall_speed_cap)
from sbs_planner.reference import Reference
import sbs_planner.cost as cost_module
from sbs_planner.world import (cloud_points, grid_points, inflate,
                               unknown_points, wall_tree)
from sbs_planner.rollout import (RolloutConfig, VehicleModel, generate,
                                 integrate)
from dataclasses import replace

CONCEPTS = ("independent", "anticipated", "social", "gamma",
            "optimistic", "maximin")

#: Half a map cell. Clearance is measured to occupied cell *centres* on a
#: 0.05 m grid, so the true surface lies within this of the measured value and
#: a body reading 0.000 m may be touching or may be 25 mm clear. Nothing that
#: has to decide something at contact can be stricter than this and still mean
#: anything.
RESOLUTION = 0.025


def yaw_of(q) -> float:
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


class JointPlanner(Node):
    def __init__(self):
        super().__init__('joint_planner')
        p = self.declare_parameter

        model = VehicleModel(
            tau_v=p('tau_v', 0.41).value, tau_w=p('tau_w', 0.49).value,
            v_max=p('v_max', 1.30).value, w_max=p('w_max', 0.80).value,
            a_max=p('a_max', 0.60).value,
            d_max=p('d_max', 0.60).value,
            d_stop=p('d_stop', 1.50).value,
            lat_acc_max=p('lat_acc_max', 1.50).value)
        self.model_tau_v = float(model.tau_v)
        self.model_a_max = float(model.a_max)
        #: The deceleration the speed caps assume. Comfortable on purpose: a
        #: cap returns the fastest speed whose stopping distance fits the gap
        #: ahead, so assuming the emergency value here would let the vehicle
        #: run closer to obstacles rather than stop harder near them.
        self.model_d_max = float(model.d_max)
        #: The deceleration available when the alternative is a collision.
        #: Answers "could this be stopped", not "how fast may it go".
        self.model_d_stop = float(model.d_stop)
        #: Measured over 48 runs against a control differing only in this:
        #:
        #:     walls   3 contacts over 25 mm (worst 84 mm) -> 1 (worst 32 mm)
        #:     cones   8 contacts            (worst 162)   -> 7 (worst 154)
        #:
        #: So it works on walls and does almost nothing for cones, which are
        #: the larger failure. It is not free either: cw progress 0.88 -> 0.85,
        #: conversational time 63 -> 58%, and 0.08 stalls per run against none.
        #: Kept on because a wall contact is a collision and five points of
        #: conversational geometry is not, but it is a trade and not a win.
        #:
        #: Declared here because the rollout config needs it too; declaring
        #: a parameter twice raises ParameterAlreadyDeclaredException and the
        #: node dies at startup, which is how the first attempt was caught.
        self.braking_authority = p('braking_authority', True).value
        self.cfg = RolloutConfig(
            asymmetric_speed=self.braking_authority,
            horizon=p('horizon', 3.0).value, dt=p('dt', 0.1).value,
            segment_split=p('segment_split', 1.5).value,
            v_span=p('v_span', 0.45).value, w_span=p('w_span', 0.60).value,
            n_v=p('n_v', 7).value, n_w=p('n_w', 5).value,
            n_v2=p('n_v2', 2).value, n_w2=p('n_w2', 2).value, model=model)

        self.weights = Weights(
            rel_range_rate=p('w_rel_range_rate', 1.0).value,
            rel_dist_end=p('w_rel_dist_end', 1.0).value,
            rel_yaw_rate=p('w_rel_yaw_rate', 1.0).value,
            motion_accel=p('w_motion_accel', 1.0).value,
            motion_speed=p('w_motion_speed', 1.0).value,
            path_dist=p('w_path_dist', 1.0).value,
            path_progress=p('w_path_progress', 0.5).value,
            rel_along_track=p('w_rel_along_track', 1.0).value,
            rel_yaw=p('w_rel_yaw', 1.0).value,
            path_cross_track=p('w_path_cross_track', 0.0).value,
            path_lane_yaw=p('w_path_lane_yaw', 0.0).value,
            rel_angle=p('w_rel_angle', 0.0).value)
        #: `split` is the fraction of the geometric speed difference the
        #: outer vehicle is given through a bend. The corpus fits 0.326 --
        #: the drivers supplied only a third of it -- and measured here that
        #: leaves a systematic 0.126 m/s pulling the pair apart on every
        #: corner, against station keeping that already saturates trying to
        #: close it. 1.0 is rigid formation. Default is the fitted value,
        #: which is what every result before this was measured under.
        #: `speed_v0`: the straight-line value of the speed law v*(kappa). The
        #: fit (1.094) is to binned *medians* of speeds the WHILL governor
        #: capped at 1.24 m/s, so it is biased low: on straights the drivers
        #: spent 23% of the time at 1.2 m/s or more. Default is the governor.
        self.model = ContextModel(split=p('speed_split', 0.326).value,
                                  v0=p('speed_v0', 1.24).value)

        self.concept = p('concept', 'social').value
        if self.concept not in CONCEPTS:
            raise ValueError(f'concept must be one of {CONCEPTS}')
        self.gamma = p('gamma', 0.5).value
        self.side = p('side', 'outer').value
        self.partner_side = 'inner' if self.side == 'outer' else 'outer'
        #: Wall clearance required of the vehicle *body*, not of base_link.
        #: 0.15 m from the centre was the old meaning and permitted the body
        #: 0.125 m inside a wall, which is why the vehicles hit walls while
        #: satisfying the constraint.
        #:
        #: Zero, i.e. "no part of the body inside a mapped cell", is the
        #: drivers' own limit. Measured on the recorded pair against this map:
        #: body clearance reached 0.003 m and 0.001 m, and ran below 0.05 m
        #: for 7-8% of the lap and below 0.15 m for 17-23%, but never once
        #: went negative. A 0.05 m margin would reject a twelfth of the
        #: humans' own driving, and it deadlocked ccw -- the vehicle stopped
        #: at 0.056 m because no rollout kept 0.05, and with v_min = 0 there
        #: is no action that recovers from a standstill.
        #: Thickening applied to the map before the planner reasons about it,
        #: matching the 0.05 m the cones already get. Quantised to the grid,
        #: so on a 0.05 m map 0.05 is the smallest value that does anything.
        #: The planner alone would see the inflated walls: measurement and
        #: detection keep the surveyed geometry.
        #:
        #: **On at 0.10 m since 2026-10-03.** It was off for a year because
        #: 0.05 m measured worse, not better -- over six ccw runs the worst
        #: contact went 38 -> 196 mm, stalls 0 -> 3, conversational 77 -> 68%
        #: -- and the reason below is still the right description of *that*
        #: planner. Re-measured on the polar cost over 24 layout runs it no
        #: longer behaves that way: wall contact 26.5 -> 4.5 mm, cone contact
        #: 10.7 -> 6.8, stalls unchanged at 0.26 -> 0.29, heading p99 43.6 ->
        #: 37.8 deg. The mask evidently stops emptying once the formation is
        #: carried by two terms instead of five, so the fallback fires rarely
        #: enough for the margin to be worth having.
        #:
        #: Do *not* also raise the lane `MARGIN`. Measured together at 0.25 +
        #: 0.10 the two gain nothing over inflation alone (wall 4.56 vs 4.49
        #: mm) and heading p99 goes to 72.5 deg -- the spin signature --
        #: because inflation already consumes the standoff a wider margin
        #: would provide and stacking them leaves no sensible manoeuvre at a
        #: pinch.
        #:
        #: The original reasoning, still true of the five-term cost:
        #: The reason is the fallback. Neither `feasible` nor `admissible`
        #: ever returns an empty set -- when nothing clears they return the
        #: roomiest candidate, which is inside the obstacle by construction.
        #: Inflating narrows the corridor by 0.10 m, so the mask empties
        #: sooner and more often, the fallback fires, and the candidate it
        #: hands back is further into the real wall than anything the
        #: uninflated planner would have chosen. A stricter constraint with a
        #: permissive fallback is worse than a looser one.
        #:
        #: Worth revisiting only after the fallback stops permitting
        #: penetration -- at which point the margin has somewhere safe to go.
        self.wall_inflate = p('wall_inflate', 0.0).value
        #: Require every plan to end somewhere it could still be stopped
        #: (`can_stop`), with `least_impact` as the fallback when none can.
        #:
        #: **Off by default: measured, and it did not work.** The diagnosis it
        #: rests on is solid -- the 0.5 s wall window inspects 0.53 m at
        #: 1.05 m/s while stopping takes 1.35 m, and at the ccw window site it
        #: called all 120 candidates safe until 0.7 s before a 78 mm contact
        #: and none after, where a stopping-distance check separates them
        #: 2.2 s out with 74 of 100 still clear. Two formulations were tried
        #: over six ccw runs each:
        #:
        #:   omnidirectional clearance >= stopping_distance
        #:       worst contact 38 -> 0.6 mm, but 42 stall episodes and
        #:       progress 0.88 -> 0.11. Too strict by construction: that
        #:       clearance is to the nearest obstacle in any direction, so it
        #:       demands stopping room for walls the vehicle is driving past.
        #:
        #:   braking tail appended to each plan (the current code)
        #:       progress and formation intact, but worst contact 38 ->
        #:       137 mm. Stricter where it binds, so the mask empties more
        #:       often and the fallback -- even ranked by impact speed --
        #:       still permits deeper contact than not having the constraint.
        #:
        #: The pattern is the same one `wall_inflate` hit: a stricter mask on
        #: top of a fallback that must return *something* makes contact worse,
        #: not better. Revisit only with a fallback that can refuse to move.
        #: Weight on the foresight term: how strongly the cost prefers a plan
        #: that stays clear over its whole horizon. A preference, never a
        #: constraint -- see the note in `tick`. Zero disables it.
        #:
        #: It has to stay below progress, and that bound is the whole tuning.
        #: A plan that does not move never approaches anything, so its
        #: foresight penalty is zero: the term *rewards standing still* unless
        #: the reward for moving outweighs it. Progress contributes at most
        #: about 1.3 here, while 2.0 yields up to 2.0 * (0.35/0.25)^2 = 3.9 --
        #: and measured at 2.0 the pair stalled for 103 s in clear space, the
        #: penalty having made stopping the best available plan. At 0.5 the
        #: worst it can contribute is 0.98 and the ordering is right.
        self.w_foresight = p('w_foresight', 0.5).value
        #: Clearance above which foresight is silent. Below it the penalty
        #: grows as the square of the shortfall.
        self.foresight_clear = p('foresight_clear', 0.35).value
        self.foresight_scale = p('foresight_scale', 0.25).value
        self.brake_constraint = p('brake_constraint', False).value
        self.min_clearance = p('min_clearance', 0.0).value
        self.clearance_steps = p('clearance_steps', 5).value
        self.publish_matrix = p('publish_matrix', False).value
        #: Per-tick line separating "the constraint would not let me go
        #: faster" from "the cost chose to go slower". Off by default; it
        #: prints at the tick rate.
        self.diagnose = p('diagnose', False).value
        self.publish_fan = p('publish_fan', True).value
        self.fan_max = p('fan_max', 80).value
        # Per-vehicle, so two fans overlaid in RViz can be told apart.
        self.fan_color = p('fan_color', [0.15, 0.80, 1.0]).value
        rate = p('rate', 10.0).value
        self.period = 1.0 / rate
        self.v_min_fraction = p('v_min_fraction', 0.30).value
        # Margin between the vehicle's *outline* and a detected object --
        # `admissible` measures object clearance from the body, not from
        # base_link. The old 0.45 m was measured from base_link, which sits
        # 0.35 m behind the front face, so it permitted a cone 0.10 m inside
        # the front of the vehicle; a logged run reached 0.15 m. As a body
        # margin, 0.45 m would refuse to pass a cone 0.72 m off the lane
        # centre, so the number changes with the measurement it belongs to.
        #
        # 0.10 m, measured at the *instant of closest approach* rather than
        # as a median over an approach window: the drivers cleared the mapped
        # cells by 0.00-0.34 m as they passed, and the placement itself is
        # accurate to a few centimetres. The earlier 0.375 m came from a
        # windowed median that averaged in the approach, and demanding it made
        # every side obstacle unavoidable -- the required position fell beyond
        # the centreline, so the rule declined to move at all.
        self.object_clearance = p('object_clearance', 0.10).value
        # Objects are checked over the whole horizon, walls only over
        # `clearance_steps`. Measured in closed loop against the four cones on
        # this route, minimum body clearance was 0.000 m at 0.5 s (both
        # vehicles drove through them) and 0.20 m at 3.0 s, with lane error
        # unchanged at 0.18 m. 1.5 s is worse than either: the cone is seen too
        # late to steer around, so the vehicle stops instead and the inner one
        # spent 28% of the run stationary. A cone sits on the route and needs
        # to be seen early; a wall runs alongside it and must not be, which is
        # why these are two numbers and not one.
        self.object_steps = p('object_steps', self.cfg.n_steps).value
        #: How far ahead an obstacle may move the lateral target, as a TIME
        #: rather than a distance, so it scales with speed.
        #:
        #: What the manoeuvre needs is time, not metres: the yielding vehicle
        #: must shed one stagger length, and `file_speed_scale` buys about a
        #: 0.35 m/s deficit, so 1.45 / 0.35 = 4.1 s regardless of how fast the
        #: pair is travelling. A fixed distance is only right at one speed --
        #: 8 m is 8.9 s at 0.9 m/s and 16 s at 0.5 m/s, and the pair then
        #: commits absurdly early wherever it slows for a corner, which is
        #: exactly where it slows.
        #:
        #: Measured at a fixed 8 m the pair was in file at three of four
        #: obstacle sites and never re-formed between them, including at the
        #: centre obstacle it should have split around.
        self.object_horizon = p('object_horizon', 5.0).value
        #: Recovery, not driving. The recorded drivers never reversed --
        #: `rollout.py` sets `v_min = 0.0` for that reason -- and this does
        #: not change the driving envelope: reverse is sampled only on the
        #: ticks where nothing moving forward is admissible at all. Without
        #: it a standstill is absorbing: `admissible` sacrifices the speed
        #: floor first, `v = 0` becomes admissible, and no action in the grid
        #: recovers. Measured, a wedged vehicle stayed wedged for the rest of
        #: the run, and its partner then destroyed its own formation trying
        #: to hold station with it.
        #: Clearance the corridor must hold before a vehicle may be sped up
        #: to take a gap. The p90 cross-track error, measured over 27k moving
        #: samples: what a fast pass typically costs in tracking, as opposed
        #: to `object_clearance`, which is what a well-tracked vehicle needs.
        self.boost_clearance = p('boost_clearance', 0.17).value
        #: How far ahead the side-by-side flag is read, as a TIME, for the same
        #: reason `object_lookahead` is one: what the manoeuvre needs is time,
        #: not metres. Reaching the pinch in file means shedding a whole
        #: stagger first, and `file_speed_scale` buys about a 0.35 m/s
        #: deficit, so 1.45 / 0.35 = 4.1 s whatever the pair's speed.
        #:
        #: This was a fixed 2.6 m and that is what it cost. At 1.2 m/s it
        #: gives 2.2 s -- about half the manoeuvre -- so the gate fired, the
        #: pair began separating, and arrived at the pinch still side by side. The
        #: outer vehicle was then squeezed between its partner and the wall:
        #: measured over six ccw runs, separation collapsing 1.31 -> 0.89 m
        #: while the lateral target moved 0.25 m inward that the vehicle could
        #: not follow, ending 0.39 m off its lane and 49-78 mm inside the wall
        #: in four of them. Negative ignores the flag entirely, which is the
        #: control arm.
        self.side_by_side_lookahead = p('side_by_side_lookahead', 4.1).value
        #: Floor on the distance that time converts to. Not a small number:
        #: a pair accelerating from rest is slow precisely when it most needs
        #: warning, and `pace * side_by_side_lookahead` collapses to nothing there.
        #: Measured, the ccw pinch sits 4.7 m into the route, so on the first
        #: lap the pair meets it from a standing start -- and every wall
        #: contact at that site was on pass 1, with passes 2 and 3 clear once
        #: the pair was up to speed and the time-based reach was working. The
        #: floor therefore has to cover the manoeuvre at a plausible cruise,
        #: not at the speed of the moment.
        self.side_by_side_reach_min = p('side_by_side_reach_min', 4.0).value
        #: Measure the corridor's width at runtime as well as reading it from
        #: the profile, and let it cap the separation target.
        #:
        #: `fits_side_by_side` is derived offline from the map and one recorded set
        #: of objects, so it is blind to a cone that has moved since -- which
        #: is the whole case for moving obstacles. Measured over 32 layout
        #: runs, every wall contact deeper than the map's own resolution was
        #: the outer vehicle pressing outward to hold a separation target the
        #: corridor could not give, at sites the shipped profile still marks
        #: side by side. This reads the same question live, from the two trees the
        #: constraints already use.
        #:
        #: It enters exactly as the offline flag does -- as a *necessary*
        #: condition, AND-ed in -- so it can only ever withdraw the side by side
        #: option, never grant one the profile denied. False is the control
        #: arm and restores the previous behaviour exactly.
        self.runtime_room = p('runtime_room', True).value
        #: Snap the lateral target to free space actually measured ahead --
        #: see `_reachable_target`. False is the control arm.
        #:
        #: Measured over 48 runs, 4 layouts x 2 directions x 3 repeats:
        #: worst wall contact 22 -> 9 mm ccw and 43 -> 24 mm cw, progress
        #: 0.69 -> 0.77 and 0.81 -> 0.83, time stuck 22 -> 16 s and 5 -> 2 s,
        #: conversational 41 -> 47% and 52 -> 52%. Six of eight cells improve
        #: or hold.
        #:
        #: The site it was built for is fixed in both directions, and as tail
        #: removal rather than a mean shift: layout 05 ccw went
        #: 0.60 [0.46-0.86] to 0.87 [0.87-0.88] and cw 0.66 [0.34-0.82] to
        #: 0.82 [0.81-0.84], stalls to zero in both. The control sometimes
        #: finishes and sometimes collapses; this lands in a 0.03-wide band.
        #:
        #: One site is worse and is not understood: cw layout 07 goes
        #: 0.87 [0.86-0.88] to 0.78 [0.62-0.88], stalling in one run of three
        #: at (29.95, -0.55), wedged 5-9 mm from a wall with a cone 56-121 mm
        #: away. It is intermittent, the aggregate is clearly positive, and
        #: the one hypothesis tried -- that the snap parks the vehicle on a
        #: band edge that holds no margin -- was measured and made it worse.
        self.band_target = p('band_target', True).value
        #: Cap the speed target on the clearance along the path the vehicle is
        #: already committed to -- see `_heading_cap`. **Off: measured
        #: harmful, and the idea is wrong rather than mistuned.**
        #:
        #: On the surveyed ccw route it took progress 0.89 -> 0.28, wall
        #: contact 0 -> 275 mm and conversational time 68 -> 28%, with the
        #: pair 2.18 m off its lanes in the first six seconds.
        #:
        #: The arc it projects is the *current* command's, chosen over a
        #: straight line precisely so that a bend would not read as a wall
        #: ahead. That does not work: the command's curvature is not the
        #: route's, because the vehicle steers continuously, so the projection
        #: meets a wall nearly everywhere the corridor curves and the cap
        #: collapses the target to the speed floor. A crawling pair then fails
        #: in its own ways. This is the same failure as the omnidirectional
        #: clearance-versus-stopping-distance rule that produced 42 stalls,
        #: reached by a different route.
        #:
        #: What it was aimed at is real and still open: two of nine real wall
        #: contacts in a 48-run campaign were a vehicle off its lane heading
        #: at a wall, where the lane-based `_speed_cap` reads v_max because
        #: the lane ahead is clear. Answering that needs the clearance along
        #: the path the planner is about to *choose*, which is not available
        #: where the cap is computed.
        self.heading_cap = p('heading_cap', False).value
        # `braking_authority` is declared above, with the rollout config that
        # also reads it. It gates two things that are one mechanism: the speed
        # grid reaches `d_stop` of deceleration, and `_foresight` asks for the
        # candidate's own stopping distance rather than a fixed 0.35 m. The
        # threshold makes the planner want to slow a second earlier; the grid
        # gives it the authority to do so. Either alone is half of it.
        #: Weight on the *worse-off* vehicle's clearance, charged to both
        #: utilities -- see `_centring`. This is the one thing in the cost
        #: that asks the vehicle with room to be the one that yields.
        #:
        #: Two earlier formulations were measured harmful and withdrawn; this
        #: one is built on the obstacle clearance itself rather than on band
        #: edges or on what the far side had spare, because the diagnosis that
        #: motivates it is specific: at the site producing most object
        #: contacts, `band_target` snaps the blocked vehicle to the only open
        #: band's lower edge, which holds exactly the minimum margin, and its
        #: partner occupies the rest of that band. Raising the worse clearance
        #: is worth something to whichever vehicle can do it.
        #:
        #: Bounded like foresight: the shortfall cannot exceed
        #: `foresight_clear`, so at 0.5 the worst it contributes is 0.98
        #: against a progress reward of about 1.3.
        #: **Off. Three formulations, three withdrawals.**
        #:
        #:   1. shortfall scored against what the far side had spare. Gameable
        #:      -- destroying the spare zeroed the penalty -- and the pair flew
        #:      apart, 0.62 m to 3.56 m in four seconds, 273 mm into a wall.
        #:   2. margin to the free band's edge. Improved by compressing, so it
        #:      pushed the pair together rather than across; a stall in each
        #:      direction and 0.05 of progress.
        #:   3. the minimum of the two vehicles' obstacle clearances, charged
        #:      to both matrices -- built directly on the diagnosis. Over 47
        #:      runs it made cone contact *worse in both directions*, 9 -> 73
        #:      and 115 -> 152 mm, with 222 mm of wall contact in cw, progress
        #:      0.87 -> 0.82, and stalls where the control had none.
        #:
        #: Three different quantities, the same outcome. The conclusion is no
        #: longer about formulation: a pair cost term is the wrong instrument
        #: for this. It asks each vehicle to pay for a clearance it may be
        #: unable to influence, which distorts its choice without helping its
        #: partner, and it does so in the joint matrix where the two vehicles'
        #: reasoning is already coupled.
        #:
        #: The failure it aims at is real and documented in the paper's open
        #: issues. The better lead is upstream: `shifted_lateral_target` sends
        #: the blocked vehicle the *wrong way* -- measured -0.46 to -1.01 m
        #: while the only open band was [+0.30, +0.95] -- because it takes the
        #: option nearest the lane. `band_target` then rescues it onto the
        #: band's near edge, which holds exactly the minimum margin. Choosing
        #: the open side at the source, and aiming inside it rather than at
        #: its edge, addresses the same failure without a pair term.
        self.w_centre = p('w_centre', 0.0).value
        self.escape_speed = p('escape_speed', 0.25).value
        #: Ticks of no forward option before reversing, so a momentary
        #: collapse of the candidate set does not lurch the vehicle backwards.
        self.escape_after = p('escape_after', 5).value
        #: Cap on one *futile* escape, in ticks. The budget resets whenever
        #: the recovery actually gains clearance, so a manoeuvre that is
        #: working is never cut off -- as a flat cap did: a vehicle backing
        #: out of a tight corner exhausted 20 ticks, gave up, and then stood
        #: still for 103 s while 0.004 m clear of the wall. What the cap is
        #: for is the case where reversing changes nothing, and that case is
        #: still bounded.
        self.escape_max = p('escape_max', 60).value
        #: How far below its standing clearance a recovery candidate may dip,
        #: and only when that standing clearance is itself inside the map's
        #: resolution -- see `_recover`. Zero restores the behaviour that
        #: refused every candidate 225 times at a median standing clearance of
        #: +0.000 m, which is the control arm for measuring this.
        #:
        #: On, and the history is worth keeping. It was a clear win in cw
        #: (stalls 1.29 -> 0.58 over 47 runs) and worse on every ccw aggregate,
        #: all of that one layout repeatably -- because with the tolerance the
        #: vehicle crept into the pinch at (20.94, 15.77) where the lateral
        #: target aimed outside the free space, and without it the strict
        #: filter was an accidental guard against that bug. So it was defaulted
        #: off pending the target fix, and re-measured after it rather than
        #: simply switched on.
        #:
        #: Measured with `band_target` on, over 48 runs on layouts that are
        #: passable *and* reachable: ccw stalls 0.17 -> 0.00 per run, time
        #: stuck 7 -> 0 s, progress 0.82 -> 0.90, conversational 49 -> 54%;
        #: cw time stuck 4 -> 2 s, progress 0.84 -> 0.86, conversational
        #: 58 -> 62%. The ccw penalty is gone, which is what the trap being
        #: fixed predicted.
        #:
        #: **And then off again, on layouts whose gate also checks that a
        #: passable band is reachable.** Over 48 such runs the two arms are
        #: identical on stalls (0.00 ccw, 0.08 cw), progress (0.90, 0.88) and
        #: conversational time -- and differ only in contact:
        #:
        #:     off   1 contact over 25 mm in 24 runs, worst  26 mm
        #:     on    5 contacts             in 24 runs, worst 130 mm
        #:
        #: The deep one is the recovery itself reversing into a wall at
        #: -0.22 m/s for 33 ticks, with no object anywhere near: the tolerance
        #: lets a recovery candidate dip one cell below its standing
        #: clearance, and tracking error turns that into 130 mm.
        #:
        #: Nothing about the tolerance changed; what changed is that there is
        #: no longer anything for it to buy. It rescued a recovery that could
        #: not extract a wedged vehicle, and with fair layouts and
        #: `band_target` the vehicles no longer wedge -- the control arm
        #: already stalls 0.00 times per ccw run. A parameter that was right
        #: while a failure existed is not right once the failure is gone, and
        #: its cost is paid either way.
        self.escape_tolerance = p('escape_tolerance', 0.0).value
        #: Compare escape candidates against the floor rather than against a
        #: 1 mm improvement, where the standing pose is within one map cell of
        #: a surface and the comparison is therefore below the resolution.
        #:
        #: **Measured and rejected, 2026-10-06.** The diagnosis it rests on is
        #: correct: escape refused 939 times over 24 layout runs at a median
        #: standing clearance of 0.002 m, asking for an improvement 25x finer
        #: than clearance can be measured on a 0.05 m grid. Relaxing it cut
        #: the refusals to 51. But stalls went 0.17 -> 0.38 per run
        #: (+0.21 +/- 0.15) and the zero-contact result was lost: cone 0.29 ->
        #: 1.80 mm, wall 0.00 -> 0.53 mm, conv 50.0 -> 46.9, filing
        #: 17.8 -> 24.1.
        #:
        #: Fewer refusals, more stall *episodes*, because the escape now fires
        #: and the move it finds is not actually better -- at 2 mm of
        #: clearance there is no good candidate to find, and relaxing what
        #: counts as acceptable does not create one. The problem is that the
        #: pair reaches such poses at all, not how it recovers from them.
        #: This is the third recovery-ladder change measured and reverted.
        self.escape_unstick = p('escape_unstick', False).value
        #: Publish a stop instead of a chosen candidate that is not genuinely
        #: clear of walls and objects over the executed prefix. False
        #: reproduces the 48-run baseline, which executed the least-bad
        #: candidate in that case and put the body 42-103 mm into a wall.
        self.stop_when_blocked = p('stop_when_blocked', True).value
        #: Seconds between the state a command is computed from and the
        #: moment it acts, added to the one tick `_brake_clear` executes
        #: before braking. 0.0 is arm `stop_brake/`, which stopped 0.17 s
        #: late and put the body 57 mm into a wall at 1.0 m/s.
        self.brake_latency = p('brake_latency', 0.2).value
        #: Stretch the brake test's execution time to twice the longest of the
        #: last 20 tick intervals when that exceeds dt + `brake_latency`. A
        #: command acts from the end of its tick to the end of the next, so
        #: an overrunning planner commits for longer than it assumed: the old
        #: planner overran 100 ms on 35% of ticks (p90 121 ms) and its 94 mm
        #: contact was a stop decided ~0.3 s late (hold/layout_07_h_ccw_r2).
        self.adaptive_latency = p('adaptive_latency', True).value
        self.tick_intervals = []
        self.state_age = 0.0
        self.last_tick_start = None
        #: When stuck, reverse along this vehicle's own recorded path for
        #: `backup_distance` metres, then rejoin below `rejoin_speed` for
        #: `rejoin_time` seconds. False is arm `stop_latency/`: zero contact in
        #: 24 runs, but 5 runs ended stopped inside the wall's safety band,
        #: where every forward move fails the brake test.
        self.backup_retrace = p('backup_retrace', True).value
        self.backup_distance = p('backup_distance', 0.8).value
        self.rejoin_speed = p('rejoin_speed', 0.4).value
        self.rejoin_time = p('rejoin_time', 4.0).value
        #: Seconds a detected object point is remembered after it was last
        #: seen. 0.0 keeps only the current scan, which is blind behind the
        #: vehicle. See `_objects`.
        #: Body clearance from candidates around the whole outline (True) or
        #: only those nearest base_link (False, every arm before 2026-10-22).
        #: The base_link rule missed corners.
        cost_module.OUTLINE_ENABLED = p('exact_clearance', True).value
        #: Ticks between recomputations of the runtime corridor room; 1 is
        #: every tick, as before 2026-10-22.
        self.room_every = max(1, int(p('room_every', 3).value))
        self.room_cached = None
        self.room_age = 0
        self.object_memory = p('object_memory', 30.0).value
        self.object_seen = {}
        self.objects_at = None      # monotonic time of the last objects message
        #: The vehicle ahead along the route stops when its partner is more
        #: than `hold_gap` metres away, and resumes under `resume_gap` (the
        #: drivers' p99 and p90 separation among cones), or after `hold_max`
        #: seconds. False is arm `backup/`.
        self.partner_hold = p('partner_hold', True).value
        self.hold_gap = p('hold_gap', 3.0).value
        self.resume_gap = p('resume_gap', 2.2).value
        self.hold_max = p('hold_max', 60.0).value
        #: The leader starts a hold only with no object within this many
        #: metres of it. Holding as soon as the gap passed 3 m parked it at the
        #: exit of the gap the yielding partner needed (layout_07 ccw, yieldw
        #: r2), and the partner went round the other side into the pocket.
        self.hold_clear = p('hold_clear', 2.5).value
        self.hold_ticks = 0
        self.hold_spent = False
        #: Filing, the vehicle whose lane is blocked stops short of the
        #: obstacle until its partner is `file_stagger` ahead, instead of
        #: creeping on. The speed floor (30% of the preferred speed) forbids
        #: waiting through the cost: at the layout_07 ccw cones the blocked
        #: vehicle crept at 0.21 m/s -- the floor -- while its partner was
        #: still alongside, and the only moving plans led round the cones
        #: into the pocket by the pillar. A real stop, like the leader hold
        #: (a-zero-speed-cap-does-nothing). Released when the partner is far
        #: enough ahead, when side by side fits again, or after `yield_max`.
        self.yield_wait = p('yield_wait', False).value
        self.yield_distance = p('yield_distance', 3.0).value
        self.yield_max = p('yield_max', 20.0).value
        self.yield_ticks = 0
        #: Remove candidates whose plan goes more than `band_margin` outside
        #: the band between the two lanes (the drivers' p99 near cones), or
        #: further out than the vehicle already is. False is arm `hold/`.
        self.lane_band = p('lane_band', True).value
        self.band_margin = p('band_margin', 0.70).value
        #: Along-track offset to hold when the pair cannot fit side by side.
        #: 1.45 m is the median the drivers used at the instant of the pass
        #: (measured 1.20-1.55 m); the bodies are 0.90 m long, so this is a
        #: genuine file with clear air between them rather than a lean.
        #: Along-track offset the pair takes to pass an obstacle. Refitted
        #: from **180 recorded passes** -- median 1.62 m, mean 1.56, p25-p75
        #: 1.01-2.06 -- where the previous 1.45 came from six observations.
        #: The drivers also closed up laterally as they passed, to 0.73 m.
        self.file_stagger = p('file_stagger', 1.60).value
        #: Along-track offset asked for while *side by side*. 0.0 is the
        #: behaviour every result so far was measured under.
        self.sbs_stagger = p('sbs_stagger', 0.0).value
        #: Use the measured per-formation separation constants instead of
        #: the pooled object-distance law. False is what every result
        #: before 2026-09-29 was measured under.
        self.formation_separation = p('formation_separation', False).value
        #: Floor on the window, so an obstacle cannot slip out of view when
        #: the vehicle is barely moving -- which is when it most needs to see
        #: what is in front of it. One file length is the natural size: below
        #: that the pair could not complete the manoeuvre anyway.
        #: Shortest distance ahead the lateral target looks for objects. Was
        #: `file_stagger` (1.6 m), so as the pair slowed for a blocked lane
        #: the warning shrank to 1.6-2.5 m, too late to file: at the layout_07
        #: ccw cones the blocked vehicle could not drop behind and went round
        #: the cones into the pocket by the pillar. Filing needs ~4 m
        #: (file-manoeuvre-needs-time-and-authority), the floor the
        #: side-by-side lookahead already uses. Default 1.6 is the old rule.
        self.object_reach_min = p('object_reach_min', self.file_stagger).value
        #: Horizon for the body-to-body check between the two vehicles, in
        #: steps. A prefix rather than the whole horizon, for the reason
        #: `feasible` gives for walls: a hard constraint applied 3 s out
        #: empties the set, and both vehicles replan at 10 Hz so the near
        #: future is what has to be safe. 1.0 s covers roughly a body length
        #: of closing at the speeds these vehicles reach.
        #: Speed target multiplier for the vehicle that yields while the
        #: pair passes in file. It has to fall about one stagger length
        #: behind over the approach -- 1.45 m in the ~4 s the lookahead
        #: gives -- so it needs roughly a 0.35 m/s deficit, which is what
        #: this buys against a ~1.05 m/s target.
        #: Impose that the vehicle whose lane is blocked takes the position
        #: behind. On by default because it is what the recorded drivers did;
        #: `yield_when_blocked:=false` leaves the ordering to the cost, which
        #: prefers the blocked vehicle to lead whenever the inner lane is the
        #: one blocked. Neither is more correct -- the first reproduces the
        #: Impose a pass order at an obstacle: the vehicle whose lane is
        #: blocked takes the position behind. **Off by default, because the
        #: drivers did not do it.** Over 148 recorded passes the blocked
        #: vehicle led 51% of the time -- a coin flip -- and the order was
        #: simply inherited: 67% of passes kept the order the pair had 5 m
        #: earlier, rising to 78% where one was already clearly ahead.
        #:
        #: They did not need a rule because they were already in file. Their
        #: median stagger at the pass is 1.62 m over 180 passes, and about
        #: three quarters exceed 1.0 m, at which point the geometry has
        #: decided who goes first. This pair reaches 1.10 m and 72%, close
        #: enough that the same thing happens.
        #:
        #: The rule was not wrong to exist -- with the old `file_stagger` of
        #: 1.45 m, removing it brought back a 66.6 mm wall contact. It was
        #: compensating for not filing as far as the drivers did. Refitting
        #: the stagger removed the need: with 1.60 m and no rule, the worst
        #: contact is 9.8 mm and obstacle clearance triples, +0.030 ->
        #: +0.091 m. True remains available and buys 12 points of
        #: conversational geometry (78% against 66% in ccw).
        self.yield_when_blocked = p('yield_when_blocked', False).value
        self.file_speed_scale = p('file_speed_scale', 0.65).value
        #: Along-track error at which station keeping saturates. It used to
        #: borrow `file_stagger`, 1.60 m -- the target offset of a *file*
        #: manoeuvre -- as the normaliser for *side-by-side* station keeping.
        #: Measured, the offsets that actually occur are 0.5-0.8 m, so the
        #: correction ran at a third of its authority and the vehicle ahead
        #: slowed by 10-17% when it could have slowed by 35%. Zero keeps the
        #: old behaviour, which is what every result before this was measured
        #: under.
        self.station_gain = p('station_gain', 0.0).value
        #: Whether station keeping may drive a vehicle *above* the speed the
        #: fitted corpus law allows for the local curvature. False keeps the
        #: pair inside the drivers' own envelope: the stagger is closed by
        #: slowing the leader rather than by speeding the follower.
        self.boost_above_preferred = p('boost_above_preferred', False).value
        self.pair_steps = p('pair_steps', 10).value
        #: Guaranteed gap between the two bodies. The constraint holds them
        #: exactly here when squeezed, so this parameter *is* the minimum
        #: body-to-body clearance: measured in closed loop, 0.05 m left 2 cm
        #: of overlap on 1.2% of ticks and 0.12 m left none, at a cost of
        #: about 1 m of progress in 118 and no change in tick time.
        self.pair_margin = p('pair_margin', 0.12).value
        #: Log a per-phase breakdown of the tick, for when it overruns.
        self.profile = p('profile', False).value

        self.reference = Reference.load(p('reference', '').value)
        self.get_logger().info(
            f'concept={self.concept} side={self.side} '
            f'horizon={self.cfg.horizon}s reference={len(self.reference.centre)} pts')

        # Last answer from side_sign_at, per vehicle: the hysteresis that
        # stops the outer/inner speed split chattering lives in the caller,
        # so `Reference` stays a pure lookup.
        self.sign_own = 0.0
        self.sign_partner = 0.0

        self.odom = None
        self.partner_odom = None
        self.pose = None
        #: Same rollouts, with reverse allowed. Built once; used only while
        #: escaping.
        self.escape_cfg = replace(
            self.cfg, model=replace(self.cfg.model,
                                    v_min=-abs(self.escape_speed)))
        self.stuck_ticks = 0
        self.escape_ticks = 0
        self.escape_best = -np.inf
        self.partner_tree = None
        #: Where this vehicle has driven forward, newest last: the one region
        #: known to be free, because its own body just passed through it.
        self.trail = []
        #: Metres still to reverse along `trail`, or None when not backing up.
        self.backup_left = None
        self.backup_ticks = 0
        self.rejoin_ticks = 0
        self.last_stuck = None
        self.backup_now = 0.0
        self.wall_tree = None
        #: Cells the map has never observed. Blocking for the free-band
        #: measurement only -- see `_corridor_room` -- never for clearance.
        self.unknown_tree = None
        #: Fixed geometry arrives from two places and both feed `wall_tree`:
        #: the surveyed map, and a latched `structure` cloud for the aisle.
        self.map_points = None
        self.structure_points = None
        self.object_tree = None

        # Depth 1: plan from the newest odometry, never from a backlog. With a
        # queue of 10 at 40 Hz the single-threaded executor handed each tick
        # the *oldest* queued message: measured 2026-10-22, the state a tick
        # planned from was 230 ms old at the median and 251 ms at most --
        # a full queue -- which is why stops kept arriving ~0.2-0.5 s late.
        # With `state_thread`, the state subscriptions get their own callback
        # group and `main` runs a two-thread executor, so fresh odometry is
        # taken in *during* a tick instead of after it. Depth 1 alone brought
        # the state a tick plans from to 22 ms old at the median, but a tick
        # that overran still left the next one starting from 170-260 ms-old
        # state, and a stop decided that late put a body 54 mm into a wall
        # (arxiv_layouts/layout_07_a_ccw_r3). Objects go in the same group:
        # left in the default group with the timer, they were starved -- the
        # two-thread executor kept picking the always-ready timer, and the
        # planner received an objects message every ~10 s instead of every
        # scan (median age 3-6 s, max 30 s). It drove through cones it had
        # never been sent: 11 runs inside a cone, against none before this
        # executor. `_objects`
        # alone touches `object_seen`, and the tick reads `object_tree`,
        # which is swapped in by one assignment. The map stays in the
        # default group.
        self.state_thread = p('state_thread', True).value
        group = (MutuallyExclusiveCallbackGroup() if self.state_thread
                 else None)
        self.create_subscription(Odometry, 'odom', self._own_odom, 1,
                                 callback_group=group)
        self.create_subscription(Odometry, 'partner_odom', self._partner_odom,
                                 1, callback_group=group)
        self.create_subscription(PoseStamped, 'pose', self._pose, 10,
                                 callback_group=group)
        # `objects_starved` puts it back in the default group, to watch the
        # bug; never for an evaluation.
        starved = p('objects_starved', False).value
        self.create_subscription(PointCloud, 'objects', self._objects, 1,
                                 callback_group=None if starved else group)
        self.create_subscription(
            OccupancyGrid, 'map', self._map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(
            PointCloud, 'structure', self._structure,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        # The planner's own speed target, and the two things that shape it.
        # The metrics node used to recompute `v_star` from the model, which
        # gave the *unscaled* value: it knew nothing of the station-keeping
        # scale or the wall speed cap, so a log showing v* = 1.05 while the
        # planner was asking for 1.35 or being held to 0.6 was unreadable,
        # and it blocked the diagnosis of why a trailing vehicle could not
        # catch up. Published rather than mirrored, because three mirrors of
        # planner logic have drifted in this codebase already.
        self.debug_pub = self.create_publisher(
            Float32MultiArray, 'plan_debug', 1)
        self.plan_pub = self.create_publisher(Path, 'chosen_plan', 1)
        self.fan_pub = self.create_publisher(Marker, 'rollout_fan', 1)
        self.matrix_pub = (self.create_publisher(CostMatrix, 'cost_matrix', 1)
                           if self.publish_matrix else None)
        self.create_timer(1.0 / rate, self.tick)

    # --- inputs -----------------------------------------------------------
    def _own_odom(self, msg): self.odom = msg
    def _partner_odom(self, msg): self.partner_odom = msg
    def _pose(self, msg): self.pose = msg

    def _objects(self, msg):
        self.objects_at = time.monotonic()
        points = np.array([[q.x, q.y] for q in msg.points], float)
        if self.object_memory <= 0.0:
            self.object_tree = cKDTree(points) if len(points) else None
            return
        # Remember what has been seen. The laser covers +/-135 degrees, so a
        # cone behind the vehicle is absent from the current scan, and every
        # check made while reversing passed it as clear: a backup drove 60 mm
        # into one (backup/layout_07_k_ccw_r1). Cones do not move, and the
        # detector already masks the partner, so points are kept on a 5 cm
        # grid for `object_memory` seconds after they were last seen.
        now = time.monotonic()
        for x, y in points:
            self.object_seen[(round(x / 0.05), round(y / 0.05))] = (x, y, now)
        stale = [k for k, (_, _, t) in self.object_seen.items()
                 if now - t > self.object_memory]
        for k in stale:
            del self.object_seen[k]
        kept = np.array([(x, y) for x, y, _ in self.object_seen.values()])
        self.object_tree = cKDTree(kept) if len(kept) else None

    def _map(self, msg):
        """Wall cells as a k-d tree, for the feasibility mask.

        Thickened by `wall_inflate` before the planner uses it, the same way
        the cones are thickened before they are published. The map is a grid
        and clearance is measured to cell *centres*, so a body flush against
        the measured points is already up to half a diagonal into the wall;
        the planner should be keeping away from the wall, not from the
        samples of it. The metrics node and the detector deliberately do not
        do this -- see `world.inflate`.
        """
        points = grid_points(msg)
        if not len(points):
            return
        # Bounded to the route's own extent. Unbounded this call builds a
        # tree over 15.3 million cells, takes 3.5 s, and does it inside this
        # callback -- see `unknown_points`.
        bounds = None
        if self.reference is not None:
            centre = self.reference.centre
            margin = self._ROOM_SPAN + 1.0
            bounds = (centre[:, 0].min() - margin, centre[:, 1].min() - margin,
                      centre[:, 0].max() + margin, centre[:, 1].max() + margin)
        blind = unknown_points(msg, bounds)
        self.unknown_tree = cKDTree(blind) if len(blind) else None
        raw = len(points)
        self.map_points = inflate(points, self.wall_inflate,
                                  msg.info.resolution)
        self._rebuild_walls()
        self.get_logger().info(
            'map: %d occupied cells, %d after %.2f m inflation, '
            '%d never observed near the route'
            % (raw, len(self.map_points), self.wall_inflate, len(blind)))

    def _structure(self, msg):
        """Fixed geometry the survey lacks: the aisle, published latched.

        Merged into the wall notion, not the object one. An object causes a
        lateral target to move; structure may only veto one. Entered as an
        object, aisle walls would have the pair swerving around the aisle.
        """
        self.structure_points = cloud_points(msg)
        self._rebuild_walls()
        self.get_logger().info(
            'structure: %d cells' % len(self.structure_points))

    def _rebuild_walls(self):
        self.wall_tree = wall_tree(self.map_points, self.structure_points)


    def _state(self, odom, pose=None):
        source = pose.pose if pose is not None else odom.pose.pose
        return (source.position.x, source.position.y, yaw_of(source.orientation),
                odom.twist.twist.linear.x, odom.twist.twist.angular.z)

    def _lane_blocked(self, state, side):
        """Is there an object in this vehicle's own lane, ahead?

        Asked directly against the lane rather than inferred from how far the
        lateral target moved. That inference was a proxy, and it stopped
        discriminating once the offsets became a profile: both vehicles shift
        for the same cone, `moved_own` and `moved_partner` land within 0.05 m
        of each other, and the yield decision falls through to a fixed
        convention -- after which whichever vehicle happens to be slightly
        ahead takes the gap, whatever lane the cone is really in.
        """
        if self.object_tree is None:
            return False
        points = np.asarray(self.object_tree.data)
        if not len(points):
            return False
        dx = points[:, 0] - state[0]
        dy = points[:, 1] - state[1]
        along = dx * np.cos(state[2]) + dy * np.sin(state[2])
        reach = max(abs(state[3]) * self.object_horizon,
                    self.object_reach_min)
        ahead = points[(along > 0.0) & (along < reach)]
        if not len(ahead):
            return False
        cross = self.reference.signed_cross_track(ahead[:, 0:1],
                                                  ahead[:, 1:2])
        lane = self.reference.lane_offset_at(state[:2], side)
        return bool(np.any(np.abs(np.asarray(cross).ravel() - lane)
                           < FOOTPRINT_BOX[2] + self.object_clearance))

    def _heading_cap(self, state, v_floor):
        """Speed the vehicle's own committed path still allows it to stop from.

        `_speed_cap` asks when the *corridor* gets too tight for the error the
        vehicle is carrying, sampled along its lane. That cannot see a vehicle
        which has left its lane and is heading across the corridor: the lane
        ahead is clear, so the cap reads `v_max` while the body closes on a
        wall. Measured in a 48-run campaign, two of the nine real wall
        contacts were exactly that -- one accelerating 0.44 -> 0.98 m/s while
        its wall clearance fell 1.60 -> 0.24 m, ending 105 mm in. At 0.98 m/s
        stopping takes 1.20 m, so it committed to the approach at the moment
        it could no longer stop, and by then every candidate entered the wall,
        which saturates `_foresight` and leaves progress to decide.

        So this asks the other question: hold the current command, and how far
        can the body go before it touches something? Projected along the arc
        the vehicle is *already on* -- constant v and w -- rather than straight
        ahead. Straight would hit the outer wall on every bend and throttle
        the whole lap, which is how the omnidirectional version of this idea
        failed: a clearance-versus-stopping-distance rule measured that way
        produced 42 stalls and progress 0.11.

        A cap on the target, never a mask on the candidates, for the reason
        the rest of this file keeps returning to: a mask can empty and a cap
        cannot. It is also floored at `v_floor`, so it can never ask for a
        speed the `moving` filter forbids.
        """
        if self.wall_tree is None or not self.heading_cap:
            return float('inf')
        v, w = abs(state[3]), state[4]
        # Nothing to decide below the floor, and a stationary vehicle has no
        # committed path to project.
        if v < max(v_floor, 0.05):
            return float('inf')
        span = stopping_distance(self.cfg.v_max, self.model_tau_v,
                                 self.model_d_stop)
        step = 0.05
        arc = np.arange(step, span + step, step)
        # Constant-curvature projection of the pose, the path the current
        # command is already committed to.
        kappa = w / max(v, 1e-6)
        yaw = state[2] + kappa * arc
        if abs(kappa) < 1e-6:
            xs = state[0] + arc * math.cos(state[2])
            ys = state[1] + arc * math.sin(state[2])
        else:
            radius = 1.0 / kappa
            xs = state[0] + radius * (np.sin(yaw) - math.sin(state[2]))
            ys = state[1] - radius * (np.cos(yaw) - math.cos(state[2]))
        pose = {'x': xs[:, None], 'y': ys[:, None], 'yaw': yaw[:, None]}
        room = footprint_clearance(pose, self.wall_tree, None, FOOTPRINT_BOX)
        touch = np.nonzero(room < 0.0)[0]
        if not len(touch):
            return float('inf')
        # The body's own front reach is excluded for the same reason
        # `_speed_cap` excludes it: a contact already inside the footprint is
        # not something slowing down can help.
        distance = max(float(arc[touch[0]]) - FOOTPRINT_BOX[1], 0.0)
        cap = self.model_d_stop * (
            -self.model_tau_v
            + math.sqrt(self.model_tau_v ** 2
                        + 2.0 * distance / self.model_d_stop))
        return max(cap, v_floor)

    def _speed_cap(self, state, side, cross_target, v_floor,
                   boosting=False):
        """Speed the corridor ahead still allows this vehicle to slow from.

        The clearance required is the cross-track error the vehicle is
        carrying: on its lane the body always fits, so a fixed margin either
        never fires or throttles the whole lap. A vehicle tracking well is
        barely capped; one drifting off-lane -- which is how it reaches a
        wall -- slows in the tight places.

        `boosting` is the exception, and it is the case this cap was missing.
        A vehicle being sped up to take a gap has not yet made the error its
        new speed will produce, so the cap sees a well-tracked vehicle, asks
        only for `object_clearance`, and does not limit. Measured: the file
        rule boosted a vehicle to 1.30 m/s into a stretch whose best body
        clearance is 0.111 m; the cap read 1.30 for three seconds, dropped to
        0.64 only once the vehicle was already 0.40 m off its target, and it
        wedged against the wall 1.2 s later. While boosting, the corridor is
        therefore asked to hold the error a fast pass actually produces --
        the measured p90 cross-track error, not the current one.

        Looked ahead exactly as far as it can matter: the stopping distance
        at v_max. A pinch beyond that does not constrain the speed now, so
        there is nothing to gain by searching further and no lookahead
        parameter to choose.

        Two bounds keep the cap from causing the failure it prevents. The
        near field is excluded -- a pinch within the body's own front reach
        is not something to slow for, since the vehicle is already in it and
        only the hard mask can help. And the cap never falls below the speed
        floor `admissible` enforces: below it the cost would be asking for a
        speed the `moving` filter forbids, and that contradiction is resolved
        by dropping the floor, which is how the vehicle came to stop dead in
        the first place.
        """
        if self.wall_tree is None:
            return float('inf')
        vehicle = self.cfg.model
        look = stopping_distance(vehicle.v_max, vehicle.tau_v, vehicle.d_max)
        here = float(self.reference.signed_cross_track(
            np.array([[state[0]]]), np.array([[state[1]]]))[0])
        need = max(abs(here - cross_target), self.object_clearance)
        if boosting:
            need = max(need, self.boost_clearance)

        xs, ys, yaws, reach = self.reference.lane_ahead(
            state[:2], side, look)
        beyond = reach >= FOOTPRINT_BOX[1]
        if not beyond.any():
            return float('inf')
        xs, ys, yaws, reach = (xs[beyond], ys[beyond], yaws[beyond],
                               reach[beyond])
        pose = {'x': xs[:, None], 'y': ys[:, None], 'yaw': yaws[:, None]}
        # Walls only, deliberately. Objects were included first, on the
        # reasoning that they narrow the same corridor, and it cost the inner
        # vehicle 34% of one run stopped in front of a cone -- stops with an
        # object in range, not a wall in range. Lateral displacement is
        # bought with forward travel, so capping the speed on approach to a
        # cone withdraws the motion `shifted_lateral_target` needs to steer
        # around it, and the vehicle creeps into the one place it cannot
        # leave. Avoiding objects is the moved target's job; not arriving at
        # a wall too fast to turn is this one's. The offline sweep of all
        # four lane files agrees: against walls the cap fires on ccw outer --
        # the lane that fails -- and on nothing else.
        clearance = footprint_clearance(pose, self.wall_tree, None,
                                        FOOTPRINT_BOX)
        cap = wall_speed_cap(clearance, reach, need, vehicle.tau_v,
                             vehicle.d_max, vehicle.v_max)

        # NOTE: capping this by the curvature ahead was tried and reverted.
        # It did stop the file rule's boost -- peak speed in the first five
        # seconds fell 1.29 -> 1.07 m/s -- but the 25 degree heading
        # transient it was aimed at barely moved (25.0 -> 24.0), so the
        # transient is not caused by entering the bend fast. It cost about
        # 10% of progress (0.92 -> 0.83 m/s) and 3 points of conversational
        # geometry for wall overlap 0.35% -> 0.13%. The heading tail at the
        # first ccw bend still has an unidentified cause.
        return max(cap, float(v_floor))

    def _foresight(self, rollouts):
        """Penalty for plans whose own body comes close to something later.

        One-sided: a plan that keeps `foresight_clear` throughout costs
        nothing, so this is silent in open corridor and only speaks where a
        trajectory is heading somewhere tight. Squared in the shortfall, like
        every other term, so the preference sharpens as the margin goes.
        """
        # A preference, not a safety check: the cheap selection (`outline`).
        room = self._worst_clearance(rollouts, None, None, outline=False)
        # The margin asked for is the candidate's own stopping distance, not a
        # constant. `foresight_clear` is a *distance*, and at 0.93 m/s it is
        # 0.38 s of warning against a 0.41 s actuation lag -- so it arrives
        # after the vehicle is committed. Traced on the 146 mm cone
        # penetration: clearance fell 0.78 -> 0.43 -> 0.02 m while the vehicle
        # accelerated 0.53 -> 0.73 -> 0.93 m/s, and the fixed threshold first
        # spoke at 17 mm. Judged against the stopping distance it speaks at
        # 0.43 m, a full second earlier, while 68 of 140 candidates were still
        # admissible and something could still be chosen.
        #
        # `foresight_clear` stays as the floor, so in slow or open driving
        # this asks for exactly what it asked for before.
        if self.braking_authority:
            speed = np.max(np.abs(rollouts['v']), axis=1)
            need = np.maximum(
                stopping_distance(speed, self.model_tau_v, self.model_d_stop),
                self.foresight_clear)
        else:
            need = np.full(len(room), self.foresight_clear)
        # Normalised by `need`, so the ceiling is unchanged: a full shortfall
        # costs exactly what a full shortfall of `foresight_clear` used to.
        # This widens *when* the term speaks, never how loudly -- raising the
        # weight is a separate question, and 2.0 once caused a 103 s stall
        # because a stationary plan has no foresight penalty at all.
        short = np.maximum(0.0, need - room)
        ratio = short / need
        return self.w_foresight * (
            ratio * self.foresight_clear / self.foresight_scale) ** 2

    #: Sampling of the corridor-width measurement. `_ROOM_STEP` along the
    #: route is under half the width of a cone group, so an object cannot
    #: fall between two samples; `_ROOM_SPAN` either side of the centreline
    #: covers any corridor the pair can hold formation in; `_ROOM_RES` is the
    #: map's own cell size, below which the answer is not meaningful anyway.
    #: Measured at 0.14 ms per sample, so the whole lookahead costs about
    #: 3 ms against a 100 ms tick.
    #: Same bound `shifted_lateral_target` uses: beyond this the vehicle
    #: cannot track the swerve and promising it drives through the obstacle.
    _MAX_SHIFT = 1.00
    #: How well the pair holds a lateral target, and so how far inside a free
    #: band to aim rather than sitting on its lip.
    _TRACK = 0.08
    _ROOM_STEP = 0.20
    _ROOM_SPAN = 1.75
    _ROOM_RES = 0.05

    def _corridor_room(self, midpoint, reach):
        """Widest the pair can be placed apart, over the route ahead.

        The same question `fits_side_by_side` answers from the profile, asked of
        what the vehicles can actually see -- so a cone that has moved since
        the profile was derived narrows the corridor here too. Returns metres
        between body *centres*, minimised over the corridor within `reach`,
        together with the free band the pair is riding in right now. Infinity
        and None when there is nothing to measure against.

        Measured as the span from the first free lateral band to the last,
        which is the whole corridor where an object splits it and the band
        width where it does not. Both readings are what the manoeuvre needs.
        Where a cone group stands in the middle of the aisle -- which is every
        object site on this route -- the drivers passed with one vehicle each
        side, still level, so the span across the group is available and this
        does not bind. Where the corridor itself is narrow, or a cone stands
        against one wall leaving a single band, the span is that band and the
        pair cannot be as wide as it would like.

        Two things it deliberately is not. Not a test on the two lateral
        targets: that was tried at runtime and read 0.33 m at a site whose
        free band is 0.78 m wide, because both targets shift for the same
        cone, where a band is geometry and does not move when the pair does.
        And not a reading at the pair's current offset: requiring the
        placement to bracket where the pair happens to be riding reported
        0.00 m at every cone site, because the pair rides down the middle and
        the middle is where the cones are.
        """
        # Unknown cells bound the bands. They are not obstacles and the
        # clearance constraints never see them, but free space cannot be
        # inferred from the absence of an occupied cell: unexplored map has no
        # occupied cells and so reads as perfectly clear. Measured, 30.2% of
        # the 3 m around the site where a vehicle pinned is unobserved, and
        # the band drawn through it made the corridor look wider than anyone
        # has ever seen it -- which is exactly when `separation_cap` stops
        # binding and the side-by-side gate stops withdrawing.
        trees = [tree for tree in (self.wall_tree, self.object_tree,
                                   self.unknown_tree) if tree is not None]
        if not trees or self.reference is None:
            return float('inf'), None
        xs, ys, yaws, _ = self.reference.centre_ahead(
            midpoint, reach, self._ROOM_STEP)
        # Where the pair is riding, as a signed offset of the centreline.
        # Shaped (1, 1): `signed_cross_track` takes rollouts and means over
        # the horizon axis, so a flat array is an error, not a point. Taken at
        # the midpoint so both vehicles read the same corridor.
        here = float(self.reference.signed_cross_track(
            np.array([[midpoint[0]]]), np.array([[midpoint[1]]]))[0])
        worst, band = float('inf'), None
        for i, (x, y, yaw) in enumerate(zip(xs, ys, yaws)):
            bands = free_bands(float(x), float(y), float(yaw), trees,
                               self.object_clearance, FOOTPRINT_BOX,
                               span=self._ROOM_SPAN, step=self._ROOM_RES)
            # The span from the first free band to the last, *minus the
            # blocked gaps between them*. The span alone is what a straddle
            # needs -- one vehicle each side of a central cone group, which
            # is what the drivers did -- but it is only available once each
            # vehicle is already in a different band. It is not available
            # during the transit into that arrangement, and nothing in the
            # cost makes the band assignment, so the pair was being told it
            # could be as wide as the straddle while still crossing the gap.
            # Measured, that is where every object contact deeper than 100 mm
            # happens: the blocked vehicle holds a stable cross-track target
            # for five seconds and drives through it instead, because
            # `sep_target` rises from 1.60 to 1.81 m as the corridor pinches
            # and its partner, pinned by a cone with 0.12 m of clearance,
            # cannot supply any of the deficit.
            #
            # Summing the bands charges for the gap: where the corridor is
            # one clear band this is unchanged, and where a cone splits it
            # the cap falls to what the pair can actually use without
            # straddling. The straddle itself is gated separately, by
            # `straddles()` on the offline profile, which knows the
            # assignment this measurement cannot.
            if bands:
                usable = sum(hi - lo for lo, hi in bands)
            else:
                usable = 0.0
            worst = min(worst, usable)
            if i == 0:
                # The band the pair is in *now*, for `_centring`. None where
                # the midpoint falls in no band, which is what happens where
                # a cone group stands in the middle of the aisle -- and there
                # the question that band is used for does not arise.
                band = next(((lo, hi) for lo, hi in bands
                             if lo - self._ROOM_RES <= here
                             <= hi + self._ROOM_RES), None)
        return worst, band

    def _centring(self, own_room, partner_room):
        """Price the *worse-off* vehicle's clearance into both utilities.

        Nothing else asks the vehicle with room to be the one that yields.
        `_foresight` charges each vehicle for its own clearance and enters
        only its own matrix, so a candidate that leaves the partner squeezed
        costs the roomy vehicle nothing. The separation terms are satisfied by
        either vehicle providing the gap, and providing it is cheap for the
        one with a lane to leave and expensive for the one against an
        obstacle.

        Measured at the site that produces most of our object contacts: a cone
        pair blocks v2's side, the only band open through the narrowing is
        [+0.30, +0.95], and `band_target` correctly snaps v2 to its nearest
        edge, +0.30 -- which holds exactly the 0.10 m minimum and no more.
        v2 cannot go deeper into the band because v1 sits at +0.55 to +0.75
        inside it, and through the site v1 moves *towards* the obstacle rather
        than away. Tracking error of 0.08-0.15 m then puts v2 into the cone,
        in every run, twice per run.

        So the term is the minimum of the two clearances, charged to both
        matrices. Raising the worse one is then worth something to whichever
        vehicle can do it, which at that site means v1 moving up the band and
        letting v2 off its edge.

        Two earlier formulations failed and are worth not repeating. Scoring
        the shortfall against what the far side had spare, as
        `min(short, spare - tight)`, was gameable: shoving the far vehicle out
        past its own edge destroyed the spare and took the penalty to zero
        while the tight side gained nothing -- separation went 0.62 m to
        3.56 m in four seconds. Scoring the tight *band edge* instead pushed
        the pair together rather than across, because the margin to a band
        edge is improved by compressing. Clearance to the obstacle itself has
        neither defect: it cannot be improved by giving up room, and it is
        already computed for `_foresight`.
        """
        tight = np.minimum(own_room[:, None], partner_room[None, :])
        short = np.clip(self.foresight_clear - tight,
                        0.0, self.foresight_clear)
        return self.w_centre * (short / self.foresight_scale) ** 2

    def _wall_room(self, state):
        """Body clearance to the nearest wall at a pose."""
        if self.wall_tree is None:
            return float('inf')
        pose = {'x': np.array([[state[0]]]), 'y': np.array([[state[1]]]),
                'yaw': np.array([[state[2]]])}
        return float(footprint_clearance(pose, self.wall_tree, None,
                                         FOOTPRINT_BOX)[0])

    def _hold_off(self, target, lane_partner, sign, staggered, give_way):
        """Keep the target out of the room the partner's lane needs.

        The wall contacts at the ccw pinch were not a wall-tracking failure.
        The obstacle blocks the inner lane, the inner vehicle shifts to the
        centreline to pass it -- correctly, and it is what the drivers did --
        and the outer vehicle, already against the wall, is left pinned: its
        own target moves inward but the pair-body constraint forbids it,
        because the space is taken. Measured, the outer vehicle sat 0.39 m
        outside its target for two seconds and ended 66 mm inside the wall,
        while the inner one kept 0.68 m of clearance.

        The drivers did not do this. The note in `shifted_lateral_target`
        records what they did: they crossed *while in file*, staggered by
        1.20-1.55 m, so the lateral overlap cost the partner nothing. Here the
        pair crosses while still side_by_side, having built 0.48 m of the 1.45 m
        the file rule asked for.

        So the crossing is made a consequence of the stagger rather than a
        parallel goal: with none built, a vehicle may not encroach within a
        body-pair's width of where its partner's lane runs; with the full
        stagger, it may cross freely. In between it is proportional. A vehicle
        that cannot pass an obstacle without encroaching therefore holds its
        lane and slows, which is recoverable, instead of squeezing its partner
        into a wall, which is not.

        `give_way` is the complement: when one vehicle has more room to spare
        than the other, it is the one that yields. Both compute the comparison
        from the same observed state, so exactly one of them gives way and
        neither has to be told.
        """
        if not give_way:
            return target
        keep = (2.0 * FOOTPRINT_BOX[2] + self.pair_margin) * (1.0 - staggered)
        if keep <= 0.0:
            return target
        if sign > 0.0:
            return max(target, lane_partner + keep)
        return min(target, lane_partner - keep)

    def _open_interval(self, point, reach):
        """Lateral positions open over the whole stretch ahead, as intervals.

        The intersection of the free bands at every station within `reach`, so
        a position survives only if a body can hold it the whole way through.
        A band that opens after the pinch is no use to a vehicle that has to
        get past the pinch first.

        Empty means no single lateral position carries the pair through: the
        corridor closes on one side and opens on the other, which is the
        straddle and file case and not this rule's business.
        """
        trees = [tree for tree in (self.wall_tree, self.object_tree,
                                   self.unknown_tree) if tree is not None]
        if not trees or self.reference is None:
            return []
        xs, ys, yaws, _ = self.reference.centre_ahead(
            point, reach, self._ROOM_STEP)
        keep = None
        for x, y, yaw in zip(xs, ys, yaws):
            bands = free_bands(float(x), float(y), float(yaw), trees,
                               self.object_clearance, FOOTPRINT_BOX,
                               span=self._ROOM_SPAN, step=self._ROOM_RES)
            if keep is None:
                keep = bands
                continue
            merged = []
            for lo, hi in keep:
                for a, b in bands:
                    left, right = max(lo, a), min(hi, b)
                    if right >= left:
                        merged.append((left, right))
            keep = merged
            if not keep:
                break
        return keep or []

    def _reachable_target(self, target, lane, point, reach):
        """`target` if a body can hold it through the stretch, else the
        nearest position that it can.

        `shifted_lateral_target` reasons about obstacles, not about the
        corridor: it builds candidates at `cross +/- need` for each obstacle,
        keeps those clearing every obstacle, and takes the one nearest the
        lane. With a *pair* of cones straddling the middle -- which is what
        the drivers met, and what the layouts now generate -- the two gaps lie
        on opposite sides of a blocked centre, and nearest-to-lane is
        routinely the closed one.

        Measured at (20.99, 15.72): the cones sit at cross -0.30 and +0.23,
        the outer lane at -0.37, and the rule asks for **-0.68** while the
        only band open across the next 0.8 m is **[+0.60, +1.15]**. The target
        is not merely on the far side of the obstacle, it is in the gap
        between two bands -- a position no body can occupy. The vehicle
        tracked it, pinned beside the cone with half a metre of wall clearance
        to spare, and oscillated there for 95 s.

        So the answer is taken from the free space actually measured, and
        taken *inside* it rather than on its lip. Nearest to the *lane*, not
        to the proposed target: the lane is where the
        vehicle would be if nothing were in the way, and "the closest place to
        my lane that I can actually hold" is the same rule the drivers'
        behaviour is described by. Beyond `_MAX_SHIFT` the honest answer is
        still to hold the proposal and let the clearance constraint slow the
        vehicle, rather than promise a swerve it cannot make -- which is the
        bound `shifted_lateral_target` already applies for the same reason.
        """
        open_set = self._open_interval(point, reach)
        if not open_set:
            return target

        def aim(lo, hi):
            """Where in this band to sit: as near the lane as possible, but
            not on the lip.

            `clip(lane, lo, hi)` lands exactly on the edge whenever the lane
            is outside the band, and a band edge holds precisely
            `object_clearance` and nothing more, while the pair tracks its
            target to about 0.08 m. Measured at the site that produces most of
            our object contacts, the blocked vehicle was aimed at +0.30 -- the
            near lip of the only open band -- and tracking error took it into
            the cone in every run, twice per run.

            The inset is capped at half the band, so a narrow band degrades to
            its middle rather than inverting, and the result is re-clipped to
            the band so it can never fall outside.
            """
            inset = min(self._TRACK, 0.5 * (hi - lo))
            return float(np.clip(np.clip(lane, lo, hi), lo + inset, hi - inset))

        # Always aim *inside* a band, not merely inside one. The old rule
        # returned the proposal untouched whenever it already lay in a band,
        # which includes lying on its very edge.
        best = min((aim(lo, hi) for lo, hi in open_set),
                   key=lambda p: abs(p - lane))
        if abs(best - lane) > self._MAX_SHIFT:
            return target
        if abs(best - target) < self._ROOM_RES:
            return target
        self.get_logger().info(
            'lateral target %+.2f is not in free space; nearest holdable is '
            '%+.2f (lane %+.2f)' % (target, best, lane),
            throttle_duration_sec=2.0)
        return best

    def _lateral_target(self, state, side, occupied=None):
        """Lane target, moved aside for obstacles the vehicle is heading into.

        Only obstacles *ahead* count, within `object_lookahead`, so the target
        returns to the lane once one is passed rather than being held aside by
        something behind.
        """
        # The lane offset *here*, from the profile `derive_lane_offsets.py`
        # writes: the corridor's own spacing, 1.20-1.35 m where there is room
        # and closing at pinches, against the constant 1.03 m the medians
        # gave. The drivers did the same -- 1.35 m in open corridor, 0.83 m
        # in tight, with their stagger doubling there -- and our 0.97 m held
        # the tight-corridor spacing everywhere.
        #
        # Substituting a local offset was tried before and reverted: it
        # wedged 1 of 3 runs. Two things are different now. The profile is
        # smooth (5 mm of lateral change per 10 cm of travel, against the
        # 100 mm steps of the runtime nudge that destabilised the pair), and
        # a stall is recoverable, so the same failure no longer ends a run.
        lane = self.reference.lane_offset_at(state[:2], side)
        if self.object_tree is None or not self.weights.path_cross_track:
            if occupied is None:
                return float(lane)
            return shifted_lateral_target(
                lane, np.array([float(occupied) - FOOTPRINT_BOX[2],
                                float(occupied) + FOOTPRINT_BOX[2]]),
                self.object_clearance, FOOTPRINT_BOX[2])

        # Walls veto candidate targets; they never cause one. Without them at
        # all the rule routes between an obstacle and the wall -- at one site
        # it aimed into a 0.12 m gap and wedged there. Treated as ordinary
        # obstacles they displace the target wherever the corridor is tight,
        # which is most of the outer lane, and the pair then files for the
        # corridor rather than for anything in its way.
        def ahead_of(tree, reach):
            if tree is None:
                return np.zeros((0, 2))
            points = np.asarray(tree.data)
            dx = points[:, 0] - state[0]
            dy = points[:, 1] - state[1]
            along = dx * np.cos(state[2]) + dy * np.sin(state[2])
            side_by = np.abs(-dx * np.sin(state[2]) + dy * np.cos(state[2]))
            # The trailing edge is not -0.5 m. The pair may be staggered by
            # a full file length, so an obstacle that the leading vehicle has
            # just passed is still ahead of the trailing one -- and releasing
            # it there snapped the target back to the lane and dropped the
            # ordering mid-manoeuvre, which is what left the blocked vehicle
            # ahead at the pass. It stays relevant until the trailing vehicle
            # and its body are clear.
            behind = -(self.file_stagger + FOOTPRINT_BOX[1]
                       - FOOTPRINT_BOX[0])
            return points[(along > behind) & (along < reach)
                          & (side_by < 3.0)]

        reach = max(abs(state[3]) * self.object_horizon,
                    self.object_reach_min)
        objects = ahead_of(self.object_tree, reach)
        if not len(objects):
            return float(lane)
        walls = ahead_of(self.wall_tree, reach)

        def cross_of(points):
            if not len(points):
                return None
            return self.reference.signed_cross_track(points[:, 0:1],
                                                     points[:, 1:2])

        here = float(self.reference.signed_cross_track(
            np.array([[state[0]]]), np.array([[state[1]]]))[0])
        cross = cross_of(objects)
        if occupied is not None and cross is not None:
            # The partner's chosen target, as something to clear. It is the
            # one obstacle that is neither in the map nor the cloud, and
            # without it both vehicles pick the same gap.
            #
            # Entered as its two edges, not its centre: `shifted_lateral_target`
            # keeps the body centre `half_width + clearance` from each
            # obstacle point, which for a point obstacle is right and for a
            # partner is half of what is needed. Clearing both edges puts the
            # two centres `2 * half_width + clearance` apart, which is exactly
            # the test `side_by_side` applies -- so a pair that fits side by side
            # is no longer told to file.
            cross = np.concatenate([
                np.asarray(cross).ravel(),
                [float(occupied) - FOOTPRINT_BOX[2],
                 float(occupied) + FOOTPRINT_BOX[2]]])
        target = shifted_lateral_target(
            lane, cross, self.object_clearance, FOOTPRINT_BOX[2],
            current=here, veto_cross=cross_of(walls))
        if self.band_target:
            target = self._reachable_target(target, lane, state[:2], reach)
        return target

    # --- selection --------------------------------------------------------

    def _rel_angle_side(self, own_state, partner_state):
        """+1 if the partner's lane is to our left, -1 if to our right.

        Read from the derived lane offsets, which are signed about the
        centreline, so this is the *assignment* and nothing more. Zero when
        `rel_angle` is off or the offsets are unavailable, which selects the
        sign-blind magnitude form and reproduces the previous cost exactly.
        """
        if not self.weights.rel_angle or self.reference is None:
            return 0.0
        own = self.reference.lane_offset_at(own_state[:2], self.side)
        partner = self.reference.lane_offset_at(partner_state[:2],
                                                self.partner_side)
        if own is None or partner is None:
            return 0.0
        return 1.0 if float(partner) > float(own) else -1.0

    def _choose(self, u_own, u_partner, terms):
        if self.concept == 'independent':
            # Ignore the partner outright. Rebuilding a zeroed Weights here
            # used to drop `progress` back to its default instead of carrying
            # the configured value, and reading a column of a pair matrix
            # scored the ego against a standing partner.
            solo = np.where(self.ok, terms['u_solo'], -np.inf)
            return int(np.argmax(solo))
        if self.concept == 'anticipated':
            j = int(np.argmax(u_partner.max(axis=0)))
            return int(np.argmax(u_own[:, j]))
        if self.concept == 'social':
            return int(np.unravel_index(
                np.argmax(u_own + u_partner), u_own.shape)[0])
        if self.concept == 'gamma':
            blend = (1.0 - self.gamma) * u_own + self.gamma * u_partner
            return int(np.unravel_index(np.argmax(blend), blend.shape)[0])
        if self.concept == 'maximin':
            # The conservative baseline the other concepts lack: every one of
            # them except `anticipated` resolves the partner's column with a
            # maximum, so none asks what happens if the partner does the worst
            # thing available.
            #
            # Minimised over the partner's *available* columns, not over the
            # whole row. `self.ok[:, None]` blanks whole rows, but the pair
            # mask removes individual cells, so a plain `u_own.min(axis=1)`
            # scores -inf for any row holding one removed pair -- and since
            # some partner rollout collides with almost any rollout of ours,
            # that is nearly every row. The argmax then returns index 0
            # whatever the cost says, silently. Those cells are not a risk the
            # partner might take: both vehicles compute the same mask from the
            # same observed state and remove the same entries, so a removed
            # pair is off the table for both. The worst case is the worst of
            # what remains.
            finite = np.isfinite(u_own)
            worst = np.where(finite, u_own, np.inf).min(axis=1)
            return int(np.argmax(np.where(finite.any(axis=1), worst, -np.inf)))
        return int(np.argmax(u_own.max(axis=1)))

    # --- main loop --------------------------------------------------------
    def tick(self):
        if self.odom is None or self.partner_odom is None:
            return

        own_state = self._state(self.odom, self.pose)
        partner_state = self._state(self.partner_odom)

        # How old the odometry is that this tick plans from, for the profile.
        stamp = self.odom.header.stamp
        self.state_age = (self.get_clock().now().nanoseconds * 1e-9
                          - (stamp.sec + stamp.nanosec * 1e-9))
        # The cadence actually achieved, for `_brake_clear`.
        now = time.monotonic()
        if self.last_tick_start is not None:
            self.tick_intervals.append(now - self.last_tick_start)
            del self.tick_intervals[:-20]
        self.last_tick_start = now

        # Breadcrumbs for backing up: forward motion only, 5 cm apart, the
        # last 4 m. Not recorded while reversing, so a backup consumes them.
        if (self.backup_retrace and self.backup_left is None
                and own_state[3] > 0.05
                and (not self.trail or math.hypot(
                    own_state[0] - self.trail[-1][0],
                    own_state[1] - self.trail[-1][1]) >= 0.05)):
            self.trail.append((own_state[0], own_state[1]))
            del self.trail[:-80]

        started = time.perf_counter()
        marks = [('start', started)]

        def mark(name):
            marks.append((name, time.perf_counter()))

        commands, self.own = generate(own_state, self.cfg)
        partner_commands, self.partner = generate(partner_state, self.cfg)
        mark('generate')
        if len(commands) < 5:
            self.get_logger().warn('no rollouts from current state')
            return

        midpoint = [(own_state[0] + partner_state[0]) / 2.0,
                    (own_state[1] + partner_state[1]) / 2.0]
        kappa = self.reference.curvature_at(midpoint)
        obj = (float(self.object_tree.query([own_state[:2]])[0][0])
               if self.object_tree is not None else self.model.object_max)
        self.ctx_object = min(obj, self.model.object_max)

        self.sign_own = self.reference.side_sign_at(
            midpoint, self.side, self.sign_own)
        self.sign_partner = self.reference.side_sign_at(
            midpoint, self.partner_side, self.sign_partner)

        # Signed lateral position and heading against the route. Both are
        # per-vehicle terms scored by the same rule for the ego and for the
        # partner, so the two agents' matrices stay transposes of each other
        # and the cooperative concepts still reduce to one bimatrix.
        # NOTE: computed independently, which lets both vehicles converge on
        # the same gap at a cone pair -- room falls to 0.33 m from a nominal
        # 1.31 m and the file rule fires on 84-96% of the first five seconds
        # of a ccw run, though the band there is 0.78 m wide and two bodies
        # need only 0.55 m. Choosing them jointly (outer first, the partner
        # entered as a body) was tried and reverted: it halved the early file
        # rate and moved separation to 1.31 m, but room still bottomed at
        # 0.33 m and heading p99 went 25 -> 49 degrees, progress 0.95 -> 0.85,
        # conversational 74% -> 67%. Making the second target depend on the
        # first makes it chase a moving quantity, which is the same reason
        # `wall_fit` and a per-tick `lane_offset_at` were both reverted.
        cross_target_own = self._lateral_target(own_state, self.side)
        cross_target_partner = self._lateral_target(partner_state,
                                                    self.partner_side)

        # How much of the file stagger the pair has actually built, and which
        # of the two has room to spare. Both are read from the shared state,
        # so the partner reaches the mirror of this conclusion unaided.
        gap_x = partner_state[0] - own_state[0]
        gap_y = partner_state[1] - own_state[1]
        built = abs(gap_x * math.cos(own_state[2])
                    + gap_y * math.sin(own_state[2]))
        staggered = float(np.clip(built / self.file_stagger, 0.0, 1.0))
        room_own = self._wall_room(own_state)
        room_partner = self._wall_room(partner_state)
        lane_here_own = self.reference.lane_offset_at(own_state[:2], self.side)
        lane_here_partner = self.reference.lane_offset_at(
            partner_state[:2], self.partner_side)
        sign_own = 1.0 if lane_here_own > lane_here_partner else -1.0
        cross_target_own = self._hold_off(
            cross_target_own, lane_here_partner, sign_own, staggered,
            room_own >= room_partner)
        cross_target_partner = self._hold_off(
            cross_target_partner, lane_here_own, -sign_own, staggered,
            room_partner >= room_own)

        # Can the two targets hold both bodies side by side? A body width
        # apart is the geometric limit; the slack that used to be a separate
        # parameter is now the object clearance, which is the same margin the
        # rest of the obstacle logic uses.
        room = abs(cross_target_own - cross_target_partner)
        # `Reference.fits_side_by_side` -- the offline "can two bodies stand here"
        # flag -- enters as a *necessary* condition, never a sufficient one.
        # OR-ing it in was tried and reverted: 95% of the route admits
        # side_by_side, so it made `side_by_side` true almost everywhere, the file
        # rule stopped firing, and ccw ordering fell from 24 of 24 to 8 of 18.
        # AND-ing can only withdraw the side-by-side option, never grant it, so
        # that failure cannot recur.
        #
        # This is what the recurring wedge at (13.0, -7.35) turned out to be.
        # The profile marks `side_by_side` 0 over s = 4.6-5.4 m -- the first
        # obstacle of a ccw run -- because an object narrows the corridor
        # there. `room` cannot see it: both targets stay on their lanes, so
        # room reads 1.15 m and the pair is told to hold formation through a
        # pinch that does not admit two bodies. Measured on the approach, the
        # outer vehicle is squeezed between its partner and the wall --
        # separation falls 1.35 -> 0.73 m while the object closes to 0.86 m
        # -- and it arrives 0.23 m off a lane that was itself clear. Raising
        # the lane margin does not help, because the lane was never the
        # binding constraint: 0.25 m wedged at the same site with the same
        # +0.02 m wall clearance, and cost 0.03 m of spacing for nothing.
        # Asked at the pair's midpoint, the same point `curvature_at` and
        # `side_sign_at` use, so both vehicles read one answer and the two
        # matrices stay transposes of each other.
        needs_pair = 2.0 * FOOTPRINT_BOX[2] + self.object_clearance
        side_by_side = room >= needs_pair
        # The pair's own speed, so both vehicles convert the time to the same
        # distance from the same observed state.
        pace = 0.5 * (abs(own_state[3]) + abs(partner_state[3]))
        reach = max(self.side_by_side_reach_min, pace * self.side_by_side_lookahead)
        if side_by_side and self.side_by_side_lookahead >= 0.0:
            side_by_side = self.reference.fits_side_by_side(midpoint, reach)
        # The same question, measured rather than read from the profile, so
        # that a cone which has moved since the profile was derived still
        # narrows the corridor. AND-ed in, exactly like the offline flag: it
        # can withdraw the side-by-side option, never grant one.
        # Recomputed every `room_every` ticks and reused between: it was the
        # largest phase of the tick (40 ms median, 81 ms p90) and the
        # corridor's width changes slowly with the pair's position.
        if not self.runtime_room:
            room_ahead, band = float('inf'), None
        else:
            self.room_age += 1
            if self.room_cached is None or self.room_age >= self.room_every:
                self.room_cached = self._corridor_room(midpoint, reach)
                self.room_age = 0
            room_ahead, band = self.room_cached
        # Two half-widths, and *not* `needs_pair`: `free_bands` already holds
        # every offset in a band `object_clearance` clear of everything, so
        # adding that margin again asks for it twice. Measured, the double
        # count withdrew side-by-side at the ccw pinch (13.0, -7.35), whose band is
        # 0.60 m wide -- the pair fits and passes it side by side at 1 mm of wall
        # contact, while filing there put the outer vehicle 57 mm into the
        # wall, which is the old failure of that site exactly.
        if side_by_side and room_ahead < 2.0 * FOOTPRINT_BOX[2]:
            side_by_side = False
            self.get_logger().info(
                'corridor measures %.2f m ahead, under the %.2f m two bodies '
                'need: filing'
                % (room_ahead, 2.0 * FOOTPRINT_BOX[2]),
                throttle_duration_sec=2.0)
        # And where it does fit, it still bounds how wide the pair may ask to
        # be. This is the half that stops the outer vehicle buying separation
        # with its wall margin; the gate above only decides the formation.
        mark('room')
        if (self.yield_wait and self.backup_left is None
                and self._yield_now(own_state, partner_state, side_by_side,
                                    cross_target_own, cross_target_partner)):
            self.stuck_ticks = 0        # waiting is not being stuck
            self.cmd_pub.publish(Twist())
            return
        separation_cap = float('inf')
        if side_by_side and math.isfinite(room_ahead):
            separation_cap = max(2.0 * FOOTPRINT_BOX[2], room_ahead)

        # A vehicle whose lane is blocked and which has no viable shift is
        # stuck: it must fall in behind. `room` cannot see this case, because
        # room only collapses when a target *moves*, and here the target
        # correctly does not. Measured at the cones 5-7 site, the outward
        # option clears the cone by 0.375 m and the wall by 0.433 m but sits
        # in a free interval 0.058 m wide, so `min_slack` rejects it -- and
        # rightly, since the pair tracks to about 0.08 m. The target then
        # holds the lane, room stays at 1.30 m, the yield rule never fires in
        # 0% of approach ticks, and the pair arrives side by side at the tightest
        # cone group with the pass order left to whichever nose is in front.
        lane_own = self.reference.lane_offset_at(own_state[:2], self.side)
        lane_partner = self.reference.lane_offset_at(partner_state[:2],
                                                     self.partner_side)
        stuck_own = (self._lane_blocked(own_state, self.side)
                     and abs(cross_target_own - lane_own) < 1e-6)
        stuck_partner = (self._lane_blocked(partner_state, self.partner_side)
                         and abs(cross_target_partner - lane_partner) < 1e-6)
        # Side by side, the along-track target has always been zero: hold the
        # pair exactly level. The drivers did not. Measured against the corpus
        # by condition, they sat 0.59-0.63 m apart along track in a clear
        # aisle and 0.97-0.99 m apart at an obstacle, while we hold 0.24-0.43 m
        # -- the largest and most systematic of our fidelity gaps, and the one
        # feature whose target and weight both bypassed the corpus.
        #
        # Default 0.0 keeps the previous behaviour exactly, so this is an arm
        # to measure rather than a change of configuration. `either` is on
        # with it: the corpus splits 51/49 on who leads, so asking for a
        # magnitude without fixing the sign is what the drivers actually did.
        along_track_target = self.sbs_stagger
        along_track_either = bool(self.sbs_stagger)
        scale_own = scale_partner = 1.0
        if stuck_own != stuck_partner and self.yield_when_blocked:
            # Exactly one of them has nowhere to go. That one goes second,
            # whatever `room` says.
            along_track_target = (self.file_stagger if stuck_own
                              else -self.file_stagger)
        elif not side_by_side and not self.yield_when_blocked:
            # No role imposed: ask for a stagger of this size and let the rest
            # of the cost settle which vehicle takes which end.
            along_track_target = self.file_stagger
            along_track_either = True
        elif not side_by_side:
            # Which of the two yields is not a choice, it is a reading of the
            # geometry: the vehicle whose lane the obstacle blocks is the one
            # whose lateral target has moved, and that vehicle goes second.
            # If it goes first it has to cut across the one with a clear lane,
            # which is what was happening while the target was unsigned.
            # NOTE: a direct `_lane_blocked` test was tried here and
            # reverted. Scored against the offline ground truth -- which lane
            # each cone group blocks, read from the map and objects.yaml --
            # the displacement comparison below is already right at every
            # site in both directions, 48 of 48, and the direct test dropped
            # ccw to 7 of 13. The appearance of a problem came from the
            # analysis script, which inferred "which lane is blocked" from
            # the same displacement proxy and so reported "both blocked"
            # everywhere; the geometry says exactly one lane is blocked at
            # every site. `_lane_blocked` is kept, unused, for diagnostics.
            moved_own = abs(cross_target_own
                            - self.reference.lane_offset_at(
                                own_state[:2], self.side))
            moved_partner = abs(cross_target_partner
                                - self.reference.lane_offset_at(
                                    partner_state[:2], self.partner_side))
            if abs(moved_own - moved_partner) < 0.05:
                # Both displaced about equally: fall back to a convention, so
                # that both vehicles still reach the same answer alone.
                yields_own = self.side == 'inner'
            else:
                yields_own = moved_own > moved_partner
            # Positive target puts the partner ahead, which is what the ego
            # wants when the ego is the one yielding.
            along_track_target = (self.file_stagger if yields_own
                              else -self.file_stagger)
        # Station keeping. Both vehicles track `v*` to about 0.1 m/s, and
        # `v*` knows nothing about the partner, so closing a stagger means
        # leaving `v*` -- which `speed` charges at 3.00 to gain 1.00 on
        # `side by side`. Measured over ten runs the pair therefore held whatever
        # stagger it acquired: 0.80 m median, and corr(stagger, speed
        # difference) = +0.06, meaning the one ahead never slowed and the one
        # behind was never asked to hurry.
        #
        # So the target moves instead of being fought, which is the same
        # lever `file_speed_scale` already used for a pass, made continuous:
        # the vehicle behind is asked for more speed and the one ahead for
        # less, in proportion to the error against the stagger the pair is
        # actually trying to hold. `speed` then drives the correction rather
        # than opposing it, and no weight has to be raised -- raising
        # `side by side` far enough to overpower `speed` cost the inner vehicle's
        # obstacle clearance down to 9 mm.
        #
        # It reuses the two constants that already describe a file, so
        # nothing new is introduced and the pass case comes out unchanged:
        # with `along_track_target` one stagger and the pair side_by_side, the yielder
        # gets exactly `file_speed_scale`.
        # `along_track_either` used to gate this block off entirely, which
        # coupled two unrelated things: asking for a stagger *magnitude*
        # rather than a signed ordering also silently disabled station
        # keeping, and the pair then held whatever offset it acquired -- the
        # exact symptom the comment above describes. The sign the controller
        # needs does not have to be prescribed; it can be read from the pair's
        # current state, which is legitimate because the side is a single bit
        # that is constant within a run (measured: 2 of 94 recorded runs
        # mixed). So an unsigned target keeps station keeping, resolving its
        # sign from where the pair actually is.
        if along_track_target or not along_track_either:
            depth = 1.0 - self.file_speed_scale
            cos_own, sin_own = math.cos(own_state[2]), math.sin(own_state[2])
            cos_p, sin_p = (math.cos(partner_state[2]),
                            math.sin(partner_state[2]))
            dx = partner_state[0] - own_state[0]
            dy = partner_state[1] - own_state[1]
            # Signed along-track offset of the partner in each vehicle's own
            # frame: positive means "the other one is ahead of me".
            along_own = dx * cos_own + dy * sin_own
            along_partner = -dx * cos_p - dy * sin_p
            # Each vehicle's own error against its own target. The partner's
            # target is the negative of the ego's, so both arrive at the same
            # pair of scales from the same observed state.
            # With `either`, the target is a magnitude: point it at the
            # ordering the pair currently holds instead of imposing one.
            signed_target = along_track_target
            if along_track_either:
                signed_target = math.copysign(along_track_target, along_own) \
                    if along_own != 0.0 else along_track_target
            err_own = along_own - signed_target
            err_partner = along_partner + signed_target
            span = self.station_gain or self.file_stagger
            scale_own = 1.0 + depth * float(
                np.clip(err_own / span, -1.0, 1.0))
            scale_partner = 1.0 + depth * float(
                np.clip(err_partner / span, -1.0, 1.0))
            if not self.boost_above_preferred:
                # Close the stagger by slowing the leader, never by driving
                # the follower faster than the drivers went. The speed law is
                # fitted from 242k corpus samples and reproduces them across a
                # 3x range -- 1.12 m/s on a straight down to 0.37 m/s at a
                # 0.6 m radius -- and then station keeping multiplied it by up
                # to 1.54x, so the catching vehicle ran at 1.26 m/s where no
                # driver exceeded about 1.12. That is also the speed at which
                # it could no longer stop: 1.84 m of stopping distance against
                # the 1.41 m it had, two seconds before a 78 mm wall contact.
                # Halves the closing rate, to about 0.38 m/s, which still
                # sheds a 1.45 m stagger inside the lookahead.
                scale_own = min(scale_own, 1.0)
                scale_partner = min(scale_partner, 1.0)

        # Same floor `admissible` uses for its `moving` filter, so the cap
        # and the constraint cannot ask for opposite things.
        v_floor = self.v_min_fraction * float(
            self.model.preferred_speed(kappa))
        # `boost_clearance` asks the corridor to hold the tracking error a
        # fast *lateral* pass produces. Filing is not that: the vehicles are
        # separating along the route, not crossing it, and the vehicle being
        # sped up is the one that has to shed a stagger. Applying it here
        # throttles exactly the vehicle the manoeuvre depends on. Measured at
        # the ccw pinch, the catching vehicle's cap fell 1.30 -> 0.64 while
        # its scale still read 1.21, the stagger stopped growing at 0.58 m of
        # the 1.45 m asked, and the pair arrived side by side at a gap that does
        # not take two bodies -- the outer one ending 49-78 mm inside the wall
        # in four of six runs.
        boosting_own = scale_own > 1.0 + 1e-6 and side_by_side
        boosting_partner = scale_partner > 1.0 + 1e-6 and side_by_side
        cap_own = min(
            self._speed_cap(own_state, self.side, cross_target_own,
                            v_floor, boosting=boosting_own),
            self._heading_cap(own_state, v_floor))
        cap_partner = min(
            self._speed_cap(partner_state, self.partner_side,
                            cross_target_partner, v_floor,
                            boosting=boosting_partner),
            self._heading_cap(partner_state, v_floor))

        self.ctx = PairContext(
            dt=self.cfg.dt, kappa_own=kappa, kappa_partner=kappa,
            object_distance=self.ctx_object,
            path_dist_own=self.reference.lane_distance(
                self.own['x'], self.own['y'], self.side),
            path_dist_partner=self.reference.lane_distance(
                self.partner['x'], self.partner['y'], self.partner_side),
            path_progress_own=self.reference.progress(
                self.own['x'], self.own['y'], self.side),
            path_progress_partner=self.reference.progress(
                self.partner['x'], self.partner['y'], self.partner_side),
            # Through a bend the outer vehicle needs the higher speed target.
            # Zero on a straight, where there is no outside to be on.
            side_sign_own=self.sign_own,
            side_sign_partner=self.sign_partner,
            cross_own=(self.reference.signed_cross_track(
                self.own['x'], self.own['y'])
                if self.weights.path_cross_track else None),
            cross_partner=(self.reference.signed_cross_track(
                self.partner['x'], self.partner['y'])
                if self.weights.path_cross_track else None),
            cross_target_own=cross_target_own,
            cross_target_partner=cross_target_partner,
            #: +1 when the partner's lane sits to our left. A discrete
            #: assignment read from the two offsets, not a pin to them:
            #: it says which side, never how far out.
            rel_angle_side_own=self._rel_angle_side(own_state,
                                                    partner_state),
            #: None unless the formation-conditioned separation law is
            #: asked for, so the default reproduces the pooled law.
            side_by_side=(bool(side_by_side) if self.formation_separation
                          else None),
            along_track_target=along_track_target,
            along_track_either=along_track_either,
            speed_scale_own=scale_own,
            speed_scale_partner=scale_partner,
            speed_cap_own=cap_own,
            speed_cap_partner=cap_partner,
            separation_cap=separation_cap,
            tangent_own=(self.reference.tangent_error(
                self.own['x'], self.own['y'], self.own['yaw'], self.side)
                if self.weights.path_lane_yaw else None),
            tangent_partner=(self.reference.tangent_error(
                self.partner['x'], self.partner['y'], self.partner['yaw'],
                self.partner_side) if self.weights.path_lane_yaw else None))

        # What the cost is actually asked for, computed exactly as `evaluate`
        # does it, so the log carries the planner's number and not a guess.
        separation = min(float(self.model.preferred_separation(
            self.ctx.object_distance)), self.ctx.separation_cap)
        unscaled = float(self.model.preferred_speed_for_side(
            self.ctx.kappa_own, separation, self.ctx.side_sign_own))
        v_star_own = min(self.ctx.speed_scale_own * unscaled, cap_own)
        message = Float32MultiArray()
        message.data = [float(v_star_own), float(unscaled),
                        float(scale_own), float(cap_own),
                        float(cross_target_own), float(along_track_target)]
        self.debug_pub.publish(message)

        mark('context')
        u_own, u_partner, terms = evaluate(
            self.own, self.partner, self.ctx, self.weights, self.model)
        # Foresight: prefer plans that stay clear, rather than forbidding the
        # ones that do not.
        #
        # The wall mask inspects 0.5 s because a hard check over the whole
        # horizon empties the set on a bend. That leaves the planner unable to
        # tell a candidate that clears from one that walks into a wall at
        # 1.5 s: measured approaching a stall at a centre-aisle cone, 96 of 120
        # candidates passed the 0.5 s check and 68 of those entered a wall
        # later in their own rollout; a second later every admissible candidate
        # did. The information is there a full second before the vehicle
        # commits -- it simply cannot be used as a constraint, because by then
        # only 2 of 120 survive a full-horizon test and a mask that thin is
        # worse than none.
        #
        # So it enters as a cost instead. Nothing is rejected, the admissible
        # set is exactly as permissive as before, and the mask cannot empty --
        # which is what defeated four earlier attempts to fix this by
        # tightening the constraint, each of which made wall contact worse by
        # driving the planner into its own last-resort fallback. Here the
        # gradient is available from metres away and costs nothing when the
        # corridor is open.
        if self.w_foresight > 0.0:
            u_own = u_own - self._foresight(self.own)[:, None]
            u_partner = u_partner - self._foresight(self.partner)[None, :]
        # Who gives way, which foresight cannot say: it charges each vehicle
        # for its own clearance, and the vehicle with a wall is the one that
        # has nowhere cheap to go.
        if self.w_centre > 0.0:
            # The clearances `_foresight` already measured, reused rather than
            # recomputed: the same worst-over-horizon body clearance to walls
            # and objects, per rollout.
            crowding = self._centring(
                self._worst_clearance(self.own, None, None, outline=False),
                self._worst_clearance(self.partner, None, None,
                                      outline=False))
            u_own = u_own - crowding
            u_partner = u_partner - crowding
        mark('evaluate')
        # A speed floor tied to the local preferred speed, so standing still
        # is not one of the choices on a clear route.
        v_min = self.v_min_fraction * float(
            self.model.preferred_speed(self.ctx.kappa_own))
        report = {}
        self.ok = admissible(self.own, self.wall_tree, v_min,
                             self.min_clearance, self.clearance_steps,
                             self.object_tree, self.object_clearance,
                             object_steps=self.object_steps, report=report,
                             tau_v=(self.model_tau_v if self.brake_constraint
                                    else None),
                             a_max=(self.model_a_max if self.brake_constraint
                                    else None))
        # Silence here was the problem: when no candidate cleared the objects,
        # the constraint was dropped and the vehicle drove through a cone at
        # full speed with nothing in the log to say the rule had changed.
        if report.get('rule') not in (None, 'walls+objects+moving',
                                      'walls+near-objects+moving'):
            self.get_logger().warn(
                'object clearance unsatisfiable: %d/%d candidates cleared '
                'walls, %d cleared objects; falling back to %s'
                % (report['n_safe'], len(self.ok), report['n_clear'],
                   report['rule']), throttle_duration_sec=2.0)
        if self.ok.sum() == 1 and len(self.ok) > 1:
            # the constraint fell back to a single candidate.
            self.get_logger().warn(
                'no rollout clears %.2f m in the next %d steps; taking the '
                'roomiest' % (self.min_clearance, self.clearance_steps),
                throttle_duration_sec=2.0)
        mark('admissible')
        u_own = np.where(self.ok[:, None], u_own, -np.inf)

        # Pairs whose bodies would collide are removed from the game. Both
        # vehicles compute the same mask from the same observed state, so both
        # remove the same entries and the matrices stay transposes.
        if self.pair_steps:
            gap = pair_gap(self.own, self.partner, self.pair_steps)
            allowed = gap >= self.pair_margin
            if not allowed.any():
                # Already overlapping, or boxed in so tightly that every pair
                # touches. Dropping the mask outright was wrong: the pair then
                # carried on overlapped, because nothing was asking them to
                # separate. Measured on a real run, 7 cm of overlap persisted
                # for 5.5% of ticks. So instead keep the pairs that open the
                # largest gap, which is the same shape of fallback the wall
                # and object checks use, and makes an overlap self-correcting.
                allowed = gap >= gap.max() - 1e-6
                self.get_logger().warn(
                    'no candidate pair clears %.2f m; best is %.3f m, '
                    'separating' % (self.pair_margin, float(gap.max())),
                    throttle_duration_sec=2.0)
            u_own = np.where(allowed, u_own, -np.inf)
            u_partner = np.where(allowed, u_partner, -np.inf)

        # Is there anything to do that both moves and is genuinely clear?
        #
        # Two separate ways the answer can be no. The speed floor is the
        # first thing `admissible` gives up, so its absence does not show in
        # the rule name -- it shows up as a candidate set whose every member
        # stands still. And when nothing clears at all, `admissible` returns
        # the least-bad candidate to keep the planner running; driving that
        # one *is* driving into the obstacle. It was the right fallback while
        # a standstill was absorbing, because creeping beat stopping forever.
        # With a reverse available it no longer is: stopping and backing out
        # is strictly better than entering a cone, so a least-bad candidate
        # no longer counts as an option.
        # "Do not enter an obstacle" and "keep a 10 cm buffer" are different
        # rules, and only the first is a safety requirement. Gating on the
        # buffer deadlocked both directions at cone sites where the corridor
        # does not offer it: measured, vehicles held for 41 s and 87 s while
        # 22 mm and 99 mm *clear* of the cone, because no moving candidate
        # kept 0.10 m. So the buffer still decides which candidates the cost
        # prefers -- `admissible` returns `safe & clear & go` -- while the
        # refusal to move is reserved for a candidate that would actually put
        # the body inside something. The drivers passed cones closer than
        # their alternatives, so a hard 0.10 m was never theirs either.
        # `safe` and `clear` are masks that never empty -- both fall back to
        # a least-bad candidate -- so a candidate can be "safe" while inside
        # a wall. The clearance arrays are the truth, and entering an
        # obstacle is not an option at any level.
        genuine = np.ones(len(commands), dtype=bool)
        if 'wall_room' in report:
            genuine = ((report['wall_room'] >= self.min_clearance)
                       & (report['room_near'] >= 0.0))
        if self.stop_when_blocked:
            # Clear over the next half-second is not enough at speed: the
            # first stop_when_blocked arm stopped in time to stay out of the
            # walls but slid 36 mm into a cone, because the stop came at
            # 0.91 m/s with 0.37 m of braking ahead. A command is executable
            # only if executing it for one tick and then braking stays clear;
            # replanning every tick then keeps a clear stop always available.
            # The partner's candidates get the same test, so both vehicles
            # mask the same pairs and their decisions stay consistent.
            genuine &= self._brake_clear(own_state, commands)
            # The partner's candidates are a prediction, and the partner's own
            # planner checks its moves exactly; the cheap selection halves
            # the cost of this test (see `footprint_clearance`).
            partner_ok = self._brake_clear(partner_state, partner_commands,
                                           outline=False)
            if np.isfinite(np.where(genuine[:, None], u_own, -np.inf)).any():
                u_own = np.where(genuine[:, None], u_own, -np.inf)
                u_partner = np.where(genuine[:, None], u_partner, -np.inf)
            if partner_ok.any() and np.isfinite(
                    np.where(partner_ok[None, :], u_own, -np.inf)).any():
                u_own = np.where(partner_ok[None, :], u_own, -np.inf)
                u_partner = np.where(partner_ok[None, :], u_partner, -np.inf)
        if self.lane_band:
            # Stay within the two lanes. The drivers, within 1.5 m of a cone,
            # were never more than 0.70 m outside the band between the lanes
            # 99% of the time; the planner's worst stall site was a pocket
            # 0.91-1.15 m outside it, beside the cones, reached by going round
            # them on the far side (7 of 34 backups in arm `backup/`). A
            # vehicle already further out may not go further, so the rule
            # cannot itself empty the set; and it is only applied while an
            # allowed pair survives, so the safety rules still govern.
            for rollouts, state, axis in ((self.own, own_state, 0),
                                          (self.partner, partner_state, 1)):
                out = self.reference.band_excursion(rollouts['x'],
                                                    rollouts['y'])
                if out is None:
                    continue
                now = self.reference.band_excursion(
                    np.array([[state[0]]]), np.array([[state[1]]]))[0]
                inside = out <= max(self.band_margin, now + 1e-3)
                mask = inside[:, None] if axis == 0 else inside[None, :]
                if np.isfinite(np.where(mask, u_own, -np.inf)).any():
                    u_own = np.where(mask, u_own, -np.inf)
                    u_partner = np.where(mask, u_partner, -np.inf)
        forward = (np.isfinite(u_own).any(axis=1)
                   & (commands[:, 0] >= 0.05)
                   & genuine)
        if forward.any():
            self.stuck_ticks = 0
            self.escape_ticks = 0
            self.escape_best = -np.inf
        else:
            self.stuck_ticks += 1

        # The leader waits for its partner. Shared planning means neither
        # drives off: when the vehicle ahead along the route is more than
        # `hold_gap` from the one behind, it stops until the gap is back
        # under `resume_gap`. The drivers, among cones, were beyond 3.0 m
        # for 1.3% of the time (p99 3.12 m); the planner was for 13-17%,
        # out to 20 m, mostly while one vehicle recovered. A real stop, not
        # a speed cap: a 2.5 m wait built as a cap was overridden by the
        # speed floor and the leader kept going at ~0.4 m/s
        # (a-zero-speed-cap-does-nothing). Never while backing up oneself,
        # and never for longer than `hold_max`.
        if self.partner_hold and self.backup_left is None:
            gap = math.hypot(partner_state[0] - own_state[0],
                             partner_state[1] - own_state[1])
            total = self.reference._lane_total['outer']
            lead = (self.reference.arc_length_at(own_state[:2], 'outer')
                    - self.reference.arc_length_at(partner_state[:2], 'outer'))
            lead -= total * round(lead / total)
            if gap < self.resume_gap:
                self.hold_spent = False
            if self.hold_ticks:
                timed_out = self.hold_ticks * self.period > self.hold_max
                if gap < self.resume_gap or lead <= 0.0 or timed_out:
                    self.get_logger().warn(
                        'partner %.2f m away after %.1f s; resuming%s'
                        % (gap, self.hold_ticks * self.period,
                           ' (timed out)' if timed_out else ''))
                    self.hold_ticks = 0
                    # A timed-out hold stays off until the pair has closed
                    # up again; otherwise it re-engages on the next tick.
                    self.hold_spent = timed_out
            elif (gap > self.hold_gap and lead > 0.0 and not self.hold_spent
                  and self._clear_of_objects(own_state, self.hold_clear)):
                self.hold_ticks = 1
                self.get_logger().warn(
                    'partner %.2f m behind (%.2f m along the route); '
                    'holding until it is within %.2f m'
                    % (gap, lead, self.resume_gap))
            if self.hold_ticks:
                self.hold_ticks += 1
                self.stuck_ticks = 0    # waiting is not being stuck
                self.cmd_pub.publish(Twist())
                return
        # Backing up runs to completion once started: releasing on the first
        # forward candidate is what made the old escape a limit cycle
        # (2.85 m travelled, 0.036 m net). Unlike the rejected hold-until-
        # clear, it reverses only along ground this body has just occupied,
        # and every step passes the same brake test as forward driving.
        if self.backup_retrace and (
                self.backup_left is not None
                or (self.stuck_ticks >= self.escape_after
                    and len(self.trail) >= 2)):
            if self.backup_left is None:
                # Stuck again where the last backup started: the approach
                # repeated itself, so back up further, to before whatever
                # choice led here. Measured (backup/layout_07_k_ccw_r1): a
                # vehicle that went round a cone on the far side backed up
                # 0.8 m thirteen times into the same pocket.
                here = (own_state[0], own_state[1])
                if (self.last_stuck is not None and math.hypot(
                        here[0] - self.last_stuck[0],
                        here[1] - self.last_stuck[1]) < 0.5):
                    self.backup_now = min(2.0 * self.backup_now, 3.2)
                else:
                    self.backup_now = self.backup_distance
                self.last_stuck = here
                self.backup_left = self.backup_now
                self.backup_ticks = 0
                self.get_logger().warn(
                    'stuck for %d ticks; backing up %.2f m along the path '
                    'driven' % (self.stuck_ticks, self.backup_now))
            if self._backup_step(own_state, partner_state):
                return
        if self.stuck_ticks >= self.escape_after:
            # Which term empties decides what the fix is, so record it rather
            # than only that the vehicle is stuck.
            fwd_cmd = commands[:, 0] >= 0.05
            self.get_logger().warn(
                'stuck under %s: of %d forward candidates %d clear walls, '
                '%d clear objects over the horizon, %d over the executed '
                'prefix' % (report.get('rule'), int(fwd_cmd.sum()),
                            int((report['safe'] & fwd_cmd).sum()),
                            int((report['clear'] & fwd_cmd).sum()),
                            int(((report['room_near'] >= report['margin'])
                                 & fwd_cmd).sum())),
                throttle_duration_sec=3.0)

        if (self.stuck_ticks >= self.escape_after
                and self.escape_ticks < self.escape_max):
            pose = {'x': np.array([[own_state[0]]]),
                    'y': np.array([[own_state[1]]]),
                    'yaw': np.array([[own_state[2]]])}
            self.partner_tree = self._partner_body(partner_state)
            here = float(self._worst_clearance(pose, None, None)[0])
            if self.partner_tree is not None:
                here = min(here, float(footprint_clearance(
                    pose, self.partner_tree, None, FOOTPRINT_BOX)[0]))
            # A recovery that is gaining ground keeps its budget; only a
            # futile one spends it.
            if here > self.escape_best + 1e-3:
                self.escape_best = here
                self.escape_ticks = 0
            escape = self._recover(own_state, here)
            if escape is not None and self.stop_when_blocked:
                # The recovery move gets the same execute-then-brake test,
                # except that it may start where the body already is: the
                # first arm's recovery drove forward at 0.56 m/s past a cone
                # it had just touched.
                standing = float(self._worst_clearance(pose, None, None)[0])
                if not self._brake_clear(own_state, np.array([escape]),
                                         floor=min(0.0, standing))[0]:
                    self.get_logger().warn(
                        'recovery move %.2f m/s, w %+.2f cannot brake clear; '
                        'holding' % escape, throttle_duration_sec=1.0)
                    escape = None
            if escape is not None:
                self.escape_ticks += 1
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = escape
                self.cmd_pub.publish(cmd)
                self.get_logger().warn(
                    'no clear candidate for %d ticks; recovering at '
                    '%.2f m/s, w %+.2f (%d/%d)'
                    % (self.stuck_ticks, escape[0], escape[1],
                       self.escape_ticks, self.escape_max),
                    throttle_duration_sec=1.0)
                return
            self.get_logger().error(
                'escape found nothing: standing clearance %.3f, so no '
                'rollout improved on doing nothing' % (here,))
            # Nothing clear forward, nothing clear backward. Hold station
            # rather than take the least-bad candidate: that candidate is
            # inside something by construction, and a stop is recoverable
            # where a collision is not.
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            self.get_logger().error(
                'no clear candidate forward and no clear reverse either; '
                'holding at a standstill (%d ticks)' % self.stuck_ticks,
                throttle_duration_sec=2.0)
            return

        if self.rejoin_ticks > 0:
            # Just backed out: approach slowly, where the brake test allows
            # passes it refuses at speed.
            self.rejoin_ticks -= 1
            slow = commands[:, 0] <= self.rejoin_speed
            if np.isfinite(np.where(slow[:, None], u_own, -np.inf)).any():
                u_own = np.where(slow[:, None], u_own, -np.inf)
                u_partner = np.where(slow[:, None], u_partner, -np.inf)

        pick = self._choose(u_own, u_partner, terms)
        mark('choose')

        # Never execute a candidate that is not genuinely clear. When nothing
        # clears, `admissible` hands back the least-bad candidate, which is
        # inside something by construction, and two paths still drove it: the
        # first `escape_after` ticks of being blocked, and every tick after
        # the escape budget ran out. All three wall contacts in the 48-run
        # baseline began with "taking the roomiest" and ended with the body
        # 42-103 mm inside the wall. Stop instead, and wait for a clear plan;
        # the stuck counter and recovery run exactly as before.
        if self.stop_when_blocked and not genuine[pick]:
            self.cmd_pub.publish(Twist())
            self.get_logger().warn(
                'chosen candidate is not clear (walls %.3f m, objects %.3f m '
                'over the executed prefix); stopping and waiting'
                % (float(report.get('wall_room', np.full(pick + 1, np.nan))[pick]),
                   float(report.get('room_near', np.full(pick + 1, np.nan))[pick])),
                throttle_duration_sec=1.0)
            return

        if self.diagnose:
            # v_adm is the fastest command surviving the wall and object
            # constraints; v_pair the fastest surviving the pair mask as
            # well. If the chosen speed equals v_pair the vehicle is being
            # held back by a constraint; if v_pair is much higher, the cost
            # chose to slow down. That is the whole question at a cone.
            finite = np.isfinite(u_own).any(axis=1)
            adm = commands[self.ok, 0]
            allowed_v = commands[finite, 0]
            obj = float('nan')
            if self.object_tree is not None:
                obj = float(self.object_tree.query(
                    [[own_state[0], own_state[1]]])[0][0])
            self.get_logger().info(
                'DIAG t=%.3f v_cmd=%.2f v_adm=%.2f v_pair=%.2f n_adm=%d '
                'n_pair=%d rule=%s obj=%.2f cross_target=%+.2f'
                % (self.get_clock().now().nanoseconds * 1e-9,
                   float(commands[pick, 0]),
                   float(adm.max()) if len(adm) else float('nan'),
                   float(allowed_v.max()) if len(allowed_v) else float('nan'),
                   int(self.ok.sum()), int(finite.sum()),
                   report.get('rule', 'walls+objects+moving'), obj,
                   cross_target_own))

        # Only the first segment is executed; the rest is lookahead.
        cmd = Twist()
        cmd.linear.x = float(commands[pick, 0])
        cmd.angular.z = float(commands[pick, 1])
        self.cmd_pub.publish(cmd)
        mark('publish_cmd')

        elapsed = time.perf_counter() - started
        if self.profile:
            parts = ' '.join(
                '%s %.0fms' % (name, (t - marks[i][1]) * 1e3)
                for i, (name, t) in enumerate(marks[1:]))
            objects_age = (time.monotonic() - self.objects_at
                           if self.objects_at is not None else float('inf'))
            self.get_logger().info('tick %.0f ms | %s | state age %.0f ms'
                                   ' | objects age %.0f ms'
                                   % (elapsed * 1e3, parts,
                                      self.state_age * 1e3,
                                      objects_age * 1e3),
                                   throttle_duration_sec=2.0)
        if elapsed > self.period:
            self.get_logger().warn(
                'tick took %.0f ms, budget %.0f ms: planning at %.1f Hz, not '
                '%.1f Hz' % (elapsed * 1e3, self.period * 1e3, 1.0 / elapsed,
                             1.0 / self.period),
                throttle_duration_sec=5.0)

        self._publish_plan(pick)
        if self.publish_fan:
            self._publish_fan(u_own, pick)
        if self.matrix_pub is not None:
            self._publish_matrix(u_own, u_partner, terms, kappa)

    def _clear_of_objects(self, state, distance):
        """No detected object within `distance` of base_link."""
        if self.object_tree is None or distance <= 0.0:
            return True
        return bool(self.object_tree.query([state[0], state[1]])[0] > distance)

    def _yield_now(self, own, partner, side_by_side, target_own,
                   target_partner):
        """Should this vehicle stand and let its partner file past first?

        Only while filing, only for the vehicle whose lane is blocked -- read
        the same way the yield rule reads it, from whose target the obstacle
        moved further -- only with an object close ahead in its own path, and
        only until the partner is `file_stagger` ahead. Both vehicles reach the
        same answer from the same observed state.
        """
        def release():
            self.yield_ticks = 0
            return False

        if side_by_side or self.object_tree is None:
            return release()
        moved_own = abs(target_own
                        - self.reference.lane_offset_at(own[:2], self.side))
        moved_partner = abs(target_partner - self.reference.lane_offset_at(
            partner[:2], self.partner_side))
        if abs(moved_own - moved_partner) < 0.05:
            yields = self.side == 'inner'
        else:
            yields = moved_own > moved_partner
        if not yields:
            return release()
        # How far ahead the partner is, along the *route*, as the leader hold
        # measures it. Projected on own heading, a partner already round a
        # corner read as level: at the layout_07 ccw corner the two waited on
        # each other for ~19 s, this one for the partner to get ahead and the
        # partner (3 m ahead along the route) holding for this one.
        total = self.reference._lane_total['outer']
        lead = (self.reference.arc_length_at(partner[:2], 'outer')
                - self.reference.arc_length_at(own[:2], 'outer'))
        lead -= total * round(lead / total)
        if lead >= self.file_stagger:
            return release()
        cos, sin = math.cos(own[2]), math.sin(own[2])
        points = np.asarray(self.object_tree.data)
        dx, dy = points[:, 0] - own[0], points[:, 1] - own[1]
        along = dx * cos + dy * sin
        across = -dx * sin + dy * cos
        close = ((along > 0.0) & (along < self.yield_distance)
                 & (np.abs(across) < FOOTPRINT_BOX[2] + 0.3))
        if not close.any():
            return release()
        self.yield_ticks += 1
        if self.yield_ticks * self.period > self.yield_max:
            return False          # timed out; stays released until reset
        if self.yield_ticks == 1:
            self.get_logger().warn(
                'lane blocked %.2f m ahead and partner only %.2f m ahead; '
                'waiting for it to file past' % (float(along[close].min()),
                                                  lead))
        return True

    def _backup_step(self, state, partner_state):
        """One tick of reversing along `trail`. False when the backup ends.

        Reverse candidates come from the escape fan. Each must pass the brake
        test against walls, objects and the partner's body, never ending
        deeper than the vehicle already stands; among those, the one whose
        rollout stays closest to the trail is taken. Ends when the distance
        is covered, the trail runs out, nothing passes, or its time budget (1.5x the reverse time, plus 2 s) runs out.
        """
        commands, rollouts = generate(state, self.escape_cfg)
        reverse = commands[:, 0] < -0.02
        trail = np.asarray(self.trail)
        pose = {k: np.array([[state[i]]]) for i, k in enumerate(('x', 'y', 'yaw'))}
        standing = float(self._worst_clearance(pose, None, None)[0])
        ok = reverse & self._brake_clear(state, commands,
                                         floor=min(0.0, standing))
        partner = self._partner_body(partner_state)
        if ok.any():
            ok &= footprint_clearance(rollouts, partner, 10,
                                      FOOTPRINT_BOX) >= 0.0
        self.backup_ticks += 1
        if (not ok.any() or len(trail) < 2 or self.backup_left <= 0.0
                or self.backup_ticks > 20 + 1.5 * self.backup_now
                / max(abs(self.escape_speed), 0.05) / self.period):
            done = 'covered' if self.backup_left <= 0.0 else (
                'no clear reverse' if not ok.any() else
                'trail used up' if len(trail) < 2 else 'timed out')
            self.get_logger().warn(
                'backup ended (%s, %.2f m left); rejoining below %.2f m/s'
                % (done, max(self.backup_left, 0.0), self.rejoin_speed))
            self.backup_left = None
            self.stuck_ticks = 0
            self.rejoin_ticks = int(round(self.rejoin_time / self.period))
            return False
        # Follow the trail: mean distance of the first second of each
        # rollout to the recorded path, plus how far its end is from the
        # trail point `backup_left` metres back.
        steps = min(10, rollouts['x'].shape[1])
        tree = cKDTree(trail)
        pts = np.column_stack([rollouts['x'][:, :steps].ravel(),
                               rollouts['y'][:, :steps].ravel()])
        off = tree.query(pts)[0].reshape(-1, steps).mean(axis=1)
        seg = np.hypot(*np.diff(trail, axis=0).T)
        back = np.concatenate([[0.0], np.cumsum(seg[::-1])])
        goal = trail[::-1][min(int(np.searchsorted(back, self.backup_left)),
                               len(trail) - 1)]
        end = np.hypot(rollouts['x'][:, steps - 1] - goal[0],
                       rollouts['y'][:, steps - 1] - goal[1])
        pick = int(np.argmin(np.where(ok, off + 0.5 * end, np.inf)))
        cmd = Twist()
        cmd.linear.x = float(commands[pick, 0])
        cmd.angular.z = float(commands[pick, 1])
        self.cmd_pub.publish(cmd)
        # Consume the trail behind the vehicle as it is passed.
        moved = abs(state[3]) * self.period
        self.backup_left -= moved
        while len(self.trail) >= 2 and math.hypot(
                self.trail[-1][0] - state[0],
                self.trail[-1][1] - state[1]) < 0.10:
            self.trail.pop()
        return True

    def _brake_clear(self, state, commands, floor=0.0, outline=None):
        """Per command: does one tick of it, then braking, stay clear?

        Integrated through the same actuation lag as every rollout, against
        the same inflated walls and dilated objects, for long enough to come
        to rest (1.6 s is ~4 time constants of tau_v). `floor` lets recovery
        accept a path that does not go deeper than where the body already is.
        """
        # The command does not act on the state it was computed from: the
        # state waits behind a tick of up to ~100 ms on a single-threaded
        # executor, and the command is published at the end of it. Measured
        # at a 57 mm wall contact (stop_brake, layout_07 cw r1): the replayed
        # recorded state said "nothing passes" at -0.90 s, the planner stopped
        # at -0.73 s. So the command is assumed to run for one tick *plus*
        # `brake_latency` before the brake bites.
        hold = self.cfg.dt + self.brake_latency
        if self.adaptive_latency and self.tick_intervals:
            # The state is `state_age` old when planning starts, and the
            # command then runs until the next one is published: with queue
            # depth 1 the age is 22 ms at the median but reaches 170-260 ms
            # when a tick overruns and the next starts before fresh odometry
            # is delivered.
            hold = max(hold, max(self.state_age, 0.0)
                       + 2.0 * max(self.tick_intervals))
        hold = self.cfg.dt * max(1, round(hold / self.cfg.dt))
        cfg = replace(self.cfg, horizon=1.6 + hold, segment_split=hold)
        cmds = np.column_stack([commands[:, 0], commands[:, 1],
                                np.zeros(len(commands)),
                                np.zeros(len(commands))])
        stop = integrate(tuple(float(s) for s in state), cmds, cfg)
        # Every step while the command runs, then every other step of the
        # braking tail: at braking speeds the body moves a few cm per step
        # and is 0.9 m long. Halves the cost of the test.
        split = cfg.split_step
        keep = sorted(set(range(split)) | set(range(split, cfg.n_steps, 2))
                      | {cfg.n_steps - 1})
        stop = {k: v[:, keep] for k, v in stop.items()}
        return self._worst_clearance(stop, None, None,
                                     outline=outline) >= floor

    # --- outputs ----------------------------------------------------------
    def _worst_clearance(self, rollouts, wall_steps, object_steps,
                         outline=None):
        """Body clearance to the nearest of wall or object, per rollout.

        `outline=False` only for estimates and preferences; every safety
        check uses the default (see `footprint_clearance`)."""
        room = footprint_clearance(rollouts, self.wall_tree, wall_steps,
                                   FOOTPRINT_BOX, outline=outline)
        if self.object_tree is not None:
            room = np.minimum(room, footprint_clearance(
                rollouts, self.object_tree, object_steps, FOOTPRINT_BOX,
                outline=outline))
        return room

    def _partner_body(self, state):
        """The partner's current footprint as points, for clearance queries.

        Held still rather than predicted. In recovery the hard pair check --
        safe against *every* partner command -- is self-fulfilling: with the
        pair 0.81 m apart almost no candidate satisfies it, both vehicles
        reason identically, and both hold for ever. Measured, that froze both
        at the same coordinates for the last 45 s of two ccw runs while one
        of them sat 0.022 m inside a wall. Treating the partner as an
        obstacle to be cleared, rather than a veto, lets each move away while
        assuming the other stays put -- and moving *towards* it scores badly,
        so they do not trade places into each other.
        """
        back, front, half = FOOTPRINT_BOX
        us = np.linspace(back, front, 5)
        vs = np.linspace(-half, half, 3)
        uu, vv = np.meshgrid(us, vs, indexing='ij')
        cos, sin = math.cos(state[2]), math.sin(state[2])
        return cKDTree(np.column_stack([
            (state[0] + uu * cos - vv * sin).ravel(),
            (state[1] + uu * sin + vv * cos).ravel()]))

    def _recover(self, state, here):
        """The move that most improves clearance, forwards or backwards.

        One rule replacing three. Earlier there was a reverse for "no forward
        option", a second reverse for "already overlapping", and a hold for
        neither -- and between them the vehicle could sit still *inside* a
        wall for 95% of a run, because the hold fired while every escape
        declined. Direction is not the question; clearance is. So every
        candidate the vehicle could execute, forward or reverse, is scored by
        the clearance it achieves, and the best one is taken provided it
        strictly beats standing still.

        Reverse is sampled only here, so the driving envelope is unchanged:
        the recorded drivers never reversed, which is why `v_min = 0.0`.

        Scored on where the move *ends*, not on the worst point along it.
        Every rollout starts at the pose the vehicle is already in, so when
        that pose is the tight one -- which is the only time recovery runs --
        the worst point along the path is essentially the standing clearance,
        and "strictly better than standing still" becomes unsatisfiable.
        Measured at the pose that held a vehicle for 92 s: backing out takes
        it from 0.051 m to 0.699 m of room, and the old objective scored that
        as a 5 mm gain, against a 1e-3 threshold. The smallest difference in
        the live object set then flipped it to "no rollout improved on doing
        nothing", and the vehicle sat there.

        The path minimum is still enforced, as a filter rather than the
        objective: a candidate may not penetrate on the way, and when the body
        is already inside something it may not go deeper. So this cannot
        reverse through a wall to reach open space beyond it.

        The filter carries one map cell of tolerance, and without it recovery
        does not work at the only pose that needs it. `floor = min(0, here)`
        means "never graze" whenever the vehicle is outside a surface, however
        narrowly -- and a vehicle that needs recovery is by definition against
        something. Measured over 32 runs on pair-cone layouts, recovery
        declined every candidate 225 times, at a median standing clearance of
        **+0.000 m**, with 89% of those poses inside the map's own resolution
        of a surface. The vehicle then held station: 94% of 31 stalls were a
        body within 0.05 m of a wall, none of them blocked by a cone, for a
        median of 28 s and once for 138 s of a 168 s run. Asking a candidate
        to keep a clearance the map cannot measure is asking it to prove
        something unprovable, and the answer is always no.

        A cell of tolerance is the smallest change that makes the question
        answerable, and it is bounded: a candidate may end up one resolution
        unit below where the vehicle already is, never further, so this still
        cannot drive into open geometry.
        """
        def terminal(rollouts):
            return {k: rollouts[k][:, -1:] for k in ('x', 'y', 'yaw')}

        # The tolerance applies only where the measurement is genuinely
        # uninformative -- a standing pose within one cell of a surface, which
        # may be touching or may be 25 mm clear. Applied unconditionally it
        # also licenses a 25 mm penetration for a vehicle that has room, and
        # measured that way it took wall contact to 37 and 52 mm on runs whose
        # only previous defect was standing still.
        floor = min(0.0, here)
        if here < RESOLUTION:
            floor -= self.escape_tolerance
        # What a candidate must beat. `here + 1e-3` asks for a 1 mm
        # improvement, which is 25x finer than the measurement: clearance is
        # taken to occupied cell centres on a 0.05 m grid, so anything below
        # RESOLUTION is uninformative about which of two poses is actually
        # further from the surface. Measured over 939 escape failures on the
        # layouts, the median standing clearance was 0.002 m -- the vehicle
        # was refusing to move because it could not *prove* an improvement
        # that no measurement at that scale could show.
        #
        # With `escape_unstick`, a pose inside one cell of a surface compares
        # against the floor instead, so the roomiest candidate is taken as
        # long as it does not end below where the vehicle already is. Bounded
        # the same way as before: terminal clearance must still clear the
        # floor, and the path minimum is still filtered by `viable`.
        if self.escape_unstick and here < RESOLUTION:
            best, best_end = None, floor
        else:
            best, best_end = None, here + 1e-3
        for cfg in (self.cfg, self.escape_cfg):
            commands, rollouts = generate(state, cfg)
            room = self._worst_clearance(rollouts, self.clearance_steps,
                                         self.object_steps)
            end = self._worst_clearance(terminal(rollouts), None, None)
            if self.partner_tree is not None:
                room = np.minimum(room, footprint_clearance(
                    rollouts, self.partner_tree, self.clearance_steps,
                    FOOTPRINT_BOX))
                end = np.minimum(end, footprint_clearance(
                    terminal(rollouts), self.partner_tree, None,
                    FOOTPRINT_BOX))
            viable = room >= floor
            if not viable.any():
                continue
            pick = int(np.argmax(np.where(viable, end, -np.inf)))
            if end[pick] > best_end:
                best_end = float(end[pick])
                best = (float(commands[pick, 0]), float(commands[pick, 1]))
        return best

    def _publish_plan(self, pick):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        for k in range(self.cfg.n_steps):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(self.own['x'][pick, k])
            pose.pose.position.y = float(self.own['y'][pick, k])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.plan_pub.publish(path)

    def _publish_fan(self, u_own, pick):
        """The admissible candidates as a line list, for RViz.

        Only the admissible ones. The rejected rollouts are still computed --
        the feasibility mask needs them -- but drawing them shows the planner
        apparently considering trajectories straight through an obstacle, when
        it has already ruled those out. The picture then misrepresents a
        correct computation, and that is worse than a sparser picture.

        A useful side effect: when the mask collapses to the roomiest
        candidate the fan becomes a single line, so a cornered planner is
        visible at a glance rather than buried in a log line.

        Two things were still being drawn that the planner cannot do. Pairs
        removed by the body-to-body check were shown, because that mask is
        applied to the utilities and not to `self.ok`. And every rollout was
        drawn to its full 3 s, while walls are only checked over
        `clearance_steps` -- 0.5 s -- so five sixths of each line was never
        validated against a wall and plenty of them ran straight through one.
        Extending the hard wall check instead is the wrong fix, for the reason
        `feasible` gives: a 3 s wall constraint empties the set at every
        corner. So each line is now cut where its body first touches
        something, which shows how far the candidate is actually clear and
        leaves the constraint alone.
        """
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.01
        marker.color.a = 0.25
        marker.color.r = float(self.fan_color[0])
        marker.color.g = float(self.fan_color[1])
        marker.color.b = float(self.fan_color[2])
        # Admissible *and* surviving the pair mask: a rollout the ego cannot
        # take against any partner choice is not a candidate.
        usable = self.ok & np.isfinite(u_own).any(axis=1)
        keep = np.flatnonzero(usable)
        if not len(keep):
            keep = np.array([pick])
        step = max(1, len(keep) // max(self.fan_max, 1))
        keep = keep[::step]

        # Where each kept rollout first puts its body into something. The
        # partner counts as something: without it a line was never shortened
        # at the other vehicle, so the fan appeared to drive through it even
        # where the pair mask had ruled that out. The partner's body is taken
        # at its observed pose, which is where it is rather than where it will
        # be -- a line crossing behind a partner that is moving away is a
        # candidate the planner may legitimately take, so this trims the
        # picture conservatively and the pair mask remains the constraint.
        sliced = {k: v[keep] for k, v in self.own.items()}
        room = None
        for tree in (self.wall_tree, self.object_tree, self.partner_tree):
            if tree is None:
                continue
            near = footprint_clearance(sliced, tree, None, FOOTPRINT_BOX,
                                       per_step=True, outline=False)
            room = near if room is None else np.minimum(room, near)
        if room is None:
            cut = np.full(len(keep), self.cfg.n_steps)
        else:
            touching = room < 0.0
            any_touch = touching.any(axis=1)
            first = np.argmax(touching, axis=1)
            cut = np.where(any_touch, first + 1, self.cfg.n_steps)

        for row, i in enumerate(keep):
            for k in range(int(cut[row]) - 1):
                for kk in (k, k + 1):
                    marker.points.append(Point(
                        x=float(self.own['x'][i, kk]),
                        y=float(self.own['y'][i, kk]), z=0.0))
        self.fan_pub.publish(marker)

    def _publish_matrix(self, u_own, u_partner, terms, kappa):
        msg = CostMatrix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.n_own, msg.n_partner = u_own.shape
        finite = np.where(np.isfinite(u_own), u_own, -1e9)
        msg.u_own = finite.astype(np.float32).ravel().tolist()
        msg.u_partner = u_partner.astype(np.float32).ravel().tolist()
        msg.feasible_own = self.ok.tolist()
        msg.term_names = self.weights.names()
        msg.term_weights = self.weights.vector().astype(np.float32).tolist()
        msg.curvature = float(kappa)
        msg.object_distance = float(self.ctx.object_distance)
        msg.target_separation = float(terms['target_separation'])
        msg.preferred_speed = float(terms['preferred_speed_own'])
        self.matrix_pub.publish(msg)


def main():
    rclpy.init()
    node = JointPlanner()
    try:
        if node.state_thread:
            executor = MultiThreadedExecutor(num_threads=2)
            executor.add_node(node)
            executor.spin()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
