#!/usr/bin/env python3
"""Log what the pair is actually doing, at planner rate, to one CSV.

Watches both vehicles from the outside -- it subscribes, never publishes a
command -- and records per vehicle state, the pair's relative state, and the
clearances that drive the feasibility mask.

Quantities are computed with the *same* calls the cost uses, so a row says what
the planner saw, not an independent reconstruction of it: `Reference` for lane
error and arc length, `ContextModel` for the separation and speed targets, and
the along/cross-track decomposition from the `along_track` term in `cost.py`. When
the log and the cost disagree about the formation, the disagreement is real
rather than an artefact of measuring it twice in two ways.

Two departures from the cost, both for reading the log rather than optimising:

`along` and `lateral` are SIGNED, where the cost takes the absolute value.
Sign is what distinguishes the partner being ahead from behind, which is the
difference between a vehicle that should ease off and one that should push --
`side by side` cannot tell them apart, and neither can a log of its value.

`sep_target` and `v_star` are recorded next to the achieved values, so a row
shows tracking error directly. A separation of 1.2 m means nothing alone; 1.2 m
against a 1.46 m target that the lane geometry cannot deliver means a great
deal.

    ros2 run sbs_planner formation_metrics --ros-args \
        -p reference:=<centerline.csv> -p output:=/tmp/run.csv
"""

import csv
import math
import os
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters, ListParameters
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Float32MultiArray

from sbs_planner.cost import (FOOTPRINT_BOX, ContextModel,
                              footprint_clearance, shifted_lateral_target,
                              stopping_distance)
from sbs_planner.reference import Reference
from sbs_planner.world import cloud_points, grid_points, wall_tree

#: One row per planner tick. Order is fixed: analysis reads by name, but a
#: human reads by position, and grouping per vehicle keeps it scannable.
COLUMNS = [
    "t",
    "v1_x", "v1_y", "v1_yaw", "v1_v", "v1_w",
    "v2_x", "v2_y", "v2_yaw", "v2_v", "v2_w",
    # pair
    "sep", "sep_target", "range_rate",
    "v1_along", "v1_lateral", "v2_along", "v2_lateral",
    "rel_yaw", "rel_yaw_rate",
    # against the route
    "v1_lane_err", "v2_lane_err", "v1_s", "v2_s",
    # signed lateral position against the centreline, and its target, and
    # heading against the lane tangent -- the quantities the route-relative
    # cost terms act on. Signed, so a side swap shows as a sign change
    # rather than as a distance that happens to be right.
    "v1_cross", "v1_cross_target", "v2_cross", "v2_cross_target",
    "v1_tangent", "v2_tangent",
    "kappa", "v1_side_sign", "v2_side_sign", "v1_v_star", "v2_v_star",
    # against the world
    # From the planner itself, not recomputed here: `v_star` as the cost
    # actually received it, the unscaled target it came from, the
    # station-keeping scale and the wall speed cap. `vN_v_star` below is the
    # old mirrored value, kept so old logs stay readable, and it is the
    # *unscaled* one.
    "v1_v_star_planner", "v2_v_star_planner",
    "v1_speed_scale", "v2_speed_scale",
    "v1_speed_cap", "v2_speed_cap",
    "v1_wall", "v2_wall", "v1_wall_body", "v2_wall_body",
    "v1_obj", "v2_obj",
    # Object clearance measured from the *body*, not from base_link. The
    # centre-based columns above cannot say whether the vehicle actually hit
    # anything: base_link sits 0.35 m behind the front face, so 0.45 m of
    # centre clearance to a cone dead ahead is 0.10 m of overlap.
    "v1_obj_body", "v2_obj_body",
]


def yaw_of(q) -> float:
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


def wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


_OBJECTS = object()      # default for _body_clear's tree argument


class FormationMetrics(Node):
    def __init__(self):
        super().__init__('formation_metrics')
        p = self.declare_parameter

        self.reference = Reference.load(p('reference', '').value)
        self.model = ContextModel()
        self.sides = {'v1': p('v1_side', 'outer').value,
                      'v2': p('v2_side', 'inner').value}
        rate = p('rate', 10.0).value

        self.planners = p('planners', ['/v1/joint_planner',
                                       '/v2/joint_planner']).value
        self.param_wait = p('param_wait', 8.0).value

        # Named by the time of the run, so successive runs never overwrite
        # each other and a file can be matched to what was tried. `output`
        # may be a directory (timestamped file inside it), a full path (used
        # verbatim), or empty (timestamped under /tmp).
        stamp = time.strftime('%Y%m%d_%H%M%S')
        path = p('output', '').value
        if not path:
            # Default into the workspace, not /tmp. The dev container mounts
            # only the workspace, so a file written to the container's /tmp is
            # invisible from the host and gone when the container exits.
            workspace = os.environ.get('SBS_WS')
            path = (os.path.join(workspace, 'runs') if workspace
                    else '/tmp')
        if os.path.isdir(path) or path.endswith(os.sep) or not path.endswith(
                '.csv'):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, f'sbs_formation_{stamp}.csv')
        self.path = path
        self.sink = None
        self.writer = None
        # The file is opened only once the planners' parameters have been
        # fetched, because they are written above the column header and a CSV
        # header cannot be prepended after the fact. A run whose weights are
        # not recorded in it is a run that cannot be compared to another one
        # later, which is most of what these logs are for.
        self.wanted = {name: None for name in self.planners}
        # Clients are created once and kept. A client created fresh on each
        # poll is never ready: discovery needs a round trip, and throwing the
        # client away before it completes one means service_is_ready() is
        # false forever.
        self.listers = {name: self.create_client(
            ListParameters, f'{name}/list_parameters')
            for name in self.planners}
        self.getters = {name: self.create_client(
            GetParameters, f'{name}/get_parameters')
            for name in self.planners}
        self.started = time.monotonic()
        self.startup = self.create_timer(0.5, self._collect_parameters)

        self.odom = {'v1': None, 'v2': None}
        self.objects = {'v1': None, 'v2': None}
        self.wall_tree = None
        #: Fixed geometry arrives from two places and both feed `wall_tree`:
        #: the surveyed map, and a latched `structure` cloud for the aisle.
        #: Latest `plan_debug` from each planner: v_star, unscaled, scale,
        #: cap, cross_target, along_track_target. NaN until one arrives.
        self.debug = {'v1': [float('nan')] * 6, 'v2': [float('nan')] * 6}
        self.map_points = None
        self.structure_points = None
        self.objects_tree = None
        self.previous = None            # (t, sep), for the range rate
        # Mirrors the planner's own hysteresis, so the logged side_sign is the
        # value the planner acted on rather than a stateless re-evaluation.
        self.signs = {'v1': 0.0, 'v2': 0.0}
        # Mirror the planner's obstacle rule so the logged target is the one
        # it acted on, not an independent guess at it. The constants come from
        # the planner over the parameter service in _open(); declaring a
        # second copy here let them drift -- object_clearance sat at 0.25
        # against the planner's 0.10, and a command-line object_horizon never
        # reached the log at all.
        self.clearance = 0.10
        self.horizon = 5.0
        self.stagger = 1.45
        self.rows = 0

        for ns in ('v1', 'v2'):
            self.create_subscription(
                Odometry, f'/{ns}/odom',
                lambda msg, key=ns: self.odom.__setitem__(key, msg), 10)
            self.create_subscription(
                PointCloud, f'/{ns}/objects',
                lambda msg, key=ns: self._objects(key, msg), 1)
        for ns in ('v1', 'v2'):
            self.create_subscription(
                Float32MultiArray, f'/{ns}/plan_debug',
                lambda msg, key=ns: self.debug.__setitem__(
                    key, list(msg.data) + [float('nan')] * 6), 10)
        self.create_subscription(
            OccupancyGrid, '/map', self._map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(
            # The metrics node is not namespaced; v1's copy will do, since
            # both vehicles are given the same structure.
            PointCloud, '/v1/world_structure', self._structure,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f'will log to {path} at {rate:.0f} Hz once the '
            'planner parameters are in')

    # --- startup ----------------------------------------------------------
    def _collect_parameters(self):
        """Ask each planner for its parameters, then open the log.

        Asked over the parameter service rather than duplicated in the launch
        file, so the log describes the planner that actually ran -- including
        an override typed on the command line, which a launch-side copy would
        silently miss.
        """
        for name, client in self.listers.items():
            if self.wanted[name] is not None or not client.service_is_ready():
                continue
            self.wanted[name] = 'asked'
            client.call_async(ListParameters.Request()).add_done_callback(
                lambda fut, node=name: self._got_names(node, fut))

        waiting = [k for k, v in self.wanted.items() if v in (None, 'asked')]
        if waiting and time.monotonic() - self.started <= self.param_wait:
            return
        if waiting:
            self.get_logger().warn(
                'no parameters from %s; logging without them'
                % ', '.join(waiting))
            for name in waiting:
                self.wanted[name] = {}
        self.startup.cancel()
        self._open()

    def _got_names(self, name, future):
        try:
            names = list(future.result().result.names)
        except Exception as error:
            self.get_logger().warn(f'{name}: list_parameters failed: {error}')
            self.wanted[name] = {}
            return
        request = GetParameters.Request()
        request.names = names
        self.getters[name].call_async(request).add_done_callback(
            lambda fut, node=name, keys=names: self._got_values(
                node, keys, fut))

    def _got_values(self, name, names, future):
        try:
            values = future.result().values
        except Exception as error:
            self.get_logger().warn(f'{name}: get_parameters failed: {error}')
            self.wanted[name] = {}
            return
        self.wanted[name] = {
            key: self._plain(value) for key, value in zip(names, values)}

    def _adopt_planner_constants(self):
        """Copy the obstacle-rule constants off the planner that ran.

        Both vehicles are launched with the same values; the first namespace
        that answered is taken, and a namespace that did not answer leaves the
        default in place.
        """
        for params in self.wanted.values():
            if not params:
                continue
            for attribute, key in (('clearance', 'object_clearance'),
                                   ('horizon', 'object_horizon'),
                                   ('stagger', 'file_stagger')):
                value = params.get(key)
                if isinstance(value, (int, float)):
                    setattr(self, attribute, float(value))
            return

    @staticmethod
    def _plain(value):
        """One rcl ParameterValue as a plain Python value."""
        kinds = {1: 'bool_value', 2: 'integer_value', 3: 'double_value',
                 4: 'string_value', 5: 'byte_array_value',
                 6: 'bool_array_value', 7: 'integer_array_value',
                 8: 'double_array_value', 9: 'string_array_value'}
        field = kinds.get(value.type)
        if field is None:
            return None
        got = getattr(value, field)
        return list(got) if field.endswith('array_value') else got

    def _open(self):
        self._adopt_planner_constants()
        self.sink = open(self.path, 'w', newline='', buffering=1)
        for name, params in sorted(self.wanted.items()):
            if not params:
                self.sink.write(f'# {name}: parameters unavailable\n')
                continue
            for key, value in sorted(params.items()):
                self.sink.write(f'# {name} {key}={value}\n')
        self.writer = csv.writer(self.sink)
        self.writer.writerow(COLUMNS)
        self.get_logger().info(f'logging to {self.path}')

    # --- inputs -----------------------------------------------------------
    def _objects(self, key, msg):
        points = np.array([[q.x, q.y] for q in msg.points], float)
        self.objects[key] = cKDTree(points) if len(points) else None

    def _map(self, msg):
        """Walls via the shared extraction, so clearances match the planner."""
        points = grid_points(msg)
        if not len(points):
            return
        self.map_points = points
        self._rebuild_walls()
        self.get_logger().info(f'map: {len(points)} occupied cells')

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


    # --- helpers ----------------------------------------------------------
    def _state(self, msg):
        pose = msg.pose.pose
        return (pose.position.x, pose.position.y, yaw_of(pose.orientation),
                msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def _nearest(self, tree, x, y):
        if tree is None:
            return float('nan')
        return float(tree.query([[x, y]])[0][0])

    def _lane_error(self, x, y, side):
        """Distance to this vehicle's lane, via the cost's own call."""
        return float(self.reference.lane_distance(
            np.array([[x]]), np.array([[y]]), side)[0])

    def _cross(self, x, y):
        return float(self.reference.signed_cross_track(
            np.array([[x]]), np.array([[y]]))[0])

    def _lateral_target(self, state, side):
        """The target the planner is holding, by the planner's own rule."""
        # Local offset from the derived profile, as the planner uses.
        lane = self.reference.lane_offset_at(state[:2], side)
        if self.objects_tree is None:
            return float(lane)
        def ahead_of(tree):
            if tree is None:
                return np.zeros((0, 2))
            points = np.asarray(tree.data)
            dx, dy = points[:, 0] - state[0], points[:, 1] - state[1]
            along = dx * np.cos(state[2]) + dy * np.sin(state[2])
            side = np.abs(-dx * np.sin(state[2]) + dy * np.cos(state[2]))
            reach = max(abs(state[3]) * self.horizon, self.stagger)
            # Same trailing edge as the planner: an obstacle stays relevant
            # until the trailing vehicle of a staggered pair is clear of it.
            behind = -(self.stagger + FOOTPRINT_BOX[1] - FOOTPRINT_BOX[0])
            return points[(along > behind) & (along < reach) & (side < 3.0)]

        objects = ahead_of(self.objects_tree)
        if not len(objects):
            return float(lane)
        walls = ahead_of(self.wall_tree)
        cross = self.reference.signed_cross_track(objects[:, 0:1],
                                                  objects[:, 1:2])
        veto = (self.reference.signed_cross_track(walls[:, 0:1],
                                                  walls[:, 1:2])
                if len(walls) else None)
        here = float(self.reference.signed_cross_track(
            np.array([[state[0]]]), np.array([[state[1]]]))[0])
        return shifted_lateral_target(lane, cross, self.clearance,
                                      FOOTPRINT_BOX[2], current=here,
                                      veto_cross=veto)

    def _body_clear(self, x, y, yaw, tree=_OBJECTS):
        """Body-to-tree clearance, the measure the wall constraint now uses.

        `feasible` moved from base_link to the body, so a centre-to-wall
        distance no longer says whether the constraint was met: 0.02 m of
        centre clearance is the body a quarter of a metre inside the wall.
        Both measures are logged -- `vN_wall` from the centre, `vN_wall_body`
        from the body -- because the gap between them is what hid the
        collisions.
        """
        if tree is _OBJECTS:
            tree = self.objects_tree
        if tree is None:
            return float('nan')
        pose = {'x': np.array([[x]]), 'y': np.array([[y]]),
                'yaw': np.array([[yaw]])}
        return float(footprint_clearance(pose, tree, None, FOOTPRINT_BOX)[0])

    def _tangent(self, x, y, yaw, side):
        return float(self.reference.tangent_error(
            np.array([[x]]), np.array([[y]]), np.array([[yaw]]), side)[0])

    # --- main loop --------------------------------------------------------
    def tick(self):
        if self.writer is None:
            return
        if self.odom['v1'] is None or self.odom['v2'] is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        s1 = self._state(self.odom['v1'])
        s2 = self._state(self.odom['v2'])

        dx, dy = s2[0] - s1[0], s2[1] - s1[1]
        sep = math.hypot(dx, dy)

        # Along/cross-track offset of the partner in each vehicle's own frame,
        # signed: +along = partner is ahead, +lateral = partner is to the left.
        def decompose(state, sign):
            ddx, ddy = sign * dx, sign * dy
            c, s = math.cos(state[2]), math.sin(state[2])
            return ddx * c + ddy * s, -ddx * s + ddy * c

        v1_along, v1_lateral = decompose(s1, +1.0)
        v2_along, v2_lateral = decompose(s2, -1.0)

        midpoint = [(s1[0] + s2[0]) / 2.0, (s1[1] + s2[1]) / 2.0]
        kappa = self.reference.curvature_at(midpoint)
        sign1 = self.signs['v1'] = self.reference.side_sign_at(
            midpoint, self.sides['v1'], self.signs['v1'])
        sign2 = self.signs['v2'] = self.reference.side_sign_at(
            midpoint, self.sides['v2'], self.signs['v2'])

        trees = [t for t in self.objects.values() if t is not None]
        if trees:
            stacked = np.vstack([np.asarray(t.data) for t in trees])
            self.objects_tree = cKDTree(stacked)
        else:
            self.objects_tree = None
        obj1 = self._nearest(self.objects['v1'], s1[0], s1[1])
        obj2 = self._nearest(self.objects['v2'], s2[0], s2[1])
        # The planner clips object distance before it reaches the target, and
        # falls back to object_max when it has seen nothing at all.
        seen = self.model.object_max if math.isnan(obj1) else obj1
        target = float(self.model.preferred_separation(
            min(seen, self.model.object_max)))

        range_rate = float('nan')
        if self.previous is not None:
            t0, sep0 = self.previous
            if now > t0:
                range_rate = (sep - sep0) / (now - t0)
        self.previous = (now, sep)

        self.writer.writerow(['%.4f' % now] + ['%.4f' % value for value in [
            *s1, *s2,
            sep, target, range_rate,
            v1_along, v1_lateral, v2_along, v2_lateral,
            wrap(s2[2] - s1[2]), s1[4] - s2[4],
            self._lane_error(s1[0], s1[1], self.sides['v1']),
            self._lane_error(s2[0], s2[1], self.sides['v2']),
            # Both on the SAME curve. Measured on each vehicle's own lane
            # these are not comparable -- the lanes are resampled
            # independently and differ in length (935 vs 869 rows on ccw), so
            # v1_s - v2_s drifts and is not an along-track stagger, however
            # much it looks like one. Two analyses read a 5 m stagger off it
            # for a pair 1.2 m apart. Use `v1_along` for the stagger; these
            # columns are for locating a lap.
            self.reference.arc_length_at(s1[:2], self.sides['v1']),
            self.reference.arc_length_at(s2[:2], self.sides['v1']),
            self._cross(s1[0], s1[1]),
            self._lateral_target(s1, self.sides['v1']),
            self._cross(s2[0], s2[1]),
            self._lateral_target(s2, self.sides['v2']),
            self._tangent(s1[0], s1[1], s1[2], self.sides['v1']),
            self._tangent(s2[0], s2[1], s2[2], self.sides['v2']),
            kappa, sign1, sign2,
            float(self.model.preferred_speed_for_side(kappa, target, sign1)),
            float(self.model.preferred_speed_for_side(kappa, target, sign2)),
            self.debug['v1'][0], self.debug['v2'][0],
            self.debug['v1'][2], self.debug['v2'][2],
            self.debug['v1'][3], self.debug['v2'][3],
            self._nearest(self.wall_tree, s1[0], s1[1]),
            self._nearest(self.wall_tree, s2[0], s2[1]),
            self._body_clear(s1[0], s1[1], s1[2], self.wall_tree),
            self._body_clear(s2[0], s2[1], s2[2], self.wall_tree),
            obj1, obj2,
            self._body_clear(s1[0], s1[1], s1[2]),
            self._body_clear(s2[0], s2[1], s2[2]),
        ]])
        self.rows += 1

        # A live line, so a run can be read as it happens rather than only
        # afterwards. Throttled: the CSV is the record, this is the glance.
        self.get_logger().info(
            'sep %.2f/%.2f  along %+.2f  lane %.2f/%.2f  wall %.2f/%.2f  '
            'v %.2f/%.2f  w %+.2f/%+.2f' % (
                sep, target, v1_along,
                self._lane_error(s1[0], s1[1], self.sides['v1']),
                self._lane_error(s2[0], s2[1], self.sides['v2']),
                self._nearest(self.wall_tree, s1[0], s1[1]),
                self._nearest(self.wall_tree, s2[0], s2[1]),
                s1[3], s2[3], s1[4], s2[4]),
            throttle_duration_sec=2.0)

    def destroy_node(self):
        if self.sink is not None:
            self.sink.close()
        self.get_logger().info(f'wrote {self.rows} rows to {self.path}')
        super().destroy_node()


def main():
    rclpy.init()
    node = FormationMetrics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
