#!/usr/bin/env python3
"""Everything in the world that the survey does not contain, as one cloud.

    static objects (from objects.yaml)  +  other vehicles (from their odom)
        -> sensor_msgs/PointCloud on `dynamic_obstacles`

`utm_simulator` already implements the architecture this feeds: it holds the
map as a static grid, and on every `dynamic_obstacles` message it resets to
that grid and stamps the cloud's points as occupied before raycasting. So the
map stays the survey, and anything the survey lacks is data.

The cones used to bypass that entirely -- baked into a second occupancy grid,
`map_obstacle.pgm`, served by a second map server, while the detector
subtracted the first. That only worked while the two grids differed by exactly
the objects and nothing else, an invariant held by hand across two binary
images with nothing checking it. Here the laser is given clean map plus
objects and the detector subtracts the clean map, so what is left is the
objects by construction rather than by coincidence. That second grid is gone
from the repository along with the `map_variant` switch that served it; there
is one map, and it is the survey.

ONE PUBLISHER PER TOPIC, deliberately. `dynamic_obstacles` is
last-message-wins: the simulator discards the previous cloud on every message.
Two nodes publishing to it -- one for the partner, one for the cones -- would
overwrite each other and each set would be visible about half the time, which
is a flicker that would be thoroughly unpleasant to diagnose. So this node
owns the topic and publishes the union.

Points are FILLED at the map resolution, not outlined. The simulator marks one
cell per point, so a ray slips between outline points spaced wider than a
cell.
"""

import math
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point32
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud

#: Same rectangle the limiter, the footprint publisher and the planner's
#: clearance check use.
FOOTPRINT = [0.55, 0.275, 0.55, -0.275, -0.35, -0.275, -0.35, 0.275]


def fill_polygon(flat, spacing):
    """Grid-sample the interior of a polygon given as [x0, y0, x1, y1, ...]."""
    corners = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]

    def inside(x, y):
        hit = False
        for i in range(len(corners)):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % len(corners)]
            if (y1 > y) != (y2 > y):
                cross = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
                if x < cross:
                    hit = not hit
        return hit

    samples = []
    steps_x = int((max(xs) - min(xs)) / spacing) + 1
    steps_y = int((max(ys) - min(ys)) / spacing) + 1
    for i in range(steps_x):
        for j in range(steps_y):
            x = min(xs) + i * spacing
            y = min(ys) + j * spacing
            if inside(x, y):
                samples.append((x, y))
    return samples


def fill_circle(x0, y0, radius, spacing):
    steps = int(radius / spacing) + 1
    out = []
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            x, y = i * spacing, j * spacing
            if x * x + y * y <= radius * radius:
                out.append((x0 + x, y0 + y))
    return out


def inflate_cells(points, radius, spacing):
    """Thicken a cell set by a radius, on the same grid.

    The map holds only the surface the survey's LiDAR saw, so a cone comes out
    as a thin arc with a hollow interior -- and the simulator marks one cell
    per point, so a ray can pass straight through a gap. Dilating fills the
    interior and gives the obstacle a body, which is what it physically has.

    Kept as a parameter rather than written into objects.yaml, so that file
    stays a faithful record of what was surveyed.
    """
    if radius <= 0:
        return list(points)
    steps = int(round(radius / spacing))
    offsets = [(i * spacing, j * spacing)
               for i in range(-steps, steps + 1)
               for j in range(-steps, steps + 1)
               if (i * spacing) ** 2 + (j * spacing) ** 2 <= radius ** 2]
    out = set()
    for x, y in points:
        for dx, dy in offsets:
            out.add((round((x + dx) / spacing), round((y + dy) / spacing)))
    return [(kx * spacing, ky * spacing) for kx, ky in sorted(out)]


def load_objects(path, spacing, section='objects'):
    """Cells from one section of an objects file: exact cells or primitives.

    `section` is `objects` for things to negotiate -- cones, another vehicle --
    or `structure` for the aisle: fixed geometry the survey happens not to
    contain. The two are read the same way and behave differently downstream,
    which is the whole reason they are separate sections. See the planner's
    `_structure` for why: an object *causes* a lateral target to move, so a
    wall entered as an object would have the pair swerving around the aisle
    it is supposed to be driving down.
    """
    payload = yaml.safe_load(Path(path).read_text())
    spacing = float(payload.get('resolution', spacing))
    points, names = [], []
    for entry in payload.get(section) or []:
        name = entry.get('name', '?')
        if 'cells' in entry:
            cells = [(float(a), float(b)) for a, b in entry['cells']]
        elif 'circle' in entry:
            c = entry['circle']
            cells = fill_circle(float(c['x']), float(c['y']),
                                float(c['radius']), spacing)
        elif 'rect' in entry:
            r = entry['rect']
            half_l, half_w = float(r['length']) / 2.0, float(r['width']) / 2.0
            yaw = float(r.get('yaw', 0.0))
            cos, sin = math.cos(yaw), math.sin(yaw)
            corners = []
            for dx, dy in ((half_l, half_w), (half_l, -half_w),
                           (-half_l, -half_w), (-half_l, half_w)):
                corners += [float(r['x']) + dx * cos - dy * sin,
                            float(r['y']) + dx * sin + dy * cos]
            cells = fill_polygon(corners, spacing)
        else:
            raise ValueError(
                f'{section} entry {name!r} has no cells, circle or rect')
        points.append((name, cells))
        names.append(f'{name}({len(cells)})')
    return points, names, spacing


class WorldObjects(Node):
    def __init__(self):
        super().__init__('world_objects')
        p = self.declare_parameter
        self.frame_id = p('frame_id', 'map').value
        spacing = p('spacing', 0.05).value
        rate = p('rate', 10.0).value
        #: Odometry topics of vehicles to inject. A vehicle must not inject
        #: itself: its own body would sit on top of its own laser and every
        #: ray would terminate at zero range.
        # Declared with a one-empty-string default, not []: rclpy infers a
        # parameter's type from its default, and an empty list infers
        # BYTE_ARRAY, so passing topic names to it fails with
        # InvalidParameterTypeException. The empties are filtered back out.
        self.vehicles = [topic for topic in p('vehicles', ['']).value
                         if topic]
        objects_file = p('objects', '').value
        #: Which sections of the objects file to load. An obstacle-free run
        #: drops `objects` and keeps `structure`: the cones are the experiment
        #: and the aisle is the building, so removing the obstacles must not
        #: remove the walls that define the route.
        sections = [name for name in p('sections',
                                       ['objects', 'structure']).value if name]

        inflate = p('inflate', 0.05).value
        groups, names, spacing = (
            load_objects(objects_file, spacing) if objects_file
            and 'objects' in sections else ([], [], spacing))
        self.static_points = []
        for _, cells in groups:
            self.static_points.extend(inflate_cells(cells, inflate, spacing))
        self.inflate = inflate

        # Structure is not inflated. `inflate` exists because the surveyed
        # cones are a handful of cells and a thin object lets rays through;
        # structure is authored as filled primitives at the map resolution, so
        # it is already solid, and growing it would silently narrow the aisle
        # by the inflation radius on both sides.
        structure, structure_names, _ = (
            load_objects(objects_file, spacing, 'structure') if objects_file
            and 'structure' in sections else ([], [], spacing))
        self.structure_points = [pt for _, cells in structure for pt in cells]
        self.body = fill_polygon(p('footprint', FOOTPRINT).value, spacing)

        self.poses = {}
        for topic in self.vehicles:
            self.create_subscription(
                Odometry, topic,
                lambda msg, key=topic: self._odom(key, msg), 10)

        self.publisher = self.create_publisher(
            PointCloud, 'dynamic_obstacles', 1)

        # Structure never moves, so it is latched once rather than republished
        # at the tick rate -- the same arrangement the map itself uses, and it
        # means a planner that starts late still receives it.
        self.structure_pub = self.create_publisher(
            PointCloud, 'world_structure',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        if self.structure_points:
            self.structure_pub.publish(
                self._cloud(self.structure_points))

        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            '%d static cells after %.2f m inflation [%s]; %d structure cells '
            '[%s]; injecting %d vehicle(s) %s at %.0f Hz'
            % (len(self.static_points), inflate, ' '.join(names) or '-',
               len(self.structure_points), ' '.join(structure_names) or '-',
               len(self.vehicles), self.vehicles, rate))

    def _cloud(self, points):
        cloud = PointCloud()
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.header.frame_id = self.frame_id
        for x, y in points:
            cloud.points.append(Point32(x=float(x), y=float(y), z=0.0))
        return cloud

    def _odom(self, topic, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.poses[topic] = (msg.pose.pose.position.x,
                             msg.pose.pose.position.y, yaw)

    def tick(self):
        cloud = PointCloud()
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.header.frame_id = self.frame_id
        points = list(self.static_points) + list(self.structure_points)
        for topic in self.vehicles:
            pose = self.poses.get(topic)
            if pose is None:
                continue
            x, y, yaw = pose
            cos, sin = math.cos(yaw), math.sin(yaw)
            points += [(sx * cos - sy * sin + x, sx * sin + sy * cos + y)
                       for sx, sy in self.body]
        cloud.points = [Point32(x=float(x), y=float(y), z=0.0)
                        for x, y in points]
        self.publisher.publish(cloud)


def main():
    rclpy.init()
    node = WorldObjects()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
