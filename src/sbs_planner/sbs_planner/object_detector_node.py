#!/usr/bin/env python3
"""Detect objects from the laser, using the map only to say what is *not* an object.

A return that coincides with a mapped cell is the building. A return that does
not is something the map does not contain -- a cone, a person, the partner.
So the map serves two purposes and neither is "the obstacle list": it is the
planner's reference, and it is the background this subtracts.

That matters because the map cannot be the obstacle source. It is a slice at
LiDAR height (~1.6 m) containing structure the vehicle passes under, and it
knows nothing about anything that moved since the survey.

The partner is masked out by its observed pose. It is already handled by the
formation terms, and feeding it in as an obstacle would have the pair repelling
each other while the cost tries to hold them 1.41 m apart.

Method and tolerance come from the offline analysis (sbs/lidar.py), where
scan-to-map alignment against this map reached 94.2% at a 0.40 m tolerance.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import Point32
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial import cKDTree
from sensor_msgs.msg import LaserScan, PointCloud

from sbs_planner.world import cloud_points, grid_points, wall_tree
from tf2_ros import Buffer, TransformListener


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        p = self.declare_parameter
        self.frame_id = p('frame_id', 'map').value
        # 0.40 m: the analysis reached 94.2% wall agreement at this tolerance
        # against this map. Tighter flags real walls as objects; the thinned
        # map.csv at 0.25 m mislabelled 13.4% of returns.
        self.wall_tolerance = p('wall_tolerance', 0.40).value
        self.partner_radius = p('partner_radius', 0.80).value
        self.min_range = p('min_range', 0.30).value
        self.max_range = p('max_range', 8.0).value
        self.min_cluster = p('min_cluster', 2).value
        rate = p('rate', 5.0).value

        self.wall_tree = None
        #: Fixed geometry arrives from two places and both feed `wall_tree`:
        #: the surveyed map, and a latched `structure` cloud for the aisle.
        self.map_points = None
        self.structure_points = None
        self.scan = None
        self.partner = None

        self.create_subscription(LaserScan, 'scan', self._scan, 5)
        self.create_subscription(Odometry, 'partner_odom', self._partner, 5)
        self.create_subscription(
            OccupancyGrid, 'map', self._map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(
            PointCloud, 'structure', self._structure,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.pub = self.create_publisher(PointCloud, 'objects', 1)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.create_timer(1.0 / rate, self.tick)

    def _scan(self, msg): self.scan = msg
    def _partner(self, msg): self.partner = msg

    def _map(self, msg):
        points = grid_points(msg)
        if not len(points):
            return
        self.map_points = points
        self._rebuild_walls()
        self.get_logger().info(f'background: {len(points)} mapped cells')

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


    def tick(self):
        if self.scan is None or self.wall_tree is None:
            return
        scan = self.scan
        try:
            tf = self.buffer.lookup_transform(
                self.frame_id, scan.header.frame_id, rclpy.time.Time())
        except Exception as error:                   # noqa: BLE001
            self.get_logger().warn(f'no transform: {error}',
                                   throttle_duration_sec=5.0)
            return

        ranges = np.asarray(scan.ranges, dtype=float)
        angles = scan.angle_min + scan.angle_increment * np.arange(len(ranges))
        keep = (np.isfinite(ranges) & (ranges > max(scan.range_min,
                                                    self.min_range))
                & (ranges < min(scan.range_max, self.max_range)))
        if not keep.any():
            return

        local = np.column_stack([ranges[keep] * np.cos(angles[keep]),
                                 ranges[keep] * np.sin(angles[keep])])
        q = tf.transform.rotation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        c, s = np.cos(yaw), np.sin(yaw)
        points = local @ np.array([[c, s], [-s, c]]) + np.array([
            tf.transform.translation.x, tf.transform.translation.y])

        # Everything the map already explains is background.
        is_object = self.wall_tree.query(points)[0] > self.wall_tolerance

        # The partner is not an obstacle: the formation terms own it.
        if self.partner is not None:
            position = self.partner.pose.pose.position
            near_partner = np.linalg.norm(
                points - np.array([position.x, position.y]),
                axis=1) <= self.partner_radius
            is_object &= ~near_partner

        objects = points[is_object]
        if len(objects) >= self.min_cluster:
            objects = self._drop_isolated(objects)

        cloud = PointCloud()
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.header.frame_id = self.frame_id
        cloud.points = [Point32(x=float(x), y=float(y), z=0.0)
                        for x, y in objects]
        self.pub.publish(cloud)

    def _drop_isolated(self, points):
        """Discard single stray returns; a real object gives several."""
        tree = cKDTree(points)
        counts = np.array([len(tree.query_ball_point(p, 0.25)) for p in points])
        return points[counts >= self.min_cluster]


def main():
    rclpy.init()
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
