#!/usr/bin/env python3
"""Publish the reference paths once, latched, so RViz can draw them.

Three paths: the pair's centreline and the two vehicle lanes. Latched
(transient-local) because they never change, so RViz picks them up whenever it
connects rather than only if it was listening at startup.

Run once per system, not per vehicle: both agents share the same reference, and
seeing it drawn is how you tell whether a vehicle is off its lane or the lane
itself is wrong -- which is exactly the bug that put 27.5% of an
offset-generated lane inside a wall.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from sbs_planner.reference import Reference


class ReferencePublisher(Node):
    def __init__(self):
        super().__init__('reference_publisher')
        path = self.declare_parameter('reference', '').value
        self.frame_id = self.declare_parameter('frame_id', 'map').value
        if not path:
            raise SystemExit('reference parameter is required')

        reference = Reference.load(path)
        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publishers_by_name = {
            'reference_centre': reference.centre,
            'reference_outer': reference.lanes['outer'],
            'reference_inner': reference.lanes['inner'],
        }
        self.handles = {name: self.create_publisher(Path, name, latched)
                        for name in self.publishers_by_name}
        for name, points in self.publishers_by_name.items():
            self.handles[name].publish(self._path(points))

        gap = float(np.median(np.hypot(
            *(reference.lanes['outer'][:len(reference.lanes['inner'])]
              - reference.lanes['inner'][:len(reference.lanes['outer'])]).T)))
        self.get_logger().info(
            f'{len(reference.centre)} centreline points; lanes published '
            f'(recorded pair separation median '
            f'{np.median(reference.separation):.2f} m)')
        # Re-publish slowly: a latched topic is enough for RViz, but a restart
        # of RViz mid-run should not have to wait on anything.
        self.create_timer(5.0, self._republish)

    def _path(self, points) -> Path:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        for x, y in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def _republish(self):
        for name, points in self.publishers_by_name.items():
            self.handles[name].publish(self._path(points))


def main():
    rclpy.init()
    node = ReferencePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
