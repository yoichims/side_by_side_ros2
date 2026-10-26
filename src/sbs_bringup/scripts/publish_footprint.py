#!/usr/bin/env python3
"""Publish a vehicle's footprint polygon, so it is visible in RViz.

The joint-planning stack has no safety limiter -- the planner is the
controller -- and the limiter was the only node that published a footprint. So
`joint_planning.rviz` carried footprint displays that never received anything
and the vehicles showed as bare axes, which makes clearance impossible to judge
by eye: an odometry arrow 0.20 m from a wall looks fine, and the same pose with
a 0.55 m wide body drawn on it obviously is not.

The polygon is published in the vehicle's own `base_link` frame and never
moved. RViz transforms it through TF every frame, so the footprint follows the
vehicle without this node touching odometry at all -- there is no pose maths
here to disagree with the simulator's.

Republished at a low rate rather than latched once, because RViz resolves the
transform at the message stamp, and a stamp from the start of the run is
outside the TF buffer for the rest of it.
"""

import rclpy
from geometry_msgs.msg import Point32, PolygonStamped
from rclpy.node import Node

#: Flat [x0, y0, x1, y1, ...] in base_link, metres. Same rectangle the safety
#: limiter and the partner-visibility publisher use: 0.90 m long, 0.55 m wide,
#: with base_link 0.35 m behind the front face.
FOOTPRINT = [0.55, 0.275, 0.55, -0.275, -0.35, -0.275, -0.35, 0.275]


class FootprintPublisher(Node):
    def __init__(self):
        super().__init__('footprint_publisher')
        flat = self.declare_parameter('footprint', FOOTPRINT).value
        frame = self.declare_parameter('frame_id', 'base_link').value
        rate = self.declare_parameter('rate', 5.0).value
        if len(flat) < 6 or len(flat) % 2:
            raise ValueError('footprint must be at least three x,y pairs')

        self.message = PolygonStamped()
        self.message.header.frame_id = frame
        self.message.polygon.points = [
            Point32(x=float(flat[i]), y=float(flat[i + 1]), z=0.0)
            for i in range(0, len(flat), 2)]

        self.publisher = self.create_publisher(PolygonStamped, 'footprint', 1)
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f'{len(flat) // 2}-point footprint in {frame} at {rate:.0f} Hz')

    def tick(self):
        self.message.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.message)


def main():
    rclpy.init()
    node = FootprintPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
