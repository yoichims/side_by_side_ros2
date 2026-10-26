#!/usr/bin/env python3
"""Make the partner vehicle visible to this vehicle's simulated laser.

utm_simulator raycasts the occupancy grid, so it sees the map and nothing else:
without this, the two PMVs pass straight through each other and the safety
limiter only ever reacts to walls. That is the wrong baseline for a side-by-side
system, where the partner is the obstacle that matters most.

The partner's footprint is sampled on the map grid and published as a
sensor_msgs/PointCloud in the map frame, which is what
utm_simulator's dynamic_obstacles callback expects -- it converts each point
straight to a cell index with the map origin and resolution.

The footprint is filled rather than outlined on purpose. utm_simulator marks
individual cells, and a raycast will slip between outline points spaced wider
than the 0.05 m grid.
"""

import math

import rclpy
from geometry_msgs.msg import Point32
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud


def footprint_samples(flat, spacing):
    """Grid-sample the interior of a polygon given as [x0, y0, x1, y1, ...]."""
    points = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    def inside(x, y):
        crossings = 0
        for i in range(len(points)):
            ax, ay = points[i]
            bx, by = points[(i + 1) % len(points)]
            if (ay <= y < by) or (by <= y < ay):
                if x < ax + (bx - ax) * (y - ay) / (by - ay):
                    crossings += 1
        return crossings % 2 == 1

    samples = []
    steps_x = max(int((max(xs) - min(xs)) / spacing) + 1, 2)
    steps_y = max(int((max(ys) - min(ys)) / spacing) + 1, 2)
    for i in range(steps_x):
        x = min(xs) + i * spacing
        for j in range(steps_y):
            y = min(ys) + j * spacing
            if inside(x, y):
                samples.append((x, y))
    return samples


class PartnerObstacle(Node):
    def __init__(self):
        super().__init__('publish_partner_obstacle')
        self.frame_id = self.declare_parameter('frame_id', 'map').value
        spacing = self.declare_parameter('spacing', 0.05).value
        flat = self.declare_parameter(
            'footprint', [0.55, 0.30, 0.55, -0.30, -0.35, -0.30, -0.35, 0.30]).value
        rate = self.declare_parameter('rate', 10.0).value

        self.samples = footprint_samples(flat, spacing)
        self.pose = None

        self.sub = self.create_subscription(
            Odometry, 'partner_odom', self.odom_callback, 10)
        self.pub = self.create_publisher(PointCloud, 'dynamic_obstacles', 1)
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f'{len(self.samples)} footprint samples at {spacing} m, '
            f'publishing in {self.frame_id}')

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def tick(self):
        if self.pose is None:
            return
        x, y, yaw = self.pose
        sin, cos = math.sin(yaw), math.cos(yaw)

        cloud = PointCloud()
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.header.frame_id = self.frame_id
        for sx, sy in self.samples:
            point = Point32()
            point.x = float(sx * cos - sy * sin + x)
            point.y = float(sx * sin + sy * cos + y)
            point.z = 0.0
            cloud.points.append(point)
        self.pub.publish(cloud)


def main():
    rclpy.init()
    node = PartnerObstacle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
