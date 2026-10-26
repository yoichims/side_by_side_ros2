#!/usr/bin/env python3
"""Publish this agent's candidate futures.

One node, used twice per vehicle:

    rollout_generator   odom := own odom          -> rollouts
    rollout_generator   odom := /<partner>/odom   -> partner_rollouts

The second instance is this agent's *prediction* of what its partner could do,
built from the partner's observed state alone. Nothing is exchanged between
vehicles except state a driver could see with their eyes -- no plans, no
intentions, no chosen trajectory. If a topic carrying another agent's decision
ever appears, the design has leaked.

Pose comes from `pose` when available and from `odom` otherwise. On the real
vehicle the NDT pose is the one to trust and odom drifts; in simulation
odo_simulator integrates from the initial pose so odom is already map-frame and
the pose topic can be left unconnected.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sbs_msgs.msg import RolloutSet

from sbs_planner.rollout import RolloutConfig, VehicleModel, generate


def yaw_of(orientation) -> float:
    q = orientation
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


class RolloutGenerator(Node):
    def __init__(self):
        super().__init__('rollout_generator')

        model = VehicleModel(
            tau_v=self.declare_parameter('tau_v', 0.41).value,
            tau_w=self.declare_parameter('tau_w', 0.49).value,
            v_max=self.declare_parameter('v_max', 1.30).value,
            v_min=self.declare_parameter('v_min', 0.0).value,
            w_max=self.declare_parameter('w_max', 0.80).value,
            a_max=self.declare_parameter('a_max', 0.60).value,
            lat_acc_max=self.declare_parameter('lat_acc_max', 1.50).value,
        )
        self.cfg = RolloutConfig(
            horizon=self.declare_parameter('horizon', 3.0).value,
            dt=self.declare_parameter('dt', 0.1).value,
            segment_split=self.declare_parameter('segment_split', 1.5).value,
            n_v=self.declare_parameter('n_v', 4).value,
            n_w=self.declare_parameter('n_w', 5).value,
            model=model,
        )
        self.frame_id = self.declare_parameter('frame_id', 'map').value
        rate = self.declare_parameter('rate', 10.0).value

        self.odom = None
        self.pose = None
        self.create_subscription(Odometry, 'odom', self.on_odom, 10)
        self.create_subscription(PoseStamped, 'pose', self.on_pose, 10)
        self.pub = self.create_publisher(RolloutSet, 'rollouts', 1)
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            f'horizon {self.cfg.horizon}s dt {self.cfg.dt} '
            f'({self.cfg.n_steps} steps, split {self.cfg.split_step}), '
            f'tau_v {model.tau_v} tau_w {model.tau_w}')

    def on_odom(self, msg):
        self.odom = msg

    def on_pose(self, msg):
        self.pose = msg

    def state(self):
        if self.odom is None:
            return None
        if self.pose is not None:
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y
            yaw = yaw_of(self.pose.pose.orientation)
        else:
            x = self.odom.pose.pose.position.x
            y = self.odom.pose.pose.position.y
            yaw = yaw_of(self.odom.pose.pose.orientation)
        return (x, y, yaw,
                self.odom.twist.twist.linear.x,
                self.odom.twist.twist.angular.z)

    def tick(self):
        state = self.state()
        if state is None:
            return
        commands, traj = generate(state, self.cfg)
        if not len(commands):
            self.get_logger().warn('no feasible rollouts from current state')
            return

        msg = RolloutSet()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.n_rollouts = len(commands)
        msg.n_steps = self.cfg.n_steps
        msg.dt = float(self.cfg.dt)
        msg.segment_split = float(self.cfg.segment_split)
        msg.commands = commands.astype(np.float32).ravel().tolist()
        for name in ('x', 'y', 'yaw', 'v', 'w'):
            setattr(msg, name, traj[name].astype(np.float32).ravel().tolist())
        (msg.origin_x, msg.origin_y, msg.origin_yaw,
         msg.origin_v, msg.origin_w) = (float(s) for s in state)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = RolloutGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
