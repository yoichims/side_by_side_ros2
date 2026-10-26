#!/usr/bin/env python3
"""Enforce the vehicle's own limits on whatever is commanding it.

    cmd_vel_raw  ->  [limiter]  ->  cmd_vel

Independent of the planner, so it still holds when the planner is wrong -- and
over one week this planner was wrong in several ways that a bound downstream
would have contained: a speed floor unreachable from rest, a recovery rule that
could not improve on standing still, a lookahead given in metres where the
manoeuvre needed seconds.

**It should never bind.** The limits come from the same `VehicleModel` the
planner samples its rollouts from, so in a correct system this node is a
no-op and every clamp it applies is a bug report about whatever is upstream.
That is the point: a limiter that fires routinely is quietly reshaping
behaviour where nothing else can see it.

Two things it deliberately does not do:

*It does not smooth.* The actuation lag is already identified from the corpus
-- tau_v = 0.41 s, tau_w = 0.49 s -- and the planner integrates it explicitly
when it predicts where a candidate goes. A rate limiter here would add a
second, unmodelled lag and make every one of those predictions wrong.

*It does not clamp the components independently.* Clipping omega without v
changes omega/v, which is the curvature -- the vehicle would follow a
different path from the one that was checked against the walls, which is how a
safety device causes a collision. When a limit binds, v and omega are scaled
**together**, so the vehicle traverses the same geometric path more slowly and
the planner's clearance reasoning survives.
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from sbs_planner.rollout import VehicleModel


def scale_to_limits(v: float, w: float, v_max: float, w_max: float,
                    lat_acc_max: float) -> float:
    """The largest factor <= 1 that brings (v, w) inside every limit.

    One factor for both, so the path shape is preserved. The lateral bound is
    |v * w| <= lat_acc_max, which is quadratic in the factor, hence the square
    root; it is the passenger-comfort limit rather than a mechanical one, and
    it is the bound most likely to bite on a fast corner.
    """
    factor = 1.0
    if v_max > 0.0 and abs(v) > v_max:
        factor = min(factor, v_max / abs(v))
    if w_max > 0.0 and abs(w) > w_max:
        factor = min(factor, w_max / abs(w))
    lateral = abs(v * w)
    if lat_acc_max > 0.0 and lateral > lat_acc_max:
        factor = min(factor, math.sqrt(lat_acc_max / lateral))
    return factor


class VelocityLimiter(Node):
    def __init__(self):
        super().__init__('velocity_limiter')
        p = self.declare_parameter
        model = VehicleModel()
        self.v_max = p('v_max', float(model.v_max)).value
        self.w_max = p('w_max', float(model.w_max)).value
        self.lat_acc_max = p('lat_acc_max', float(model.lat_acc_max)).value
        #: Reverse is a recovery behaviour, not a driving one, so the bound is
        #: separate: the recorded drivers never reversed.
        self.v_min = p('v_min', -0.30).value
        self.report = p('report', True).value
        self.clamped = 0
        self.worst = 1.0

        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.create_subscription(Twist, 'cmd_vel_raw', self._limit, 1)
        self.create_timer(5.0, self._summary)
        self.get_logger().info(
            'limiting to v %.2f, w %.2f, |v*w| %.2f'
            % (self.v_max, self.w_max, self.lat_acc_max))

    def _limit(self, msg):
        v, w = float(msg.linear.x), float(msg.angular.z)
        factor = scale_to_limits(v, w, self.v_max, self.w_max,
                                 self.lat_acc_max)
        out = Twist()
        out.linear.x = max(self.v_min, v * factor)
        out.angular.z = w * factor
        if factor < 1.0 - 1e-6:
            self.clamped += 1
            self.worst = min(self.worst, factor)
        self.pub.publish(out)

    def _summary(self):
        if self.report and self.clamped:
            self.get_logger().warn(
                'limiter bound %d commands, worst scale %.2f -- the planner '
                'is asking for more than the vehicle allows'
                % (self.clamped, self.worst))
            self.clamped = 0
            self.worst = 1.0


def main():
    rclpy.init()
    node = VelocityLimiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
