# State of the safety limiter that sits between the velocity multiplexer and
# the vehicle. Replaces safety_limiter/Status.msg from the ROS 1 tree.

byte NORMAL = 0        # command passed through unchanged
byte DECELERATING = 1  # command reduced because the footprint would hit something
byte DISABLED = 2      # limiter switched off, command passed through

std_msgs/Header header
int32 status
