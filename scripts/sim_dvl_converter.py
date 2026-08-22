#!/usr/bin/env python3
"""Converts stonefish_ros2 DVL output to nav_msgs/Odometry for AQUA-SLAM.

Stonefish DVL publishes stonefish_ros2/DVL on /girona500/dvl.
AQUA-SLAM expects nav_msgs/Odometry on /bluerov2/dvl (same as real hardware path).

stonefish_ros2/DVL fields used:
    header          std_msgs/Header
    velocity        geometry_msgs/Vector3   [m/s, body frame]
    altitude        float32                 [m]
"""

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from stonefish_ros2.msg import DVL


class SimDvlConverter(Node):
    def __init__(self):
        super().__init__('sim_dvl_converter')
        self._sub = self.create_subscription(DVL, '/girona500/dvl', self._cb, 50)
        self._pub = self.create_publisher(Odometry, '/bluerov2/dvl', 50)

    def _cb(self, msg: DVL):
        # Current stonefish_ros2 DVL.msg has no status field. Treat non-finite
        # velocities or non-positive altitude as invalid.
        if (
            not math.isfinite(msg.velocity.x)
            or not math.isfinite(msg.velocity.y)
            or not math.isfinite(msg.velocity.z)
            or not math.isfinite(msg.altitude)
            or msg.altitude <= 0.0
        ):
            return

        odom = Odometry()
        odom.header = msg.header
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.twist.twist.linear.x = msg.velocity.x
        odom.twist.twist.linear.y = msg.velocity.y
        odom.twist.twist.linear.z = msg.velocity.z

        self._pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = SimDvlConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
