#!/usr/bin/env python3
"""Keyboard teleoperation for the Stonefish Girona500 pool ROV.

The Stonefish scenario accepts five normalized thruster setpoints in this order:
surge port, surge starboard, heave bow, heave stern, sway.
"""

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class RovKeyboardTeleop(Node):
    """Publish latched manual setpoints and always stop the vehicle on exit."""

    def __init__(self):
        super().__init__('rov_keyboard_teleop')
        self.declare_parameter('topic', '/girona500/controller/thruster_setpoints_sim')
        self.declare_parameter('step', 0.08)
        self.declare_parameter('max_setpoint', 0.45)
        self._topic = self.get_parameter('topic').value
        self._step = float(self.get_parameter('step').value)
        self._limit = float(self.get_parameter('max_setpoint').value)
        self._setpoints = [0.0] * 5
        self._publisher = self.create_publisher(Float64MultiArray, self._topic, 10)
        self._timer = self.create_timer(0.05, self._publish)

    def _publish(self):
        message = Float64MultiArray()
        message.data = self._setpoints
        self._publisher.publish(message)

    def stop(self):
        self._setpoints = [0.0] * 5
        self._publish()

    def _adjust(self, indices, amount):
        for index, sign in indices:
            self._setpoints[index] = max(
                -self._limit,
                min(self._limit, self._setpoints[index] + sign * amount),
            )

    def handle_key(self, key):
        # The pairs below follow the actuator order in girona500_aqua.scn.
        bindings = {
            'w': ((0, 1), (1, 1)),       # surge forward
            's': ((0, -1), (1, -1)),     # surge reverse
            'a': ((0, -1), (1, 1)),      # yaw left
            'd': ((0, 1), (1, -1)),      # yaw right
            'q': ((4, -1),),             # sway left
            'e': ((4, 1),),              # sway right
            'r': ((2, -1), (3, -1)),     # heave up
            'f': ((2, 1), (3, 1)),       # heave down
        }
        if key == ' ':
            self.stop()
            return
        if key in bindings:
            self._adjust(bindings[key], self._step)


def main(args=None):
    if not sys.stdin.isatty():
        raise RuntimeError('rov_keyboard_teleop must be started from an interactive terminal')

    rclpy.init(args=args)
    node = RovKeyboardTeleop()
    original_terminal = termios.tcgetattr(sys.stdin)
    print('Controls: W/S surge, A/D yaw, Q/E sway, R/F heave, Space stop, Ctrl-C exit')
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if readable:
                node.handle_key(sys.stdin.read(1).lower())
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal)
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
