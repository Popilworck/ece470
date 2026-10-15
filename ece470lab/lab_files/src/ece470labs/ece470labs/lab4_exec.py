#!/usr/bin/env python3
"""Move the UR3e to a world-frame pose, or verify it with --dry-run."""

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .lab_func import lab_invk


class UR3e(Node):
    """Publish a pose trajectory and check measured joint feedback."""

    def __init__(self):
        super().__init__('ur3e_lab4')
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
        ]
        self.current_positions = None
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory', 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

    def joint_state_callback(self, msg):
        """Order complete, finite feedback by joint name."""
        try:
            positions = [msg.position[msg.name.index(name)]
                         for name in self.joint_names]
        except (ValueError, IndexError):
            return
        if all(math.isfinite(value) for value in positions):
            self.current_positions = positions

    def wait_until_ready(self, timeout=15.0):
        """Wait for both state feedback and a trajectory subscriber."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if (self.current_positions is not None and
                    self.trajectory_pub.get_subscription_count() > 0):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def move_arm(self, target):
        if (self.current_positions is None or len(target) != 6 or
                not all(math.isfinite(value) for value in target)):
            self.get_logger().error('Missing joint feedback or invalid target.')
            return False

        # A quintic segment with zero endpoint velocity/acceleration has
        # peak velocity 1.875*d/T and acceleration (10/sqrt(3))*d/T^2.
        # Do not cap T: that could exceed the intended speed/acceleration.
        max_delta = max(abs(a - b)
                        for a, b in zip(self.current_positions, target))
        velocity_limit, acceleration_limit = 1.0, 0.8
        duration = max(1.0, 1.875 * max_delta / velocity_limit,
                       math.sqrt(10 / math.sqrt(3) * max_delta /
                                 acceleration_limit))
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        for positions, seconds in [(self.current_positions, 0.0),
                                   (target, duration)]:
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in positions]
            point.velocities = [0.0] * 6
            point.accelerations = [0.0] * 6
            nanoseconds = round(seconds * 1e9)
            point.time_from_start.sec = nanoseconds // 1_000_000_000
            point.time_from_start.nanosec = nanoseconds % 1_000_000_000
            trajectory.points.append(point)
        self.trajectory_pub.publish(trajectory)
        self.get_logger().info(
            f'Moving to encoder degrees: {np.degrees(target).round(3)}')

        # Allow for the controller's reduced speed scaling.
        deadline = time.monotonic() + max(30.0, 5 * duration + 5)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(abs(a - b) < 0.001
                   for a, b in zip(self.current_positions, target)):
                return True
        return False


def main(args=None):
    argv = sys.argv if args is None else ['lab4_exec', *args]
    parser = argparse.ArgumentParser(
        description='UR3e elbow-up IK: world position in meters, yaw in degrees.')
    parser.add_argument('x', type=float)
    parser.add_argument('y', type=float)
    parser.add_argument('z', type=float)
    parser.add_argument('yaw', type=float)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print IK/FK results without creating a ROS node.')
    pose = parser.parse_args(remove_ros_args(args=argv)[1:])
    try:
        target = lab_invk(pose.x, pose.y, pose.z, pose.yaw)
    except ValueError as error:
        parser.error(str(error))
    print('Encoder angles (degrees):', np.degrees(target))
    if pose.dry_run:
        return

    try:
        input("Check the arm's path is clear and move the monitor to the left.\n"
              "Check Remote mode, Normal state, and the ROS2 driver.\n"
              'Press <Enter> to command this pose, or Ctrl-C to cancel. ')
    except (KeyboardInterrupt, EOFError):
        return

    rclpy.init(args=args)
    node = None
    try:
        node = UR3e()
        node.get_logger().info('Waiting for robot feedback and controller...')
        if not node.wait_until_ready():
            node.get_logger().error(
                'Robot/controller unavailable. Check driver and ROS_DOMAIN_ID.')
            raise SystemExit(1)
        if not node.move_arm(target):
            node.get_logger().error('Timed out before reaching the requested pose.')
            raise SystemExit(1)
        node.get_logger().info('Requested joint pose reached.')
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
