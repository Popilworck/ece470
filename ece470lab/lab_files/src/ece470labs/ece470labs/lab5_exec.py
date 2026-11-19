#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from ur_msgs.srv import SetIO
from ur_msgs.msg import IOStates
import time
import copy
import numpy as np
from math import pi
import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from .blob_search import *
from .lab_func import *
from rclpy.qos import qos_profile_sensor_data

class JointAngles:
    def __init__(self):
        self.name = ["", "", "", "", "", ""]  #could have also done [""] * 6
        self.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# UR3 home position
home = np.radians([120, -90, 90, -90, -90, 0])
# Position for UR3 not blocking the camera
go_away = [270*pi/180.0, -90*pi/180.0, 90*pi/180.0, -90*pi/180.0, -90*pi/180.0, 135*pi/180.0]

# Starting values in world meters. Confirm them at the bench.
PICK_Z = 0.030
BLOCK_HEIGHT = 0.030
CLEARANCE_Z = 0.150
GREEN_STACK = (0.18, 0.08)
YELLOW_STACK = (0.34, 0.08)

class UR3e(Node):
    def __init__(self):
        super().__init__('ur3e')

        # Publishers
        self.trajectory_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.io_state_sub = self.create_subscription(IOStates, '/io_and_status_controller/io_states', self.io_state_callback, 10)

        
        # Service clients
        self.io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')
        while not self.io_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('IO service not available, waiting...')
        
        # === Vision-related members ===
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, qos_profile_sensor_data)

        self.image_count = 0
        self.last_image_time = None
        # store world coordinates for blocks (same idea as xw_yw_G, xw_yw_Y in lab5)
        self.green_blocks = []   # list of (x, y) or (x, y, z)
        self.yellow_blocks = []  # adjust to your needs


        # State variables
        self.current_joint_state = None
        self.analog_in_0_value = 0
        self.current_JointAngles = JointAngles()
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ] # shoulder_pan_joint is the base rotation joint

    def joint_state_callback(self, msg):
        self.current_joint_state = msg  # Currently only used to check if messages have arrived
        index_inOrder = 0
        for name in self.joint_names:
            index_outofOrder = msg.name.index(name)
            self.current_JointAngles.name[index_inOrder] = name
            self.current_JointAngles.position[index_inOrder] = msg.position[index_outofOrder]
            index_inOrder = index_inOrder + 1 


    def io_state_callback(self, msg):
        for state in msg.analog_in_states:
            if state.pin == 0:
                self.analog_in_0_value = state.state   


    def image_callback(self, msg: Image):
        try:
            raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        # If you need to flip like in lab5:
        # cv_image = cv2.flip(raw_image, 1)
        cv_image = raw_image  # if no flip is needed

        # Optional overlay / debugging lines:
        # cv2.line(cv_image, (0,50), (640,50), (0,0,0), 5)

        # Use your existing blob_search to detect block centers in world frame
        self.green_blocks = blob_search(cv_image, "green")
        self.yellow_blocks = blob_search(cv_image, "yellow")
        self.last_image_time = time.monotonic()



    def set_io(self, pin, state):
        req = SetIO.Request()
        req.fun = 1
        req.pin = pin
        req.state = state
        future = self.io_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


    def move_arm(self, target):
        if self.current_joint_state is None:
            self.get_logger().error("No joint state received!")
            return False

        V_MAX = 1#2.09    # rad/s
        A_MAX = 0.8#2.79   # rad/s^2
        MIN_DURATION = 1
        MAX_DURATION = 8.0

        deltas = []
        for i in range(6):
            deltas.append(abs(self.current_JointAngles.position[i] - target[i]))


        max_delta = max(deltas)
        t_acc = V_MAX / A_MAX
        d_acc = 0.5 * A_MAX * (t_acc ** 2)
        if max_delta > 2 * d_acc:
            # trapezoidal velocity profile
            t_total = 2 * t_acc + (max_delta - 2 * d_acc) / V_MAX
        else:
            # triangular velocity profile
            t_total = 2 * (max_delta / A_MAX) ** 0.5

        duration = max(MIN_DURATION, min(t_total, MAX_DURATION))

        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        # Start immediately when the controller receives it
        trajectory_msg.header.stamp.sec = 0
        trajectory_msg.header.stamp.nanosec = 0

        # Anchor point: current measured joint state at t = 0
        p0 = JointTrajectoryPoint()
        p0.positions = self.current_JointAngles.position
        p0.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # starting at rest
        p0.accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # starting at rest
        p0.time_from_start.sec = 0
        p0.time_from_start.nanosec = 0
        trajectory_msg.points.append(p0)

        # Goal point
        p1 = JointTrajectoryPoint()
        p1.positions = target
        p1.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # end at rest, 2 point trajectory
        p1.accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #end at rest.
        p1.time_from_start.sec = int(duration)
        p1.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        trajectory_msg.points.append(p1)

        self.trajectory_pub.publish(trajectory_msg)

        self.get_logger().info(f'Moving to position: {np.degrees(target)}')

        # Wait for movement completion
        start_time = time.time()
        while time.time() - start_time < duration + 2:
            rclpy.spin_once(self, timeout_sec=0.1)

            deltas = []
            for i in range(6):
                deltas.append(abs(self.current_JointAngles.position[i] - target[i]))
            if all(delta < 0.001 for delta in deltas):
                time.sleep(0.25)
                return True
        return False


    def move_block_vision(self, start_xy, target_xyz):
        """Pick up one block and place it at target_xyz in the world frame."""
        # ========================= Student's code starts here =========================
        x_end, y_end, z_end = target_xyz
        yaw = 90
`
        # Find the joint angles at the block and at the drop-off point.
        try:
            x, y = start_xy[:2]
            q_above_block = lab_invk(x, y, CLEARANCE_Z, yaw)
            q_pick = lab_invk(x, y, PICK_Z, yaw)
            q_above_stack = lab_invk(x_end, y_end, CLEARANCE_Z, yaw)
            q_place = lab_invk(x_end, y_end, z_end, yaw)
        except (TypeError, ValueError) as error:
            print("Cannot move block at", start_xy, error)
            return False

        for to_move in (q_above_block, q_pick):
            res = self.move_arm(to_move)
            if not res:
                raise RuntimeError(f"Could not reach the block at {start_xy}.")

        # Turn on suction and wait for the cup to grab the block.
        response = self.set_io(0, 1.0)
        if response is None or not response.success:
            raise RuntimeError("Could not turn suction on.")
        self.analog_in_0_value = None
        start_time = time.monotonic()
        while rclpy.ok() and time.monotonic() - start_time < 0.5:
            rclpy.spin_once(self, timeout_sec=0.05)
        vacuum = self.analog_in_0_value
        if vacuum is None or not np.isfinite(vacuum):
            raise RuntimeError("No valid suction feedback on analog input 0.")

        print(f"Suction reading at {start_xy}: {vacuum:.2f}")
        found_block = vacuum >= 2  # Same threshold as Lab 2.
        if not found_block:
            print("Missing block at", start_xy)
            q_up = q_above_block
        else:
            # Lift up, move across, then lower onto the stack.
            for to_move in (q_above_block, q_above_stack):
                res = self.move_arm(to_move)
                if not res:
                    raise RuntimeError("Could not carry the block above its stack.")
            if self.analog_in_0_value < 2:
                raise RuntimeError(f"Lost suction while carrying the block from {start_xy}.")
            res = self.move_arm(q_place)
            if not res:
                raise RuntimeError("Could not lower the block onto its stack.")
            q_up = q_above_stack

        # Turn off suction before lifting the arm.
        response = self.set_io(0, 0.0)
        if response is None or not response.success:
            raise RuntimeError("Could not turn suction off.")
        time.sleep(0.2)
        res = self.move_arm(q_up)
        if not res:
            raise RuntimeError("Could not lift the arm after releasing suction.")
        return found_block
        # ========================= Student's code ends here =========================




def main(args=None):
    input("Check if the UR3e is in 'Remote' Mode?\n\
    Check if the UR3e is initialized and in 'Normal' state.\n\
    Have you run the ROS2 launch statement?\n\
    If there was an UR3e emergency stop or error, Ctrl-C the ros2 launch and rerun.\n\
    \n\
    Press <Enter> to Continue.")
    rclpy.init(args=args)
    node = UR3e()

    try:
        # Wait for joint state updates
        while rclpy.ok() and node.current_joint_state is None:
            rclpy.spin_once(node, timeout_sec=0.1)

        # Move to go_away (your move_arm uses rclpy.spin_once internally)
        parked = rclpy.ok() and node.move_arm(go_away)

        # Let vision settle for 5 seconds
        settle_start = time.time()
        while rclpy.ok() and (time.time() - settle_start < 5.0):
            rclpy.spin_once(node, timeout_sec=0.1)

        # Snapshot once 
        green_snapshot = list(node.green_blocks)
        node.get_logger().info(f"Snapshot green blocks: {green_snapshot}")

        # Execute task once based on snapshot
        # ========================= Student's code starts here =========================

        if not rclpy.ok():
            return
        if not parked:
            node.get_logger().error("Could not move the arm out of the camera view.")
            return
        if node.last_image_time is None or time.monotonic() - node.last_image_time > 1.0:
            node.get_logger().error("No recent camera image while the arm was out of view.")
            return

        yellow_snapshot = list(node.yellow_blocks)
        print("Snapshot yellow blocks:", yellow_snapshot)
        node.destroy_subscription(node.image_sub)

        total = len(green_snapshot) + len(yellow_snapshot)
        if total > 4 or max(len(green_snapshot), len(yellow_snapshot)) > 3:
            node.get_logger().error("Too many blocks detected. Check the camera masks.")
            return
        for _ in range(4 - total):
            print("Missing block in the camera snapshot.")

        # Stack the green blocks first, then the yellow blocks.
        stacks = [(green_snapshot, GREEN_STACK), (yellow_snapshot, YELLOW_STACK)]
        placed = 0
        try:
            for blocks, stack_xy in stacks:
                x, y = stack_xy
                z = PICK_Z
                for start_xy in blocks:
                    res = node.move_block_vision(start_xy, (x, y, z))
                    if res:
                        z = z + BLOCK_HEIGHT
                        placed = placed + 1
        except RuntimeError as error:
            node.get_logger().error(str(error))
            return

        # ========================= Student's code ends here ===========================

        node.move_arm(go_away)
        node.get_logger().info(f"Finished. Placed {placed} of 4 blocks.")

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
