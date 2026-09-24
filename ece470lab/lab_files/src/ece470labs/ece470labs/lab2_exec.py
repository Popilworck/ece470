#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from ur_msgs.srv import SetIO
from ur_msgs.msg import IOStates, Analog
import time
import numpy as np
from math import pi
import sys
class JointAngles:
    def __init__(self):
        self.name = ["", "", "", "", "", ""]  #could have also done [""] * 6
        self.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# UR3e home position
home = np.radians([180, -80, 90, -90, -90, 13])

# Hanoi tower location 
Q11 = [168.15*pi/180.0, -48.60*pi/180.0, 101.60*pi/180.0, -143.37*pi/180.0, -89.47*pi/180.0, 1.34*pi/180.0]   #a1
Q12 = [181.81*pi/180.0, -44.79*pi/180.0, 91.40*pi/180.0, -134.34*pi/180.0, -89.72*pi/180.0, 15.65*pi/180.0]      #b1
Q13 = [196*pi/180.0, -34.3*pi/180.0, 66.92*pi/180.0, -120.94*pi/180.0, -94.07*pi/180.0, 15.39*pi/180.0]     #c1
Q21 = [167.75*pi/180.0, -56.22*pi/180.0, 103.88*pi/180.0, -141.61*pi/180.0, -92.85*pi/180.0, 6.12*pi/180.0]
Q22 = [182.62*pi/180.0, -49.87*pi/180.0, 91.08*pi/180.0, -134.02*pi/180.0, -92.76*pi/180.0, 7.80*pi/180.0]
Q23 = [195.35*pi/180.0, -40.52*pi/180.0, 71.65*pi/180.0, -123.39*pi/180.0, -93.44*pi/180.0, 14.55*pi/180.0]
Q31 = [167.26*pi/180.0, -60.30*pi/180.0, 98.17*pi/180.0, -128.76*pi/180.0, -91.71*pi/180.0, 3.45*pi/180.0]
Q32 = [181.25*pi/180.0, -53.58*pi/180.0, 86.61*pi/180.0, -124.66*pi/180.0, -91.09*pi/180.0, 3.44*pi/180.0]
Q33 = [194.92*pi/180.0, -40.91*pi/180.0, 62.47*pi/180.0, -112.53*pi/180.0, -92.29*pi/180.0, 3.04*pi/180.0]
############## Your Code Start Here ##############
"""
: Initialize Q matrix
"""

Q = [ [Q11, Q12, Q13], \
      [Q21, Q22, Q23], \
      [Q31, Q32, Q33] ]
############### Your Code End Here ###############
class UR3e(Node):
    def __init__(self):
        super().__init__('ur3e')

        # Publishers
        self.trajectory_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        ############## Your Code Start Here ##############
        # TODO: define a ROS subscriber for gripper input message and corresponding callback function
        # ROS2 gripper input topic: /io_and_status_controller/io_states

        self.gripper_input_sub = self.create_subscription(IOStates,'io_and_status_controller/io_states', self.io_state_callback, 10)

        ############### Your Code End Here ###############

        # Service clients
        self.io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')
        while not self.io_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('IO service not available, waiting...')

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
    ############## Your Code Start Here ##############
        """
        TODO: define a ROS topic callback funtion that 
        receives and stores the state of  the suction cup
        Whenever /io_and_status_controller/io_states 
        publishes this info, this callback function is
        called.
        """
        if (len(msg.analog_in_states) < 1): 
            print("Invalid IO state message")
            return
        # self.analog_in_0_value = msg.analog_in_states[0].state > 2
        for pin_state in msg.analog_in_states:
            if (pin_state.pin == 0):
                self.analog_in_0_value = pin_state.state
                break

    ############### Your Code End Here ###############

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


    def move_block(self, start_tower, start_height, end_tower, end_height):
        global Q
    ############## Your Code Start Here ##############
    # TODO: add code to move block from start tower and height to end tower and height
    ### Hint: Use the Q array to map out your towers by location and "height".

        error = False
        to_move = Q[start_height][start_tower]
        res = self.move_arm(to_move)
        if not res:
            print("Did not move")
            return True
         
        self.set_io(0,1.0)
        start_time = time.time()
        while time.time() - start_time < 0.5:
            rclpy.spin_once(self, timeout_sec=0.1)

        print(f'hahhahahahaha {self.analog_in_0_value}')
        if(self.analog_in_0_value < 2):
            print("Quitting... not found block ")
            self.set_io(0,0.0)
            raise InterruptedError

        res = self.move_arm(home)
        if not res:
            print("Did not move")
            return True
        res = self.move_arm(Q[end_height][end_tower])
        if not res:
            print("Did not move")
            return True
        time.sleep(0.1)
        self.set_io(0,0.0)

        self.move_arm(home)
        return False

    def tower_of_hanoi(self, n, source, target, auxiliary, rods_state):

        if n == 0:
            return

        # Step 1: Move n-1 disks from source to auxiliary
        self.tower_of_hanoi(n - 1, source, auxiliary, target, rods_state)

        # Step 2: Move the remaining largest disk from source to target
        self.move_block(source, len(rods_state[source])-1, target, len(rods_state[target]))
        disk = rods_state[source].pop()
        rods_state[target].append(disk)
        
        # Print the current step and state of the rods
        # print(f"Move disk {disk} from {source} to {target}")
        # print(f"Current State -> A: {rods_state['A']}, B: {rods_state['B']}, C: {rods_state['C']}\n")

        # Step 3: Move the n-1 disks from auxiliary to target
        self.tower_of_hanoi(n - 1, auxiliary, target, source, rods_state)

        
    ############### Your Code End Here ###############


def main(args=None):
    input("Check if the UR3e is in 'Remote' Mode?\n\
    Check if the UR3e is initialized and in 'Normal' state.\n\
    Have you run the ROS2 launch statement?\n\
    If there was an UR3e emergency stop or error, Ctrl-C the ros2 launch and rerun.\n\
    \n\
    Press <Enter> to Continue.")
    rclpy.init(args=args)
    node = UR3e()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    ############## Your Code Start Here ##############
    # TODO: modify the code below so that program can get user input
    loop_count = 0
    # Wait for initial state updates
    while node.current_joint_state is None:
        executor.spin_once(timeout_sec=0.05)
        node.get_logger().info("Waiting for initial state updates...")
        time.sleep(0.5)

    try:
        # Get user input
        input_string = input("Enter 'start stop' with a space in the middle  <Either 1 2 3 or 0 to quit> ")
        print("You entered " + input_string + "\n")
               
       
        if (len(input_string)>1):
            start,stop = input_string.split()
            start = int(start) -1
            stop = int(stop) -1 
        elif (int(input_string) == 0):
            print("Quitting... ")
            sys.exit()
        else:
            print("Please just enter the character 1 2 3 or 0 to quit \n\n")

        aux = [i for i in [0,1,2] if i not in [start,stop]][0]

        rods = {start:[],stop:[], aux:[]}
        rods[start] = [1,2,3]
        node.move_arm(home)
        node.tower_of_hanoi(3, start, stop, aux, rods)
        ############## Your Code Start Here ##############
        # TODO: modify the code so that UR3e can move tower accordingly from user input

        # while(loop_count > 0):

        #     node.move_arm(home)

        #     node.get_logger().info(f'Sending goal 1 ...')

        #     if not node.move_arm(Q[0][0]):
        #         node.get_logger().error("Failed to move to goal" + str(Q[0][0]))
        #         break

        #     node.set_io(0, 1.0)  # Turn/ on suction
        #     # Delay to make sure suction cup has grasped the block
        #     time.sleep(1.0)

        #     node.get_logger().info(f'Sending goal 2 ...')
        #     if not node.move_arm(Q[1][1]):
        #         node.get_logger().error("Failed to move to goal"+str(Q[1][1]))
        #         break

        #     node.get_logger().info(f'Sending goal 3 ...')
        #     if not node.move_arm(Q[2][2]):
        #         node.get_logger().error("Failed to move to goal"+str(Q[2][2]))
        #         break
        #     loop_count = loop_count - 1
        #     node.set_io(0, 0.0)  # Turn off suction


    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
