# MIT License

# Copyright (c) 2023 Matthew Lock

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Todo: Cleanup imports.

from abc import ABC, abstractmethod
from copy import deepcopy

from crazyflie_interfaces.msg import LogDataGeneric
from crazyflie_interfaces.msg import Position
from crazyflie_interfaces.srv import Land
from crazyflie_interfaces.srv import Takeoff

from crazyflie_mpc_msgs.srv import ReferenceTrajectory

from crazyflie_mpc.controller_utils import *
from crazyflie_mpc.controllers.oqsp_mpc import oqsp_MPC

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped

from math import cos
from math import sin

from nav_msgs.msg import Odometry
from nav_msgs.msg import Path

from rclpy.duration import Duration
from rclpy.node import Node

from std_msgs.msg import Float64
from tf_transformations import euler_from_quaternion

import casadi as ca
import control
import math
import numpy as np
import rclpy
import time

# This controller does not use the thrust as an input, but rather directly uses the desired altitude.
# The thrust offset below is left for legacy reasons.

CF_MASS = 0.074
CF_THRUST_OFFSET = 48200


class CrazyflieMPC(ABC, Node):

    def time(self):
        """Returns the current time in seconds."""
        if self.get_clock().now().nanoseconds == 0:
            rclpy.spin_once(self, timeout_sec=0)
        return self.get_clock().now().nanoseconds / 1e9

    def sleep(self, duration):
        """Sleeps for the provided duration in seconds."""
        start = self.time()
        end = start + duration
        while self.time() < end:
            # self.logger.info(f"Sleeping for {end - self.time()} seconds.")
            rclpy.spin_once(self, timeout_sec=0)

    def takeoff(self, targetHeight, duration, groupMask=0):
        req = Takeoff.Request()
        req.group_mask = groupMask
        req.height = targetHeight
        req.duration = Duration(seconds=duration).to_msg()
        self.take_off_client.call_async(req)

    def land(self, targetHeight, duration, groupMask=0):
        req = Land.Request()
        req.group_mask = groupMask
        req.height = targetHeight
        req.duration = Duration(seconds=duration).to_msg()
        self.land_client.call_async(req)
        
    def init_publishers(self):
        """Initialize the publishers."""
        
        cmd_vel_topic = "cmd_vel_legacy"
        mpc_pred_path_topic = "mpc/predicted_path"
        cmd_position_topic = "cmd_position"
        control_topic = "acc_u"
        cmd_vel_z_distance_topic = "cmd_vel_z_distance"
        
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 1)
        self.mpc_path_pub = self.create_publisher(
            Path, mpc_pred_path_topic, 10)
        self.cmd_position_pub = self.create_publisher(
            Position, cmd_position_topic, 10)
        self.control_pub = self.create_publisher(PoseStamped, control_topic, 10)
        self.cmd_vel_z_distance_pub = self.create_publisher(
            Twist, cmd_vel_z_distance_topic, 1)
        
    def init_subscribers(self):
        """Initialize the subscribers."""
        
        pose_topic = "pose"
        velocity_topic = "velocity"
        
        self.pose_sub = self.create_subscription(
            PoseStamped, pose_topic, self.pose_callback, 1)
        self.velocity_sub = self.create_subscription(
            LogDataGeneric, velocity_topic, self.velocity_callback, 1)
        
    def init_clients(self):
        """Initialize the clients."""
        
        self.take_off_client = self.create_client(Takeoff, "takeoff") # Connects to the takeoff service (crazyswarm).
        self.land_client = self.create_client(Land, "land") # Connects to the land service (crazyswarm)

    def __init__(self, control_frequency=50, N=100):
        """
        Initialize the MPC controller.

        Parameters
        ----------
        control_frequency : int
            The control frequency of the controller.
        N : int
            The prediction horizon (number of steps).
        """
        super().__init__('simple_mpc_controller')
        
        self.init_publishers()
        self.init_subscribers()
        self.init_clients()

        self.last_pose = PoseStamped()
        self.last_twist = TwistStamped()

        self.logger = self.get_logger()
        self.logger.info("Initializing MPC controller.")

        self.mass = CF_MASS
        self.g = 9.81
        self.dt = 1.0 / control_frequency
        self.N = N

        self.init_mpc_controller(freq=control_frequency)
        self.logger.info("MPC controller initialized.")

        # Create timer to run MPC
        self.timer_period = self.dt
        self.control_timer = self.create_timer(
            self.timer_period, self.control_timer_callback, clock=self.get_clock())

        self.ready_to_solve = False
        self.unlocked_counter = 0

    def init_mpc_controller(self, freq: int):
        """
        Initialize the MPC planner.

        Parameters
        ----------
        freq : int
            The control frequency of the controller.
        """

        self.controller = oqsp_MPC(
            freq=freq,
            N=self.N,
            logger=self.logger
        )

    def pose_callback(self, msg: PoseStamped):
        """Callback for the pose subscriber."""
        # self.logger.info("Got pose message.")

        self.last_pose = deepcopy(msg)

    def velocity_callback(self, msg: LogDataGeneric):
        """
        Callback for the velocity subscriber.
        """

        self.last_twist.header.stamp = msg.header.stamp
        self.last_twist.twist.linear.x = msg.values[0]
        self.last_twist.twist.linear.y = msg.values[1]
        self.last_twist.twist.linear.z = msg.values[2]

        self.ready_to_solve = True

    def publish_predicted_path(self, path):
        """
        Publish the predicted path from the MPC controller.

        Parameters
        ----------
        path : np.array
            The predicted path from the MPC controller. np.array of shape (3, N).
        """

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "world"
        path_msg.poses = []

        for i in range(self.controller.N):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "world"
            pose.pose.position.x = path[0, i]
            pose.pose.position.y = path[1, i]
            pose.pose.position.z = path[2, i]
            path_msg.poses.append(pose)

        self.mpc_path_pub.publish(path_msg)

    def publish_cmd_position(self, x, y, z, yaw):

        cmd_position = Position()
        cmd_position.header.stamp = self.get_clock().now().to_msg()
        cmd_position.header.frame_id = "world"
        cmd_position.x = x
        cmd_position.y = y
        cmd_position.z = z
        cmd_position.yaw = yaw

        self.cmd_position_pub.publish(cmd_position)

    @abstractmethod
    def get_reference_trajectory(self):
        """
        Get a reference trajectory from the reference trajectory service.

        Returns
        -------
        np.array
            The reference trajectory. np.array of shape (N, 6).
        np.array
            The final reference state. np.array of shape (6,).
        """
        
        # Example reference trajectory of zeros.
        
        # x_ref = [0 for i in range(100)]
        # y_ref = [0 for i in range(100)]
        # z_ref = [0 for i in range(100)]
        
        # ref_traj = np.array(list(([x, y, z, 0, 0, 0])
        #                     for x, y, z in zip(x_ref, y_ref, z_ref)))
        # ref_traj_f = np.array([x_ref[-1], y_ref[-1], z_ref[-1], 0, 0, 0])
        
        pass

    def control_timer_callback(self):
        """Callback for the timer."""
        
        if not self.ready_to_solve:
            self.logger.warn("Not ready to solve!")
            return

        # Get current state
        x = np.array([
            self.last_pose.pose.position.x,
            self.last_pose.pose.position.y,
            self.last_pose.pose.position.z,
            self.last_twist.twist.linear.x,
            self.last_twist.twist.linear.y,
            self.last_twist.twist.linear.z
        ])
        
        ref_traj, ref_traj_f = self.get_reference_trajectory()

        self.controller.set_reference_trajectory(x_ref=ref_traj, x_ref_f=ref_traj_f)
        self.controller.set_initial_state(x)

        # height_ctrl = z_ref[20]

        # Solve MPC
        
        # The retruned control input 'u' is the acceleration in the body frame. I.e
        # u = [u_x, u_y, u_z]. The control input is then converted to thrust and roll/pitch
        
        pred_traj, u, _ , solve_time = self.controller.solve()
        self.last_u = u

        if solve_time >= self.dt:
            self.logger.warn( f"Solve time exceeds control period: {solve_time} s")

        # Publish control input
        u_acc = PoseStamped()
        u_acc.pose.position.x = u[0]
        u_acc.pose.position.y = u[1]
        u_acc.pose.position.z = u[2]
        self.control_pub.publish(u_acc)

        # Publish predicted path
        self.publish_predicted_path(pred_traj)

        if self.unlocked_counter < 1:
            # The crazyflie requires some zero inputs to unlock.
            self.emergency_stop()
            self.unlocked_counter += 1
            return

        euler = euler_from_quaternion([self.last_pose.pose.orientation.x, self.last_pose.pose.orientation.y,
                                      self.last_pose.pose.orientation.z, self.last_pose.pose.orientation.w])
        roll, pitch, yaw = euler

        # Convert control input to thrust and roll/pitch.
        # Note: The thrust is not used as a control input anymore.
        
        thrust_des, roll_des, pitch_des = acc2TRP(
            [u[0], u[1], u[2]], yaw, self.mass, CF_THRUST_OFFSET)
        desired_altitude = ref_traj[10, 2] # Todo : Find a better way to set this value. Currently offset to account for communication delay.
        # thrust_des = T2cmd(thrust_des)

        roll_des = math.degrees(roll_des)
        pitch_des = math.degrees(pitch_des)

        twist = Twist()
        twist.linear.x = pitch_des
        twist.linear.y = roll_des
        twist.angular.z = 0.0
        twist.linear.z = desired_altitude
        self.cmd_vel_z_distance_pub.publish(twist)

    def emergency_stop(self):

        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        twist.linear.z = 0.0
        self.cmd_vel_pub.publish(twist)