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

from crazyflie_mpc.controllers.oqsp_mpc import oqsp_MPC
from crazyflie_mpc.controller_utils import *

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped

from nav_msgs.msg import Odometry
from nav_msgs.msg import Path

from tf_transformations import euler_from_quaternion

import numpy as np
import casadi as ca
import control

import math
from math import sin
from math import cos
import time

from crazyflie_interfaces.msg import LogDataGeneric
from crazyflie_interfaces.msg import VelocityWorld
from crazyflie_interfaces.msg import FullState

from crazyflie_interfaces.msg import Position
from crazyflie_interfaces.srv import Takeoff
from crazyflie_interfaces.srv import Land

from copy import deepcopy
from scipy import sparse
import rowan



TARGET_HEIGHT = 0.5
FREQUENCY = 40

# THRUST_OFFSET = 38500     # small crazyflie
# MASS = 0.038              # small crazyflie

# THRUST_OFFSET = 42000       # big crazyflie
# MASS = 0.054                # big crazyflie

THRUST_OFFSET = 49200       # big crazyflie with mocap deck
MASS = 0.074                # big crazyflie with mocap deck

# THRUST_OFFSET = 38500   # simulation
# THRUST_OFFSET = 49000   # simulation
# MASS = 0.04            # simulation

class SimpleMPC(Node):
    
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
            
    def takeoff(self,targetHeight, duration, groupMask = 0):
        req = Takeoff.Request()
        req.group_mask = groupMask
        req.height = targetHeight
        req.duration = Duration(seconds=duration).to_msg()
        self.take_off_client.call_async(req)
        
    def land(self, targetHeight, duration, groupMask = 0):
        req = Land.Request()
        req.group_mask = groupMask
        req.height = targetHeight
        req.duration = Duration(seconds=duration).to_msg()
        self.land_client.call_async(req)

    def __init__(self):
        super().__init__('simple_mpc_controller')
        
        # Publishers and subscribers
        self.pose_sub           = self.create_subscription(PoseStamped, 'pose', self.pose_callback, 1)
        self.velocity_sub       = self.create_subscription(LogDataGeneric, 'velocity', self.velocity_callback, 1)

        self.cmd_vel_pub        = self.create_publisher(Twist, 'cmd_vel_legacy', 1)
        self.cmd_vel_world_pub  = self.create_publisher(VelocityWorld, 'cmd_velocity_world', 1)
        self.cmd_full_state_pub = self.create_publisher(FullState, 'cmd_full_state', 1)
        self.cmd_vel_z_disrance_pub = self.create_publisher(Twist, 'cmd_vel_z_distance', 1)
        
        self.path_pub           = self.create_publisher(Path, 'mpc_controller/path', 10)
        self.cmd_position_pub   = self.create_publisher(Position, 'cmd_position', 10)
        self.ref_pub            = self.create_publisher(PoseStamped, 'ref_pose', 10)
        self.u_pub              = self.create_publisher(PoseStamped, 'acc_u', 10)
        
        # Clients
        self.take_off_client    = self.create_client(Takeoff, "takeoff")
        self.land_client        = self.create_client(Land, "land")
        
        self.last_odom = Odometry()
        self.last_pose = PoseStamped()
        self.x0,    self.y0,    self.z0     = 0, 0, 0
        self.xv0,   self.yv0,   self.zv0    = 0, 0, 0
        self.az0    = 0
        self.quaternion_x, self.quaternion_y, self.quaternion_z, self.quaternion_w = 0, 0, 0, 0
        self.last_u = None
        
        self.logger = self.get_logger()
        self.logger.info("Initializing MPC controller.")

        self.mass = MASS
        self.g      = 9.81
        self.dt     = 1.0 / FREQUENCY
        self.N      = 100
        
        self.init_mpc_planner()    
        self.logger.info("MPC controller initialized.")
        self.target_height = TARGET_HEIGHT
        
        self.REFERENCE_FOLLOWING_TIME = 10.0
        self.LAND_TIME = 20.0
        
        # Takeoff
        # self.logger.info("Taking off.")
        # self.takeoff(self.target_height, duration=2.0)
        # self.sleep(self.target_height+2.0)
        
        # # Hover for 2 seconds
        # self.logger.info("Hovering.")
        # self.sleep(2.0)
        
        # Land
        # self.logger.info("Landing.")
        # self.land(targetHeight=0.03, duration=self.target_height+1.0)

        # # Send zeros to unlock the controller
        self.unlocked_counter = 0
        
        # Create timer to run MPC
        self.timer_period = self.dt
        self.timer = self.create_timer(self.timer_period, self.timer_callback, clock=self.get_clock())
        self.logger.info("Timer created.")
        self.ready_to_solve = False
        self.start_time = -1
        
    def init_mpc_planner(self):
                
        self.mpc_planner = oqsp_MPC(
            N=self.N,
            fz_max=2.0,
            logger=self.logger
        )
        
    def pose_callback(self, msg : PoseStamped):
        """Callback for the pose subscriber."""
        # self.logger.info("Got pose message.")
        
        self.x0 = msg.pose.position.x
        self.y0 = msg.pose.position.y
        self.z0 = msg.pose.position.z

        self.quaternion_x = msg.pose.orientation.x
        self.quaternion_y = msg.pose.orientation.y
        self.quaternion_z = msg.pose.orientation.z
        self.quaternion_w = msg.pose.orientation.w

        self.last_pose = deepcopy(msg)

    def velocity_callback(self, msg : LogDataGeneric):

        self.xv0 = msg.values[0]
        self.yv0 = msg.values[1]
        self.zv0 = msg.values[2]
        self.ready_to_solve = True

    def publish_predicted_path(self,x):
        
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "world"
        path_msg.poses = []
        
        for i in range(self.mpc_planner.N):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "world"
            pose.pose.position.x = x[0,i]
            pose.pose.position.y = x[1,i]
            pose.pose.position.z = x[2,i]
            path_msg.poses.append(pose)
            
        self.path_pub.publish(path_msg)      
        
    def publish_cmd_position(self, x,y,z,yaw):
        
        cmd_position = Position()
        cmd_position.header.stamp = self.get_clock().now().to_msg()
        cmd_position.header.frame_id = "world"
        cmd_position.x = x
        cmd_position.y = y
        cmd_position.z = z
        cmd_position.yaw = yaw
        
        self.cmd_position_pub.publish(cmd_position)
        
    def timer_callback(self):
        """Callback for the timer."""

        if self.start_time ==-1:
            self.start_time = self.time()

        # Get current state
        x = np.array([
            self.x0,
            self.y0,
            self.z0,
            self.xv0,
            self.yv0,
            self.zv0
        ])

        # Follow circle after 10 seconds

        height_ctrl = 0.0

        if self.start_time !=-1:
            time_since_start = self.time() - self.start_time
            if time_since_start >=  self.REFERENCE_FOLLOWING_TIME and time_since_start <  self.LAND_TIME :

                # Create reference for x and y with N steps look ahead, with circle defined by sin(x * 2 * pi * (1/5))
                time_references = [(time_since_start + i*self.dt) for i in range(self.N)]
                # x_ref = [ 0.0 for i in range(self.N)]
                # y_ref = [ 0.0 for i in range(self.N)]
                x_ref = [(0.5*math.sin(t*2*math.pi*(1/5))) for t in time_references]
                y_ref = [(0.5*math.cos(t*2*math.pi*(1/5))) for t in time_references]
                # z_ref = [ self.target_height for i in range(self.N)]
                z_ref = [(self.target_height +0.25*math.cos(t*2*math.pi*(1/2))) for t in time_references]

                ref_pose = PoseStamped()
                ref_pose.pose.position.x = x_ref[0]
                ref_pose.pose.position.y = y_ref[0]
                ref_pose.pose.position.z = z_ref[0]
                self.ref_pub.publish(ref_pose)    

                self.mpc_planner.set_reference_trajectory(
                    x_ref = np.array(list(([x, y, z, 0, 0 , 0]) for x, y, z in zip(x_ref, y_ref, z_ref))),
                    x_ref_f = np.array([x_ref[-1], y_ref[-1], z_ref[-1],0,0,0])
                )

                height_ctrl = z_ref[20]

            elif time_since_start >=  self.LAND_TIME:

                land_height = 0.05

                x_ref = [ 0.0 for i in range(self.N)]
                y_ref = [ 0.0 for i in range(self.N)]
                z_ref = [ land_height for i in range(self.N)]
                
                self.mpc_planner.set_reference_trajectory(
                        x_ref = np.array(list(([x, y, z, 0, 0 , 0]) for x, y, z in zip(x_ref, y_ref, z_ref))),
                        x_ref_f = np.array([x_ref[-1], y_ref[-1], z_ref[-1],0,0,0])
                    )
                
                ref_pose = PoseStamped()
                ref_pose.pose.position.x = x_ref[0]
                ref_pose.pose.position.y = y_ref[0]
                ref_pose.pose.position.z = z_ref[0]
                self.ref_pub.publish(ref_pose) 

                height_ctrl = z_ref[2]

            else:
                ref_pose = PoseStamped()
                ref_pose.pose.position.x = 0.0
                ref_pose.pose.position.y = 0.0
                ref_pose.pose.position.z = self.target_height
                self.ref_pub.publish(ref_pose) 
                height_ctrl = self.target_height


        if not self.ready_to_solve:
            self.logger.warn("Not ready to solve!")
            return
        
        self.mpc_planner.set_initial_state(x)

        x_pred,u, u2, solve_time = self.mpc_planner.solve(verbose=False)
        self.last_u = u
        
        if solve_time >= self.dt:
            self.logger.warn(f"========================Solve time exceeds control rate {1/solve_time} ========================")
        
        u_acc = PoseStamped()
        u_acc.pose.position.x = u[0]
        u_acc.pose.position.y = u[1]
        u_acc.pose.position.z = u[2]
        self.u_pub.publish(u_acc)

        # self.publish_predicted_path(x_pred)
        # self.logger.info(f"Predicted state: {x_pred[:,0]}")
        
        px_pred = x_pred[0,0]
        py_pred = x_pred[1,0]
        pz_pred = x_pred[2,0]
        
        vx_pred = x_pred[3,0]
        vy_pred = x_pred[4,0]
        vz_pred = x_pred[5,0]

        if time_since_start >=  self.LAND_TIME:
            height_ctrl = x_pred[2,10]

        # self.logger.info(f"Predicted position: {px_pred}, {py_pred}, {pz_pred}")
        # self.logger.info(f"Predicted velocity: {vx_pred}, {vy_pred}, {vz_pred}")
               
        # if self.unlocked_counter < 0:
        #     self.emergency_stop()
        #     self.unlocked_counter +=1
        #     self.logger.info(f"Unlocked counter {self.unlocked_counter}")
        #     return
        
        roll, pitch, yaw = euler_from_quaternion([ self.quaternion_x, self.quaternion_y, self.quaternion_z, self.quaternion_w])
        thrust_des, roll_des, pitch_des = acc2TRP([u[0],u[1],u[2]], yaw, self.mass, THRUST_OFFSET)
        thrust_des = T2cmd(thrust_des) 
        yaw_des     = 0.0
        roll_des    = math.degrees(roll_des)
        pitch_des   = math.degrees(pitch_des)
        
        self.logger.info(f"Desired roll: {roll_des}, pitch: {pitch_des}, thrust: {thrust_des}")
        
        thrust_des_2, roll_des_2, pitch_des_2 = acc2TRP([u2[0],u2[1],u2[2]], yaw, self.mass, THRUST_OFFSET)
        thrust_des_2 = T2cmd(thrust_des_2)
        roll_des_2    = math.degrees(roll_des_2)
        pitch_des_2   = math.degrees(pitch_des_2)
        
        self.logger.info(f"Desired roll: {roll_des_2}, pitch: {pitch_des_2}, thrust: {thrust_des_2}")
        
        des_roll_rate = (roll_des_2 - roll_des) / self.dt
        des_pitch_rate = (pitch_des_2 - pitch_des) / self.dt
        des_yaw_rate = 0.0
        
        # cmd_vel_legacy
        # twist  = Twist()
        # twist.linear.x = -pitch_des
        # twist.linear.y = roll_des
        # twist.angular.z = 0.0
        # twist.linear.z = 0.0  
        # # twist.linear.z = thrust_des  
        # self.cmd_vel_pub.publish(twist)
        
        # cmd_velocity_world
        # cmd_vel_world = VelocityWorld()
        # cmd_vel_world.vel.x = vx_pred * 0
        # cmd_vel_world.vel.y = vy_pred * 0
        # cmd_vel_world.vel.z = vz_pred * 0 + 0.5
        # cmd_vel_world.yaw_rate = 0.0
        # self.cmd_vel_world_pub.publish(cmd_vel_world)        
        
        # cmd_full_state_pub
        # cmd_full_state = FullState()
        
        # cmd_full_state.pose.position.x = px_pred
        # cmd_full_state.pose.position.y = py_pred
        # cmd_full_state.pose.position.z = pz_pred
        
        # cmd_full_state.twist.linear.x = vx_pred
        # cmd_full_state.twist.linear.y = vy_pred
        # cmd_full_state.twist.linear.z = vz_pred
        
        # q = rowan.from_euler(roll, pitch, yaw)
        
        # cmd_full_state.acc.x = u[0]
        # cmd_full_state.acc.y = u[1]
        # cmd_full_state.acc.z = u[2]
        
        # cmd_full_state.twist.angular.x = - des_pitch_rate
        # cmd_full_state.twist.angular.y = des_roll_rate
        # cmd_full_state.twist.angular.z = 0.0
        
        # cmd_full_state.pose.orientation.x = q[0]
        # cmd_full_state.pose.orientation.y = q[1]
        # cmd_full_state.pose.orientation.z = q[2]
        # cmd_full_state.pose.orientation.w = q[3]
           
        # self.cmd_full_state_pub.publish(cmd_full_state)        
        
        # cmd_vel_z_disrance_pub
        twist  = Twist()
        twist.linear.x = pitch_des
        twist.linear.y = roll_des
        twist.angular.z = 0.0
        twist.linear.z = height_ctrl
        self.cmd_vel_z_disrance_pub.publish(twist)
        
        
    def emergency_stop(self):
        
        twist  = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        twist.linear.z = 0.0
        self.cmd_vel_pub.publish(twist)
    
def main(args=None):
    
    rclpy.init(args=args)
    minimal_publisher = SimpleMPC()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()