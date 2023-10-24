from crazyflie_mpc.controllers.simple_mpc import MPC

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

from crazyflie_interfaces.msg import Position
from crazyflie_interfaces.srv import Takeoff
from crazyflie_interfaces.srv import Land

from copy import deepcopy

TARGET_HEIGHT = 0.5
FREQUENCY = 20

THRUST_OFFSET = 38500     # small crazyflie
MASS = 0.034              # small crazyflie

# THRUST_OFFSET = 43000       # big crazyflie
# MASS = 0.044                # big crazyflie

# THRUST_OFFSET = 42000       # big crazyflie
# MASS = 0.054                # big crazyflie


# THRUST_OFFSET = 43000

# THRUST_OFFSET = 49000   # simulation
# MASS = 0.065            # simulation


# MASS = 0.044
# MASS = 0.034


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
        
    def TRP2acc(self,TRP, psi):
        """
        Thrust, Roll, Pitch Command to desired accelerations
        where T is already in [N] and angles in [rad]
        """
        T = TRP[0]
        phi = TRP[1]
        theta = TRP[2]
        ax = self.g * (sin(psi) * phi + cos(psi) * theta)
        ay = self.g * (-cos(psi) * phi + sin(psi) * theta)
        az = T / self.mass
        return np.array([ax, ay, az])
    
    def acc2TRP(self,a, psi):
        """
        desired accelerations in global frame to Thrust, Roll, Pitch command
        """
        
        T = self.mass * a[2] + self.cmd2T(THRUST_OFFSET)
        phi = (a[0] * sin(psi) - a[1] * cos(psi)) / self.g
        theta = (a[0] * cos(psi) + a[1] * sin(psi)) / self.g
        
        return np.array([T, phi, theta])

    def __init__(self):
        super().__init__('simple_mpc_controller')
        
        # Publishers and subscribers
        self.odom_sub           = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.pose_sub           = self.create_subscription(PoseStamped, 'pose', self.pose_callback, 10)
        self.cmd_vel_pub        = self.create_publisher(Twist, 'cmd_vel_legacy', 1)
        self.path_pub           = self.create_publisher(Path, 'mpc_controller/path', 10)
        self.cmd_position_pub   = self.create_publisher(Position, 'cmd_position', 10)
        
        # Clients
        self.take_off_client    = self.create_client(Takeoff, "takeoff")
        self.land_client        = self.create_client(Land, "land")
        
        self.last_odom = Odometry()
        self.last_pose = PoseStamped()
        self.x0,    self.y0,    self.z0     = 0, 0, 0
        self.xv0,   self.yv0,   self.zv0    = 0, 0, 0
        self.quaternion_x, self.quaternion_y, self.quaternion_z, self.quaternion_w = 0, 0, 0, 0
        
        self.logger = self.get_logger()
        self.logger.info("Initializing MPC controller.")
        
        # self.mass   = 0.045
        # self.mass   = 0.034
        # self.mass = 0.0288
        self.mass = MASS
        self.g      = 9.81
        self.dt     = 1.0 / FREQUENCY
        self.init_model()   
        
        # Thrust mapping: Force --> PWM value (inverse mapping from system ID paper)
        a2 = 2.130295 * 1e-11
        a1 = 1.032633 * 1e-6
        a0 = 5.484560 * 1e-4
        self.cmd2T = lambda cmd: 4 * (a2 * (cmd) ** 2 + a1 * (cmd) + a0)
        self.T2cmd = lambda T: (- (a1 / (2 * a2)) + math.sqrt(a1**2 / (4 * a2**2) - (a0 - (max(0, T) / 4)) / a2))
        
        self.is_low_level_flight_active = False
        
        self.init_mpc_planner()    
        self.logger.info("MPC controller initialized.")
        self.target_height = TARGET_HEIGHT
        
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
        
    def init_model(self):
        
        dimensions  = 3
        self.n      = dimensions*2 # number for double integrator
        self.m      = dimensions
        
        self.Ac = ca.DM.zeros(self.n, self.n)
        self.Bc = ca.DM.zeros(self.n, self.m)
        
        self.Ac[0, 3] = 1
        self.Ac[1, 4] = 1
        self.Ac[2, 5] = 1
        
        self.Bc[3, 0] = 1
        self.Bc[4, 1] = 1
        self.Bc[5, 2] = 1
        
        self.continous_system = control.StateSpace(self.Ac, self.Bc, ca.DM.eye(self.n), ca.DM.zeros(self.n, self.m))
        self.discrete_system = self.continous_system.sample(self.dt)
        
        x_lim = 5
        y_lim = 5
        z_lim = 5
        v_lim = 3
        u_lim_1 = 1
        u_lim = 1
        
        self.x_upper_bound = ca.DM([x_lim, y_lim, z_lim, v_lim, v_lim, v_lim])
        self.u_upper_bound = ca.DM([u_lim_1, u_lim_1, u_lim])
        self.x_lower_bound = -self.x_upper_bound
        self.u_lower_bound = -self.u_upper_bound
        
        def model(x, u):
            return self.discrete_system.A@x + self.discrete_system.B@u
        self.model = model
        
        
    def init_mpc_planner(self):
                
        self.mpc_planner = MPC(
            x0=np.array([0, 0, 0, 0, 0, 0]),
            x_upper_bound=self.x_upper_bound,
            x_lower_bound=self.x_lower_bound,
            u_upper_bound=self.u_upper_bound,
            u_lower_bound=self.u_lower_bound,
            model=self.model,
            N=10,
            x_ref_states=np.array([0, 1, 2]),
            logger=self.logger
        )
        
        
    def odom_callback(self, msg : Odometry):
        """Callback for the odometry subscriber."""
        self.last_odom = msg
        # self.logger.info("Got odometry message.")      
        
        self.x0     = msg.pose.pose.position.x
        self.y0     = msg.pose.pose.position.y
        self.z0     = msg.pose.pose.position.z
        self.xv0    = msg.twist.twist.linear.x
        self.yv0    = msg.twist.twist.linear.y
        self.zv0    = msg.twist.twist.linear.z
        
        self.quaternion_x = msg.pose.pose.orientation.x
        self.quaternion_y = msg.pose.pose.orientation.y
        self.quaternion_z = msg.pose.pose.orientation.z
        self.quaternion_w = msg.pose.pose.orientation.w
        
    def pose_callback(self, msg : PoseStamped):
        """Callback for the pose subscriber."""
        # self.logger.info("Got pose message.")
        
        self.x0 = msg.pose.position.x
        self.y0 = msg.pose.position.y
        self.z0 = msg.pose.position.z

        if self.last_pose is not None:
            
            dt = (msg.header.stamp.nanosec - self.last_pose.header.stamp.nanosec)/1e9
            if dt != 0:
                self.xv0 = (msg.pose.position.x - self.last_pose.pose.position.x)/dt
                self.yv0 = (msg.pose.position.y - self.last_pose.pose.position.y)/dt
                self.zv0 = (msg.pose.position.z - self.last_pose.pose.position.z)/dt
            else:
                self.logger.warn(f"No velicity avaiable, dt = {dt}")
            
        self.quaternion_x = msg.pose.orientation.x
        self.quaternion_y = msg.pose.orientation.y
        self.quaternion_z = msg.pose.orientation.z
        self.quaternion_w = msg.pose.orientation.w

        self.last_pose = deepcopy(msg)
        
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
         
        # Get current state
        x = np.array([
            self.x0,
            self.y0,
            self.z0,
            self.xv0,
            self.yv0,
            self.zv0
        ])

        # Round x to 4th decimal
        x = np.around(x, decimals=4)
        
        self.logger.info(f"x {x}")
        
        x_ref = np.array([self.x0, self.y0, self.target_height])
        x_ref = np.tile(x_ref, (self.mpc_planner.N+1, 1)).T
        
        self.mpc_planner.set_initial_state(x)
        self.mpc_planner.set_reference_trajectory(x_ref)
        x_pred,u, solve_time = self.mpc_planner.solve(verbose=True)

        if solve_time >= self.dt:
            self.logger.warn(f"========================Solve time exceeds control rate {1/solve_time} ========================")
        
        self.publish_predicted_path(x_pred)
               
        if self.unlocked_counter < 1:
            self.emergency_stop()
            self.unlocked_counter +=1
            self.logger.info(f"Unlocked counter {self.unlocked_counter}")
            # return
        
        # self.logger.info(f"u {u[:,0]}")
        
        euler = euler_from_quaternion([ self.quaternion_x, self.quaternion_y, self.quaternion_z, self.quaternion_w])

        roll    = euler[0]
        pitch   = euler[1]
        yaw     = euler[2]

        self.logger.info(f"u{u[:,0]}")
        
        thrust_des, pitch_des, roll_des = self.acc2TRP([u[0,0],u[1,0],u[2,0]*1], yaw)
        thrust_des = self.T2cmd(thrust_des) 
        
        # roll_des, pitch_des, thrust_des = self.acceleration_to_drone_command(u[:,0])
        
        # x_next = x_pred[0,1]
        # y_next = x_pred[1,1]
        # z_next = x_pred[2,1]
        # yaw_next = 0
        # self.cmd_position_pub(x_next,y_next,z_next,yaw_next)
        
        twist  = Twist()
        # twist.linear.x = pitch_des
        # twist.linear.y = roll_des
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        twist.linear.z = thrust_des  
        # twist.linear.z = 42400.0

        self.logger.info(f"thrust_des {thrust_des}")
        
        # # self.logger.info(f" roll_des {roll_des} pitch_des {pitch_des} thrust_des {thrust_des}")
        self.cmd_vel_pub.publish(twist)
        
        # self.logger.info(f"z {x[2:]}")
        # self.logger.info(f"z_dot {x[5:]}")
        # self.logger.info(f"u {u[2:]}")
        
        # Extract the first input
        
    def emergency_stop(self):
        
        twist  = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        twist.linear.z = 0.0
        self.cmd_vel_pub.publish(twist)
        
    def full_speed(self):
        
        twist  = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        twist.linear.z = 5000.0
        self.cmd_vel_pub.publish(twist)
        
    def thrust_to_pwn(self, thrust):
        
        # F/4 = 2.130295e-11 * PWM^2 + 1.032633e-6 * PWM + 5.484560e-4
        # Solve quadratic equation for PWM with F = thrust
        a = 2.130295e-11
        b = 1.032633e-6
        c = 5.484560e-4 - thrust/4
        
        # a,b,c = [ 1.71479058e-09,  8.80284482e-05, -2.21152097e-01 - thrust]
        
        PWM = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        PWM = max(0.0, PWM)
        PWM = min(65535.0, PWM)
        
        # a2 = 2.130295 * 1e-11
        # a1 = 1.032633 * 1e-6
        # a0 = 5.484560 * 1e-4
        
        # self.T2cmd = lambda T: (- (a1 / (2 * a2)) + np.sqrt(a1**2 / (4 * a2**2) - (a0 - (max(0, T) / 4)) / a2))
        
        return PWM
        
    def acceleration_to_drone_command(self, acc):
        """Convert acceleration to drone command."""
        
        euler = euler_from_quaternion([ self.quaternion_x, self.quaternion_y, self.quaternion_z, self.quaternion_w])

        roll    = euler[0]
        pitch   = euler[1]
        yaw     = euler[2]
        
        # self.logger.info(f"acc {acc}")
        
        roll_des    = 1/self.g * (acc[0]*np.sin(yaw) - acc[1]*np.cos(yaw))
        pitch_des   = 1/self.g * (acc[0]*np.cos(yaw) + acc[1]*np.sin(yaw))
        thrust_des  = self.mass * (acc[2] + self.g)
        
        roll_des = math.degrees(roll_des)
        pitch_des = math.degrees(pitch_des)
        thrust_des = self.thrust_to_pwn(thrust_des)
        
        # self.logger.info(f"roll_des {roll_des} pitch_des {pitch_des} thrust_des {thrust_des}")
        
        return roll_des, pitch_des, thrust_des
    
def main(args=None):
    
    rclpy.init(args=args)
    minimal_publisher = SimpleMPC()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()