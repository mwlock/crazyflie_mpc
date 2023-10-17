from crazyflie_mpc.controllers.simple_mpc import MPC

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped

from nav_msgs.msg import Odometry
from nav_msgs.msg import Path

from tf_transformations import euler_from_quaternion

import numpy as np
import casadi as ca
import control

import math
import time

class SimpleMPC(Node):

    def __init__(self):
        super().__init__('simple_mpc_controller')
        
        # Publishers and subscribers
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, 'pose', self.pose_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel_legacy', 1)
        self.path_pub = self.create_publisher(Path, 'mpc_controller/path', 10)
        
        self.last_odom = Odometry()
        self.last_pose = PoseStamped()
        self.x0,    self.y0,    self.z0     = 0, 0, 0
        self.xv0,   self.yv0,   self.zv0    = 0, 0, 0
        self.quaternion_x, self.quaternion_y, self.quaternion_z, self.quaternion_w = 0, 0, 0, 0
        
        self.logger = self.get_logger()
        
        self.logger.info("Initializing MPC controller.")
        self.init_model()   
        self.init_mpc_planner()    
        self.logger.info("MPC controller initialized.")
        
        # Send zeros to unlock the controller
        self.unlocked_counter = 0
        
        # Create timer to run MPC
        self.timer_period = self.dt
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.logger.info("Timer created.")
        
    def init_model(self):
        
        dimensions  = 3
        self.n      = dimensions*2 # number for double integrator
        self.m      = dimensions
        
        # self.mass   = 0.045
        self.mass   = 0.034
        self.g      = 9.81
        self.dt     = 1/20
        
        self.Ac = ca.DM.zeros(self.n, self.n)
        self.Bc = ca.DM.zeros(self.n, self.m)
        
        self.Ac[0, 3] = 1
        self.Ac[1, 4] = 1
        self.Ac[2, 5] = 1
        
        # self.Bc[3, 0] = 1/self.mass
        # self.Bc[4, 1] = 1/self.mass
        # self.Bc[5, 2] = 1/self.mass
        
        self.Bc[3, 0] = 1
        self.Bc[4, 1] = 1
        self.Bc[5, 2] = 1
        
        self.continous_system = control.StateSpace(self.Ac, self.Bc, ca.DM.eye(self.n), ca.DM.zeros(self.n, self.m))
        self.discrete_system = self.continous_system.sample(self.dt)
        
        x_lim = 5
        y_lim = 5
        z_lim = 3
        v_lim = 1
        u_lim_1 = 0.5
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
            N=7,
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
        self.last_pose = msg
        
        self.x0 = msg.pose.position.x
        self.y0 = msg.pose.position.y
        self.z0 = msg.pose.position.z
        
        dt = (msg.header.stamp.sec - self.last_pose.header.stamp.sec)
        if dt != 0:
            self.xv0 = (msg.pose.position.x - self.last_pose.pose.position.x)/dt
            self.yv0 = (msg.pose.position.y - self.last_pose.pose.position.y)/dt
            self.zv0 = (msg.pose.position.z - self.last_pose.pose.position.z)/dt
            
        self.quaternion_x = msg.pose.orientation.x
        self.quaternion_y = msg.pose.orientation.y
        self.quaternion_z = msg.pose.orientation.z
        self.quaternion_w = msg.pose.orientation.w
        
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
        
        x_ref = np.array([0.0, 0.0, 0.5])
        x_ref = np.tile(x_ref, (self.mpc_planner.N+1, 1)).T
        
        self.mpc_planner.set_initial_state(x)
        self.mpc_planner.set_reference_trajectory(x_ref)
        x_pred,u = self.mpc_planner.solve(verbose=True)
        
        self.publish_predicted_path(x_pred)
        
        # self.logger.info(f"u {u[:,0]}")
        
        roll_des, pitch_des, thrust_des = self.acceleration_to_drone_command(u[:,0])
        
        twist  = Twist()
        twist.linear.x = pitch_des
        twist.linear.y = roll_des
        twist.angular.z = 0.0
        twist.linear.z = thrust_des
        
        # self.logger.info(f" roll_des {roll_des} pitch_des {pitch_des} thrust_des {thrust_des}")
        if self.unlocked_counter < 50:
            self.emergency_stop()
            self.unlocked_counter +=1
            self.logger.info(f"Unlocked counter {self.unlocked_counter}")
        else:
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
        twist.linear.z = 60000.0
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
        
        self.logger.info(f"roll_des {roll_des} pitch_des {pitch_des} thrust_des {thrust_des}")
        
        return roll_des, pitch_des, thrust_des
    
def main(args=None):
    
    rclpy.init(args=args)
    minimal_publisher = SimpleMPC()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()