from crazyflie_mpc.controllers.simple_mpc import MPC

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from tf_transformations import euler_from_quaternion

import numpy as np
import casadi as ca
import control

class SimpleMPC(Node):

    def __init__(self):
        super().__init__('simple_mpc_controller')
        
        # Publishers and subscribers
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel_legacy', 10)
        self.last_odom = Odometry()
        
        self.logger = self.get_logger()
        
        self.init_model()   
        self.init_mpc_planner()    
        
        # Create timer to run MPC
        self.timer_period = self.dt
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def init_model(self):
        
        dimensions  = 3
        self.n      = dimensions*2 # number for double integrator
        self.m      = dimensions
        
        self.mass   = 0.05
        self.g      = 9.81
        self.dt     = 1/100
        
        self.Ac = ca.DM.zeros(self.n, self.n)
        self.Bc = ca.DM.zeros(self.n, self.m)
        
        self.Ac[0, 3] = 1
        self.Ac[1, 4] = 1
        self.Ac[2, 5] = 1
        
        self.Bc[3, 0] = 1/self.mass
        self.Bc[4, 1] = 1/self.mass
        self.Bc[5, 2] = 1/self.mass
        
        self.continous_system = control.StateSpace(self.Ac, self.Bc, ca.DM.eye(self.n), ca.DM.zeros(self.n, self.m))
        self.discrete_system = self.continous_system.sample(self.dt)
        
        x_lim = 3
        y_lim = 3
        z_lim = 3
        v_lim = 3
        u_lim = 1
        
        self.x_upper_bound = ca.DM([x_lim, y_lim, z_lim, v_lim, v_lim, v_lim])
        self.u_upper_bound = ca.DM([u_lim, u_lim, u_lim])
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
        
    def timer_callback(self):
        """Callback for the timer."""
        
        # Get current state
        x = np.array([
            self.last_odom.pose.pose.position.x,
            self.last_odom.pose.pose.position.y,
            self.last_odom.pose.pose.position.z,
            self.last_odom.twist.twist.linear.x,
            self.last_odom.twist.twist.linear.y,
            self.last_odom.twist.twist.linear.z,
        ])
        
        x_ref = np.array([0, 0, 1, 0, 0, 0])
        x_ref = np.tile(x_ref, (self.mpc_planner.N+1, 1)).T
        
        self.mpc_planner.set_initial_state(x)
        self.mpc_planner.set_reference_trajectory(x_ref)
        x,u = self.mpc_planner.solve(verbose=False)
        
        self.logger.info(f"u {u[:,0]}")
        
        roll_des, pitch_des, thrust_des = self.acceleration_to_drone_command(u[:,0])
        
        twist  = Twist()
        twist.linear.x = pitch_des
        twist.linear.y = roll_des
        twist.angular.z = 0
        twist.linear.z = thrust_des
        
        self.logger.info(f" roll_des {roll_des} pitch_des {pitch_des} thrust_des {thrust_des}")
        
        self.cmd_vel_pub.publish(twist)
        
        # self.logger.info(f"z {x[2:]}")
        # self.logger.info(f"z_dot {x[5:]}")
        # self.logger.info(f"u {u[2:]}")
        
        # Extract the first input
        
    def thrust_to_pwn(self, thrust):
        
        # F/4 = 2.130295e-11 * PWM^2 + 1.032633e-6 * PWM + 5.484560e-4
        # Solve quadratic equation for PWM with F = thrust
        a = 2.130295e-11
        b = 1.032633e-6
        c = 5.484560e-4 - thrust/4
        
        PWM = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        print(f"thrust {thrust} PWM {PWM}")
        
        return PWM
        
        
        
    def acceleration_to_drone_command(self, acc):
        """Convert acceleration to drone command."""
        
        quaternion  = self.last_odom.pose.pose.orientation
        euler = euler_from_quaternion([ quaternion.x, quaternion.y, quaternion.z, quaternion.w])

        roll    = euler[0]
        pitch   = euler[1]
        yaw     = euler[2]
        
        roll_des    = 1/self.g * (acc[0]*np.sin(yaw) - acc[1]*np.cos(yaw))
        pitch_des   = 1/self.g * (acc[0]*np.cos(yaw) + acc[1]*np.sin(yaw))
        thrust_des  = self.mass * (acc[2] + self.g)
        
        return roll_des, pitch_des, thrust_des

def main(args=None):
    
    rclpy.init(args=args)
    minimal_publisher = SimpleMPC()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()