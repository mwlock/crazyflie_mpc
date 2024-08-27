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

from crazyflie_mpc.crazyflie_mpc import CrazyflieMPC
from geometry_msgs.msg import PoseStamped
import math
import numpy as np
import rclpy

def override(method):
    '''
    Psuedo decorator to indicate that a method is overriding a method in a superclass.
    '''
    def wrapper(*args, **kwargs):
        return method(*args, **kwargs)
    return wrapper

class MPCController(CrazyflieMPC):
    
    def __init__(self):
        
        super().__init__()
        
        # Set hover positions
        self.x_hover = 0.0
        self.y_hover = 0.0
        self.z_hover = 0.5
        
        # Initialize time trackers
        self.start_time = None
        self.last_log_time = None
        
        # Create a publisher for reference pose
        self.ref_pose_pub = self.create_publisher(PoseStamped, 'reference_pose', 10)
        
    @override
    def get_reference_trajectory(self):
        '''
        Get the reference trajectory for the MPC controller.
        '''
        
        # Calculate elapsed time in seconds
        if self.start_time is None:
            self.start_time = self.get_clock().now()
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.start_time).nanoseconds / 1e9
        
        x_ref = [ self.x_hover for i in range(self.N)]
        y_ref = [ self.y_hover for i in range(self.N)]
        z_ref = [ self.z_hover for i in range(self.N)]
        
        # Log with rate limit
        if self.last_log_time is None or (current_time - self.last_log_time).nanoseconds / 1e9 > 2.0:
            self.logger.info("Elapsed time: {:.2f}".format(elapsed_time))
            self.last_log_time = current_time
        
        if elapsed_time > 5.0 and elapsed_time < 30.0:
            
            time_references = [(elapsed_time + i*self.dt) for i in range(self.N)]
            x_ref = [(0.5*math.sin(t*2*math.pi*(1/5))) for t in time_references]
            y_ref = [(0.5*math.cos(t*2*math.pi*(1/5))) for t in time_references]
            z_ref = [(self.z_hover +0.25*math.cos(t*2*math.pi*(1/2))) for t in time_references]
        
        elif elapsed_time > 30.0 and elapsed_time < 32.0:
            
            x_ref = [ self.last_pose.pose.position.x for i in range(self.N)]
            y_ref = [ self.last_pose.pose.position.y for i in range(self.N)]
            z_ref = [ 0.05 for i in range(self.N)]
            
        elif elapsed_time > 32.0:
            
            self.logger.info("End of trajectory")
            self.destroy_node()

        ref_traj = np.array(list(([x, y, z, 0, 0, 0])
                            for x, y, z in zip(x_ref, y_ref, z_ref)))
        ref_traj_f = np.array([x_ref[-1], y_ref[-1], z_ref[-1], 0, 0, 0])
        
        ref_pose = PoseStamped()
        ref_pose.header.stamp = self.get_clock().now().to_msg()
        ref_pose.header.frame_id = 'world'
        ref_pose.pose.position.x = x_ref[0]
        ref_pose.pose.position.y = y_ref[0]
        ref_pose.pose.position.z = z_ref[0]
        self.ref_pose_pub.publish(ref_pose)
        
        return ref_traj, ref_traj_f

def main(args=None):

    rclpy.init(args=args)
    mpc_node = MPCController()
    rclpy.spin(mpc_node)
    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
