from launch import LaunchDescription
from launch_ros.actions import Node

robot_name = "cf1"

def generate_launch_description():
    return LaunchDescription([
        Node(
            package     = 'crazyflie_mpc',
            namespace   = robot_name,
            executable  ='mpc_external_planner_node',
            name        ='mpc_external_planner_node',
            parameters  = 
                [{
                    "use_sim_time": False,
                }],
                
        ),
    ])