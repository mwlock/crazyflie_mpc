from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration

robot_name = "cf1"

def generate_launch_description():
    
    # ========================================
    # Launch arguments
    # ========================================
    
    arg_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Whether to use simulated time'
    )

    # ========================================
    # Nodes
    # ========================================
    
    mpc_node = Node(
        package     = 'crazyflie_mpc',
        namespace   = robot_name,
        executable  ='spiral_example',
        name        ='spiral_example_node',
        parameters  = [
            {
                "use_sim_time": LaunchConfiguration('use_sim_time'),
            }],
    )
    
    # ========================================
    # Launch description
    # ========================================
    
    launch_arguments = [arg_use_sim_time]
    launch_nodes = [mpc_node]
    
    return LaunchDescription(launch_arguments + launch_nodes)
    