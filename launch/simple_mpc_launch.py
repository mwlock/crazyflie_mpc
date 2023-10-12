from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='crazyflie_mpc',
            namespace='crazyflie_1',
            executable='simple_mpc_node',
            name='simple_mpc_node'
        ),
    ])