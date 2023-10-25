from setuptools import find_packages, setup

package_name = 'crazyflie_mpc'

data_files = []
data_files.append(('share/' + package_name + '/launch', ['launch/mpc_launch.py']))
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name, ['package.xml']))

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    author='Matthew Lock',
    author_email='mlwock@kth.se',
    maintainer='Matthew Lock',
    maintainer_email='mlwock@kth.se',
    description='ROS 2 package for MPC control of Crazyflie 2.1',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_node = crazyflie_mpc.mpc_node:main'
        ],
        'launch.frontend.launch_extension': ['launch_ros = launch_ros']
    }
)
