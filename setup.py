from setuptools import find_packages, setup

package_name = 'crazyflie_mpc'

data_files = []
data_files.append(('share/' + package_name + '/launch', ['launch/simple_mpc_launch.py']))

data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name, ['package.xml']))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    author='Matthew Lock',
    author_email='mlwock@kth.se',
    maintainer='Matthew Lock',
    maintainer_email='mlwock@kth.se',
    description='TODO: Package description',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_mpc_node = crazyflie_mpc.simple_mpc_node:main'
        ],
        'launch.frontend.launch_extension': ['launch_ros = launch_ros']
    }
)
