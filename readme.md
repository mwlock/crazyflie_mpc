# crazyflie_mpc

ROS 2 package implementing a Model Predictive Control (MPC) system for precise and real-time control of the Crazyflie 2.1 quadcopter.

## Dependencies

Assumes you have a working installation of [Crazyswarm2](https://github.com/IMRCLab/crazyswarm2).

## Installation

1. Clone the repository into your ROS 2 workspace.

2. Install the required dependencies:

```bash
python -m pip install casadi
python -m pip install numpy
python -m pip install osqp
```

```bash
rosdep install --from-paths src --ignore-src -r -y
```
3. Build the package:

```bash
colcon build --symlink-install
```

## Run Example

Start the Crazyflie 2.1 simulation:

```bash
ros2 launch crazyflie launch.py backend:=sim
```

Start the MPC controller:

```bash
source install/setup.bash
ros2 launch crazyflie_mpc spiral_example.launch.py
```
