import numpy as np

from math import sin
from math import cos
from math import sqrt

# Thrust mapping: Force --> PWM value (inverse mapping from system ID paper)
a2 = 2.130295 * 1e-11
a1 = 1.032633 * 1e-6
a0 = 5.484560 * 1e-4
cmd2T = lambda cmd: 4 * (a2 * (cmd) ** 2 + a1 * (cmd) + a0) # command to thrust
T2cmd = lambda T: (- (a1 / (2 * a2)) + sqrt(a1**2 / (4 * a2**2) - (a0 - (max(0, T) / 4)) / a2)) # thrust to command

def TRP2acc(self,TRP, psi, m):
    """
    Thrust, Roll, Pitch Command to desired accelerations
    where T is already in [N] and angles in [rad]
    """
    T = TRP[0]
    phi = TRP[1]
    theta = TRP[2]
    ax = self.g * (sin(psi) * phi + cos(psi) * theta)
    ay = self.g * (-cos(psi) * phi + sin(psi) * theta)
    az = T / m
    return np.array([ax, ay, az])

def acc2TRP(self, a, psi, m, thrust_offset = 0.0):
    """
    Acceleration to Thrust, Roll, Pitch Command where T is already in [N] and angles in [rad]

    Parameters
    ----------
    a : list
        List of accelerations in [x,y,z]
    psi : 
    """
    
    T = m * a[2] + cmd2T(thrust_offset)
    phi = (a[0] * sin(psi) - a[1] * cos(psi)) / self.g
    theta = (a[0] * cos(psi) + a[1] * sin(psi)) / self.g
    
    return np.array([T, phi, theta])