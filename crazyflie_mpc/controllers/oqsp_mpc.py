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

""" QP MPC Controller for Crazyflie 2.1 """


import osqp
import numpy as np
import scipy as sp
from scipy import sparse

import control
import casadi as ca

import matplotlib.pyplot as plt


class oqsp_MPC():

    def __init__(self,
            freq = 50,
            x_max = 5.0,
            y_max = 5.0,
            z_min = -10,
            z_max = 10,
            vx_max = 2.0,
            vy_max = 2.0,
            vz_max = 2.0,
            fx_max = 1.0,
            fy_max = 1.0,
            fz_max = 1.0,
            N = 100,
            logger = None
        ) -> None:
        
        self.logger = logger

        self.dt = 1/freq

        self.Ac = ca.DM.zeros(6, 6)
        self.Bc = ca.DM.zeros(6, 3)

        self.Ac[0, 3] = 1   # x
        self.Ac[1, 4] = 1   # y
        self.Ac[2, 5] = 1   # z

        # self.Ac[5, 5] = 1 # disturbance


        self.Bc[3, 0] = 1   # vx 
        self.Bc[4, 1] = 1   # vy
        self.Bc[5, 2] = 1   # vz


        continous_system     = control.StateSpace(self.Ac, self.Bc, ca.DM.eye(6), ca.DM.zeros(6, 3))
        self.discrete_system = continous_system.sample(self.dt)

        # print(f"continous_system:\n {continous_system}")
        # print(f"discrete_system:\n {discrete_system}")
        # print(f"discrete_system_disturbance:\n {discrete_system_disturbance}")


        self.Ad = sparse.csc_matrix(self.discrete_system.A)
        self.Bd = sparse.csc_matrix(self.discrete_system.B)

        [nx, nu] = self.Bd.shape

        self.nx = nx
        self.nu = nu

        print(f"self.nx: {self.nx}")
        print(f"nu: {nu}")

        # Constraints

        umin = np.array([-fx_max, -fy_max, -fz_max])
        umax = np.array([ fx_max,  fy_max,  fz_max])
        xmin = np.array([-x_max, -y_max, z_min, -vx_max, -vy_max, -vz_max])
        xmax = np.array([ x_max,  y_max, z_max,  vx_max,  vy_max,  vz_max])

        # Objective function
        Q = sparse.diags([1., 1., 1., 0., 0., 0.])
        QN = Q
        R = 0.1*sparse.eye(3)

        # Initial and reference states
        x0 = np.zeros(6)
        xr = np.array([0., 0., 0.5, 0., 0., 0.])
        xr_seq = np.kron(np.ones(N), -Q@xr)

        self.x0 = x0

        # Prediction horizon
        self.N = N

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                            sparse.kron(sparse.eye(N), R)], format='csc')

        # - linear objective
        q = np.hstack([xr_seq, -QN@xr, np.zeros(N*nu)])

        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(self.nx)) + sparse.kron(sparse.eye(N+1, k=-1), self.Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), self.Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*self.nx)])
        ueq = leq

        # - input and state constraints
        Aineq = sparse.eye((N+1)*self.nx + N*nu)
        lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])

        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        self.l = np.hstack([leq, lineq])
        self.u = np.hstack([ueq, uineq])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(
            P, q, A, self.l, self.u, 
            warm_start=True,
            verbose=False
            )
        
    def set_reference_trajectory(self, x_ref, x_ref_f):

        # Objective function
        Q = sparse.diags([1., 1., 1., 0., 0., 0.])
        QN = Q

        xr_seq = np.hstack( [np.kron(np.ones(1), -Q@xr) for xr in x_ref])
        q = np.hstack([xr_seq, -QN@x_ref_f, np.zeros(self.N*self.nu)])
        self.prob.update(q=q)
    
    def set_initial_state(self, x0):
        """ Set the initial state of the system. """

        self.l[:self.nx] = -x0
        self.u[:self.nx] = -x0
        self.prob.update(l=self.l, u=self.u)

    def solve(self, verbose=False):
        """ Solve the optimization problem. """
        
        res = self.prob.solve()

        if res.info.status != 'solved':
            self.logger.warn("!!!!!!!!!!!!!!!!!!!!!!! OSQP did not solve the problem !!!!!!!!!!!!!!!!!!!!!!!")
            raise ValueError('OSQP did not solve the problem!')

        ctrl = res.x[-self.N*self.nu:-(self.N-1)*self.nu]
        # x0 = self.Ad@self.x0 + self.Bd@ctrl
        x0 = 0

        if verbose:
            self.logger.info("MPC solve time: {}".format(res.info.solve_time))
            self.logger.info("MPC solve frequency: {}".format(1/res.info.solve_time))

        return x0, ctrl, res.info.solve_time

